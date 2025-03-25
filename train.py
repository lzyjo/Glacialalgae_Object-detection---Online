import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from tqdm import tqdm
import matplotlib.pyplot as plt
from BCCD_model import SSD300, MultiBoxLoss
from utils import *
from dataset import GA_Dataset
from label_map import *
import argparse
from hyperparameters import *

# Parsing command-line arguments
parser = argparse.ArgumentParser(description='Model training')

## data_folder argument
parser.add_argument('--data_folder', default=r'JSON_folder', type=str, help='folder with data files')

## date_of_dataset_used argument
parser.add_argument('--date_of_dataset_used', required=True, type=str, help='date of the dataset used for training')

## save_dir argument
parser.add_argument('--save_dir', default=r'Checkpoints', type=str, help='folder to save checkpoints')

# Parse arguments
args = parser.parse_args()


# Data parameters
data_folder = args.data_folder
date_of_dataset_used = args.date_of_dataset_used
save_dir = args.save_dir
keep_difficult = True  # use objects considered difficult to detect?


# Model parameters
n_classes = len(label_map)  # number of different types of objects
device = torch.device("cpu")


cudnn.benchmark = True
def main():
    """
    Training.
    """
    # Initialize model or load checkpoint
    if checkpoint is None:
        start_epoch = 0
        model = SSD300(n_classes=n_classes)
        # Initialize the optimizer, with twice the default learning rate for biases, as in the original Caffe repo
        biases = list()
        not_biases = list()
        for param_name, param in model.named_parameters():
            if param.requires_grad:
                if param_name.endswith('.bias'):
                    biases.append(param)
                else:
                    not_biases.append(param)
        optimizer = torch.optim.SGD(params=[{'params': biases, 'lr': 2 * lr}, {'params': not_biases}],
                                    lr=lr, momentum=momentum, weight_decay=weight_decay)

    else:
        checkpoint_data = torch.load(checkpoint)
        start_epoch = checkpoint_data['epoch'] + 1
        print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)
        model = checkpoint_data['model']
        optimizer = checkpoint_data['optimizer']

    # Move to default device
    model = model.to(device)
    criterion = MultiBoxLoss(priors_cxcy=model.priors_cxcy).to(device)

    # Custom dataloaders
    train_dataset = GA_Dataset(data_folder,
                                split='train',
                                keep_difficult=False)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               collate_fn=train_dataset.collate_fn, num_workers=workers,
                                               pin_memory=True)  # note that we're passing the collate function here

    epochs = epoch
    decay_lr_at_epochs = decay_lr_at_epochs

    # Epochs
    for epoch in range(start_epoch, epochs):

        # Decay learning rate at particular epochs
        if epoch in decay_lr_at_epochs:
            adjust_learning_rate(optimizer, decay_lr_to)

        # One epoch's training
        train(train_loader=train_loader,
              model=model,
              criterion=criterion,
              optimizer=optimizer,
              epoch=epoch,
              epochs=epochs)

        # Save checkpoint
        save_checkpoint(epoch, model, optimizer, date_of_dataset_used, save_dir)


def train(train_loader, model, criterion, optimizer, epoch, epochs):
    """
    One epoch's training.

    :param train_loader: DataLoader for training data
    :param model: model
    :param criterion: MultiBox loss
    :param optimizer: optimizer
    :param epoch: epoch number
    """
    model.train()  # training mode enables dropout

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss

    start = time.time()

    # Batches
    for i, (images, boxes, labels, _) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}', unit='batch')):
        data_time.update(time.time() - start)

        # Move to default device
        images = images.to(device)  # (batch_size (N), 3, 300, 300)
        boxes = [b.to(device) for b in boxes]
        labels = [l.to(device) for l in labels]

        # Forward prop.
        predicted_locs, predicted_scores = model(images)  # (N, 8732, 4), (N, 8732, n_classes)

        # Loss
        loss = criterion(predicted_locs, predicted_scores, boxes, labels)  # scalar

        # Backward prop.
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients, if necessary
        if grad_clip is not None:
            clip_gradient(optimizer, grad_clip)

        # Update model
        optimizer.step()

        losses.update(loss.item(), images.size(0))
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, i, len(train_loader),
                                                                  batch_time=batch_time,
                                                                  data_time=data_time, loss=losses))

        # Save checkpoint
        if i % checkpoint_freq == 0:
            save_checkpoint(epoch, model, optimizer, date_of_dataset_used, save_dir)

    del predicted_locs, predicted_scores, images, boxes, labels  # free some memory since their histories may be stored

    return losses.avg


if __name__ == '__main__':
    main()
