import torch
from torch.utils.data import DataLoader
import torch.utils.data as data_utils
from dataset import PC_Dataset
from utils import calculate_metrics_and_loss, save_metrics
import hyperparameters
from label_map import label_map_Classifier, label_map_OD
import argparse
from utils import adjust_learning_rate
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import average_precision_score, jaccard_score
from tqdm import tqdm
from hyperparameters import *
import datetime
from torch.utils.tensorboard import SummaryWriter
import csv
import os
import shutil
import time 


########################### ARGUMENTS ################################

parser = argparse.ArgumentParser(description='Model training params') # Parsing command-line arguments

## arguments
parser.add_argument('--object_detector', type=str, choices=['yes', 'no'], required=True, 
                    help='use object detector label map if "yes", otherwise use classifier label map')
parser.add_argument('--data_folder', default=r'JSON_folder', type=str, help='folder with data files')
parser.add_argument('--training_output_file', type=str, help='file to save training output')
parser.add_argument('--save_dir', default=r'Checkpoints', type=str, help='folder to save checkpoints')

# Parse arguments
args = parser.parse_args()


########################################### DATA PARAMETERS ####################################################

# object detector or classifier?
if args.object_detector == 'yes':
    label_map = label_map_OD  # use object detector label map
else:
    label_map = label_map_Classifier  # use classifier label map


n_classes = len(label_map)  # number of different types of objects

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
model = model.to(device)  # Assuming 2 classes (background + object)

############################################## HYPERPARAMETERS #####################################################

# Model params 
# assign optimiser in initialization function according to which model, optimizer, and loss function are being used

# Model params
model = fasterrcnn_resnet50_fpn(pretrained=False,
                                num_classes=2)  # Assuming 2 classes (background + object)
chosen_optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
chosen_loss_fn = nn.BCELoss() # Assuming a binary classification task
                # Assuming a classification task: loss_fn = nn.CrossEntropyLoss(




########################################## Dataloaders #####################################################

# Custom dataloaders
data_folder = args.data_folder

# training dataset and dataloader
train_dataset = PC_Dataset(data_folder,
                            split='train',
                            keep_difficult=False)  # Assuming we don't want to keep difficult examples
train_loader = torch.utils.data.DataLoader(train_dataset, 
                                           batch_size=batch_size, shuffle=True,
                                            collate_fn=train_dataset.collate_fn,  # custom collate function
                                            num_workers=workers,
                                            pin_memory=True)  # note that we're passing the collate function here

# validation dataset and dataloader
validation_dataset = PC_Dataset(data_folder,
                                    split='val',
                                    keep_difficult=False)  # Assuming we don't want to keep difficult examples
validation_loader = torch.utils.data.DataLoader(validation_dataset,
                                                batch_size=batch_size, shuffle=False,
                                                collate_fn=validation_dataset.collate_fn,  # custom collate function
                                                num_workers=workers,
                                                pin_memory=True)  # note that we're passing the collate function here

# test dataset and dataloader
test_dataset = PC_Dataset(data_folder,
                            split='test',
                            keep_difficult=False)  # Assuming we don't want to keep difficult examples
test_loader = torch.utils.data.DataLoader(test_dataset,
                                           batch_size=batch_size, shuffle=False,
                                            collate_fn=test_dataset.collate_fn, # custom collate function
                                            num_workers=workers,
                                            pin_memory=True)  # note that we're passing the collate function here




############################################### TRAINING #####################################################

def initialize_training():
    """
    Initializes the training process by setting up the starting epoch, optimizer, and loss function.
    If no checkpoint is provided, the function initializes training from scratch with default settings.
    If a checkpoint is provided, it loads the training state (epoch, optimizer, and loss function) 
    from the checkpoint file.
    Returns:
        tuple: A tuple containing:
            - start_epoch (int): The starting epoch for training.
            - optimizer (torch.optim.Optimizer): The optimizer for training.
            - loss_fn (torch.nn.Module): The loss function for training.
    Args:
        checkpoint (str or None): Path to the checkpoint file. If None, training starts from scratch.
    Warnings:
        - Ensure that the optimizer and loss function are appropriate for your specific task 
          (e.g., classification, object detection, etc.). Modify them as needed for your use case.
        - The checkpoint file must contain the keys 'epoch', 'optimizer', and 'loss_fn' for proper loading.
    """

    if checkpoint is None:
        start_epoch = 0
        optimizer = chosen_optimizer 
        # why optimizer = optimizer does not work, when optimizer is defined earlier in train_custom or in hyperparameters.py (which is imported)?
        loss_fn = chosen_loss_fn
        # why loss_fn = loss_fn does not work, when loss_fn is defined earlier in train_custom or in hyperparameters.py (which is imported)?

        print('\nNo checkpoint provided. Starting training from scratch.\n')
        print(f"Start epoch: {start_epoch}")
        print(f"Optimizer: {optimizer}")
        print(f"Loss function: {loss_fn}")

        return start_epoch, optimizer, loss_fn

    if checkpoint is not None:
        checkpoint_data = torch.load(checkpoint)
        print(f'\nLoaded checkpoint from epoch {checkpoint_data["epoch"] + 1}.\n')
        print(f"Optimizer: {checkpoint_data['optimizer']}")
        print(f"Loss function: {checkpoint_data['loss_fn']}")
        start_epoch = checkpoint_data['epoch'] + 1
        optimizer = checkpoint_data['optimizer']
        loss_fn = checkpoint_data['loss_fn']
        return start_epoch, optimizer, loss_fn


def train_one_epoch(epoch,total_epochs,
                    writer, optimizer, loss_fn):
    """
    Trains the model for one epoch.
    Args:
        epoch (int): The current epoch number.
        total_epochs (int): The total number of epochs.
        writer (torch.utils.tensorboard.SummaryWriter): TensorBoard writer for logging metrics.
        optimizer (torch.optim.Optimizer): Optimizer used to update model parameters.
        loss_fn (callable): Loss function to compute the loss between predictions and labels.
    Returns:
        None
    """

    model.train()
    running_loss = 0.0

# Batch Structure:
# - The custom collate_fn for object detection returns a batch as (images, boxes, labels, difficulties),
#   not just (inputs, labels). Ignoring bounding boxes and difficulties (as in the simple classification loop)
#   will break object detection training.
#
# Object Detection Models:
# - Models like Faster R-CNN (torchvision) require both images and a targets list (one dict per image,
#   each with 'boxes' and 'labels') during training. The correct code prepares and passes this structure.
#   The simple classification loop does not, leading to errors like:
#   AssertionError: targets should not be none when in training mode.
#
# Device Handling:
# - The correct code ensures all tensors (images, boxes, labels) are moved to the correct device (CPU or GPU).
#   The simple loop only moves inputs and labels, which is not enough for detection tasks.
#
# Classification vs Detection:
# - The correct code handles both object detection (object_detector == 'yes') and classification
#   (object_detector == 'no') by branching appropriately. The simple loop only works for classification.
#
# Summary:
# - The correct code unpacks the batch according to the custom collate function, prepares the data in the
#   format required by object detection models, and handles device placement for all relevant tensors.
#   The simple classification loop is only suitable for basic classification tasks and will not work for object detection.

    for i, (images, boxes, labels, _) in enumerate(
        tqdm(train_loader, 
             desc=f'Epoch {epoch}/{epoch_num}', 
             unit='batch')
            ):
        
        images = images.to(device) # Move data to device

        if args.object_detector == 'yes':  # Torchvision models expect targets to be a list of dicts with 'boxes' and 'labels' keys
            filtered_images = []
            filtered_targets = []

            for img, b, l in zip(images, boxes, labels):
                if b.shape[0] > 0: 
                    filtered_images.append(img)
                    filtered_targets.append({
                        'boxes': b.to(device),
                        'labels': l.to(device)
                    })
            if len(filtered_images) == 0:
                continue  # skip this batch if no images have boxes

            # if want to include images with no boxes, we must find a way to convert empty boxes to an appropate format
            # but at the moment, we are skipping images with no boxes
            # because i do not know how to handle empty boxes in the targets dict.... :( 

            images_batch = torch.stack(filtered_images).to(device)
            optimizer.zero_grad() #clear gradients 
            loss_dict = model(images_batch, filtered_targets) # For torchvision models, loss_dict is a dict of losses
            loss = sum(loss for loss in loss_dict.values())
            print(loss)
    
        else: #Simple classification loop
            if isinstance(labels, (list, tuple)): #handles case where labels is a list of tensors (multi-label) and also when labels is a single tensor 
                labels = [l.to(device) for l in labels]
            else:
                labels = labels.to(device)
            optimizer.zero_grad() #clear gradients
            outputs = model(images) #forward pass
            loss = loss_fn(outputs, labels) #loss computation (separately unlike in the object detection case)

        loss.backward() #backward pass
        optimizer.step() #update weights
        running_loss += loss.item()  # accumulates the loss for the current batch

        if i % 1000 == 999:
            avg_loss = running_loss / 1000
            print(f'  batch {i + 1} loss: {avg_loss}')
            writer.add_scalar('Loss/train', avg_loss, epoch * len(train_loader) + i + 1)
            running_loss = 0.0


## From DLWPyTorch example ###############
# import torch 
# import torch.nn as nn  

# train_loader = torch.utils.data.DataLoader(cifar2, batch_size=64, shuffle=True)  
# model = nn.Sequential( nn.Linear(3072, 512), 
                        # nn.Tanh(), 
                        # nn.Linear(512, 2), 
                        # nn.LogSoftmax(dim=1))  

#learning_rate = 1e-2  

# optimizer = optim.SGD(model.parameters(), lr=learning_rate)  
# loss_fn = nn.NLLLoss()  
# n_epochs = 100  

# for epoch in range(n_epochs): 
#    for imgs, labels in train_loader:  DATASET  24, 13, 18, 7  10, 4, 11, 2  =4  =  DATA LOADER  Figure 7.14 A data loader dispensing minibatches by using a dataset to sample individual data items 186 CHAPTER 7 Telling birds from airplanes: Learning from images  batch_size = imgs.shape[0] outputs = model(imgs.view(batch_size, -1)) loss = loss_fn(outputs, labels)  optimizer.zero_grad() loss.backward() optimizer.step()  print("Epoch: %d, Loss: %f" % (epoch, float(loss)))






def main():
    """
    Main function to initialize and execute the training process for a machine learning model.
    The function performs the following steps:
    1. Initializes training parameters, including the starting epoch, optimizer, and loss function.
    2. Sets up a SummaryWriter for logging training metrics.
    3. Iterates through the specified number of epochs, adjusting the learning rate at predefined epochs.
    4. Trains the model for one epoch and calculates training and validation metrics.
    5. Saves the metrics to a CSV file and logs them using the SummaryWriter.
    6. Saves the trained model's state dictionary to a file.
    Variables:
    - start_epoch: The epoch to start training from.
    - optimizer: The optimizer used for training.
    - loss_fn: The loss function used for training.
    - training_output_file: The name of the output file for training logs.
    - train_loader: DataLoader for the training dataset.
    - validation_loader: DataLoader for the validation dataset.
    - total_epochs: Total number of epochs for training.
    - decay_lr_at_epoch: List of epochs at which to decay the learning rate.
    - decay_lr_to: The factor to decay the learning rate by.
    Outputs:
    - Logs training and validation metrics to a file and SummaryWriter.
    - Saves the trained model's state dictionary as 'model.pth'.
    """

    start_epoch, optimizer, loss_fn = initialize_training()

    training_output_file = args.training_output_file
    writer = SummaryWriter(f'5_Results/{training_output_file}')
    #train_loader = train_loader
    #validation_loader = validation_loader

    total_epochs = epoch_num
    decay_lr_at_epoch = decay_lr_at_epochs

    for epoch in range(start_epoch, total_epochs):
        if epoch in decay_lr_at_epoch:
            adjust_learning_rate(optimizer, decay_lr_to)

        epoch = epoch + 1
        train_one_epoch(epoch, total_epochs, writer, optimizer, loss_fn)
        train_metrics = calculate_metrics_and_loss(train_loader, loss_fn)
        val_metrics = calculate_metrics_and_loss(validation_loader, loss_fn)
        save_metrics(epoch, total_epochs, 
                     writer, training_output_file,
                     train_metrics, val_metrics) #include metrics to CSV file
        
    torch.save(model.state_dict(), 'model.pth')
    print("Model saved as model.pth")


def train():
    """
    Main function to train the model.
    """
    main()
    print("Training completed.")
    print("Model saved as model.pth")

if __name__ == '__main__':
    train() #run the training function when the module is run as a script




