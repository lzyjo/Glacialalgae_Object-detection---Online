from calendar import c
from turtle import up
import torch
from torch.utils.data import DataLoader
import torch.utils.data as data_utils
from dataset import PC_Dataset
from utils import AverageMeter, accuracy, calculate_metrics_and_loss, save_checkpoint, save_metrics
from utils import separate_preds_targets
import hyperparameters
from label_map import label_map_Classifier, label_map_OD
import argparse
from utils import adjust_learning_rate
from utils import calculate_metrics, log_metrics
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay #metrics for classification
from sklearn.metrics import average_precision_score, jaccard_score #metrics for object detection
from torchmetrics.detection.mean_ap import MeanAveragePrecision #metrics for object detection
from torchmetrics.detection.iou import IntersectionOverUnion #metrics for object detection
from torchmetrics import JaccardIndex #metrics for object detection
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
parser.add_argument('--model_type', type=str, choices=['object_detector', 'object_classifier'], required=True,
                    help='type of model to train: "object_detector" for object detection, "object_classifier" for classification')
parser.add_argument('--data_folder', default=r'JSON_folder', type=str, help='folder with data files')
parser.add_argument('--training_output_txt', type=str, help='file to save training output')
parser.add_argument('--training_output_csv', type=str, help='file to save training output in CSV format')
parser.add_argument('--save_dir', default=r'Checkpoints', type=str, help='folder to save checkpoints')

args = parser.parse_args() # Parse arguments



########################################### DATA PARAMETERS ####################################################

# object detector or classifier?
if args.model_type == 'object_detector':
    label_map = label_map_OD  # use object detector label map
if args.model_type == 'object_classifier':
    label_map = label_map_Classifier  # use classifier label map


n_classes = len(label_map)  # number of different types of objects

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
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


"""def train_one_epoch(epoch,total_epochs,
                    writer, optimizer, loss_fn):

    Trains the model for one epoch.
    Args:
        epoch (int): The current epoch number.
        total_epochs (int): The total number of epochs.
        writer (torch.utils.tensorboard.SummaryWriter): TensorBoard writer for logging metrics.
        optimizer (torch.optim.Optimizer): Optimizer used to update model parameters.
        loss_fn (callable): Loss function to compute the loss between predictions and labels.
    Returns:
        None

    model.train()
    running_loss = 0.0

    print_freq = 200  # Track loss every

    with tqdm(train_loader, desc=f'Epoch {epoch}/{total_epochs}', unit='batch') as tq:
        for i, batch in enumerate(tq):
            images, boxes, labels, image_files = batch

            images = images.to(device)  # Move data to device

            if args.model_type == 'object_detector':  # Torchvision models expect targets to be a list of dicts with 'boxes' and 'labels' keys
                preds, targets, _, debug_info = separate_preds_targets(
                    images, boxes, labels, image_files, device, 
                    debug=True
                )
                
                if debug_info: # check if debug_info is not empty
                    print(f"Debug info: {debug_info}")  # Print debug information if needed

                if len(preds) == 0:
                    continue  # skip this batch if no images have valid boxes

                images_batch = torch.stack(preds).to(device)
                optimizer.zero_grad()  # clear gradients 
                loss_dict = model(images_batch, targets)  # For torchvision models, loss_dict is a dict of losses
                loss = sum(loss for loss in loss_dict.values())

                # Get model outputs (detections) in eval mode for metrics
                model.eval()
                with torch.no_grad():
                    detections = model(images_batch)
                model.train()

                # Prepare predictions in the format required by torchmetrics
                preds_for_metric = []
                for det in detections:
                    preds_for_metric.append({
                        'boxes': det['boxes'].detach().cpu(),
                        'scores': det['scores'].detach().cpu(),
                        'labels': det['labels'].detach().cpu()
                    })

            else: #Simple classification loop
                if isinstance(labels, (list, tuple)): #handles case where labels is a list of tensors (multi-label) and also when labels is a single tensor 
                    labels = [l.to(device) for l in labels]
                else:
                    labels = labels.to(device)
                optimizer.zero_grad() #clear gradients
                outputs = model(images) #forward pass
                loss = loss_fn(outputs, labels) #loss computation (separately unlike in the object detection case)

                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            if (i + 1) % print_freq == 0: # Print loss every print_freq batches
                tq.set_postfix({'loss': loss.item()}) # update the progress bar with the current loss
                tq.write(f'Batch {i+1}, Loss: {loss.item()}') # Print loss for the current batch


        loss.backward() #backward pass
        optimizer.step() #update weights
        running_loss += loss.item()  # accumulates the loss for the current batch

        # Accumulate predictions and ground truths for epoch-level metrics
        if not hasattr(train_one_epoch, "epoch_gt_boxes"):
            epoch_gt_boxes = []
            epoch_gt_labels = []
            epoch_pred_boxes = []
            epoch_pred_labels = []
            epoch_pred_scores = []

        # Get model outputs (detections) in eval mode for metrics
        model.eval() # Set model to evaluation mode to disable dropout and batch normalization
        with torch.no_grad():
            detections = model(images_batch)
        model.train() # Set model back to training mode for the next iteration

        # Prepare ground truth and prediction lists for metric calculation
        gt_boxes = [t['boxes'].cpu() for t in targets] # Extract ground truth boxes from targets
        gt_labels = [t['labels'].cpu() for t in targets] # Extract ground truth labels from targets
                # Note: targets is a list of dicts, each dict contains 'boxes' and 'labels'
        pred_boxes = [d['boxes'].cpu() for d in detections] # Extract predicted boxes from detections
        pred_labels = [d['labels'].cpu() for d in detections] # Extract predicted labels from detections
        pred_scores = [d['scores'].cpu() for d in detections] # Extract predicted scores from detections

        # Extend epoch lists with current batch data
        epoch_gt_boxes.extend(gt_boxes)
        epoch_gt_labels.extend(gt_labels)
        epoch_pred_boxes.extend(pred_boxes)
        epoch_pred_labels.extend(pred_labels)
        epoch_pred_scores.extend(pred_scores)

        gt_labels = torch.cat(epoch_gt_labels) # Concatenate all ground truth labels for the epoch
        pred_labels = torch.cat(epoch_pred_labels) # Concatenate all predicted labels for the epoch

        
        # Calculate metrics for object detection or classification

        if args.model_type == 'object_classifier':  # If using a classifier
            # Calculate accuracy for classification
            accuracy_value = accuracy_score(
                gt_labels, # ground truth labels
                pred_labels # predicted labels
            )
            precision_value = precision_score(
                gt_labels,
                pred_labels,
                average='weighted'
            )
            recall_value = recall_score(
                gt_labels,
                pred_labels,
                average='weighted'
            )
            f1_score_value = f1_score( #fq score is mean of precision and recall
                gt_labels,
                pred_labels,
                average='weighted'
            )
            confusion_matrix_value = confusion_matrix(
                gt_labels,
                pred_labels,
            )

            # Log metrics to TensorBoard
            writer.add_scalar('Accuracy/train_batch', accuracy_value, epoch * len(train_loader) + i + 1)
            mean_ap = MeanAveragePrecision(
                iou_type='bbox',  # Use bounding box IoU
            )
            mean_ap.update(preds_for_metric, targets)  # Update MeanAveragePrecision with predictions and targets
            mean_ap_value = mean_ap.compute()
            print(f'F1 Score: {f1_score_value}')
            print(f'Precision: {precision_value}')
            print(f'Recall: {recall_value}')
            print(f'Confusion Matrix: {confusion_matrix_value}')
            confusion_matrix_display = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_value,
                                                              display_labels=label_map.keys())
            confusion_matrix_display.plot(cmap='Blues')

        if args.model_type == 'object_detector':  # If using an object detector
            # Calculate metrics for object detection
            mean_ap = MeanAveragePrecision(
                iou_type='bbox',  # Use bounding box IoU
            )
            mean_ap.update(preds, targets)  # Update MeanAveragePrecision with predictions and targets
            mean_ap_value = mean_ap.compute()

            jaccard_value = jaccard_score(
                gt_labels.numpy(),
                pred_labels.numpy(),
                average='weighted'
            )

            # Log metrics to TensorBoard
            writer.add_scalar('Mean Average Precision/train_batch', mean_ap_value['map'], epoch * len(train_loader) + i + 1)
            writer.add_scalar('Jaccard/train_batch', jaccard_value, epoch * len(train_loader) + i + 1)    
            print(f'Mean Average Precision: {mean_ap_value["map"]}')
            print(f'Jaccard Index: {jaccard_value}')

            print("Saving metrics to file...")
            try:
                # save metrics to txt file
                with open(f'5_Results/{args.training_output_txt}.txt', 'a') as f:
                    f.write(f'Epoch: {epoch}, Batch: {i + 1}, Mean Average Precision: {mean_ap_value.get("map", "N/A")}, Jaccard Index: {jaccard_value if "jaccard_value" in locals() else "N/A"}\n')

                # save metrics to CSV file
                with open(f'5_Results/{args.training_output_csv}.csv', 'a', newline='') as csvfile:
                    writer_csv = csv.writer(csvfile)
                    # Write header if file is empty
                    if csvfile.tell() == 0:
                        writer_csv.writerow(['Epoch', 'Batch', 'Mean Average Precision', 'Jaccard Index'])
                    # Write metrics for the current batch
                    writer_csv.writerow([
                        epoch,
                        i + 1,
                        mean_ap_value.get('map', 'N/A'),
                        jaccard_value if "jaccard_value" in locals() else "N/A"
                    ])
                # Save the model state dictionary
                torch.save(model.state_dict(), f'5_Results/{args.training_output_txt}_model.pth')
                print(f'Model state dictionary saved as {args.training_output_csv}_model.pth')
            except Exception as e:
                print(f"Error saving metrics or model: {e}")


        # Print status
        if i == len(train_loader) - 1: # If this is the last batch of the epoch
            print('Epoch: [{0}/{1}]\t' 
              'Average Loss: {loss:.4f}'.format(
                  epoch, total_epochs, loss=running_loss / len(train_loader)))"""


def train_one_epoch(epoch, total_epochs, writer, optimizer, loss_fn):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    print_freq = 200

    with tqdm(train_loader, desc=f'Epoch {epoch}/{total_epochs}', unit='batch') as tq:
        for i, batch in enumerate(tq):
            images, boxes, labels, image_files = batch
            images = images.to(device) #images = [img.to(device) for img in images]
                        # If images is a list of tensors: move each to device.
                        # If images is a batched tensor: convert to list, then move each to device.
                        # NOTE:
                        # The custom collate_fn currently stacks images into a single tensor of shape (N, 3, H, W),
                        # but returns boxes, labels, and difficulties as lists of tensors (one per image).
                        # Torchvision object detection models (like Faster R-CNN) expect a list of image tensors,
                        # not a single batched tensor. Passing a batched tensor instead of a list will cause errors.
                        # To fix: convert the images tensor back to a list before passing to the model:
                        #     images = [img.to(device) for img in images]

            if args.model_type == 'object_detector':
                preds, targets, _, debug_info = separate_preds_targets(
                    images, boxes, labels, image_files, device, debug=True
                )
                if debug_info:
                    print(f"Debug info: {debug_info}")
                if len(preds) == 0:
                    continue

                images_batch = torch.stack(preds).to(device)
                optimizer.zero_grad()
                loss_dict = model(images_batch, targets) # For torchvision models, outputs is a dict of losses
                #The types of loss are: loss_classifier, loss_box_reg, loss_objectness, loss_rpn_box_reg
                #Print(loss_dict) would return something like:
                # {'loss_classifier': tensor(0.0490, grad_fn=<NllLossBackward0>), 'loss_box_reg': tensor(0.0344, grad_fn=<DivBackward0>), 'loss_objectness': tensor(0.0069, grad_fn=<BinaryCrossEntropyWithLogitsBackward0>), 'loss_rpn_box_reg': tensor(0.0021, grad_fn=<DivBackward0>)} 
                # {'boxes': tensor([], size=(0, 4), grad_fn=<StackBackward0>), 'labels': tensor([], dtype=torch.int64), 'scores': tensor([], grad_fn=<IndexBackward0>)}
                loss = sum(loss for loss in loss_dict.values()) #custom loss function such as nn.CrossEntropyLoss() or nn.BCEWithLogitsLoss() cannot be used directly with torchvision models.
                                                                #to use a specific loss function
                                                                #modify the model’s source code or write your own detection head

            else:
                if isinstance(labels, (list, tuple)):
                    labels = [l.to(device) for l in labels]
                else:
                    labels = labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = loss_fn(outputs, labels)
                
            loss.backward() # backward pass
            optimizer.step() # update weights
            running_loss += loss.item() # accumulates the loss for the current batch

            model.eval()
            with torch.no_grad():
                detections = model(images_batch)
            model.train()

            preds_for_metric = [] # Prepare predictions in the format required by torchmetrics
            for det in detections:
                preds_for_metric.append({
                    'boxes': det['boxes'].detach().cpu(),
                    'scores': det['scores'].detach().cpu(),
                    'labels': det['labels'].detach().cpu()
                })

            if (i + 1) % print_freq == 0:
                tq.set_postfix({'loss': loss.item()})
                tq.write(f'Batch {i+1}, Loss: {loss.item()}')

        # Accumulate predictions and ground truths for epoch-level metrics
        if not hasattr(train_one_epoch, "epoch_gt_boxes"):
            epoch_gt_boxes = []
            epoch_gt_labels = []
            epoch_pred_boxes = []
            epoch_pred_labels = []
            epoch_pred_scores = []

        model.eval()
        with torch.no_grad():
            detections = model(images_batch)
        model.train()

        gt_boxes = [t['boxes'].cpu() for t in targets]
        gt_labels = [t['labels'].cpu() for t in targets]
        pred_boxes = [d['boxes'].cpu() for d in detections]
        pred_labels = [d['labels'].cpu() for d in detections]
        pred_scores = [d['scores'].cpu() for d in detections]

        epoch_gt_boxes.extend(gt_boxes)
        epoch_gt_labels.extend(gt_labels)
        epoch_pred_boxes.extend(pred_boxes)
        epoch_pred_labels.extend(pred_labels)
        epoch_pred_scores.extend(pred_scores)

        gt_labels_cat = torch.cat(epoch_gt_labels)
        pred_labels_cat = torch.cat(epoch_pred_labels)

    
        if i == len(train_loader) - 1:
            print('Epoch: [{0}/{1}]\tAverage Loss: {loss:.4f}'.format(
                epoch, total_epochs, loss=running_loss / len(train_loader)))

    if args.model_type == 'object_detector':
        return loss, epoch_gt_boxes, gt_labels_cat, pred_labels_cat, targets, i
    if args.model_type == 'object_classifier':
        return loss, gt_labels_cat, pred_labels_cat, epoch_pred_scores, epoch_gt_boxes


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

    writer = SummaryWriter(f'5_Results/')
    #train_loader = train_loader
    #validation_loader = validation_loader

    total_epochs = epoch_num
    decay_lr_at_epoch = decay_lr_at_epochs

    for epoch in range(start_epoch, total_epochs):
        if epoch in decay_lr_at_epoch:
            adjust_learning_rate(optimizer, decay_lr_to)

        epoch = epoch + 1
        loss, gt_labels_cat, pred_labels_cat, preds_for_metric, targets, i = train_one_epoch(epoch, total_epochs, writer, optimizer, loss_fn)

         # Calculate metrics
        if args.model_type == 'object_classifier':
            metrics = calculate_metrics(
                args, gt_labels_cat, pred_labels_cat
            )
            log_metrics(
                args, writer, epoch, i, train_loader,
                metrics, label_map=label_map
            )
        elif args.model_type == 'object_detector':
            metrics = calculate_metrics(
                args, gt_labels_cat, pred_labels_cat, 
                loss=loss, preds_for_metric=preds_for_metric, targets=targets
            )
            log_metrics(
                args, writer, epoch, i, train_loader,
                metrics
            )

        save_checkpoint(epoch, model, optimizer, data_folder, metrics)
        print("Model saved as model.pth")

        # train_metrics = calculate_metrics_and_loss(train_loader, loss_fn)
        # val_metrics = calculate_metrics_and_loss(validation_loader, loss_fn)
        # save_metrics(epoch, total_epochs, writer, training_output_file, train_metrics, val_metrics) #include metrics to CSV file

        print(f'Epoch {epoch}/{total_epochs} completed. Metrics logged and model saved.')


def train():
    """
    Main function to train the model.
    """
    main()
    print("Training completed.")
    print("Model saved as model.pth")

if __name__ == '__main__':
    train() #run the training function when the module is run as a script




