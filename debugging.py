
from os import write
from dataset import PC_Dataset
from utils import separate_preds_targets
import torch
from hyperparameters import *
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import torch
import numpy as np
from types import SimpleNamespace
from utils import calculate_metrics
import csv
from sklearn.metrics import ConfusionMatrixDisplay
""
""" 
data_folder = r'3_TrainingData\20250513_Augmented\Split'  # Folder containing the training data
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 


# training dataset and dataloader
train_dataset = PC_Dataset(data_folder,
                            split='train',
                            keep_difficult=False)  # Assuming we don't want to keep difficult examples
train_loader = torch.utils.data.DataLoader(train_dataset, 
                                           batch_size=batch_size, shuffle=True,
                                            collate_fn=train_dataset.collate_fn,  # custom collate function
                                            num_workers=workers,
                                            pin_memory=True)  # note that we're passing the collate function here



# Get a batch from your DataLoader
batch = next(iter(train_loader))
images, boxes, labels, image_files = batch

# Use your function to get preds and targets
preds, targets, _, debug_info = separate_preds_targets(images, boxes, labels, image_files, device, debug=True)

# Example: create fake detections for each image in the batch
detections = []
for t in targets:
    detections.append({
        'boxes': t['boxes'],  # use ground truth boxes as fake predictions
        'scores': torch.ones(t['boxes'].shape[0]),  # fake confidence scores
        'labels': t['labels']
    })

preds_for_metric = []
for det in detections:
    preds_for_metric.append({
        'boxes': det['boxes'].detach().cpu(),
        'scores': det['scores'].detach().cpu(),
        'labels': det['labels'].detach().cpu()
    })



print("Sample detection:", detections[0])
print("Sample target:", targets[0])
print("Sample prediction:", preds[0])
print("Sample preds_for_metric:", preds_for_metric[0])  



# Dummy args
args = SimpleNamespace(model_type='object_detector')

# Dummy label_map for JaccardIndex
label_map = {0: 'background', 1: 'object'}

# Dummy ground truth and predicted labels (for classification metrics)
gt_labels_cat = np.array([1, 0, 1, 1])
pred_labels_cat = np.array([1, 0, 0, 1])

# Dummy targets and preds_for_metric (for detection metrics)
# targets = [
#    {'boxes': torch.tensor([[0, 0, 10, 10]]), 'labels': torch.tensor([1])},
#    {'boxes': torch.tensor([[5, 5, 15, 15]]), 'labels': torch.tensor([1])}
#]
#preds_for_metric = [
#    {'boxes': torch.tensor([[0, 0, 10, 10]]), 'scores': torch.tensor([0.9]), 'labels': torch.tensor([1])},
#    {'boxes': torch.tensor([[5, 5, 15, 15]]), 'scores': torch.tensor([0.8]), 'labels': torch.tensor([1])}
#]

# Call your function
metrics = calculate_metrics(
    args, gt_labels_cat, pred_labels_cat, preds_for_metric=preds_for_metric, targets=targets
)

print(metrics)


# Add missing imports for debugging
import matplotlib.pyplot as plt
from utils import log_metrics

from utils import manage_training_output_file
date_of_dataset_used = '20250513'  # Date of dataset used for training
results_folder = r'5_Results'  # Folder to save results
training_output_txt, training_output_csv = manage_training_output_file(
    results_folder=results_folder,
    date_of_dataset_used=date_of_dataset_used,
    augmented=True,
    model_type='detector')  # augmented_data if augmented dataset used



args.model_type = 'object_detector'  # Set model type for logging
args.training_output_txt = training_output_txt
args.training_output_csv = training_output_csv

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir='./')  # or any directory you want

log_metrics(args,
            epoch=1,
            i=0,
            train_loader=train_loader,
            writer=writer,  # now this is an instance, not the class
            metrics=metrics)
"""""



import matplotlib.pyplot as plt
import torch
import tifffile  # Use tifffile directly, not imread
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from torchvision.models.detection import fasterrcnn_resnet50_fpn

"""def visualize_detection_on_image(
    image_path,
    model_path,
    device="cpu",
    label_map=None,
    mask_threshold=0.7,
    threshold=0.5,
    figsize=(12, 12)
):
    device = torch.device(device)

    print(f"Reading image from: {image_path}")
    # Read image using tifffile
    image_np = tifffile.imread(image_path)
    print(f"Original image shape: {image_np.shape}")

    # Convert to torch tensor and preserve original shape
    image = torch.from_numpy(image_np)
    # If image is HWC, convert to CHW
    if image.ndim == 3 and image.shape[-1] <= 4:
        image = image.permute(2, 0, 1)
    elif image.ndim == 2:
        image = image.unsqueeze(0)
    print(f"Image tensor shape (CHW): {image.shape}")

    print(f"Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    num_classes = checkpoint.get('num_classes', 2)
    # Load the model architecture and weights from model_path
    if 'model' in checkpoint:
        model = checkpoint['model']
        model.to(device)
        print("Model loaded from provided model_path and moved to device.")
    else:
        model = fasterrcnn_resnet50_fpn(num_classes=num_classes)
        model.to(device)
        print("Model architecture created and moved to device.")

    # Print available metrics and loss if present in checkpoint
    if 'metrics' in checkpoint:
        print("Training metrics:")
        for k, v in checkpoint['metrics'].items():
            print(f"  {k}: {v}")
    if 'loss' in checkpoint:
        print(f"Final training loss: {checkpoint['loss']}")
    elif 'train_loss' in checkpoint:
        print(f"Final training loss: {checkpoint['train_loss']}")
    else:
        print("No loss value found in checkpoint.")

    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        # Use the original image shape for prediction
        x = image[:3, ...].float() / 255.0  # Normalize the image
        x = x.to(device)
        print("Running model inference...")
        predictions = model([x])
        pred = predictions[0]  # Get the first prediction
    print("Inference complete.")

    keep = pred["scores"] > threshold
    pred_boxes = pred["boxes"][keep]

    # Filter labels and scores using keep mask
    filtered_labels = pred["labels"][keep]
    filtered_scores = pred["scores"][keep]

    # For visualization, scale to uint8 and keep original shape
    vis_image = image.clone()
    if vis_image.dtype != torch.uint8:
        vis_image = (255.0 * (vis_image - vis_image.min()) / (vis_image.max() - vis_image.min())).to(torch.uint8)
    vis_image = vis_image[:3, ...]  # Only use first 3 channels for visualization
    print("Image normalized for visualization.")

    pred_labels = [f"cell: {score:.3f}" for score in filtered_scores]
    pred_boxes = pred_boxes.long()
    print(f"Number of predicted boxes: {len(pred_boxes)}")

    if "masks" in pred:
        masks = (pred["masks"] > mask_threshold).squeeze(1)
        print(f"Number of predicted masks: {masks.shape[0]}")
        output_image = draw_bounding_boxes(vis_image, pred_boxes, pred_labels, colors="red")
        output_image = draw_segmentation_masks(output_image, masks, alpha=0.5, colors="blue")
        # Show original and detected images side by side
        fig, axs = plt.subplots(1, 2, figsize=(figsize[0]*2, figsize[1]))
        axs[0].imshow(image_np if image_np.ndim == 2 else image_np[..., :3])
        axs[0].set_title("Original Image")
        axs[0].axis("off")
        axs[1].imshow(output_image.permute(1, 2, 0).cpu().numpy())
        axs[1].set_title("Detections")
        axs[1].axis("off")
        plt.tight_layout()
        plt.show()
    else:
        output_image = draw_bounding_boxes(vis_image, pred_boxes, pred_labels, colors="red")
        print("No masks found in prediction.")
        # Display original and detected images side by side
        fig, axs = plt.subplots(1, 2, figsize=(figsize[0]*2, figsize[1]))
        axs[0].imshow(image_np if image_np.ndim == 2 else image_np[..., :3])
        axs[0].set_title("Original Image")
        axs[0].axis("off")
        axs[1].imshow(output_image.permute(1, 2, 0).cpu().numpy())
        axs[1].set_title("Detections")
        axs[1].axis("off")
        print(f"Number of detections: {len(pred_boxes)}") #print(f"Number of detections: {len(pred['boxes'])}") returns the number of boxes in the prediction
                                                            # which is by default 100
        plt.tight_layout()
        plt.show()
    
    print("All scores:", pred["scores"].cpu().numpy())
    print(
        "Min score:", pred["scores"].min().item(),
        "Mean score:", pred["scores"].mean().item(),
        "Max score:", pred["scores"].max().item(),
        "25th percentile score:", torch.quantile(pred["scores"], 0.25).item(),
        "50th percentile score:", torch.quantile(pred["scores"], 0.5).item(),
        "75th percentile score:", torch.quantile(pred["scores"], 0.75).item()
    )
          
    
if __name__ == "__main__":
    visualize_detection_on_image(
        image_path=r'3_TrainingData\20250318_Augmented\Split\test\images\1.tif',
        model_path=r'5_Results/5_Results/training_results_20250513_Augmented.txt_model.pth',
        device="cpu",
        label_map=None,
        mask_threshold=0.7,
        threshold=0.5,
        figsize=(12, 12)
    )"""




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


# --- Training loop code (not inside functions) ---

# Initialize training
start_epoch, optimizer, loss_fn = initialize_training()

writer = SummaryWriter(f'5_Results/')
total_epochs = epoch_num
decay_lr_at_epoch = decay_lr_at_epochs

for epoch in range(start_epoch, total_epochs):
    if epoch in decay_lr_at_epoch:
        adjust_learning_rate(optimizer, decay_lr_to)

    # Train one epoch
    loss, gt_labels_cat, pred_labels_cat, preds_for_metric, targets = train_one_epoch(
        epoch, total_epochs, writer, optimizer, loss_fn
    )

    # Calculate and log metrics
    if args.model_type == 'object_classifier':
        metrics = calculate_metrics(args, gt_labels_cat, pred_labels_cat)
        log_metrics(args, writer, epoch, 0, train_loader, metrics, label_map=label_map)
    elif args.model_type == 'object_detector':
        metrics = calculate_metrics(
            args, gt_labels_cat, pred_labels_cat,
            loss=loss, preds_for_metric=preds_for_metric, targets=targets
        )
        log_metrics(args, writer, epoch, 0, train_loader, metrics)

    save_checkpoint(epoch, model, optimizer, data_folder, metrics)

torch.save(model.state_dict(), 'model.pth')
print("Training completed.")
print("Model saved as model.pth")
