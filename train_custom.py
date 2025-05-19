import torch
from torch.utils.data import DataLoader
from dataset import GA_Dataset, collate_fn
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




# Custom dataloaders
data_folder = args.data_folder

# training dataset and dataloader
train_dataset = GA_Dataset(data_folder,
                            split='train',
                            keep_difficult=False)
train_loader = torch.utils.data.DataLoader(train_dataset, 
                                           batch_size=batch_size, shuffle=True,
                                            collate_fn=train_dataset.collate_fn,  # custom collate function
                                            num_workers=workers,
                                            pin_memory=True)  # note that we're passing the collate function here

# validation dataset and dataloader
validation_dataset = GA_Dataset(data_folder,
                                    split='validation',
                                    keep_difficult=False) 
validation_loader = torch.utils.data.DataLoader(validation_dataset,
                                                batch_size=batch_size, shuffle=False,
                                                collate_fn=validation_dataset.collate_fn,  # custom collate function
                                                num_workers=workers,
                                                pin_memory=True)  # note that we're passing the collate function here

# test dataset and dataloader
test_dataset = GA_Dataset(data_folder,
                            split='test',
                            keep_difficult=False)
test_loader = torch.utils.data.DataLoader(test_dataset,
                                           batch_size=batch_size, shuffle=False,
                                            collate_fn=test_dataset.collate_fn, # custom collate function
                                            num_workers=workers,
                                            pin_memory=True)  # note that we're passing the collate function here



########################################### DATALOADER CHECKING ####################################################

def print_batch_structure(loader, loader_name="train_loader"):
    """
    Print the type and shape of each item in a batch from the given DataLoader.
    Args:
        loader: DataLoader to sample a batch from.
        loader_name: Name of the loader (for display purposes).
    """
    print(f"\nSample batch from {loader_name} (collate_fn output):")
    batch = next(iter(loader))
    if isinstance(batch, (tuple, list)):
        for idx, item in enumerate(batch):
            if idx == 0:
                print(f"  Batch item {idx}: type={type(item)}, shape={getattr(item, 'shape', 'N/A')}")
            else:
                if isinstance(item, list):
                    for j, subitem in enumerate(item):
                        print(f"  Batch item {idx}:")
                        print(f"    Subitem {j}: type={type(subitem)}, shape={getattr(subitem, 'shape', 'N/A')}")
                        print('-' * 50)
                else:
                    print(f"  Batch item {idx}: type={type(item)}, shape={getattr(item, 'shape', 'N/A')}")
    else:
        print(f"Type: {type(batch)}, shape={getattr(batch, 'shape', 'N/A')}")
    # Note: This code prints the type and shape of each batch item. 
    # If a batch item is a list (e.g., a list of tensors), it iterates through the list and prints the shape of each element inside.
    # This avoids trying to access .shape on a list directly (which would be 'N/A'), and instead shows the shapes of the actual tensors inside the list.




################################################ INITIALIZE TRAINING ################################################

def check_model_trained(checkpoint_dir='6_Checkpoints',
                        date_of_dataset_used='20250318', 
                        augmented=False,):
    """
    Check if a model has already been trained by verifying the presence of a checkpoint file.

    Args:
        date_of_dataset_used (str): The date of the dataset used.
        augmented (bool): Whether the dataset is augmented. Defaults to False.
        checkpoint_dir (str): Directory where checkpoints are stored. Defaults to '6_Checkpoints'.

    """
    # Get the list of all files in the checkpoint directory
    if augmented: # Check if the dataset is augmented
        date_of_dataset_used += '_Augmented' # Append '_Augmented' to the date string if the dataset is augmented

    model_files = [f for f in os.listdir(checkpoint_dir) if date_of_dataset_used in f] # Filter the model files based on the updated date_of_dataset_used
        
    if model_files:
        model_path = os.path.join(checkpoint_dir, model_files[0])
        print(f'Model may be present. Please check: {model_path}')
        print(f'Model checkpoint found at: {model_path}')
    else:
        print(f'No model checkpoint found for date: {date_of_dataset_used} in {checkpoint_dir}')

if __name__ == '__main__':
    check_model_trained(checkpoint_dir='6_Checkpoints',
                        date_of_dataset_used='20250318', 
                        augmented=False)



def manage_training_output_file(results_folder, date_of_dataset_used,
                                augmented=False,
                                object_detector=False):
    """
    Manage the training output file by creating or appending to it, and write training parameters if not already present.

    Args:
        results_folder (str): Folder to store results.
        date_of_dataset_used (str): Date of the dataset used for training.
        augmented (bool): Whether the dataset is augmented.

    Returns:
        str: Path to the training output file.
    """
    if augmented:
        date_of_dataset_used += '_Augmented'
        augmentations_file = os.path.join('3_TrainingData', date_of_dataset_used, 'augmentations_used.txt')
        if not os.path.exists(augmentations_file):
            print(f"Error: The file {augmentations_file} does not exist.")
            return

    training_output_file = os.path.join(results_folder, f'training_results_{date_of_dataset_used}.txt')
    os.makedirs(results_folder, exist_ok=True)  # Ensure the results folder exists

    if object_detector:
        object_detector = 'Object Detector'
    else:
        object_detector = 'Classifier'
    
    if augmented:
        with open(augmentations_file, 'r') as f:
            augmentation = [line.strip() for line in f if line.strip()]
    else:
        augmentation = None

    params = [
        f'Checkpoint: {checkpoint}',
        f'Batch Size: {batch_size}',
        f'Iterations: {iterations}',
        f'Workers: {workers}',
        f'Print Frequency: {print_freq}',
        f'Learning Rate: {lr}',
        f'Decay Learning Rate At: {decay_lr_at}',
        f'Decay Learning Rate To: {decay_lr_to}',
        f'Momentum: {momentum}',
        f'Weight Decay: {weight_decay}',
        f'Gradient Clipping: {grad_clip}',
        f'Checkpoint Frequency: {checkpoint_freq}',
        f'epoch_num: {epoch_num}',
        f'Decay Learning Rate At Epochs: {decay_lr_at_epochs}',
        '-' * 50,  # Separator
        f'Number of Epochs: {epoch_num}',
        f'Date of Dataset Used: {date_of_dataset_used}',
        f'Augmentation: {augmentation}',
        f'Object Detector: {object_detector}'
    ]

    if os.path.exists(training_output_file):
        with open(training_output_file, 'r') as f:
            content = f.readlines()
        content = [line.strip() for line in content if line.strip()]

        # Ensure params are written in the same order and not added at the end
        updated_content = []
        for param in params:
            if param not in content:
                updated_content.append(param)
            else:
                updated_content.append(content[content.index(param)])
    else:
        updated_content = params

    with open(training_output_file, 'w') as f:
        f.write('\n'.join(updated_content) + '\n\n')

    return training_output_file

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
        optimizer = optimizer
        loss_fn = loss_fn

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


################################################## CHECKPOINT  ##################################################


def save_checkpoint(epoch, model, optimizer, date_of_dataset_used, save_dir):
    """
    Save model checkpoint.

    :param epoch: epoch number
    :param model: model
    :param optimizer: optimizer
    :param save_dir: directory where the checkpoint will be saved
    """
    state = {'epoch': epoch,
             'model': model,
             'optimizer': optimizer,
             'avg_loss': avg_loss,
             'accuracy' : accuracy,
             'precision' : precision,
             'recall' : recall,
             'f1' : f1,
             'mAP' : mAP,
             'IoU' : IoU}
    
    filename = os.path.join(save_dir, f'{date_of_dataset_used}_checkpoint_{epoch}.pth.tar')

    torch.save(state, filename)



def manage_top_checkpoints(epoch, model, optimizer, metrics, date_of_dataset_used, save_dir):
    """
    Save one checkpoint with the highest score for each of the following metrics:
    avg_loss, accuracy, precision, recall, f1, mAP, IoU.

    :param epoch: epoch number
    :param model: model
    :param optimizer: optimizer
    :param metrics: dictionary containing avg_loss, accuracy, precision, recall, f1, mAP, IoU
    :param date_of_dataset_used: date of the dataset used for training
    :param save_dir: directory where the checkpoint will be saved
    """
    # Save the current checkpoint
    save_checkpoint(epoch, model, optimizer, metrics, date_of_dataset_used, save_dir)

    # Track checkpoints
    checkpoint_files = [f for f in os.listdir(save_dir) if f.endswith('.pth.tar')]
    checkpoint_files = [os.path.join(save_dir, f) for f in checkpoint_files]

    # Initialize dictionaries to store the best checkpoint for each metric
    best_checkpoints = {
        'avg_loss': None,
        'accuracy': None,
        'precision': None,
        'recall': None,
        'f1': None,
        'mAP': None,
        'IoU': None
    }
    best_scores = {
        'avg_loss': float('inf'),  # Lower is better
        'accuracy': float('-inf'),  # Higher is better
        'precision': float('-inf'),
        'recall': float('-inf'),
        'f1': float('-inf'),
        'mAP': float('-inf'),
        'IoU': float('-inf')
    }

    # Evaluate each checkpoint
    for file in checkpoint_files:
        checkpoint = torch.load(file)
        for metric in best_scores:
            score = checkpoint[metric]
            if (metric == 'avg_loss' and score < best_scores[metric]) or (metric != 'avg_loss' and score > best_scores[metric]):
                best_scores[metric] = score
                best_checkpoints[metric] = file

    # Copy the best checkpoints to the 'Top_checkpoints' directory
    top_checkpoints_dir = r'6_Checkpoints\Top_checkpoints'
    os.makedirs(top_checkpoints_dir, exist_ok=True)
    for metric, file in best_checkpoints.items():
        if file:
            destination = os.path.join(top_checkpoints_dir, os.path.basename(file))
            if not os.path.exists(destination):
                shutil.copy(file, destination)

    # Remove all checkpoints except the best ones
    best_files = set(best_checkpoints.values())
    for file in checkpoint_files:
        if file not in best_files:
            os.remove(file)



################################################## METRICS CALCULATION ##################################################

def calculate_metrics_and_loss(loader, loss_fn):
    """
    Calculate accuracy, precision, recall, F1 score, loss, mAP, and IoU for a given data loader.
    """

    # Initialize variables
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_outputs = []
    running_loss = 0.0
    iou_scores = []

    with torch.no_grad():  # Disable gradient calculation for evaluation
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)  # Move data to the same device as the model
            outputs = model(inputs)  # forward pass
            loss = loss_fn(outputs, labels)  # compute loss
            running_loss += loss.item()  # accumulate the loss for the current batch

            _, preds = torch.max(outputs, 1)  # get the predicted class
            correct += (preds == labels).sum().item()  # count correct predictions
            total += labels.size(0)  # total number of samples
            all_preds.extend(preds.cpu().numpy())  # get predicted labels
            all_labels.extend(labels.cpu().numpy())  # get true labels
            all_outputs.extend(outputs.cpu().numpy())  # get raw outputs for mAP calculation

            # Calculate IoU for each sample
            for pred, label in zip(preds, labels):
                iou = calculate_iou(pred, label)
                if iou is not None:
                    iou_scores.append(iou)

    accuracy = correct / total
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    avg_loss = running_loss / len(loader)
    mAP = calculate_mAP(all_labels, all_outputs)  # mAP calculation
    mean_iou = sum(iou_scores) / len(iou_scores) if iou_scores else 0.0    # Calculate mean IoU

    # Print metrics
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(f'Average Loss: {avg_loss:.4f}')
    print(f'mAP: {mAP:.4f}')
    print(f'Mean IoU: {mean_iou:.4f}')

    return accuracy, precision, recall, f1, avg_loss, mAP, mean_iou

def precision_score(y_true, y_pred):
    """
    Calculate precision score for binary classification.

    :param y_true: True labels
    :param y_pred: Predicted labels
    :return: Precision score
    """
    return precision_score(y_true, y_pred, average='binary', zero_division=0)

def recall_score(y_true, y_pred):
    """
    Calculate recall score for binary classification.

    :param y_true: True labels
    :param y_pred: Predicted labels
    :return: Recall score
    """
    return recall_score(y_true, y_pred, average='binary', zero_division=0)

def f1_score(y_true, y_pred): 
    """
    Calculate F1 score for binary classification.

    :param y_true: True labels
    :param y_pred: Predicted labels
    :return: F1 score
    """
    return f1_score(y_true, y_pred, average='binary', zero_division=0)

def calculate_iou(pred, label):
    """
    Calculate Intersection over Union (IoU) for a single prediction and label using scikit-learn.

    :param pred: Predicted binary mask or bounding box (1D array or flattened binary mask).
    :param label: Ground truth binary mask or bounding box (1D array or flattened binary mask).
    :return: IoU value.
    """
    pred = pred.cpu().numpy().flatten()  # Convert tensor to numpy and flatten
    label = label.cpu().numpy().flatten()  # Convert tensor to numpy and flatten
    return jaccard_score(label, pred, average='binary', zero_division=0)


def calculate_mAP(all_labels, all_outputs):
    """
    Calculate AP (Average Precision) for each class and mAP (mean Average Precision).

    :param all_labels: List of true labels.
    :param all_outputs: List of model outputs.
    :return: mAP (mean Average Precision).
    """
    # Calculate AP (Average Precision) for each class
    unique_labels = set(all_labels)
    ap_scores = []
    for label in unique_labels:
        binary_labels = [1 if l == label else 0 for l in all_labels]
        binary_outputs = [o[label] for o in all_outputs]
        ap = average_precision_score(binary_labels, binary_outputs)
        ap_scores.append(ap)

    # Calculate mAP (mean Average Precision)
    mAP = sum(ap_scores) / len(ap_scores) if ap_scores else 0.0
    return mAP






################################################### METRICS LOGGING ##################################################

def save_metrics(epoch, total_epochs, 
                    writer, training_output_file,
                    train_metrics, val_metrics):
    """
    Logs and saves training and validation metrics for each epoch.
    Args:
        epoch (int): The current epoch number.
        total_epochs (int): The total number of epochs.
        writer (torch.utils.tensorboard.SummaryWriter): TensorBoard writer for logging metrics.
        training_output_file (str): Path to the file where metrics will be saved as CSV.
        train_metrics (tuple): A tuple containing training metrics in the following order:
            (accuracy, precision, recall, F1 score, loss, mAP, IoU).
        val_metrics (tuple): A tuple containing validation metrics in the following order:
            (accuracy, precision, recall, F1 score, loss, mAP, IoU).
    Prints:
        - Training and validation metrics for the current epoch.
        - Model, loss function, and optimizer details (only during the first epoch).
    Logs:
        - Training and validation metrics to TensorBoard.
    Saves:
        - Training and validation metrics to a CSV file.
    Note:
        The function assumes that `model`, `loss_fn`, and `optimizer` are defined globally.
        Ensure that these variables are accessible within the function.
    """
    model.eval()

    print(f'Epoch {epoch}/{total_epochs}')

    train_accuracy, train_precision, train_recall, train_f1, train_loss, train_mAP, train_iou = train_metrics
    val_accuracy, val_precision, val_recall, val_f1, val_loss, val_mAP, val_iou = val_metrics

    # Print metrics
    print(f'Train Loss: {train_loss}, Val Loss: {val_loss}')
    print(f'Train Accuracy: {train_accuracy}, Val Accuracy: {val_accuracy}')
    print(f'Train Precision: {train_precision}, Val Precision: {val_precision}')
    print(f'Train Recall: {train_recall}, Val Recall: {val_recall}')
    print(f'Train F1 Score: {train_f1}, Val F1 Score: {val_f1}')
    print(f'Train mAP: {train_mAP}, Val mAP: {val_mAP}')
    print(f'Train IoU: {train_iou}, Val IoU: {val_iou}')

    # Print model, loss function, and optimizer details once
    if epoch == 1:
        print(f'Model: {model}')
        print(f'Loss Function: {loss_fn}')
        print(f'Optimizer: {optimizer}')

    # Log metrics to TensorBoard
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Loss/val', val_loss, epoch)
    writer.add_scalar('Accuracy/train', train_accuracy, epoch)
    writer.add_scalar('Accuracy/val', val_accuracy, epoch)
    writer.add_scalar('Precision/train', train_precision, epoch)
    writer.add_scalar('Precision/val', val_precision, epoch)
    writer.add_scalar('Recall/train', train_recall, epoch)
    writer.add_scalar('Recall/val', val_recall, epoch)
    writer.add_scalar('F1_Score/train', train_f1, epoch)
    writer.add_scalar('F1_Score/val', val_f1, epoch)
    writer.add_scalar('mAP/train', train_metrics[5], epoch)
    writer.add_scalar('mAP/val', val_metrics[5], epoch)
    writer.add_scalar('IoU/train', train_metrics[6], epoch)
    writer.add_scalar('IoU/val', val_metrics[6], epoch)

    # Save metrics to csv
    save_metrics_to_csv(epoch, train_metrics, val_metrics, training_output_file)


def save_metrics_to_csv(epoch, train_metrics, val_metrics, training_output_file):
    """
    Save training and validation metrics to a CSV file.
    """
    csv_file = f'5_Results/{training_output_file.replace(".txt", ".csv")}'
    file_exists = os.path.isfile(csv_file)

    with open(csv_file, mode='a', newline='') as file:
        csv_writer = csv.writer(file)
        if not file_exists:
            # Write header if file does not exist
            csv_writer.writerow(['Epoch', 'Train Loss', 'Val Loss', 'Train Accuracy', 'Val Accuracy',
                                 'Train Precision', 'Val Precision', 'Train Recall', 'Val Recall',
                                 'Train F1 Score', 'Val F1 Score', 'Train mAP', 'Val mAP', 'Train IoU', 'Val IoU'])
        
        # Write metrics
        csv_writer.writerow([epoch] + train_metrics + val_metrics)




############################################### TRAINING #####################################################

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
             desc=f'Epoch {epoch}/{total_epochs}', 
             unit='batch')
            ):
        
        images = images.to(device) # Move data to device

        if args.object_detector == 'yes':  # Torchvision models expect targets to be a list of dicts with 'boxes' and 'labels' keys
            targets = [] # targets should be a list of dicts, each with 'boxes' and 'labels'
            for b, l in zip(boxes, labels): #loop through each pair of boxes and labels
                targets.append({ #for each image, create a dict with 'boxes' and 'labels'
                    'boxes': b.to(device),  
                    'labels': l.to(device)
                })
            optimizer.zero_grad() #clear gradients 
            loss_dict = model(images, targets) # For torchvision models, loss_dict is a dict of losses
                                            # this is also the forward pass
                                            # model(images, targets) is thus the forward pass and loss calculation
            loss = sum(loss for loss in loss_dict.values())

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
    train_loader = train_loader
    validation_loader = validation_loader

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




