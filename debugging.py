import torch
from torch.utils.data import DataLoader
from dataset import GA_Dataset
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
from dataset import GA_Dataset, collate_fn

# ------------------- ARGUMENTS -------------------
object_detector = 'yes'  # or 'no'
data_folder = r'4_JSON_folder\20250513_Augmented'
training_output_file = r'5_Results/training_results_20250513_Augmented.txt'
save_dir = r'Checkpoints'

# ------------------- DATA PARAMETERS -------------------
if object_detector == 'yes':
    label_map = label_map_OD
else:
    label_map = label_map_Classifier

n_classes = len(label_map)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# You must define your model here, e.g.:
# model = MyModel(...)
model = model.to(device)

data_folder = data_folder
train_dataset = GA_Dataset(data_folder, split='train', keep_difficult=False)
# View a sample annotation as parsed by the dataset
sample_ann = train_dataset[0]
print("Sample parsed annotation from GA_Dataset[0]:")
if isinstance(sample_ann, (tuple, list)):
    for idx, item in enumerate(sample_ann):
        print(f"  Item {idx}: type={type(item)}, shape={getattr(item, 'shape', 'N/A')}")
else:
    print(f"  Type: {type(sample_ann)}, shape={getattr(sample_ann, 'shape', 'N/A')}")

train_dataset = GA_Dataset(data_folder, split='train', keep_difficult=False)
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=train_dataset.collate_fn,
    num_workers=workers,
    pin_memory=True
)

validation_dataset = GA_Dataset(data_folder, split='val', keep_difficult=False)
validation_loader = torch.utils.data.DataLoader(
    validation_dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=validation_dataset.collate_fn,
    num_workers=workers,
    pin_memory=True
)   

test_dataset = GA_Dataset(data_folder, split='test', keep_difficult=False)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=test_dataset.collate_fn, #question: why do i not included batch_size as an argument here?
    num_workers=workers,
    pin_memory=True
)




# View a batch as returned by collate_fn
print("\nSample batch from train_loader (collate_fn output):")
batch = next(iter(train_loader))
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


# --- View a few batches from the loaders for inspection ---
def inspect_loader(loader, name, num_batches=1):
    print(f"\nInspecting {name} loader:")
    for i, batch in enumerate(loader):
        print(f"Batch {i+1}:")
        if isinstance(batch, (list, tuple)):
            for idx, item in enumerate(batch):
                print(f"  Item {idx} type: {type(item)}; shape: {getattr(item, 'shape', 'N/A')}")
        else:
            print(f"  Batch type: {type(batch)}; shape: {getattr(batch, 'shape', 'N/A')}")
        if i + 1 >= num_batches:
            break

# Example usage:
inspect_loader(train_loader, "train", num_batches=1) #shape (batch_size, channels, height, width)
inspect_loader(validation_loader, "validation", num_batches=1)
inspect_loader(test_loader, "test", num_batches=1)

# ------------------- INITIALIZE TRAINING -------------------
checkpoint = None  # Set this if you want to load from checkpoint

if checkpoint is None:
    start_epoch = 0
    # You must define optimizer and loss_fn here, e.g.:
    # optimizer = torch.optim.Adam(model.parameters(), lr=...)
    # loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = optimizer
    loss_fn = loss_fn
    print('\nNo checkpoint provided. Starting training from scratch.\n')
    print(f"Start epoch: {start_epoch}")
    print(f"Optimizer: {optimizer}")
    print(f"Loss function: {loss_fn}")
else:
    checkpoint_data = torch.load(checkpoint)
    print(f'\nLoaded checkpoint from epoch {checkpoint_data["epoch"] + 1}.\n')
    print(f"Optimizer: {checkpoint_data['optimizer']}")
    print(f"Loss function: {checkpoint_data['loss_fn']}")
    start_epoch = checkpoint_data['epoch'] + 1
    optimizer = checkpoint_data['optimizer']
    loss_fn = checkpoint_data['loss_fn']

# ------------------- TRAINING LOOP -------------------
training_output_file = training_output_file
writer = SummaryWriter(f'5_Results/{training_output_file}')
total_epochs = epoch_num
decay_lr_at_epoch = decay_lr_at_epochs

# Only run 1 epoch
for epoch in range(start_epoch, start_epoch + 1):
    if epoch in decay_lr_at_epoch:
        adjust_learning_rate(optimizer, decay_lr_to)
    epoch_print = epoch + 1

    # --------- Train One Epoch ---------
    print(f'\nEpoch {epoch_print}/{total_epochs}')
    model.train()
    running_loss = 0.0
  
    for i, (images, boxes, labels, _) in enumerate(
        tqdm(train_loader, desc=f'Epoch {epoch_print}/{total_epochs}', unit='batch')
    ):
        # Optionally, track data loading time if needed
        # data_time.update(time.time() - start)

        # Move data to device
        images = images.to(device)
        # Prepare targets for detection models
        if object_detector == 'yes':
            # targets should be a list of dicts, each with 'boxes' and 'labels'
            targets = []
            for b, l in zip(boxes, labels):
                targets.append({
                    'boxes': b.to(device),
                    'labels': l.to(device)
                })
            optimizer.zero_grad()
            loss_dict = model(images, targets)
            # For torchvision models, loss_dict is a dict of losses
            loss = sum(loss for loss in loss_dict.values())
        else:
            # For classification models
            if isinstance(labels, (list, tuple)):
                labels = [l.to(device) for l in labels]
            else:
                labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    if i % 1000 == 999:
        avg_loss = running_loss / 1000
        print(f'  batch {i + 1} loss: {avg_loss}')
        writer.add_scalar('Loss/train', avg_loss, epoch_print * len(train_loader) + i + 1)
        running_loss = 0.0

    # --------- Calculate Metrics and Loss (Train) ---------
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_outputs = []
    running_loss = 0.0
    iou_scores = []
    with torch.no_grad():
        for images, boxes, labels, _ in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_outputs.extend(outputs.cpu().numpy())
            for pred, label in zip(preds, labels):
                pred_np = pred.cpu().numpy().flatten()
                label_np = label.cpu().numpy().flatten()
                iou = jaccard_score(label_np, pred_np, average='binary', zero_division=0)
                if iou is not None:
                    iou_scores.append(iou)
    train_accuracy = correct / total
    train_precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    train_recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    train_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    train_loss = running_loss / len(train_loader)
    # mAP calculation
    unique_labels = set(all_labels)
    ap_scores = []
    for label in unique_labels:
        binary_labels = [1 if l == label else 0 for l in all_labels]
        binary_outputs = [o[label] for o in all_outputs]
        ap = average_precision_score(binary_labels, binary_outputs)
        ap_scores.append(ap)
    train_mAP = sum(ap_scores) / len(ap_scores) if ap_scores else 0.0
    train_iou = sum(iou_scores) / len(iou_scores) if iou_scores else 0.0

    # --------- Calculate Metrics and Loss (Validation) ---------
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_outputs = []
    running_loss = 0.0
    with torch.no_grad():
        for images, boxes, labels, _ in validation_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_outputs.extend(outputs.cpu().numpy())
            for pred, label in zip(preds, labels):
                pred_np = pred.cpu().numpy().flatten()
                label_np = label.cpu().numpy().flatten()
                iou = jaccard_score(label_np, pred_np, average='binary', zero_division=0)
                if iou is not None:
                    iou_scores.append(iou)
                    iou_scores.append(iou)
    val_accuracy = correct / total
    val_precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    val_recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    val_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    val_loss = running_loss / len(validation_loader)
    unique_labels = set(all_labels)
    ap_scores = []
    for label in unique_labels:
        binary_labels = [1 if l == label else 0 for l in all_labels]
        binary_outputs = [o[label] for o in all_outputs]
        ap = average_precision_score(binary_labels, binary_outputs)
        ap_scores.append(ap)
    val_mAP = sum(ap_scores) / len(ap_scores) if ap_scores else 0.0
    val_iou = sum(iou_scores) / len(iou_scores) if iou_scores else 0.0

    # --------- Print and Log Metrics ---------
    print(f'Epoch {epoch_print}/{total_epochs}')
    print(f'Train Loss: {train_loss}, Val Loss: {val_loss}')
    print(f'Train Accuracy: {train_accuracy}, Val Accuracy: {val_accuracy}')
    print(f'Train Precision: {train_precision}, Val Precision: {val_precision}')
    print(f'Train Recall: {train_recall}, Val Recall: {val_recall}')
    print(f'Train F1 Score: {train_f1}, Val F1 Score: {val_f1}')
    print(f'Train mAP: {train_mAP}, Val mAP: {val_mAP}')
    print(f'Train IoU: {train_iou}, Val IoU: {val_iou}')
    if epoch_print == 1:
        print(f'Model: {model}')
        print(f'Loss Function: {loss_fn}')
        print(f'Optimizer: {optimizer}')
    writer.add_scalar('Loss/train', train_loss, epoch_print)
    writer.add_scalar('Loss/val', val_loss, epoch_print)
    writer.add_scalar('Accuracy/train', train_accuracy, epoch_print)
    writer.add_scalar('Accuracy/val', val_accuracy, epoch_print)
    writer.add_scalar('Precision/train', train_precision, epoch_print)
    writer.add_scalar('Precision/val', val_precision, epoch_print)
    writer.add_scalar('Recall/train', train_recall, epoch_print)
    writer.add_scalar('Recall/val', val_recall, epoch_print)
    writer.add_scalar('F1_Score/train', train_f1, epoch_print)
    writer.add_scalar('F1_Score/val', val_f1, epoch_print)
    writer.add_scalar('mAP/train', train_mAP, epoch_print)
    writer.add_scalar('mAP/val', val_mAP, epoch_print)
    writer.add_scalar('IoU/train', train_iou, epoch_print)
    writer.add_scalar('IoU/val', val_iou, epoch_print)

    # --------- Save Metrics to CSV ---------
    csv_file = f'5_Results/{training_output_file.replace(".txt", ".csv")}'
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, mode='a', newline='') as file:
        csv_writer = csv.writer(file)
        if not file_exists:
            csv_writer.writerow(['Epoch', 'Train Loss', 'Val Loss', 'Train Accuracy', 'Val Accuracy',
                                 'Train Precision', 'Val Precision', 'Train Recall', 'Val Recall',
                                 'Train F1 Score', 'Val F1 Score', 'Train mAP', 'Val mAP', 'Train IoU', 'Val IoU'])
        csv_writer.writerow([epoch_print, train_loss, val_loss, train_accuracy, val_accuracy,
                             train_precision, val_precision, train_recall, val_recall,
                             train_f1, val_f1, train_mAP, val_mAP, train_iou, val_iou])

torch.save(model.state_dict(), 'model.pth')
print("Model saved as model.pth")
