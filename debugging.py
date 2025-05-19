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
from dataset import GA_Dataset

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

   
torch.save(model.state_dict(), 'model.pth')
print("Model saved as model.pth")
