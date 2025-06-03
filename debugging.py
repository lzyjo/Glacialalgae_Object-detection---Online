import torch
from torch.utils.data import DataLoader
from zmq import device
from dataset import GA_Dataset
from label_map import label_map_Classifier, label_map_OD
import argparse
from utils import adjust_learning_rate
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import average_precision_score, jaccard_score
from tqdm import tqdm
from hyperparameters import *
import datetime
from torch.utils.tensorboard import SummaryWriter as writer
import csv
import os
import shutil
import time
from dataset import PC_Dataset
from tqdm import tqdm


from dataset import PC_Dataset

data_folder = r'3_TrainingData\20250513_Augmented\Split'  # Folder containing the training data

# training dataset and dataloader
train_dataset = PC_Dataset(data_folder,
                            split='train')
train_loader = torch.utils.data.DataLoader(train_dataset, 
                                           batch_size=batch_size, shuffle=True,
                                            collate_fn=train_dataset.collate_fn,  # custom collate function
                                            num_workers=workers,
                                            pin_memory=True)  # note that we're passing the collate function here



device = torch.device('cpu')

# Model params
model = fasterrcnn_resnet50_fpn(pretrained=False,
                                num_classes=2)  # Assuming 2 classes (background + object)
chosen_optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
chosen_loss_fn = nn.BCELoss() # Assuming a binary classification task
                # Assuming a classification task: loss_fn = nn.CrossEntropyLoss(
start_epoch = 0
optimizer = chosen_optimizer
loss_fn = chosen_loss_fn

# model training 
model.train()
running_loss = 0.0

epoch = 1
object_detector = 'yes'
# Debugging: limit number of batches, print shapes and sample data
max_debug_batches = 3  # Only run a few batches for debugging

for i, (images, boxes, labels, _) in enumerate(
    tqdm(train_loader, 
         desc=f'Epoch {epoch}/{epoch_num}', 
         unit='batch')
    ):
    print(f"\nBatch {i+1}:")
    print(f"  images type: {type(images)}, shape: {getattr(images, 'shape', 'N/A')}")
    print(f"  boxes type: {type(boxes)}, sample: {boxes[0] if len(boxes) > 0 else 'N/A'}")
    print(f"  labels type: {type(labels)}, sample: {labels[0] if len(labels) > 0 else 'N/A'}")

    images = images.to(device) if hasattr(images, 'to') else images

    if object_detector == 'yes':
        targets = []
        for b, l in zip(boxes, labels):
            print(f"    box shape: {getattr(b, 'shape', 'N/A')}, label shape: {getattr(l, 'shape', 'N/A')}")
            targets.append({
                'boxes': b.to(device),
                'labels': l.to(device)
            })
        optimizer.zero_grad()
        loss_dict = model(images, targets)
        print(f"  loss_dict: {loss_dict}")
        loss = sum(loss for loss in loss_dict.values())
    else:
        if isinstance(labels, (list, tuple)):
            labels = [l.to(device) for l in labels]
        else:
            labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        print(f"  outputs shape: {getattr(outputs, 'shape', 'N/A')}")
        loss = loss_fn(outputs, labels)

    print(f"  loss: {loss.item()}")
    loss.backward()
    optimizer.step()
    running_loss += loss.item()

    if i % 1000 == 999:
        avg_loss = running_loss / 1000
        print(f'  batch {i + 1} loss: {avg_loss}')
        writer.add_scalar('Loss/train', avg_loss, epoch * len(train_loader) + i + 1)
        running_loss = 0.0

    if i + 1 >= max_debug_batches:
        print("Debug mode: breaking after a few batches.")
        break
