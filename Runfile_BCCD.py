

# Checking and setting cwd 
import json
import os
import pandas as pd

## Check the current working directory
cwd = os.getcwd()
print(f"Current working directory: {cwd}")

# Dataset prep and set up
from utils import convert_files_to_list, split_and_copy_files

# Split the dataset into train, test and validation sets
images, annotations = convert_files_to_list(images_folder=r'GA_Dataset\Images', annotations_folder=r'GA_Dataset\Annotations') # Convert to list 
output_folder = r'GA_Dataset\Split' 
split_and_copy_files(images, annotations, output_folder) #create_folders, copy files, then split into test and train


# Creating datalists for the train, val and test data
from utils import create_data_lists

# Label map (explicitly defined)
label_classes_path = os.path.abspath(r"label_classes.csv") # Load label classes from CSV
label_classes_df = pd.read_csv(label_classes_path) # read csv
label_classes = tuple(label_classes_df.iloc[:, 0].tolist())  # Derive labels from the first column of the CSV
label_map = {k: v + 1 for v, k in enumerate(label_classes)}
rev_label_map = {v: k for k, v in label_map.items()}  # Inverse mapping

create_data_lists(train_annotation_path=r'GA_Dataset/Split/train/annotations',
                train_image_path=r'GA_Dataset/Split/train/images',
                test_annotation_path=r'GA_Dataset/Split/test/annotations',
                test_image_path=r'GA_Dataset/Split/test/images',
                label_map=label_map,
                output_folder='./')


# Check if model is already trained and present 
if os.path.exists('checkpoint_ssd300.pth'):
    print('Model already trained: checkpoint_ssd300.pth present in cwd')
else:
    print('Model not trained or present: checkpoint_ssd300.pth not present in cwd')


# Training the model
from dataset import BCCDDataset

data_folder = './'  # folder with data files
os.system('python BCCD_train.py')

# Evaluate the model

## Check if model is already trained and present 
if os.path.exists('checkpoint_ssd300.pth.tar'):
    print('Model already trained: checkpoint_ssd300.pth present in cwd')
else:
    print('Model not trained or present: checkpoint_ssd300.pth not present in cwd')

os.system('python BCCD_eval.py')


# Inference 

os.system('python BCCD_detect.py')

import BCCD_detect # want to be able to directly use the detect function and change the file 
# directly to use the cnn to annotate the test images  

