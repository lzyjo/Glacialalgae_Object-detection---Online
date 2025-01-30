

# Checking and setting cwd 
import json
import os

## Check the current working directory
cwd = os.getcwd()
print(f"Current working directory: {cwd}")

expected_cwd = r"C:\Users\me20332\OneDrive - University of Bristol\MscR\GitHub\Testing_different_datasets"
if cwd != expected_cwd:
    print("Current working directory is not the Testing_different_datasets directory")
    os.chdir(r"C:\Users\me20332\OneDrive - University of Bristol\MscR\GitHub\Testing_different_datasets") # Set the current working directory if needed
    print("Current working directory has been set to the Testing_different_datasets directory")
else:
    print("Current working directory is the Testing_different_datasets directory")


# Run the BCCD_utils.py file
os.system('python BCCD_utils.py')


## Check if the data exists and extract it if not
from BCCD_utils import initialize_bccd_trainvaltest_data

# direct paths to the BCCD dataset and make sure they are absolute
train_path = os.path.abspath(r"BCCD.v3\Original\train")
test_path = os.path.abspath(r"BCCD.v3\Original\test")
valid_path = os.path.abspath(r"BCCD.v3\Original\valid")


initialize_bccd_trainvaltest_data(train_path=train_path,
                                  test_path=test_path,
                                  valid_path=valid_path)


# Structuring of the BCCD dataset (according to VOC) 
# from BCCD_utils import setup_annotations_for_dataset, setup_images_for_dataset

# src_dir = os.path.abspath(r"BCCD.v3\Original")
# dest_dir = os.path.abspath(r"BCCD.v3\Annotations")
# setup_annotations_for_dataset(src_dir= src_dir, dest_dir=dest_dir)

# src_dir = os.path.abspath(r"BCCD.v3\Original")
# dest_dir = os.path.abspath(r"BCCD.v3\Images")
# setup_images_for_dataset(src_dir= src_dir, dest_dir=dest_dir)


# Dataset prep and set up
from BCCD_utils import convert_files_to_list, split_and_copy_files, create_folders, copy_files


# Split the dataset into train, test and validation sets
images, annotations = convert_files_to_list(images_folder=r'GA_Dataset\Images', annotations_folder=r'GA_Dataset\Annotations') # Convert to list 
output_folder = r'GA_Dataset\Split' 
create_folders(output_folder) #create_folders(output_folder)
copy_files(images, annotations, output_folder) #copy_files(images, annotations, output_folder)
split_and_copy_files(images, annotations, output_folder)


# Creating datalists for the train, val and test data
from BCCD_utils import parse_annotation, create_data_lists
annotation_path = r"GA_Dataset\Annotations"
output_folder = cwd

# Label map
# Load labels from a file
import csv
with open('Label classes.csv', 'r') as f:
    reader = csv.reader(f)
    Labels = tuple(row[0] for row in reader)
label_map = {k: v + 1 for v, k in enumerate(Labels)}
# label_map['background'] = 0
rev_label_map = {v: k for k, v in label_map.items()}  # Inverse mapping

# Color map for bounding boxes of detected objects from https://sashat.me/2017/01/11/list-of-20-simple-distinct-colors/
distinct_colors = ['#e6194b', '#3cb44b', '#ffe119', '#0082c8']
label_color_map = {k: distinct_colors[i] for i, k in enumerate(label_map.keys())}

create_data_lists(annotation_path=r"GA_Dataset\Annotations",
                  train_path=train_path,
                  test_path=test_path,
                  valid_path=valid_path,
                  output_folder=cwd)


# Check if model is already trained and present 
if os.path.exists('checkpoint_ssd300.pth'):
    print('Model already trained: checkpoint_ssd300.pth present in cwd')
else:
    print('Model not trained or present: checkpoint_ssd300.pth not present in cwd')


# Training the model
from BCCD_dataset import BCCDDataset

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

