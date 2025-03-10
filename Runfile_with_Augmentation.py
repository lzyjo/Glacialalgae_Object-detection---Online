# Checking and setting cwd 
import json
import os
import pandas as pd
import subprocess
from torchvision.transforms import v2 as T

## Check the current working directory
cwd = os.getcwd()
print(f"Current working directory: {cwd}")



# DATA AUGMENTATION
from augmentation import *

GA_dataset_path = r'GA_Dataset'
date_of_dataset_used = '20250221'
image_dir = r'GA_Dataset\20250221\Images'
annotation_dir = r'GA_Dataset\20250221\Annotations'
transformations = T.Compose([
    T.RandomHorizontalFlip(p=1.0),
    T.RandomVerticalFlip(p=1.0)])

run_augmentation_pipeline(GA_dataset_path= GA_dataset_path,
                          date_of_dataset_used= date_of_dataset_used,
                          image_dir= image_dir,
                          annotation_dir= annotation_dir,
                          transformations= transformations,
                          num_pairs=5)

# RANDOM ROTATION RETURNS ISSUE WITH BOUNDING BOX TRANSFORMATIONS (not correctly rotated in line with image)




# DATASET PREPARATION FOR SPLIT (WITH AUGMENTATION)
from utils import create_dataset_folder, extract_files

# Create dataset folder because we are combining multiple datasets:
                                                                # GA_Dataset\20250221
                                                                # GA_Dataset\20250221_randomhorizontalflip
                                                                # GA_Dataset\20250221_randomverticalflip
                                                                # GA_Dataset\20250221_randomrotation
create_dataset_folder(folder_type='Training', folder_date='20250221')  # Only run if you want to create a new dataset folder!!

annotations_folder = r'Training_GA_Dataset\20250221\Annotations' # Change this to the correct folder for which files are to be extracted to
images_folder = r'Training_GA_Dataset\20250221\Images' # Change this to the correct folder for which files are to be extracted to

## Run only once for each dataset for file extraction
##20250221 (not augmented) DATA
extract_files(date_of_dataset_used= '20250221', # Change this to the correct dataset used, FOR REFERENCE ONLY
                annotations_folder= annotations_folder, 
                images_folder= images_folder, 
                images_src_folder=r'GA_Dataset\20250221\Images', # Change this to your source folder path 
                annotations_src_folder=r'GA_Dataset\20250221\Annotations') # Change this to your source folder path

##20250221_randomhorizontalflip DATA
extract_files(date_of_dataset_used= '20250221_randomhorizontalflip', # Change this to the correct dataset used, FOR REFERENCE ONLY
                annotations_folder= annotations_folder, 
                images_folder= images_folder, 
                images_src_folder=r'GA_Dataset\20250221_randomhorizontalflip\Images', # Change this to your source folder path 
                annotations_src_folder=r'GA_Dataset\20250221_randomhorizontalflip\Annotations') # Change this to your source folder path

##20250221_randomverticalflip DATA
extract_files(date_of_dataset_used= '20250221_randomverticalflip', # Change this to the correct dataset used, FOR REFERENCE ONLY
                annotations_folder= annotations_folder,
                images_folder= images_folder,
                images_src_folder=r'GA_Dataset\20250221_randomverticalflip\Images', # Change this to your source folder path
                annotations_src_folder=r'GA_Dataset\20250221_randomverticalflip\Annotations') # Change this to your source folder path

## GA_Dataset/20250221_randomhorizontalflip_randomverticalflip DATA
extract_files(date_of_dataset_used= '20250221_randomhorizontalflip_randomverticalflip', # Change this to the correct dataset used, FOR REFERENCE ONLY
                annotations_folder= annotations_folder,
                images_folder= images_folder,
                images_src_folder=r'GA_Dataset\20250221_randomhorizontalflip_randomverticalflip\Images', # Change this to your source folder path
                annotations_src_folder=r'GA_Dataset\20250221_randomhorizontalflip_randomverticalflip\Annotations') # Change this to your source folder path


# Split the dataset into train, test and validation sets
from utils import convert_files_to_list, split_and_copy_files
images, annotations = convert_files_to_list(images_folder=images_folder, 
                                            annotations_folder=annotations_folder) # Convert to list 

output_folder = r'Training_GA_Dataset\20250221\Split' #output folder forw here split is stored 
split_and_copy_files(images, annotations, #create_folders, copy files, then split into test and train
                     output_folder= output_folder) 



# Creating datalists for the train, val and test data
from utils import create_data_lists
from label_map import label_map # Label map (explicitly defined)
import shutil

train_annotation_path= r'Training_GA_Dataset\20250221\Split\train\annotations'
train_image_path= r'Training_GA_Dataset\20250221\Split\train\images'  
test_annotation_path= r'Training_GA_Dataset\20250221\Split\test\annotations'
test_image_path= r'Training_GA_Dataset\20250221\Split\test\images'
date_of_dataset_used='20250221_augmented'

create_data_lists(train_annotation_path=train_annotation_path,
                train_image_path=train_image_path,
                test_annotation_path=test_annotation_path,
                test_image_path=test_image_path,
                label_map=label_map,
                date_of_dataset_used=date_of_dataset_used,
                JSON_folder=r'JSON_folder')





# Check if model is already trained and present 
date_of_dataset_used = '20250221'
date_of_dataset_used = date_of_dataset_used + '_augmentation'
model_path = os.path.join(date_of_dataset_used + '_checkpoint_ssd300.pth')

if os.path.exists(model_path):
    print(f'Model for date: {date_of_dataset_used} already trained: {model_path} present in cwd')
else:
    print(f'Model for date: {date_of_dataset_used} not trained or present: {model_path} not present in cwd')



# TRAIN MODEL

## Suppress specific warnings
import warnings
from utils import manage_training_output_file
warnings.filterwarnings("ignore")

# Training the model and saving the results to a .txt file
data_folder = r'JSON_folder\20250221'
date_of_dataset_used = '20250221'
checkpoint = r'Checkpoints\20250221_checkpoint_54.pth.tar'
checkpoint_frequency = '120'
lr = '1e-5'
iterations = '1200'

#set up training output file
results_folder = r'Results'
training_output_file = manage_training_output_file(results_folder = results_folder,
                                                date_of_dataset_used= date_of_dataset_used, 
                                                checkpoint_frequency = checkpoint_frequency,
                                                lr =  lr, 
                                                iterations = iterations)


# Run the training process and save the output
with open(training_output_file, 'a') as f:
    subprocess.run(['python', 'train.py', 
                    '--data_folder', data_folder,
                    '--date_of_dataset_used', date_of_dataset_used,
                    '--save_dir', r'Checkpoints',
                    '--checkpoint', checkpoint,
                    '--checkpoint_frequency', checkpoint_frequency,
                    '--lr', lr,
                    '--iterations', iterations], stdout=f)

# Run the training process and save the output
with open(training_output_file, 'a') as f:
    for line in subprocess.run(['python', 'train.py', 
                '--data_folder', data_folder,
                '--date_of_dataset_used', date_of_dataset_used,
                '--save_dir', r'Checkpoints',
                '--checkpoint', checkpoint,
                '--checkpoint_frequency', checkpoint_frequency,
                '--lr', lr,
                '--iterations', iterations], stdout=f):
        if line.startswith('Epoch:'):  # write lines starting with 'Epoch:'
            f.write(line)


# Return the relative file path of the training output file
print(f"Training output file saved at: {os.path.relpath(training_output_file)}")

    





# keep only relevant checkpoints
from utils import keep_checkpoints

training_output_file = 
date_of_dataset_used = 
keep_checkpoints(checkpoint_dir=r'Checkpoints', 
                 log_file= training_output_file,
                 date_of_dataset_used= date_of_dataset_used)


# EVALUATE MODEL

# Evaluate the model and save the results to a .txt file
checkpoint = r'Checkpoints\20250219_checkpoint_3.pth.tar'
results_folder = r'Results'
evaluation_output_file = os.path.join(results_folder, f'evaluation_results_{os.path.basename(checkpoint)}.txt')

# Ensure the results folder exists
os.makedirs(results_folder, exist_ok=True)

with open(evaluation_output_file, 'w') as f:
    subprocess.run(['python', 'eval.py',
                    '--data_folder', r'JSON_folder\20250219',
                    '--checkpoint', checkpoint], stdout=f)


# Inference 
subprocess.run(['python', 'detect.py',
                '--checkpoint', checkpoint,
                '--img_path', r'GA_Dataset/Split/test/images/0.tif'])

