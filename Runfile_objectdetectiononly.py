# Checking and setting cwd 
import json
import os
import pandas as pd
import subprocess

## Check the current working directory
cwd = os.getcwd()
print(f"Current working directory: {cwd}")



# Dataset prep and set up
from utils import create_dataset_folder, extract_files

create_dataset_folder() #only run if you want to create a new dataset folder!!

##BLUFF_230724 DATA
extract_files(date_of_dataset_used= 'Bluff_230724', # Change this to the correct dataset used, FOR REFERENCE ONLY
                annotations_folder=r'GA_Dataset\20250221\Annotations', # Change this to the correct folder for which files were extracted 
                images_folder=r'GA_Dataset\20250221\Images', # Change this to the correct folder for which files were extracted 
                images_src_folder=r'Completed annotations/Bluff_230724/Original_Images_Unlabelled_Bluff_230724', # Change this to your source folder path 
                annotations_src_folder=r'Completed annotations\Bluff_230724') # Change this to your source folder path

##PAM_Surf_220724 DATA
extract_files(date_of_dataset_used= 'PAM_Surf_220724', # Change this to the correct dataset used, FOR REFERENCE ONLY
                annotations_folder=r'GA_Dataset\20250221\Annotations', # Change this to the correct folder for which files were extracted 
                images_folder=r'GA_Dataset\20250221\Images', # Change this to the correct folder for which files were extracted 
                images_src_folder=r'Completed annotations/PAM_Surf_220724/Original_Images_Unlabelled_PAM_Surf_220724', # Change this to your source folder path 
                annotations_src_folder=r'Completed annotations\PAM_Surf_220724') # Change this to your source folder path


# Convert all label classes to 'cell'
from objectdetector_utils import convert_labels_to_cell
convert_labels_to_cell(annotations_folder= r'GA_Dataset\20250221_objectdetector\Annotations')


# Split the dataset into train, test and validation sets
from utils import convert_files_to_list, split_and_copy_files
images, annotations = convert_files_to_list(images_folder=r'GA_Dataset\20250221_objectdetector\Images', 
                                            annotations_folder=r'GA_Dataset\20250221_objectdetector\Annotations') # Convert to list 

split_and_copy_files(images, annotations, #create_folders, copy files, then split into test and train
                     output_folder=r'GA_Dataset\20250221_objectdetector\Split') #output folder







# Creating datalists for the train, val and test data
from utils import create_data_lists
import shutil
import json
import os
import pandas as pd

label_classes = ('cell', 'UNKNOWN')  # Define label classes directly
label_map_objectdetector = {k: v + 1 for v, k in enumerate(label_classes)}
label_map_objectdetector['background'] = 0  # Background is the first class
rev_label_map_objectdetector = {v: k for k, v in label_map_objectdetector.items()}  # Inverse mapping


create_data_lists(train_annotation_path=r'GA_Dataset\20250221_objectdetector\Split\train\annotations',
                train_image_path=r'GA_Dataset\20250221_objectdetector\Split\train\images',
                test_annotation_path=r'GA_Dataset\20250221_objectdetector\Split\test\annotations',
                test_image_path=r'GA_Dataset\20250221_objectdetector\Split\test\images',
                label_map=label_map_objectdetector,
                date_of_dataset_used='20250221_objectdetector',
                JSON_folder=r'JSON_folder')


# Check if model is already trained and present 
date_of_dataset_used = '20250221_objectdetector'
model_path = os.path.join(date_of_dataset_used + '_checkpoint_ssd300.pth')

if os.path.exists(model_path):
    print(f'Model for date: {date_of_dataset_used} already trained: {model_path} present in cwd')
else:
    print(f'Model for date: {date_of_dataset_used} not trained or present: {model_path} not present in cwd')


# Training the model

## Suppress specific warnings
import warnings
warnings.filterwarnings("ignore")

subprocess.run(['python', 'train.py', 
                '--data_folder', r'JSON_folder\20250221_objectdetector',
                '--date_of_dataset_used', '20250221_objectdetector',])


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

