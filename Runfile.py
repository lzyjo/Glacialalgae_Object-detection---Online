# Checking and setting cwd 

"""
This script orchestrates the end-to-end pipeline for preparing, augmenting, splitting, and training an object detection model on glacial algae datasets. The workflow includes:
1. Setting up the working directory and importing required modules.
2. Creating dataset folders for raw and processed data.
3. Extracting raw annotation (.xml) and image (.tif) files from source directories into a master dataset folder.
4. Cleaning and standardizing annotation labels for object detection and classification tasks.
5. Splitting the dataset into training, validation, and test sets, and copying files accordingly.
6. Defining and applying a series of data augmentation transformations to increase dataset diversity.
7. Creating a final training dataset by combining original and augmented data, and organizing it into split folders.
8. Extracting and organizing files for each split (train, test, val) into the final training data structure.
9. Generating JSON data lists for use with custom PyTorch dataloaders.
10. Initializing custom dataset and dataloader objects for training, validation, and testing.
11. Checking for existing trained model checkpoints and managing training output files.
12. Running the model training process via a subprocess call to a separate training script.
13. Managing and retaining only relevant model checkpoints based on training logs.
14. Evaluating the trained model and saving evaluation results to a text file.
15. Running inference on a sample image using the trained model.
Note:
- The script assumes the existence of several utility modules (e.g., `utils`, `augmentation`, `OD_misc_utils`, `dataset`, etc.) and external scripts (`train_custom.py`, `e.py`, `detect.py`).
- Paths and parameters should be updated as needed for different datasets or experimental setups.
- Some steps (e.g., augmentation, label conversion) are tailored for object detection tasks and may need adjustment for other use cases.
"""
import json
import os
import pandas as pd
import subprocess
from torchvision.transforms import v2 as T
from datetime import datetime



## Check the current working directory
cwd = os.getcwd()
print(f"Current working directory: {cwd}")



##################################################### DATA HANDLING #####################################################

########### For local only ############

#CREATE DATASET FOLDER 
from utils import create_dataset_folder

create_dataset_folder(base_folder=r'1_GA_Dataset', #base folder where dataset is stored
                      folder_date='None', #if None, current_date = datetime.now().strftime('%Y%m%d')
                      Augmented=False,
                      Split=True) #date of dataset created

# Replace all occurrences of 1_GA_Dataset\PREVIOUS with 1_GA_Dataset\CURRENT in this file, at any point in the runfile
# Capacity to create a function to do this, but not needed for now?


# EXTRACT RAW .XMLS AND .TIFS INTO (MASTERLIST) DATASET FOLDER 
from utils import extraction_pipeline

source_folders = [
    r'0_Completed annotations\Bluff_230724',  # Original dataset, not augmented
    r'0_Completed annotations\PAM_Surf_220724'  # Augmented datasets
]

# Change to correct folder for which files are to be extracted to
annotations_folder = r'1_GA_Dataset\20250513/Annotations'
images_folder = r'1_GA_Dataset\20250513/Images' 

extraction_pipeline(source_folders=source_folders,
                    annotations_folder=annotations_folder, images_folder=images_folder,
                    raw_data=True, #Because raw data = True, train_test_val is not applicable (any argument is fine)
                    train_test_val=None,
                    include_augmentation_list=False) # Call the function to extract files from multiple folders





# CORRECT LABELS 
# for OD, labels are cell only + no unknowns
# for Classifier, labels are cells (AA_X and AN_X etc) + no unknowns
from OD_misc_utils import convert_labels_to_cell, remove_unknowns_from_labels

# Change this to the correct folder for which files are to be extracted to
annotations_folder = r'1_GA_Dataset\20250513\Annotations' 
convert_labels_to_cell(annotations_folder= annotations_folder) # Convert all labels to 'cell' in the annotation files
remove_unknowns_from_labels(annotations_folder= annotations_folder) # Remove all 'UNKNOWN' labels from the annotation files


# TRAIN, TEST,  SPLIT
from utils import convert_files_to_list, split_and_copy_files

annotations_folder = r'1_GA_Dataset\20250513\Annotations' # Change this to the correct folder for which files are to be extracted to
images_folder = r'1_GA_Dataset\20250513\Images' # Change this to the correct folder for which files are to be extracted to

images, annotations = convert_files_to_list(images_folder=images_folder, 
                                            annotations_folder=annotations_folder) # Convert to list 


output_folder = r'1_GA_Dataset\20250513\Split' #output folder forw here split is stored 
split_and_copy_files(images, annotations, #create_testtrain_folders, copy files, then split into test and train
                     output_folder= output_folder) 

################################################


# DATA AUGMENTATION
from augmentation import define_and_generate_transformations, run_augmentation_pipeline

random_augmentations = define_and_generate_transformations(num_random_augmentations=7, # Number of random augmentations to generate
                                                include_color_jitter=True, #True if want to include color jitter, False if not
                                                include_horizontal_flip=True, ##True if want to include horizontal flip, False if not
                                                include_vertical_flip=True, ##True if want to include vertical flip, False if not
                                                include_photometric_distort=False) #True if want to include photometric distort, False if not

# Run augmentation pipeline for each combination of transformations/random generated combinations 
augmented_dataset_path = r'2_DataAugmentation' # if needed, current_date = datetime.now().strftime('%Y%m%d')
date_of_dataset_used = '20250513'
image_dir =  r'1_GA_Dataset\20250513\Split\train\images' #Original SPLIT annotation folder in GA_Dataset 
annotation_dir = r'1_GA_Dataset\20250513\Split\train\annotations' #Original SPLIT image folder in GA_Dataset

run_augmentation_pipeline(augmented_dataset_path=augmented_dataset_path,
                            date_of_dataset_used=date_of_dataset_used,
                            image_dir=image_dir,
                            annotation_dir=annotation_dir,
                            augmentation=random_augmentations,  # List of transformations to apply
                            num_pairs=2,  # Number of pairs to generate for each transformation
                            object_detector=True)  # True if using object detector, False if not


# NOT ALL TRANSFORMATIONS ARE WORKING PROPERLY.. seem random?? 
# RANDOM ROTATION RETURNS ISSUE WITH BOUNDING BOX TRANSFORMATIONS (not correctly rotated in line with image)
# is it ok to augment in this way..? because realisticallty, i have now 10 sets of the 'same' datatset, but with different augmentations applied to them.... 





########################################################## TRAINING DATASET CREATION ##########################################################

#CREATE AUGMENTATED DATASET FOLDER
# Create dataset folder because we are combining multiple datasets:
                                                                # 1_GA_Dataset\20250221
                                                                # 1_GA_Dataset\20250221_randomhorizontalflip
                                                                # 1_GA_Dataset\20250221_randomverticalflip
                                                                # 1_GA_Dataset\20250221_randomrotation
                                                            # This forms the final training dataset folder that will be used for training (original + augmented)
                                                            # Training dataset: train + train augmented + test + val
from utils import create_dataset_folder

create_dataset_folder(base_folder=r'3_TrainingData', #base folder where dataset is stored
                      folder_date='20250513', #date of dataset created
                      Split=True, #True if want the folder to contain split folders (train, test, val), False if not
                      Augmented=True) 




# EXTRACT DATA AUGMENTATION FILES TO TRAINING FOLDER
# Extract files from the augmented dataset and original dataset folders to the TRAIN split

from utils import extraction_pipeline

source_folders = [
    r'1_GA_Dataset\20250513\Split',  # Original, not augmented TRAIN dataset
    r'2_DataAugmentation\20250513'  # Augmented TRIAN datasets
]
# Change to the correct folder for which files are to be extracted to
training_data_folder = r'3_TrainingData\20250513_Augmented' 
annotations_folder = r'3_TrainingData\20250513_Augmented\Split\train\annotations' 
images_folder = r'3_TrainingData\20250513_Augmented\Split\train\images'

extraction_pipeline(source_folders=source_folders, # Source to extract from
                    training_data_folder=training_data_folder,
                    annotations_folder=annotations_folder, # Destination for annotations
                    images_folder=images_folder, # Destination for images
                    train_test_val= ['train'], # For train split, train_test_val = 'train' because we are including train data
                    raw_data=False, # For train split, raw_data = False because we are using the augmented data
                    include_augmentation_list=True) # For train split, include_augmentation_list = True because we are including the augmentation list (using augmented data)


# EXTRACT FILES FROM DATASET FOLDERS TO THE TEST SPLIT
source_folders = [
    r'1_GA_Dataset\20250513\Split']  # Original, not augmented TEST dataset

# Change this to the correct folder for which files are to be extracted to
annotations_folder = r'3_TrainingData\20250513_Augmented\Split\test\annotations' 
images_folder = r'3_TrainingData\20250513_Augmented\Split\test\images' 

extraction_pipeline(source_folders= source_folders, # Source to extract from
                    training_data_folder=training_data_folder,
                    annotations_folder= annotations_folder, # Destination for annotations
                    images_folder= images_folder, # Destination for images
                    train_test_val= ['test'], # For test split, train_test_val = 'test' because we are including test data
                    raw_data=False, # For test split, raw_data = False because we are using the augmented data
                    include_augmentation_list=False) # For test split, include_augmentation_list = False because we do not need to include the augmentation list (using raw data)


### Extract files from dataset folders to the VAL split
source_folders = [
        r'1_GA_Dataset\20250513\Split']  # Original, not augmented VAL dataset

annotations_folder = r'3_TrainingData\20250513_Augmented\Split\val\annotations' # Change this to the correct folder for which files are to be extracted to
images_folder = r'3_TrainingData\20250513_Augmented\Split\val\images' # Change this to the correct folder for which files are to be extracted to

extraction_pipeline(source_folders= source_folders, # Source to extract from
                    training_data_folder=training_data_folder,
                    annotations_folder= annotations_folder, # Destination for annotations
                    images_folder= images_folder, # Destination for images
                    train_test_val= ['val'], # For val split, train_test_val = 'val' because we are including val data
                    raw_data=False, # For val split, raw_data = False because we are using the augmented data
                    include_augmentation_list=False) # For val split, include_augmentation_list = False because we do not need to include the augmentation list (using raw data)


# DATALOADER CREATION
# Creating datalists for the train,  and test data
from utils import create_data_lists
from label_map import label_map_Classifier # Label map (explicitly defined)
from label_map import label_map_OD # Label map (object detector)
import shutil

train_annotation_path= r'3_TrainingData\20250513_Augmented\Split\train\annotations'
train_image_path= r'3_TrainingData\20250513_Augmented\Split\train\images'  
test_annotation_path= r'3_TrainingData\20250513_Augmented\Split\test\annotations'
test_image_path= r'3_TrainingData\20250513_Augmented\Split\test\images'
date_of_dataset_used='20250513'

create_data_lists(train_annotation_path=train_annotation_path,
                train_image_path=train_image_path,
                test_annotation_path=test_annotation_path,
                test_image_path=test_image_path,
                object_detector=True, #True if using object detector, False if not
                date_of_dataset_used=date_of_dataset_used,
                augmented=True,
                JSON_folder=r'4_JSON_folder')


# Custom dataloaders
from dataset import PC_Dataset  # Custom dataset class for loading
data_folder = r'3_TrainingData\20250513_Augmented\Split'  # Folder containing the training data

# training dataset and dataloader
train_dataset = PC_Dataset(data_folder,
                            split='train')
train_loader = torch.utils.data.DataLoader(train_dataset, 
                                           batch_size=batch_size, shuffle=True,
                                            collate_fn=train_dataset.collate_fn,  # custom collate function
                                            num_workers=workers,
                                            pin_memory=True)  # note that we're passing the collate function here

# validation dataset and dataloader
validation_dataset = PC_Dataset(data_folder,
                                    split='val') 
validation_loader = torch.utils.data.DataLoader(validation_dataset,
                                                batch_size=batch_size, shuffle=False,
                                                collate_fn=validation_dataset.collate_fn,  # custom collate function
                                                num_workers=workers,
                                                pin_memory=True)  # note that we're passing the collate function here

# test dataset and dataloader
test_dataset = PC_Dataset(data_folder,
                            split='test')
test_loader = torch.utils.data.DataLoader(test_dataset,
                                           batch_size=batch_size, shuffle=False,
                                            collate_fn=test_dataset.collate_fn, # custom collate function
                                            num_workers=workers,
                                            pin_memory=True)  # note that we're passing the collate function here







# TRAIN MODEL
from utils import check_model_trained # on the otherhand, this is fine?
# when the function 'check_model_trained' is imported from 'train_custom'.
# Why does this function return:
# usage: [-h] --object_detector {yes,no} [--data_folder DATA_FOLDER] [--training_output_file TRAINING_OUTPUT_FILE] [--save_dir SAVE_DIR]
# : error: the following arguments are required: --object_detector
# but not when imported from 'utils'?


####  also why do i ned to keep saving in order for changes to appear in source control? ####

checkpoint_dir = r'6_Checkpoints'  # Directory where checkpoints are stored
date_of_dataset_used = '20250513'  # Date of dataset used for training
check_model_trained(checkpoint_dir=checkpoint_dir,
                    date_of_dataset_used=date_of_dataset_used,
                    augmented=True)  # Check if model is already trained and present



from hyperparameters import *
from utils import manage_training_output_file
# # when the function 'manage_training_output_file' is imported from 'train_custom'.
# Why does this function return:
# usage: [-h] --object_detector {yes,no} [--data_folder DATA_FOLDER] [--training_output_file TRAINING_OUTPUT_FILE] [--save_dir SAVE_DIR]
# : error: the following arguments are required: --object_detector
# but not when imported from 'utils'? 
# same problem as above, but with 'check_model_trained' function 

date_of_dataset_used = '20250513'  # Date of dataset used for training
results_folder = r'5_Results'  # Folder to save results
training_output_file = manage_training_output_file(results_folder=results_folder,
                                                   date_of_dataset_used=date_of_dataset_used,
                                                   augmented=True)  # augmented_data if augmented dataset used

# TRAIN MODEL: Run the training process and save the output
data_folder = r'3_TrainingData\20250513_Augmented\Split'  # Folder containing the training data
date_of_dataset_used = '20250513'  # Date of dataset used for training


subprocess.run(['python', 'train_custom.py', #change model, optimizer, loss_fn, etc. in train_custom.py 
                '--data_folder', data_folder, 
                '--training_output_file', training_output_file, 
                '--save_dir', r'6_Checkpoints',
                '--object_detector', 'yes',])

           

# Return the relative file path of the training output file
print(f"Training output file saved at: {os.path.relpath(training_output_file)}")





































# keep only relevant checkpoints
from utils import keep_checkpoints
import fileinput
import sys

training_output_file =  r'5_Results\training_results_20250221_augmented.txt'
date_of_dataset_used = '20250221_augmented'
keep_checkpoints(checkpoint_dir=r'6_Checkpoints', 
                 log_file= training_output_file,
                 date_of_dataset_used= date_of_dataset_used)


# EVALUATE MODEL

# Evaluate the model and save the results to a .txt file
checkpoint = r''
results_folder = r'5_Results'
evaluation_output_file = os.path.join(results_folder, f'evaluation_results_{os.path.basename(checkpoint)}.txt')

# Ensure the results folder exists
os.makedirs(results_folder, exist_ok=True)

with open(evaluation_output_file, 'w') as f:
    subprocess.run(['python', 'e.py',
                    '--data_folder', r'4_JSON_folder\20250219',
                    '--checkpoint', checkpoint], stdout=f)


# Inference 
subprocess.run(['python', 'detect.py',
                '--checkpoint', checkpoint,
                '--img_path', r'1_GA_Dataset/Split/test/images/0.tif'])

