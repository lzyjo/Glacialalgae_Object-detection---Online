# Checking and setting cwd 
import json
import os
import pandas as pd
import subprocess
from torchvision.transforms import v2 as T
from datetime import datetime


## Check the current working directory
cwd = os.getcwd()
print(f"Current working directory: {cwd}")



#CREATE DATASET FOLDER
from utils import create_dataset_folder

# Create dataset folder because we are combining multiple datasets:
                                                                # 1_GA_Dataset\20250221
                                                                # 1_GA_Dataset\20250221_randomhorizontalflip
                                                                # 1_GA_Dataset\20250221_randomverticalflip
                                                                # 1_GA_Dataset\20250221_randomrotation


create_dataset_folder(base_folder=r'1_GA_Dataset', #base folder where dataset is stored
                      folder_date=None, #if None, current_date = datetime.now().strftime('%Y%m%d')
                      Augmented=False,
                      Split=True) #date of dataset created


# EXTRACT RAW .XMLS AND .TIFS INTO (MASTERLIST) DATASET FOLDER
from utils import extraction_pipeline

### Extract files from the augmented dataset folders to the TRAIN split
source_folders = [
    r'0_Completed annotations\Bluff_230724',  # Original dataset, not augmented
    r'0_Completed annotations\PAM_Surf_220724'  # Augmented datasets
]

annotations_folder = r'1_GA_Dataset/20250513/Annotations' # Change this to the correct folder for which files are to be extracted to
images_folder = r'1_GA_Dataset/20250513/Images' # Change this to the correct folder for which files are to be extracted to

extraction_pipeline(source_folders=source_folders,
                    annotations_folder=annotations_folder, images_folder=images_folder,
                    train_test_val=False, # For train split, include_test = False because we are not including test data in the training data
                    raw_data=True,
                    include_augmentation_list=False) # Call the function to extract files from multiple folders







## LABELS ARE CELL ONLY + NO UNKNOWNS 
from OD_misc_utils import convert_labels_to_cell, remove_unknowns_from_labels

annotations_folder = r'1_GA_Dataset\20250513\Annotations' # Change this to the correct folder for which files are to be extracted to

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



#CREATE AUGMENTATED DATASET FOLDER
from utils import create_dataset_folder

# Create dataset folder because we are combining multiple datasets:
                                                                # 1_GA_Dataset\20250221
                                                                # 1_GA_Dataset\20250221_randomhorizontalflip
                                                                # 1_GA_Dataset\20250221_randomverticalflip
                                                                # 1_GA_Dataset\20250221_randomrotation

create_dataset_folder(base_folder=r'3_TrainingData', #base folder where dataset is stored
                      folder_date='20250513', #date of dataset created
                      Split=True, #True if want to split the dataset into train/test/, False if not
                      Augmented=True) 




# EXTRACT DATA AUGMENTATION FILES TO TRAINING FOLDER
from utils import extraction_pipeline

### Extract files from the augmented dataset folders to the TRAIN split
source_folders = [
    r'1_GA_Dataset\20250513\Split',  # Original dataset, not augmented
    r'2_DataAugmentation\20250513'  # Augmented datasets
]

training_data_folder = r'3_TrainingData\20250513_Augmented' # Change this to the correct folder for which files are to be extracted to
annotations_folder = r'3_TrainingData\20250513_Augmented\Split\train\annotations' # Change this to the correct folder for which files are to be extracted to
images_folder = r'3_TrainingData\20250513_Augmented\Split\train\images' # Change this to the correct folder for which files are to be extracted to

# Extract files from the raw and augmented dataset folders to the TRAIN split
extraction_pipeline(source_folders=source_folders, # Source to extract from
                    annotations_folder=annotations_folder, # Destination for annotations
                    images_folder=images_folder, # Destination for images
                    train_test_val= ['train'], # For train split, train_test_val = 'train' because we are including train data
                    raw_data=False, # For train split, raw_data = False because we are using the augmented data
                    include_augmentation_list=True) # For train split, include_augmentation_list = True because we are including the augmentation list (using augmented data)


### Extract files from dataset folders to the TEST split
source_folders = [
    r'1_GA_Dataset\20250513\Split']  # Original dataset, not augmented 

annotations_folder = r'3_TrainingData\20250513_Augmented\Split\test\annotations' # Change this to the correct folder for which files are to be extracted to
images_folder = r'3_TrainingData\20250513_Augmented\Split\test\images' # Change this to the correct folder for which files are to be extracted to

extraction_pipeline(source_folders= source_folders, # Source to extract from
                    annotations_folder= annotations_folder, # Destination for annotations
                    images_folder= images_folder, # Destination for images
                    train_test_val= ['test'], # For test split, train_test_val = 'test' because we are including test data
                    raw_data=False, # For test split, raw_data = False because we are using the augmented data
                    include_augmentation_list=False) # For test split, include_augmentation_list = False because we do not need to include the augmentation list (using raw data)


### Extract files from dataset folders to the VAL split
source_folders = [
        r'1_GA_Dataset\20250513\Split']  # Original dataset, not augmented 

annotations_folder = r'3_TrainingData\20250513_Augmented\Split\val\annotations' # Change this to the correct folder for which files are to be extracted to
images_folder = r'3_TrainingData\20250513_Augmented\Split\val\images' # Change this to the correct folder for which files are to be extracted to

extraction_pipeline(source_folders= source_folders, # Source to extract from
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



# TRAIN MODEL
from train_custom import check_model_trained

checkpoint_dir = r'6_Checkpoints'  # Directory where checkpoints are stored
date_of_dataset_used = '20250513'  # Date of dataset used for training
check_model_trained(checkpoint_dir=checkpoint_dir,
                    date_of_dataset_used=date_of_dataset_used,
                    augmented=True)  # Check if model is already trained and present



from hyperparameters import *
from train_custom import manage_training_output_file
date_of_dataset_used = '20250513'  # Date of dataset used for training
results_folder = r'5_Results'  # Folder to save results
training_output_file = manage_training_output_file(results_folder=results_folder,
                                                   date_of_dataset_used=date_of_dataset_used,
                                                   augmented=True)  # augmented_data if augmented dataset used

# TRAIN MODEL: Run the training process and save the output
data_folder = r'4_JSON_folder\20250513_Augmented'
date_of_dataset_used = '20250513'  # Date of dataset used for training


subprocess.run(['python', 'train_custom.py',
                '--data_folder', data_folder, 
                '--training_output_file', training_output_file, 
                '--save_dir', r'6_Checkpoints',
                '--object_detector', 'yes',])


           

# Return the relative file path of the training output file
print(f"Training output file saved at: {os.path.relpath(training_output_file)}")





































# keep only relevant checkpoints
from utils import keep_checkpoints

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

