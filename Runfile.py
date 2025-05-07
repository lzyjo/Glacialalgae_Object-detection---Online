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
from utils import extract_files_from_multiple_folders

### Extract files from the augmented dataset folders to the TRAIN split
source_folders = [
    r'0_Completed annotations\Bluff_230724 copy',  # Original dataset, not augmented
    r'0_Completed annotations\PAM_Surf_220724 copy'  # Augmented datasets
]

annotations_folder = r'1_GA_Dataset/20250506/Annotations' # Change this to the correct folder for which files are to be extracted to
images_folder = r'1_GA_Dataset/20250506/Images' # Change this to the correct folder for which files are to be extracted to

extract_files_from_multiple_folders(source_folders=source_folders,
                                    annotations_folder=annotations_folder, images_folder=images_folder,
                                    include_test=False,
                                    raw_data=True) # Call the function to extract files from multiple folders







from utils import move_raw_images_to_dataset

move_raw_images_to_dataset(master_folder=r'0_Completed annotations\Bluff_230724 copy\Bluff_230724_Raw_Images',
                           destination_folder=r'0_Completed annotations\Bluff_230724 copy')







## LABELS ARE CELL ONLY + NO UNKNOWNS 
from OD_misc_utils import convert_labels_to_cell, remove_unknowns_from_labels

annotations_folder = r'1_GA_Dataset\20250318\Annotations' # Change this to the correct folder for which files are to be extracted to

convert_labels_to_cell(annotations_folder= annotations_folder) # Convert all labels to 'cell' in the annotation files
remove_unknowns_from_labels(annotations_folder= annotations_folder) # Remove all 'UNKNOWN' labels from the annotation files


# TRAIN, TEST, VAL SPLIT
from utils import convert_files_to_list, split_and_copy_files

annotations_folder = r'1_GA_Dataset\20250318\Annotations' # Change this to the correct folder for which files are to be extracted to
images_folder = r'1_GA_Dataset\20250318\Images' # Change this to the correct folder for which files are to be extracted to

images, annotations = convert_files_to_list(images_folder=images_folder, 
                                            annotations_folder=annotations_folder) # Convert to list 


output_folder = r'1_GA_Dataset\20250318\Split' #output folder forw here split is stored 
split_and_copy_files(images, annotations, #create_testtrain_folders, copy files, then split into test and train
                     output_folder= output_folder) 








# DATA AUGMENTATION
from augmentation import *


random_augmentations = define_and_generate_transformations(num_random_augmentations=1, # Number of random augmentations to generate
                                                include_color_jitter=True, #True if want to include color jitter, False if not
                                                include_horizontal_flip=True, ##True if want to include horizontal flip, False if not
                                                include_vertical_flip=True, ##True if want to include vertical flip, False if not
                                                include_photometric_distort=False) #True if want to include photometric distort, False if not

# Run augmentation pipeline for each combination of transformations/random generated combinations 
augmented_dataset_path = r'2_DataAugmentation' # if needed, current_date = datetime.now().strftime('%Y%m%d')
date_of_dataset_used = '20250318'
image_dir =  r'1_GA_Dataset\20250318\Split\train\images' #Original SPLIT annotation folder in GA_Dataset 
annotation_dir = r'1_GA_Dataset\20250318\Split\train\annotations' #Original SPLIT image folder in GA_Dataset

run_augmentation_pipeline(augmented_dataset_path=augmented_dataset_path,
                            date_of_dataset_used=date_of_dataset_used,
                            image_dir=image_dir,
                            annotation_dir=annotation_dir,
                            augmentation=random_augmentations,  # List of transformations to apply
                            num_pairs=5,  # Number of pairs to generate for each transformation
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
                      folder_date='20250318', #date of dataset created
                      Augmented=True) 




# EXTRACT DATA AUGMENTATION FILES TO TRAINING FOLDER
from utils import extract_files, extract_files_from_multiple_folders

### Extract files from the augmented dataset folders to the TRAIN split
source_folders = [
    r'1_GA_Dataset\20250318\Split',  # Original dataset, not augmented
    r'2_DataAugmentation\20250318'  # Augmented datasets
]

training_data_folder = r'3_TrainingData\20250318_Augmented' # Change this to the correct folder for which files are to be extracted to
annotations_folder = r'3_TrainingData\20250318_Augmented\Split\train\annotations' # Change this to the correct folder for which files are to be extracted to
images_folder = r'3_TrainingData\20250318_Augmented\Split\train\images' # Change this to the correct folder for which files are to be extracted to

extract_files_from_multiple_folders(source_folders, annotations_folder, images_folder,
                                    include_test=False) # Call the function to extract files from multiple folders


### Extract files from dataset folders to the TEST split
source_folders = [
    r'1_GA_Dataset\20250318\Split']  # Original dataset, not augmented 

annnotations_folder = r'3_TrainingData\20250318_Augmented\Split\test\annotations' # Change this to the correct folder for which files are to be extracted to
images_folder = r'3_TrainingData\20250318_Augmented\Split\test\images' # Change this to the correct folder for which files are to be extracted to

extract_files_from_multiple_folders(source_folders, annnotations_folder, images_folder,
                                    include_test=True) # Call the function to extract files from multiple folders







# DATALOADER CREATION
# Creating datalists for the train, val and test data
from utils import create_data_lists
from label_map import label_map_Classifier # Label map (explicitly defined)
from label_map import label_map_OD # Label map (object detector)
import shutil

train_annotation_path= r'3_TrainingData\20250318_Augmented\Split\train\annotations'
train_image_path= r'3_TrainingData\20250318_Augmented\Split\train\images'  
test_annotation_path= r'3_TrainingData\20250318_Augmented\Split\test\annotations'
test_image_path= r'3_TrainingData\20250318_Augmented\Split\test\images'
date_of_dataset_used='20250318'
augmentation = 'augmented_data' #augmented_data if augmented dataset used,
                                #None if no augmentation used                            

create_data_lists(train_annotation_path=train_annotation_path,
                train_image_path=train_image_path,
                test_annotation_path=test_annotation_path,
                test_image_path=test_image_path,
                object_detector=True, #True if using object detector, False if not
                date_of_dataset_used=date_of_dataset_used,
                augmentation= augmentation,
                JSON_folder=r'4_JSON_folder')



# TRAIN MODEL
from utils import check_model_trained

checkpoint_dir = r'6_Checkpoints'  # Directory where checkpoints are stored
date_of_dataset_used = '20250318'  # Date of dataset used for training
check_model_trained(checkpoint_dir=checkpoint_dir,
                    date_of_dataset_used=date_of_dataset_used,
                    augmented=True)  # Check if model is already trained and present



from hyperparameters import *
from utils import manage_training_output_file
results_folder = r'5_Results'
date_of_dataset_used = '20250318'  # Date of dataset used for training
training_output_file = manage_training_output_file(results_folder=results_folder,
                                                   date_of_dataset_used=date_of_dataset_used,
                                                   augmented=True)  # augmented_data if augmented dataset used

# TRAIN MODEL: Run the training process and save the output
data_folder = r'4_JSON_folder\20250318_Augmented'
date_of_dataset_used = '20250318'  # Date of dataset used for training

# Run the training process and save the output
with open(training_output_file, 'a') as f:
    try:
        result = subprocess.run(['python', 'train.py', 
                    '--data_folder', data_folder,
                    '--date_of_dataset_used', date_of_dataset_used,
                    '--object_detector', 'yes',
                    '--save_dir', r'6_Checkpoints'],
                    capture_output=True, text=True, check=True)
        if result.stdout:
            for line in result.stdout.splitlines():
                if line.startswith('Epoch:'):  # write lines starting with 'Epoch:'
                    f.write(line + '\n')
    except subprocess.CalledProcessError as e:
        print(f"Error occurred during training: {e}")
        if e.stdout:
            print("Standard Output:")
            print(e.stdout)
            f.write("Standard Output:\n")
            f.write(e.stdout + '\n')
        if e.stderr:
            print("Standard Error:")
            print(e.stderr)
            f.write("Standard Error:\n")
            f.write(e.stderr + '\n')
        f.write(f"Error occurred during training: {e}\n")


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
    subprocess.run(['python', 'eval.py',
                    '--data_folder', r'4_JSON_folder\20250219',
                    '--checkpoint', checkpoint], stdout=f)


# Inference 
subprocess.run(['python', 'detect.py',
                '--checkpoint', checkpoint,
                '--img_path', r'1_GA_Dataset/Split/test/images/0.tif'])

