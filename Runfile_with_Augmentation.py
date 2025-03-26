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
create_dataset_folder(folder_type='no_augmentation', 
                      folder_date='20250221')  # Only run if you want to create a new dataset folder!!


# EXTRACT RAW .XMLS AND .TIFS INTO (MASTERLIST) DATASET FOLDER
from utils import extract_files


annotations_folder = r'1_GA_Dataset\20250318\Annotations' # Change this to the correct folder for which files are to be extracted to
images_folder = r'1_GA_Dataset\20250318\Images' # Change this to the correct folder for which files are to be extracted to

## Run only once for each dataset for file extraction

##BLUFF_230724 DATA
extract_files(date_of_dataset_used= 'Bluff_230724', # Change this to the correct dataset used, FOR REFERENCE ONLY
                annotations_folder= annotations_folder, 
                images_folder= images_folder, 
                images_src_folder=r'0_Completed annotations/Bluff_230724/Original_Images_Unlabelled_Bluff_230724', # Change this to your source folder path 
                annotations_src_folder=r'0_Completed annotations\Bluff_230724') # Change this to your source folder path

##PAM_Surf_220724 DATA
extract_files(date_of_dataset_used= 'PAM_Surf_220724', # Change this to the correct dataset used, FOR REFERENCE ONLY
                annotations_folder= annotations_folder, 
                images_folder= images_folder, 
                images_src_folder=r'0_Completed annotations/PAM_Surf_220724/Original_Images_Unlabelled_PAM_Surf_220724', # Change this to your source folder path 
                annotations_src_folder=r'0_Completed annotations\PAM_Surf_220724') # Change this to your source folder path




# DATA AUGMENTATION
from augmentation import *

augmented_dataset_path = r'2_Data_Augmentation'
current_date = datetime.now().strftime('%Y%m%d')
date_of_dataset_used = '20250318'
image_dir = r'1_GA_Dataset\20250318\Annotations' #Original annotation folder in GA_Dataset 
annotation_dir = r'1_GA_Dataset\20250318\Images' #Original image folder in GA_Dataset



# Define the transformations to be included + generate all possible pairings
color_jitter = T.ColorJitter(brightness=(0,1), contrast=(0,1), saturation=(0,1), hue=(0,0.5))

ColourJitter1_params = T.ColorJitter.get_params(
    color_jitter.brightness, color_jitter.contrast, color_jitter.saturation,
    color_jitter.hue) 
ColourJitter2_params = T.ColorJitter.get_params(
    color_jitter.brightness, color_jitter.contrast, color_jitter.saturation,
    color_jitter.hue) #must name as such to allow folder name to be created correctly (transformation name in position 0)
# check with levi how to properly implement get_params

transformations_to_include = [
    T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.5),
    T.ColorJitter(brightness=ColourJitter1_params[1], 
                  contrast=ColourJitter1_params[2], 
                  saturation=ColourJitter1_params[3], 
                  hue=ColourJitter1_params[4]),
    T.ColorJitter(brightness=ColourJitter2_params[1],
                  contrast=ColourJitter2_params[2],
                  saturation=ColourJitter2_params[3],
                  hue=ColourJitter2_params[4])
                  ]

# Generate all possible pairings of transformations
all_transformations = generate_all_transformations(transformations_to_include)
number_of_pairings = len(all_transformations)

# Run augmentation pipeline for each combination of transformations
for i, transformations in enumerate(all_transformations):
    print(f"Running augmentation pipeline for transformation set {i+1}/{len(all_transformations)}")
    run_augmentation_pipeline(augmented_dataset_path=augmented_dataset_path,
                              date_of_dataset_used=date_of_dataset_used,
                              image_dir=image_dir,
                              annotation_dir=annotation_dir,
                              transformations=transformations,
                              num_pairs=5)

# RANDOM ROTATION RETURNS ISSUE WITH BOUNDING BOX TRANSFORMATIONS (not correctly rotated in line with image)




#CREATE AUGMENTATED DATASET FOLDER
from utils import create_dataset_folder

# Create dataset folder because we are combining multiple datasets:
                                                                # 1_GA_Dataset\20250221
                                                                # 1_GA_Dataset\20250221_randomhorizontalflip
                                                                # 1_GA_Dataset\20250221_randomverticalflip
                                                                # 1_GA_Dataset\20250221_randomrotation
create_dataset_folder(folder_type='augmented_data', #augmented_data if augmented dataset used,
                                                    #None if no augmentation used                            
                      folder_date='20250221')  # Only run if you want to create a new dataset folder!!



# EXTRACT AUGMENTED FILES TO TRAINING FOLDER
from utils import extract_files


annotations_folder = r'3_TrainingData\20250221\Annotations' # Change this to the correct folder for which files are to be extracted to
images_folder = r'3_TrainingData\20250221\Images' # Change this to the correct folder for which files are to be extracted to

## Run only once for each dataset for file extraction
##20250221 (not augmented) DATA
extract_files(date_of_dataset_used= '20250221', # Change this to the correct dataset used, FOR REFERENCE ONLY
                annotations_folder= annotations_folder, 
                images_folder= images_folder, 
                images_src_folder=r'1_GA_Dataset\20250221\Images', # Change this to your source folder path 
                annotations_src_folder=r'1_GA_Dataset\20250221\Annotations') # Change this to your source folder path

##20250221_randomhorizontalflip DATA
extract_files(date_of_dataset_used= '20250221_randomhorizontalflip', # Change this to the correct dataset used, FOR REFERENCE ONLY
                annotations_folder= annotations_folder, 
                images_folder= images_folder, 
                images_src_folder=r'1_GA_Dataset\20250221_randomhorizontalflip\Images', # Change this to your source folder path 
                annotations_src_folder=r'1_GA_Dataset\20250221_randomhorizontalflip\Annotations') # Change this to your source folder path

##20250221_randomverticalflip DATA
extract_files(date_of_dataset_used= '20250221_randomverticalflip', # Change this to the correct dataset used, FOR REFERENCE ONLY
                annotations_folder= annotations_folder,
                images_folder= images_folder,
                images_src_folder=r'1_GA_Dataset\20250221_randomverticalflip\Images', # Change this to your source folder path
                annotations_src_folder=r'1_GA_Dataset\20250221_randomverticalflip\Annotations') # Change this to your source folder path

## 1_GA_Dataset/20250221_randomhorizontalflip_randomverticalflip DATA
extract_files(date_of_dataset_used= '20250221_randomhorizontalflip_randomverticalflip', # Change this to the correct dataset used, FOR REFERENCE ONLY
                annotations_folder= annotations_folder,
                images_folder= images_folder,
                images_src_folder=r'1_GA_Dataset\20250221_randomhorizontalflip_randomverticalflip\Images', # Change this to your source folder path
                annotations_src_folder=r'1_GA_Dataset\20250221_randomhorizontalflip_randomverticalflip\Annotations') # Change this to your source folder path

##2_Data_Augmentation/20250221_randomhorizontalflip_randomverticalflip_colorjitter_colorjitter
extract_files(date_of_dataset_used= '20250221_randomhorizontalflip_randomverticalflip_colorjitter_colorjitter', # Change this to the correct dataset used, FOR REFERENCE ONLY
                annotations_folder= annotations_folder,
                images_folder= images_folder,
                images_src_folder=r'2_Data_Augmentation\20250221_randomhorizontalflip_randomverticalflip_colorjitter_colorjitter\Images', # Change this to your source folder path
                annotations_src_folder=r'2_Data_Augmentation\20250221_randomhorizontalflip_randomverticalflip_colorjitter_colorjitter\Annotations') # Change this to your source folder path





# TRAIN, TEST, VAL SPLIT
from utils import convert_files_to_list, split_and_copy_files
images, annotations = convert_files_to_list(images_folder=images_folder, 
                                            annotations_folder=annotations_folder) # Convert to list 

output_folder = r'3_TrainingData\20250221\Split' #output folder forw here split is stored 
split_and_copy_files(images, annotations, #create_folders, copy files, then split into test and train
                     output_folder= output_folder) 



# DATALOADER CREATION
# Creating datalists for the train, val and test data
from utils import create_data_lists
from label_map import label_map # Label map (explicitly defined)
import shutil

train_annotation_path= r'3_TrainingData\20250221\Split\train\annotations'
train_image_path= r'3_TrainingData\20250221\Split\train\images'  
test_annotation_path= r'3_TrainingData\20250221\Split\test\annotations'
test_image_path= r'3_TrainingData\20250221\Split\test\images'
date_of_dataset_used='20250221'
augmentation = 'augmented_data' #augmented_data if augmented dataset used,
                                #None if no augmentation used                            

create_data_lists(train_annotation_path=train_annotation_path,
                train_image_path=train_image_path,
                test_annotation_path=test_annotation_path,
                test_image_path=test_image_path,
                label_map=label_map,
                date_of_dataset_used=date_of_dataset_used,
                augmentation= augmentation,
                JSON_folder=r'4_JSON_folder')





# Check if model is already trained and present 
date_of_dataset_used = '20250221'
date_of_dataset_used = date_of_dataset_used + '_augmented' #if augmented dataset used
model_path = os.path.join(date_of_dataset_used + '_checkpoint_ssd300.pth')

if os.path.exists(model_path):
    print(f'Model for date: {date_of_dataset_used} already trained: {model_path} present in cwd')
else:
    print(f'Model for date: {date_of_dataset_used} not trained or present: {model_path} not present in cwd')



# TRAIN MODEL

## Suppress specific warnings
import warnings
warnings.filterwarnings("ignore")


# Adjust hyperparameters in hyperparameters.py now
from hyperparameters import *
from utils import manage_training_output_file


## Training the model and saving the results to a .txt file

## Set up training output file
date_of_dataset_used = '20250221_augmented'
results_folder = r'5_Results'
training_output_file = manage_training_output_file(results_folder = results_folder,
                                                date_of_dataset_used= date_of_dataset_used)


# Run the training process and save the output
data_folder = r'4_JSON_folder\20250221_augmented'

with open(training_output_file, 'a') as f:
    subprocess.run(['python', 'train.py', 
                    '--data_folder', data_folder,
                    '--date_of_dataset_used', date_of_dataset_used,
                    '--save_dir', r'6_Checkpoints'], 
                    stdout=f)

# Run the training process and save the output
with open(training_output_file, 'a') as f:
    result = subprocess.run(['python', 'train.py', 
                '--data_folder', data_folder,
                '--date_of_dataset_used', date_of_dataset_used,
                '--save_dir', r'6_Checkpoints',
                '--checkpoint', checkpoint if checkpoint else '',
                '--checkpoint_frequency', checkpoint_frequency,
                '--lr', lr,
                '--iterations', iterations], capture_output=True, text=True)
    for line in result.stdout.splitlines():
        if line.startswith('Epoch:'):  # write lines starting with 'Epoch:'
            f.write(line + '\n')

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

