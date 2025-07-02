import os
import shutil
import json
import xml.etree.ElementTree as ET
from matplotlib.pyplot import box
import torch
import random
import torchvision.transforms.functional as FT
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score, precision_score, recall_score #metrics for classification
from sklearn.metrics import average_precision_score, jaccard_score #metrics for object detection
from torchmetrics.detection.mean_ap import MeanAveragePrecision #metrics for object detection
from torchmetrics.detection.iou import IntersectionOverUnion #metrics for object detection
                                                        # JaccardIndex is used for classification, not object detection
import pandas as pd
from datetime import datetime
from label_map import label_map_Classifier
from label_map import  label_map_OD
from hyperparameters import *
from hyperparameters import optimizer, loss_fn, model
import csv



device = torch.device("cpu")

optimizer = optimizer

####################################################### DATA HANDLING ############################################################
"""
The provided code implements a pipeline for dataset preparation, training, and evaluation in object detection.
It includes functions for creating dataset folders, extracting files, splitting datasets, generating JSON data lists, 
and managing training processes.

- create_dataset_folder: 
Creates a structured folder hierarchy for datasets, including subfolders for annotations, images, and optional train/test splits.

- extract_files: 
Extracts .xml (annotations) and .tif (images) files from source folders to destination folders, ensuring unique filenames.

- move_raw_images_to_dataset:
Moves raw image subfolders into a destination folder and renames them to raw_images.

- process_raw_data:
Processes raw data by identifying subfolders containing raw images and annotations, organizing them into a structured format.

- process_standard_data:
Identifies subfolders containing Images and Annotations in standard datasets and organizes them into a list of dictionaries.

- confirm_and_extract:
Prompts the user to confirm file extraction and calls extract_files for each dataset.

- extraction_pipeline: 
Processes multiple source folders and organizes their contents into specified destination folders.

"""

def create_dataset_folder(base_folder='1_GA_Dataset',
                          folder_date=None,
                          Augmented=False,
                          Split=False):  # Changed parameter name to 'Augmented'
    """
    Creates a dataset folder structure for training data, including subfolders for annotations, images, 
    and data splits. The folder structure is determined based on the specified date and whether the dataset is augmented.
    Parameters:
        folder_date (str or None): The date string (in 'YYYYMMDD' format) for the folder. 
                                   If None, the current date is used.
        Augmented (bool): Whether the dataset is augmented. If True, '_Augmented' is appended to the folder name.
        Split (bool): Whether to create train, test, and val subfolders for images and annotations.
    Returns:
        tuple: A tuple containing the paths to the created folders:
               (annotations_folder, images_folder, split_folder, date_folder, train_image_folder, train_annotation_folder, 
               test_image_folder, test_annotation_folder, val_image_folder, val_annotation_folder).
    """

    # Use current date if folder_date is None
    if folder_date is None:
        folder_date = datetime.now().strftime('%Y%m%d')

    # Append '_Augmented' to the folder name if the dataset is augmented
    folder_suffix = '_Augmented' if Augmented else ''
    date_folder = os.path.join(base_folder, f'{folder_date}{folder_suffix}')

    # Define the subfolders
    annotations_folder = os.path.join(date_folder, 'Annotations')
    images_folder = os.path.join(date_folder, 'Images')

    # Define train, test, and val subfolders if Split is True
    if Split:
        split_folder = os.path.join(date_folder, 'Split')
        train_image_folder = os.path.join(split_folder, 'train', 'images') 
        train_annotation_folder = os.path.join(split_folder, 'train', 'annotations') 
        test_image_folder = os.path.join(split_folder, 'test', 'images')
        test_annotation_folder = os.path.join(split_folder, 'test', 'annotations') 
        val_image_folder = os.path.join(split_folder, 'val', 'images')
        val_annotation_folder = os.path.join(split_folder, 'val', 'annotations')

    # Check if the folder already exists
    if os.path.exists(date_folder):
        print(f'Dataset folder already exists: {date_folder}')
        if Split:
            print(f'Train Images Folder: {train_image_folder}')
            print(f'Train Annotations Folder: {train_annotation_folder}')
            print(f'Test Images Folder: {test_image_folder}')
            print(f'Test Annotations Folder: {test_annotation_folder}')
            print(f'Val Images Folder: {val_image_folder}')
            print(f'Val Annotations Folder: {val_annotation_folder}')
        return annotations_folder, images_folder, split_folder, date_folder, train_image_folder, train_annotation_folder, test_image_folder, test_annotation_folder, val_image_folder, val_annotation_folder

    # Create the folders if they don't exist
    os.makedirs(annotations_folder, exist_ok=True)
    os.makedirs(images_folder, exist_ok=True)
    os.makedirs(split_folder, exist_ok=True)

    if Split:
        os.makedirs(train_image_folder, exist_ok=True)
        os.makedirs(train_annotation_folder, exist_ok=True)
        os.makedirs(test_image_folder, exist_ok=True)
        os.makedirs(test_annotation_folder, exist_ok=True)
        os.makedirs(val_image_folder, exist_ok=True)
        os.makedirs(val_annotation_folder, exist_ok=True)

    print(f'Dataset folder created: {date_folder}')
    if Split:
        print(f'Train Images Folder: {train_image_folder}')
        print(f'Train Annotations Folder: {train_annotation_folder}')
        print(f'Test Images Folder: {test_image_folder}')
        print(f'Test Annotations Folder: {test_annotation_folder}')
        print(f'Val Images Folder: {val_image_folder}')
        print(f'Val Annotations Folder: {val_annotation_folder}')
    return annotations_folder, images_folder, split_folder, date_folder, train_image_folder, train_annotation_folder, test_image_folder, test_annotation_folder, val_image_folder, val_annotation_folder


if __name__ == '__main__':
    use_current_date = input("Use current date for folder creation? (yes/no): ").strip().lower()

    if use_current_date == 'yes':
        folder_date = None  # Use current date
    else:
        folder_date = input("Enter the folder date (e.g., 20250318): ").strip()

    # Create the dataset folder
    create_dataset_folder(base_folder='1_GA_Dataset',
                          folder_date=folder_date,  # Change this to the correct folder for which files are to be extracted to
                          Augmented=True,  # Set to True if the dataset is augmented
                          Split=True)  # Set to True if the dataset is split into train/test/val folders


def move_raw_images_to_dataset(raw_image_folder, destination_folder):
    """
    Iterates through each subfolder in a master folder, copies its contents into a corresponding folder
    in the destination folder, and renames the subfolder to 'raw_images'.

    Args:
    - raw_image_folder (str): Path to the folder containing raw image subfolders to process.
    - destination_folder (str): Path to the destination folder where subfolders will be copied and renamed.

    Returns:
    - None

    Side Effects:
    - Copies subfolders and their contents to the destination folder.
    - Renames the copied subfolders to 'raw_images'.
    """
    for subfolder_name in os.listdir(raw_image_folder):
        subfolder_path = os.path.join(raw_image_folder, subfolder_name)
        if not os.path.isdir(subfolder_path):  # Skip if not a directory
            continue

        # Define the destination subfolder path
        destination_subfolder = os.path.join(destination_folder, subfolder_name, 'images')

        # Check if the destination subfolder already exists
        if os.path.exists(destination_subfolder):
            print(f"Images have already been moved to '{destination_subfolder}', skipping.")
            continue

        # Create the destination subfolder if it doesn't exist
        os.makedirs(destination_subfolder, exist_ok=True)

        # Copy all files and subdirectories from the source subfolder to the destination subfolder
        for item in os.listdir(subfolder_path):
            source_item = os.path.join(subfolder_path, item)
            destination_item = os.path.join(destination_subfolder, item)
            if os.path.isdir(source_item):
                shutil.copytree(source_item, destination_item, dirs_exist_ok=True)
            else:
                shutil.copy2(source_item, destination_item)

        print(f"Copied and renamed subfolder '{subfolder_name}' to '{destination_subfolder}'.")
    
    print(f"All subfolders from '{raw_image_folder}' have been copied and renamed'.")


def process_raw_data(source_folders):
    """
    Processes raw data by organizing raw images into the dataset folder and identifying source folders 
    for images and annotations.
    This function expects the raw data to be organized in the following format:
    - Each `source_folder` should contain a subfolder named `Raw_Images` where the raw images are stored.
    - The function moves the contents of `Raw_Images` to the root of the corresponding `source_folder`.
    - It then identifies subfolders containing images (folders with "images" in their name, case-insensitive) 
        and annotations (folders with "voc" in their name, case-insensitive).
    Args:
        source_folders (list of str): A list of paths to source folders containing raw data.
    Returns:
        list of dict: A list of dictionaries, where each dictionary contains:
            - "date_of_dataset_used" (str): The name of the dataset folder (typically a date or identifier).
            - "images_src_folder" (list of str): A list of paths to folders containing image data.
            - "annotations_src_folder" (list of str): A list of paths to folders containing annotation data.
    Example:
        Given the following folder structure:
        source_folders = [
            "/path/to/source1",
            "/path/to/source2"]

        /path/to/source1/
        ├── Raw_Images/
        │   ├── img1.tif
        │   ├── img2.tif
        ├── Dataset_2023/
        │   ├── Images/
        │   ├── VOC_Annotations/

        /path/to/source2/
        ├── Raw_Images/
        │   ├── img3.tif
        │   ├── img4.tif
        ├── Dataset_2022/
        │   ├── Images/
        │   ├── VOC_Annotations/

        The function will return:
        [
            {
                "date_of_dataset_used": "Dataset_2023",
                "images_src_folder": ["/path/to/source1/Dataset_2023/Images"],
                "annotations_src_folder": ["/path/to/source1/Dataset_2023/VOC_Annotations"]
            },
            {
                "date_of_dataset_used": "Dataset_2022",
                "images_src_folder": ["/path/to/source2/Dataset_2022/Images"],
                "annotations_src_folder": ["/path/to/source2/Dataset_2022/VOC_Annotations"]
            }"""

    for source in source_folders:
        raw_image_folder = os.path.join(source, 'Raw_Images')
        destination_folder = os.path.join(source)
        move_raw_images_to_dataset(raw_image_folder, destination_folder)
        print("-" * 50)

    datasets = []
    for source_folder in source_folders:
        for folder_name in os.listdir(source_folder):
            folder_path = os.path.join(source_folder, folder_name)
            if not os.path.isdir(folder_path):
                continue

            images_src_folder = [
                os.path.join(folder_path, sub_folder_name)
                for sub_folder_name in os.listdir(folder_path)
                if os.path.isdir(os.path.join(folder_path, sub_folder_name)) and 'images' in sub_folder_name.lower()
            ]

            annotations_src_folder = [
                os.path.join(folder_path, sub_folder_name)
                for sub_folder_name in os.listdir(folder_path)
                if os.path.isdir(os.path.join(folder_path, sub_folder_name)) and 'voc' in sub_folder_name.lower()
            ]

            datasets.append({
                "date_of_dataset_used": folder_name,
                "images_src_folder": images_src_folder,
                "annotations_src_folder": annotations_src_folder
            })

    print("Datasets found:")
    for dataset in datasets:
        print(f" - {dataset['date_of_dataset_used']}")
    return datasets


def process_standard_data(source_folders, training_data_folder, train_test_val, include_augmentation_list=False):
    """
    Processes standard data by identifying source folders for images and annotations.

    Args:
        source_folders (list of str): Paths to source folders containing datasets.
        train_test_val (list of str): Keywords to filter datasets ('train', 'test', 'val', or 'all').
        include_augmentation_list (bool): If True, saves a list of augmentations used.

    Returns:
        list of dict: Each dictionary contains:
            - "date_of_dataset_used" (str): Dataset folder name.
            - "images_src_folder" (str): Path to 'Images' subfolder.
            - "annotations_src_folder" (str): Path to 'Annotations' subfolder.
    """

    if not isinstance(train_test_val, list) or not set(train_test_val).issubset({'train', 'test', 'val', 'all'}):
        print("Invalid input for train_test_val. Use a list with 'train', 'test', 'val', or 'all'.")
        return []

    datasets = []
    for source_folder in source_folders:
        if isinstance(source_folder, (str, bytes, os.PathLike)): # Check if source_folder is a valid path
            for folder_name in os.listdir(source_folder):
                folder_path = os.path.join(source_folder, folder_name)
                if os.path.isdir(folder_path):
                    images_folder = os.path.join(folder_path, 'Images')
                    annotations_folder = os.path.join(folder_path, 'Annotations')
                    if os.path.isdir(images_folder) and os.path.isdir(annotations_folder):
                        datasets.append({
                            "date_of_dataset_used": folder_name,
                            "images_src_folder": images_folder,
                            "annotations_src_folder": annotations_folder
                        })
                    else:
                        print(f"Skipping {folder_path} as it does not contain both Images and Annotations folders.")
        else:
            print(f"Skipping invalid source_folder: {source_folder}")

    # Exclude datasets containing other splits if a specific split is requested
    if 'all' not in train_test_val:
        exclude = {'train', 'test', 'val'} - set(train_test_val)
        datasets = [
            dataset for dataset in datasets
            if not any(keyword in dataset['date_of_dataset_used'].lower() for keyword in exclude)
        ] #instead of searching for keyword of instance... 
        # as this makes this code unusable for folders which do not contain train test val folders at all (2_DataAugmentation folders)
        
    if include_augmentation_list:
        save_augmentations_used(datasets, training_data_folder)

    print("Datasets found:")
    for dataset in datasets:
        print(f" - {dataset['date_of_dataset_used']}")
    return datasets

def save_augmentations_used(datasets, training_data_folder):
    """
    Save a list of augmentations used to a text file in the appropriate folder.

    Args:
        datasets (list): List of dataset dictionaries.
        training_data_folder (str): Path to the training data folder.
    """
    # Creates a list of augmentations used from datasets
    # Excludes 'train', 'test', and 'val' from the list
    augmentations_used = [
        dataset["date_of_dataset_used"]
        for dataset in datasets
        if dataset["date_of_dataset_used"].strip().lower() not in {"train", "test", "val"}
    ]

    # Determine the output file path for saving augmentations used
    if training_data_folder:
        output_file = os.path.join(training_data_folder, "augmentations_used.txt")
    else:
        print("No training folder found. Cannot save augmentations.")
        return

    if output_file and os.path.exists(output_file):
        print(f"Augmentations file already exists at {output_file}. Skipping save.")
        return

    if augmentations_used:
        print("List of augmentations used:", augmentations_used)

    if output_file:
        os.makedirs(training_data_folder, exist_ok=True)
        with open(output_file, "w") as f:
            f.write("\n".join(augmentations_used))
        print(f"Augmentations saved to {output_file}")
    else:
        print("Warning: No training folder found.")



def confirm_and_extract(datasets,
                        annotations_folder, images_folder):
    """
    Prompts the user to confirm whether to proceed with extracting files from the provided datasets.
    If the user approves, it calls the `extract_files` function for each dataset to perform the extraction.
    Args:
        datasets (list of dict): A list of dictionaries where each dictionary contains the following keys:
            - 'date_of_dataset_used' (str): The date associated with the dataset.
            - 'images_src_folder' (str): The source folder path for the images.
            - 'annotations_src_folder' (str): The source folder path for the annotations.
    Behavior:
        - Displays dataset details to the user.
        - Asks the user for confirmation to proceed with file extraction.
        - If the user inputs 'yes', it calls the `extract_files` function for each dataset, passing the necessary parameters.
        - If the user inputs 'no', the process is aborted.
    Dependencies:
        - This function relies on the `extract_files` function to handle the actual file extraction process.
          Ensure that `extract_files` is defined and accepts the following parameters:
            - date_of_dataset_used (str)
            - annotations_folder (str)
            - images_folder (str)
            - images_src_folder (str)
            - annotations_src_folder (str)
    Notes:
        - The variables `annotations_folder` and `images_folder` must be defined in the scope where this function is called.
        - Input validation ensures the user provides a valid response ('yes' or 'no') before proceeding.
    """

    counter = 1
    for dataset in datasets:
        print(f"Dataset {counter}: {dataset['date_of_dataset_used']}")
        print(f"Images Source Folder: {dataset['images_src_folder']}")
        print(f"Annotations Source Folder: {dataset['annotations_src_folder']}")
        print("-" * 50)
        counter += 1

    proceed = input("Do you want to proceed with extracting files? (yes/no):").strip().lower()
    while proceed not in ['yes', 'no']:
        print("Invalid input. Please type 'yes' or 'no'.")
        proceed = input("Do you want to proceed with extracting files? (yes/no): ").strip().lower()

    if proceed == 'no':
        print("File extraction aborted.")
        return

    for dataset in datasets:
        extract_files(
            date_of_dataset_used=dataset["date_of_dataset_used"],
            annotations_folder=annotations_folder,
            images_folder=images_folder,
            images_src_folder=dataset["images_src_folder"],
            annotations_src_folder=dataset["annotations_src_folder"]
        )
    
    print("File extraction completed.")
     



def extract_files(date_of_dataset_used, 
                  annotations_folder, images_folder, 
                  annotations_src_folder, images_src_folder):   # Define source, annotations, and images folders
    """
    Extracts .xml and .tif files from source folders to specified destination folders, ensuring unique filenames.
    Parameters:
    date_of_dataset_used (str): The date of the dataset being used.
    annotations_folder (str): The destination folder for annotation (.xml) files.
    images_folder (str): The destination folder for image (.tif) files.
    annotations_src_folder (str): The source folder containing annotation (.xml) files.
    images_src_folder (str): The source folder containing image (.tif) files.

    Functionality:
    - Creates the destination folders if they do not exist.
    - Copies .xml files from the source annotations folder to the destination annotations folder with unique filenames.
    - Copies .tif files from the source images folder to the destination images folder with unique filenames.
    - Checks if the number of files in the annotations and images folders are the same.
    - Verifies if the files in the annotations and images folders have matching filenames (excluding extensions).
    - Prints unmatched files if there are discrepancies between the annotations and images folders.
    """
 


    # Create destination folders if they do not exist
    if not os.path.exists(annotations_folder):
        os.makedirs(annotations_folder)
    else:
        print(f"Folder {annotations_folder} already exists.")
        

    if not os.path.exists(images_folder):
        os.makedirs(images_folder)
    else:
        print(f"Folder {images_folder} already exists.")

    print("-" * 20)  # Add a separator line for better readability

    # Extract .xml files to annotations folder
    num_annotations_start = 0
    xml_files = [f for f in os.listdir(annotations_folder) if f.endswith('.xml')]
    num_annotations_start += len(xml_files)
    print(f"Total annotations from {annotations_folder}: {num_annotations_start}")

    counter = len([f for f in os.listdir(annotations_folder) if f.endswith('.xml')]) + 1
    for folder in [annotations_src_folder]:
        for file in os.listdir(folder):
            if file.endswith('.xml'):
               src_file = os.path.join(folder, file)
               dst_file = os.path.join(annotations_folder, f"{counter}.xml")
               if os.path.exists(dst_file):
                   print(f"Error: File {dst_file} already exists. / "
                         f"Source file: {src_file} / "
                         f"Destination file: {dst_file}")
                   return
               shutil.copy(src_file, dst_file)
            counter += 1

    # Count the number of annotations at the end
    num_annotations_end = len([f for f in os.listdir(annotations_folder) if f.endswith('.xml')])
    
    # Calculate the number of annotations moved
    num_annotations_moved = num_annotations_end - num_annotations_start

    # Verify if the counts add up
    if num_annotations_end == (counter -1):  
        print(f"Files extracted from {date_of_dataset_used} to {annotations_folder}/ "
              f"Start: {num_annotations_start}\n"
              f"Moved: {num_annotations_moved}\n"
              f"End: {num_annotations_end}\n")
    else:
        print(f"Error: Number of annotations in {annotations_folder} does not match the expected count.\n"
              f"Start: {num_annotations_start}\n"
              f"Moved: {num_annotations_moved}\n"
              f"End: {num_annotations_end}\n"
              f"Expected Total Moved: {counter - 1}")

    print("-" * 20)  # Add a separator line for better readability

    # Extract .tif files to images folder
    num_images_start = 0
    tif_files = [f for f in os.listdir(images_folder) if f.endswith('.tif')]
    num_images_start += len(tif_files)
    print(f"Total images from {images_folder}: {num_images_start}")


    counter = len([f for f in os.listdir(images_folder) if f.endswith('.tif')]) + 1
    for folder in [images_src_folder]: #needs list not string
        for file in os.listdir(folder):
            if file.endswith('.tif'):
               src_file = os.path.join(folder, file)
               dst_file = os.path.join(images_folder, f"{counter}.tif")
               if os.path.exists(dst_file):
                   print(f"Error: File {dst_file} already exists. / "
                         f"Source file: {src_file} / "
                         f"Destination file: {dst_file}")
                   return
               shutil.copy(src_file, dst_file)
            counter += 1
        
    # Count the number of images at the end
    num_images_end = len([f for f in os.listdir(images_folder) if f.endswith('.tif')])
    
    # Calculate the number of images moved
    num_images_moved = num_images_end - num_images_start

    # Verify if the counts add up
    if num_images_end == (counter - 1):
        print(f"Files extracted from {date_of_dataset_used} to {images_folder}/ ")
        print(f"Start: {num_images_start}")
        print(f"Moved: {num_images_moved}")
        print(f"End: {num_images_end}")
    else:
        print(f"Error: Number of images in {images_folder} does not match the expected count. ")
        print(f"Start: {num_images_start}")
        print(f"Moved: {num_images_moved}")
        print(f"End: {num_images_end}")
        print(f"Expected Total Moved: {counter - 1}")
    
    print("-" * 50)  # Add a separator line for better readability


    if len(annotations_folder) == len(images_folder):
        print("The number of annotation files and image files are the same.")
        print("Number of annotation files:", len(annotations_folder))
        print("Number of image files:", len(images_folder)) 

    def check_file_matching(annotations_folder, images_folder, annotations_src_folder, images_src_folder):
        """
        Check if the files in the annotations and images folders have matching file names (different extensions).
        If not, identify unmatched files in both folders and their source folders.

        Args:
            annotations_folder (str): Path to the folder containing annotation (.xml) files.
            images_folder (str): Path to the folder containing image (.tif) files.
            annotations_src_folder (str): Path to the source folder containing annotation (.xml) files.
            images_src_folder (str): Path to the source folder containing image (.tif) files.

        Returns:
            None
        """
        # Check if the files in these folders have the same file name (just different extension)
        files_in_annotations_folder = [f for f in os.listdir(annotations_folder) if f.endswith('.xml')]
        files_in_images_folder = [f for f in os.listdir(images_folder) if f.endswith('.tif')]
        files_in_annotations_folder_no_ext = [os.path.splitext(f)[0] for f in files_in_annotations_folder]
        files_in_images_folders_no_ext = [os.path.splitext(f)[0] for f in files_in_images_folder]

        if set(files_in_annotations_folder_no_ext) == set(files_in_images_folders_no_ext):
            print("The files in the annotations and images folders have the same file names (just different extensions).")
            print("-" * 50)  # Add a separator line for better readability
        else:
            unmatched_annotations = set(files_in_annotations_folder_no_ext) - set(files_in_images_folders_no_ext)
            unmatched_images = set(files_in_images_folders_no_ext) - set(files_in_annotations_folder_no_ext)

            print("The files in the annotations and images folders do not match.")
            print("Unmatched annotation files:", unmatched_annotations)
            print("Unmatched image files:", unmatched_images)

            # Get the list of .xml files in the annotations folder including subfolders
            xml_files = []
            for root, dirs, files in os.walk(annotations_src_folder):
                for file in files:
                    if file.endswith('.xml'):
                        xml_files.append(file)
            xml_files_no_ext = [os.path.splitext(file)[0] for file in xml_files]

            # Get the list of .tif files in the images folder including subfolders
            tif_files = []
            for root, dirs, files in os.walk(images_src_folder):
                for file in files:
                    if file.endswith('.tif'):
                        tif_files.append(file)
            tif_files_no_ext = [os.path.splitext(file)[0] for file in tif_files]

            # Find unmatched files
            unmatched_annotations = set(xml_files_no_ext) - set(tif_files_no_ext)
            unmatched_images = set(tif_files_no_ext) - set(xml_files_no_ext)

            # Print the results
            print(f"The files in the {annotations_src_folder} and {images_src_folder} do not match.")
            print("Unmatched annotation files:", unmatched_annotations)
            print("Unmatched image files:", unmatched_images)
            print("-" * 50)  # Add a separator line for better readability)

    check_file_matching(annotations_folder, images_folder, annotations_src_folder, images_src_folder)
        
if __name__ == '__main__':
    extract_files(date_of_dataset_used=date_of_dataset_used,  # Change this to the correct dataset used, FOR REFERENCE ONLY
                    annotations_folder=r'GA_Dataset\20250219\Annotations',  # Change this to the correct folder for which files were extracted 
                    images_folder=r'GA_Dataset\20250219\Images',  # Change this to the correct folder for which files were extracted 
                    images_src_folder=r'Completed annotations/Bluff_230724/Original_Images_Unlabelled_Bluff_230724',
                    annotations_src_folder=r'Completed annotations\Bluff_230724')  # Change this to your source folder path 
                    

def extraction_pipeline(source_folders, 
                        training_data_folder, annotations_folder, images_folder,
                        raw_data=False,
                        train_test_val=None, 
                        include_augmentation_list=False):
    """
    Extracts files from multiple source folders and organizes them into specified annotations and images folders.

    Args:
        source_folders (list of str): Paths to source folders containing datasets.
        annotations_folder (str): Destination folder for annotation files.
        images_folder (str): Destination folder for image files.
        train_test_val (list of str, optional): List containing any combination of 'train', 'test', 'val', or 'all'. Not applicable if raw_data is True.
        raw_data (bool): If True, process raw images and VOC annotations.
        include_augmentation_list (bool): If True, include a list of augmentations used.

    Returns:
        None
    """
    
    if raw_data:
        datasets = process_raw_data(source_folders)
    else:
        datasets = process_standard_data(source_folders, training_data_folder, train_test_val, include_augmentation_list)

    if datasets:
        confirm_and_extract(datasets, annotations_folder, images_folder)
    

if __name__ == "__main__":
    # Define the source folders to extract files from
    source_folders = [
        r'1_GA_Dataset\20250318\Split',  # Original dataset, not augmented
        r'2_DataAugmentation\20250318'  # Augmented datasets
    ]

    # Define the destination folders for annotations and images
    annotations_folder = r'3_TrainingData\20250318_Augmented\Split\train\annotations'
    images_folder = r'3_TrainingData\20250318_Augmented\Split\train\images'
    training_data_folder = r'3_TrainingData\20250318_Augmented'  # Define the training data folder

    # Call the function to extract files from multiple folders
    extraction_pipeline(source_folders, 
                        training_data_folder, annotations_folder, images_folder,
                        train_test_val=['train', 'test', 'val'], 
                        raw_data=False,
                        include_augmentation_list=True)



################################################# TRAIN TEST SPLIT ############################################################

"""
The provided code implements a pipeline for dataset preparation, training, and evaluation in object detection. 
It includes functions for splitting datasets, creating folder structures, copying files, parsing annotations, 
and generating JSON data lists. 

splitting datasets, creating folder structures, copying files, parsing annotations, 
and generating JSON data lists. 

- convert_files_to_list:
Converts image and annotation files in specified folders into separate lists of file paths

- create_testtrain_folders: 
Creates the necessary folder structure for training and testing datasets (e.g., train/test images and annotations)

- copy_files: 
Copies files (e.g., .tif or .xml) to specified destination folders, ensuring proper naming.

- split_and_copy_files: Splits the dataset into training and testing subsets and organizes them into the appropriate 
folders using create_testtrain_folders and copy_files. Uses images, annotations generated by convert_files_to_list as this is an
appropriate format to use for train_test_split function (skicit learn: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)

Training utilities include checkpoint management, learning rate adjustment, 
gradient clipping, and metric tracking. Data augmentation techniques like resizing, flipping, and cropping are 
applied during training. Evaluation utilities include mAP calculation and IoU computation. The modular design ensures
 flexibility, reusability, and efficient handling of datasets and training processes.

 """



def convert_files_to_list(images_folder, annotations_folder):
    """
    Convert all files in two folders to two separate lists of their contents.
    :param images_folder: Path to the folder containing the image files
    :param annotations_folder: Path to the folder containing the annotation files
    :return: Tuple of two lists - (image file paths, annotation file paths)
    """
    images_file_paths = []
    for file_name in os.listdir(images_folder):
        file_path = os.path.join(images_folder, file_name)
        if os.path.isfile(file_path):
            images_file_paths.append(file_path)

    annotations_file_paths = []
    for file_name in os.listdir(annotations_folder):
        file_path = os.path.join(annotations_folder, file_name)
        if os.path.isfile(file_path):
            annotations_file_paths.append(file_path)
                
    return images_file_paths, annotations_file_paths

if __name__ == '__main__':
    """Change paths and output folder accordingly to your setup"""
    images, annotations = convert_files_to_list(images_folder=r'GA_Dataset\Images', annotations_folder=r'GA_Dataset\Annotations')


def create_testtrainval_folders(output_folder):
    """
    Create necessary folders for train, test, and validation datasets.
    """

    folders = {
        "train_image_folder": os.path.join(output_folder, 'train', 'images'),
        "test_image_folder": os.path.join(output_folder, 'test', 'images'),
        "val_image_folder": os.path.join(output_folder, 'val', 'images'),
        "train_annotation_folder": os.path.join(output_folder, 'train', 'annotations'),
        "test_annotation_folder": os.path.join(output_folder, 'test', 'annotations'),
        "val_annotation_folder": os.path.join(output_folder, 'val', 'annotations')
    }

    for folder_name, folder_path in folders.items():
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        else:
            print(f"Folder {folder_path} already exists.")

def copy_files(files, destination_folder, file_extension):
    """
    Copy files to the destination folder.
    :param files: list of file paths to be copied
    :param destination_folder: folder where the files will be copied
    :param file_extension: extension of the files to be copied (e.g., '.tif' or '.xml')
    """
    
    if os.path.exists(destination_folder) and \
    any(file.endswith(file_extension) for file in os.listdir(destination_folder)) and \
    len(os.listdir(destination_folder)) > 0:
        print(f"Destination directory already exists and directory already contains {file_extension} files.")
            
    else:
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)
        for index, file in enumerate(files):
            file_name = f"{index}{file_extension}"  # Derive the file name from an object in a list and append the file extension
            dest_file_path = os.path.join(destination_folder, file_name)  # Create the destination file path by joining the destination folder and the file name
            shutil.copy(file, dest_file_path) 

        print(f'Files copied to {destination_folder}') # Print a message indicating that the files have been copied to the destination folder


def split_and_copy_files(images, annotations, output_folder,  
                         test_size= None, 
                         train_size = None,
                         random_state=None):

    if test_size is None:
        test_size = 0.15
    if train_size is None:
        train_size = 0.15
    if random_state is None:
        random_state = 42

    train_images_path = os.path.join(output_folder, 'train', 'images')
    test_images_path = os.path.join(output_folder, 'test', 'images')
    train_annotations_path = os.path.join(output_folder, 'train', 'annotations')
    test_annotations_path = os.path.join(output_folder, 'test', 'annotations')
    val_images_path = os.path.join(output_folder, 'val', 'images')
    val_annotations_path = os.path.join(output_folder, 'val', 'annotations')

    if all(os.path.exists(path) for path in [train_images_path, test_images_path, train_annotations_path, test_annotations_path, val_images_path, val_annotations_path]) and \
       all(file.endswith('.tif') for path in [train_images_path, test_images_path, val_images_path] for file in os.listdir(path)) and \
       all(file.endswith('.xml') for path in [train_annotations_path, test_annotations_path, val_annotations_path] for file in os.listdir(path)) and \
       all(len(os.listdir(path)) > 0 for path in [train_images_path, test_images_path, train_annotations_path, test_annotations_path, val_images_path, val_annotations_path]):

        print("Train and test lists already exist. Dataset has been split and contains relevant files.")

        total_annotations = len(os.listdir(train_annotations_path)) + \
                            len(os.listdir(test_annotations_path)) + \
                            len(os.listdir(val_annotations_path))
           
        if total_annotations == len(annotations):
            print("The number of XML files in train and test folders matches the total number of XML files in annotations.")
        else:
            print("Mismatch in the number of XML files between train/test folders and annotations. Train-test split may belong to an old version of the dataset. Try extracting and splitting the dataset again.")
            
        return
    
    else:
        # test train split
        train_images, test_images, train_annotations, test_annotations = train_test_split(images, annotations, 
                                                                                          test_size=test_size, 
                                                                                          random_state=random_state)

        # train val split
        train_images, val_images, train_annotations, val_annotations = train_test_split(train_images, train_annotations,
                                                                                        test_size=train_size, 
                                                                                        random_state=random_state)


        create_testtrainval_folders(output_folder)
        copy_files(train_images, os.path.join(output_folder,   'train', 'images'), file_extension='.tif')
        copy_files(test_images, os.path.join(output_folder,   'test', 'images'), file_extension= '.tif')
        copy_files(train_annotations, os.path.join(output_folder,   'train', 'annotations'), file_extension='.xml')
        copy_files(test_annotations, os.path.join(output_folder,   'test', 'annotations'),file_extension='.xml')
        copy_files(val_images, os.path.join(output_folder,   'val', 'images'), file_extension='.tif')
        copy_files(val_annotations, os.path.join(output_folder,   'val', 'annotations'), file_extension='.xml')

        print(f"Files have been split and copied to {output_folder}")

if __name__ == '__main__':
    """Change paths and output folder accordingly to your setup"""
    split_and_copy_files(images, annotations, 
                         output_folder=r'GA_Dataset\Split', 
                         test_size=0.15,
                         train_size=0.15, 
                         random_state=42)



############################################ PARSING AND DATALIST/DATALOADER CREATION ####################################################

"""
The provided code implements a pipeline for parsing annotations and creating JSON data lists for object detection tasks. 
It includes functions for extracting bounding box data, labels, and difficulties from XML files and organizing them into structured 
JSON files.

"""



def parse_annotation(annotation_file, label_map): #FILE not path, because path is to folder, and path is to indifidual file
    """
    Parse an XML annotation file to extract bounding box coordinates, labels, and difficulty levels.
    Args:
        annotation_file (str): Path to the XML annotation file.
        label_map (dict): A dictionary mapping label names to integer values.
    Returns:
        dict: A dictionary containing:
            - 'boxes' (list of list of int): Bounding box coordinates [xmin, ymin, xmax, ymax].
            - 'labels' (list of int): Corresponding labels for each bounding box.
            - 'difficulties' (list of int): Difficulty levels for each object (0 or 1).
    """

    tree = ET.parse(annotation_file)
    root = tree.getroot()

    boxes = list()
    labels = list()
    difficulties = list()

    for object in root.iter('object'):

        difficult = int(object.find('difficult').text == '1')

        label = object.find('name').text.strip()
        if label not in label_map:
            print (f"Label '{label}' not present in label_map")
            continue

        bbox = object.find('bndbox')
        xmin = int(bbox.find('xmin').text) - 1
        ymin = int(bbox.find('ymin').text) - 1
        xmax = int(bbox.find('xmax').text) - 1
        ymax = int(bbox.find('ymax').text) - 1

        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(label_map[label])
        difficulties.append(difficult)

    # print({'boxes': boxes, 'labels': labels, 'difficulties': difficulties})
    return boxes, labels, difficulties

def create_data_lists(train_annotation_path, train_image_path, 
                      test_annotation_path, test_image_path,
                      date_of_dataset_used,
                      augmented,
                      object_detector,
                      JSON_folder=r'JSON_folder'):
    """
    Create lists of images, the bounding boxes and labels of the objects in these images, and save these to file.

    :param train_annotation_path: path to the training annotations folder
    :param train_image_path: path to the training images folder
    :param test_annotation_path: path to the testing annotations folder
    :param test_image_path: path to the testing images folder
    :param object_detector: boolean, if True use label_map_OD, if False use label_map_Classifier
    :param date_of_dataset_used: date string for dataset identification
    :param augmented: boolean, if True indicates augmented data
    :param JSON_folder: folder where the JSONs must be saved
    """

    # Select the appropriate label map
    label_map = label_map_OD if object_detector else label_map_Classifier

    # Determine the output folder based on whether the data is augmented
    if augmented:
        output_folder = os.path.join(JSON_folder, f"{date_of_dataset_used}_Augmented")
    else:
        output_folder = os.path.join(JSON_folder, date_of_dataset_used)

    # Check if data lists already exist
    if all(os.path.exists(os.path.join(output_folder, file)) for file in 
           ['TEST_images.json', 
            'TEST_objects.json', 
            'TRAIN_images.json', 
            'TRAIN_objects.json', 
            'label_map.json']):
        print('Datalists already created')
        return

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    def process_data(image_path, annotation_path, label_map, output_image_file, output_object_file):
        """
        Process images and annotations to create data lists.

        :param image_path: Path to the images folder
        :param annotation_path: Path to the annotations folder
        :param label_map: Label map to use
        :param output_image_file: Output JSON file for image paths
        :param output_object_file: Output JSON file for objects
        """
        images = []
        objects = []
        n_objects = 0

        # Get image IDs
        image_files = [file for file in os.listdir(image_path) if file.endswith('.tif')]
        ids = [os.path.splitext(file)[0] for file in image_files]
        print(f"Found {len(ids)} images.")

        for id in ids:
            annotation_file = os.path.join(annotation_path, f"{id}.xml")

            if not os.path.isfile(annotation_file):
                print(f"Annotation file {annotation_file} does not exist, skipping.")
                continue

            # Parse the annotation file
            parsed_objects = parse_annotation(annotation_file, label_map)
            if not parsed_objects['boxes']:
                print(f"No objects found in {annotation_file}, skipping.")
                continue

            n_objects += len(parsed_objects['boxes'])
            objects.append(parsed_objects)
            images.append(os.path.join(image_path, f"{id}.tif"))
            print(f"Processed {annotation_file}, found {len(parsed_objects['boxes'])} objects.")

        # Save to JSON files
        with open(output_image_file, 'w') as j:
            json.dump(images, j)
        with open(output_object_file, 'w') as j:
            json.dump(objects, j)

        print(f"\nProcessed {len(images)} images containing a total of {n_objects} objects. "
              f"Files saved to {os.path.abspath(output_folder)}.")

    # Process training data
    process_data(train_image_path, train_annotation_path, label_map,
                 os.path.join(output_folder, 'TRAIN_images.json'),
                 os.path.join(output_folder, 'TRAIN_objects.json'))

    # Save the label map
    with open(os.path.join(output_folder, 'label_map.json'), 'w') as j:
        json.dump(label_map, j)
    print("Label map saved to", os.path.join(output_folder, 'label_map.json'))

    print("-" * 50)  # Print a separator line for better readability

    # Process testing data
    process_data(test_image_path, test_annotation_path, label_map,
                 os.path.join(output_folder, 'TEST_images.json'),
                 os.path.join(output_folder, 'TEST_objects.json'))
    
    print("Data processing completed.")

if __name__ == '__main__':
    """Change paths and output folder accordingly to your setup"""
    create_data_lists(train_annotation_path=r'GA_Dataset\Split\train\annotations',
                      train_image_path=r'GA_Dataset\Split\train\images',
                      test_annotation_path=r'GA_Dataset\Split\test\annotations',
                      test_image_path=r'GA_Dataset\Split\test\images',
                      object_detector=True,
                      date_of_dataset_used='20250219',
                      augmentation=None,
                      JSON_folder='./')



def create_datalists_old(train_annotation_path, train_image_path, test_annotation_path, test_image_path, 
                      object_detector, 
                      date_of_dataset_used,
                      augmentation,
                      JSON_folder=r'JSON_folder'):
    """
    Create lists of images, the bounding boxes and labels of the objects in these images, and save these to file.

    :param train_annotation_path: path to the training annotations folder
    :param train_image_path: path to the training images folder
    :param test_annotation_path: path to the testing annotations folder
    :param test_image_path: path to the testing images folder
    :param object_detector: boolean, if True use label_map_OD, if False use label_map_Classifier
    :param date_of_dataset_used: date string for dataset identification
    :param augmentation: augmentation type, if any
    :param JSON_folder: folder where the JSONs must be saved
    """

    label_map = label_map_OD if object_detector else label_map_Classifier
    
    if augmentation is None:
        output_folder = os.path.join(JSON_folder, date_of_dataset_used)
    if augmentation is None:
        output_folder = os.path.join(JSON_folder, date_of_dataset_used)

    if augmentation == 'augmented_data':
        output_folder = os.path.join(JSON_folder, date_of_dataset_used + '_Augmented')

    if os.path.exists(os.path.join(output_folder, 'TEST_images.json')) \
        and os.path.exists(os.path.join(output_folder, 'TEST_objects.json')) \
        and os.path.exists(os.path.join(output_folder, 'TRAIN_images.json')) \
        and os.path.exists(os.path.join(output_folder, 'TRAIN_objects.json')) \
        and os.path.exists(os.path.join(output_folder, 'label_map.json')):
        print('Datalists already created')

    else:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder, exist_ok=True)

        # Training data

        train_images = list()
        train_objects = list()
        n_objects = 0

        for path in [train_image_path]:

            # Find IDs of images in training data by listing files in the Images directory
            image_files = [file for file in os.listdir(train_image_path) if file.endswith('.tif')] # Iterate over each file in the 'Images' directory. # Check if the file has a '.tif' extension (i.e., it is a JPEG image).
            ids = [os.path.splitext(file)[0] for file in image_files] # Split the file name into the base name and extension, and take the base name (i.e., the part before '.tif').
            
            print(f"Found {len(ids)} training images.")
            
            for id in ids:
                annotation_file = os.path.join(train_annotation_path, id + '.xml')

                if not os.path.isfile(annotation_file):
                    print(f"Annotation file {annotation_file} does not exist, skipping.")
                    continue
                
                # Parse annotation's XML file
                objects = parse_annotation(annotation_file, label_map)
                if len(objects['boxes']) == 0:
                    print(f"No objects found in {annotation_file}, skipping.")
                    continue
                
                n_objects += len(objects['boxes'])
                train_objects.append(objects)
                train_images.append(os.path.join(train_image_path, id + '.tif'))
                
                print(f"Processed {annotation_file}, found {len(objects['boxes'])} objects.")

                assert len(train_objects) == len(train_images)

        # Save to file
        with open(os.path.join(output_folder, 'TRAIN_images.json'), 'w') as j:
            json.dump(train_images, j)
        with open(os.path.join(output_folder, 'TRAIN_objects.json'), 'w') as j:
            json.dump(train_objects, j)
        with open(os.path.join(output_folder, 'label_map.json'), 'w') as j:
            json.dump(label_map, j)  # save label map too

        print('\nThere are %d training images containing a total of %d objects. Files have been saved to %s.' % (
            len(train_images), n_objects, os.path.abspath(output_folder)))
        

        # Test data
        test_images = list()
        test_objects = list()
        n_objects = 0

        for path in [test_image_path]:

            # Find IDs of images in training data by listing files in the Images directory
            image_files = [file for file in os.listdir(test_image_path) if file.endswith('.tif')] # Iterate over each file in the 'Images' directory. # Check if the file has a '.tif' extension (i.e., it is a JPEG image).
            ids = [os.path.splitext(file)[0] for file in image_files] # Split the file name into the base name and extension, and take the base name (i.e., the part before '.tif').
            
            print(f"Found {len(ids)} test images.")
            
            for id in ids:
                annotation_file = os.path.join(test_annotation_path, id + '.xml')

                if not os.path.isfile(annotation_file):
                    print(f"Annotation file {annotation_file} does not exist, skipping.")
                    continue
                
                # Parse annotation's XML file
                objects = parse_annotation(annotation_file, label_map)
                if len(objects['boxes']) == 0:
                    print(f"No objects found in {annotation_file}, skipping.")
                    continue
                
                n_objects += len(objects['boxes'])
                test_objects.append(objects)
                test_images.append(os.path.join(test_image_path, id + '.tif'))
                
                print(f"Processed {annotation_file}, found {len(objects['boxes'])} objects.")

                assert len(test_objects) == len(test_images)

        # Save to file
        with open(os.path.join(output_folder, 'TEST_images.json'), 'w') as j:
            json.dump(test_images, j)
        with open(os.path.join(output_folder, 'TEST_objects.json'), 'w') as j:
            json.dump(test_objects, j)

        print('\nThere are %d testing images containing a total of %d objects. Files have been saved to %s.' % (
            len(test_images), n_objects, os.path.abspath(output_folder)))


if __name__ == '__main__':
    """Change paths and output folder accordingly to your setup"""
    create_data_lists(train_annotation_path=r'GA_Dataset\Split\train\annotations',
                      train_image_path=r'GA_Dataset\Split\train\images',
                      test_annotation_path=r'GA_Dataset\Split\test\annotations',
                      test_image_path=r'GA_Dataset\Split\test\images',
                      label_map=label_map,
                      date_of_dataset_used='20250219',
                      augmentation=None,
                      JSON_folder='./')




def separate_preds_targets(images, boxes, labels, image_files, device, debug=True):
    """
    Separates batch data into prediction inputs (images) and target dictionaries (boxes, labels).
    Returns:
        preds: list of image tensors (inputs for model)
        targets: list of dicts with 'boxes' and 'labels' (targets for model)
        image_files: list of filenames for reference
        debug_info (optional): list of dicts with details about skipped/invalid boxes if debug=True

    Notes:
        - Each image tensor should be of shape [C, H, W] and in 0-1 range.
        - Boxes should be FloatTensor[N, 4] in [x1, y1, x2, y2] format, with x in [0, W], y in [0, H].
        - Labels should be Int64Tensor[N].
    """
    preds = []
    targets = []
    valid_image_files = []
    debug_info = []
    for img, b, l, fname in zip(images, boxes, labels, image_files):
        if b.shape[0] > 0:
            # Filter out invalid boxes (xmax > xmin and ymax > ymin)
            mask = (b[:, 2] > b[:, 0]) & (b[:, 3] > b[:, 1])
            if mask.sum() == 0:
                if debug:
                    debug_info.append({
                        "file": fname,
                        "reason": "No valid boxes",
                        "boxes": b.cpu().tolist() if hasattr(b, "cpu") else b,
                        "labels": l.cpu().tolist() if hasattr(l, "cpu") else l
                    })
                continue  # skip if no valid boxes
            filtered_boxes = b[mask]
            filtered_labels = l[mask] if l.shape[0] == b.shape[0] else l
            # Ensure correct types and shapes
            preds.append(img) #list of image tensors
            targets.append({ # list of dicts with 'boxes' and 'labels'
                'boxes': filtered_boxes.to(device).float(),
                'labels': filtered_labels.to(device).long()
            })
            valid_image_files.append(fname)
        else:
            if debug:
                debug_info.append({
                    "file": fname,
                    "reason": "No boxes in annotation",
                    "boxes": [],
                    "labels": []
                })
    if debug:
        return preds, targets, valid_image_files, debug_info

    return preds, targets, valid_image_files





########################################### DATALOADER CHECKING ####################################################

def print_batch_structure(loader, loader_name="train_loader"):
    """
    Print the type and shape of each item in a batch from the given DataLoader.
    Args:
        loader: DataLoader to sample a batch from.
        loader_name: Name of the loader (for display purposes).
    """
    print(f"\nSample batch from {loader_name} (collate_fn output):")
    batch = next(iter(loader))
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




################################################ INITIALIZE TRAINING ################################################

def check_model_trained(checkpoint_dir='6_Checkpoints',
                        date_of_dataset_used='20250318', 
                        augmented=False,):
    """
    Check if a model has already been trained by verifying the presence of a checkpoint file.

    Args:
        date_of_dataset_used (str): The date of the dataset used.
        augmented (bool): Whether the dataset is augmented. Defaults to False.
        checkpoint_dir (str): Directory where checkpoints are stored. Defaults to '6_Checkpoints'.

    """
    # Get the list of all files in the checkpoint directory
    if augmented: # Check if the dataset is augmented
        date_of_dataset_used += '_Augmented' # Append '_Augmented' to the date string if the dataset is augmented

    model_files = [f for f in os.listdir(checkpoint_dir) if date_of_dataset_used in f] # Filter the model files based on the updated date_of_dataset_used
        
    if model_files:
        model_path = os.path.join(checkpoint_dir, model_files[0])
        print(f'Model may be present. Please check: {model_path}')
        print(f'Model checkpoint found at: {model_path}')
    else:
        print(f'No model checkpoint found for date: {date_of_dataset_used} in {checkpoint_dir}')

def manage_training_output_file(results_folder, date_of_dataset_used,
                                augmented=False,
                                model_type='detector'):
    """
    Manage the training output file by creating or appending to it, and write training parameters if not already present.
    Also creates a corresponding .csv file for results.

    Args:
        results_folder (str): Folder to store results.
        date_of_dataset_used (str): Date of the dataset used for training.
        augmented (bool): Whether the dataset is augmented.
        model_type (str): Either 'detector' or 'classifier'.

    Returns:
        str: Path to the training output .txt file and .csv file.
    """
    if augmented:
        date_of_dataset_used += '_Augmented'
        augmentations_file = os.path.join('3_TrainingData', date_of_dataset_used, 'augmentations_used.txt')
        if not os.path.exists(augmentations_file):
            print(f"Error: The file {augmentations_file} does not exist.")
            return

    training_output_txt = os.path.join(results_folder, f'training_results_{date_of_dataset_used}.txt')
    training_output_csv = os.path.join(results_folder, f'training_results_{date_of_dataset_used}.csv')
    os.makedirs(results_folder, exist_ok=True)  # Ensure the results folder exists

    if model_type.lower() == 'detector':
        model_str = 'Object Detector'
    elif model_type.lower() == 'classifier':
        model_str = 'Classifier'
    else:
        raise ValueError("model_type must be either 'detector' or 'classifier'.")

    if augmented:
        with open(augmentations_file, 'r') as f:
            augmentation = [line.strip() for line in f if line.strip()]
    else:
        augmentation = None

    params = [
        f'Checkpoint: {checkpoint}',
        f'Batch Size: {batch_size}',
        f'Iterations: {iterations}',
        f'Workers: {workers}',
        f'Print Frequency: {print_freq}',
        f'Learning Rate: {lr}',
        f'Decay Learning Rate At: {decay_lr_at}',
        f'Decay Learning Rate To: {decay_lr_to}',
        f'Momentum: {momentum}',
        f'Weight Decay: {weight_decay}',
        f'Gradient Clipping: {grad_clip}',
        f'Checkpoint Frequency: {checkpoint_freq}',
        f'epoch_num: {epoch_num}',
        f'Decay Learning Rate At Epochs: {decay_lr_at_epochs}',
        '-' * 50,  # Separator
        f'Number of Epochs: {epoch_num}',
        f'Date of Dataset Used: {date_of_dataset_used}',
        f'Augmentation: {augmentation}',
        f'Model: {model_str}'
    ]

    if os.path.exists(training_output_txt):
        with open(training_output_txt, 'r') as f:
            content = f.readlines()
        content = [line.strip() for line in content if line.strip()]

        # Ensure params are written in the same order and not added at the end
        updated_content = []
        for param in params:
            if param not in content:
                updated_content.append(param)
            else:
                updated_content.append(content[content.index(param)])
    else:
        updated_content = params

    with open(training_output_txt, 'w') as f:
        f.write('\n'.join(updated_content) + '\n\n')

    # Create a CSV file with header if it doesn't exist
    if not os.path.exists(training_output_csv):
        with open(training_output_csv, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Epoch', 'Train Loss', 'Val Loss', 'Train Accuracy', 'Val Accuracy',
                                'Train Precision', 'Val Precision', 'Train Recall', 'Val Recall',
                                'Train F1 Score', 'Val F1 Score', 'Train mAP', 'Val mAP', 'Train IoU', 'Val IoU'])

    return training_output_txt, training_output_csv




################################################## METRICS CALCULATION ##################################################



def calculate_metrics(
    args, 
    gt_labels_cat, pred_labels_cat, 
    preds_for_metric=None, targets=None,
    loss=None
):
    """
    Calculates metrics for classification or object detection.
    Returns a dictionary of metrics.
    """
    metrics = {}

    if args.model_type == 'object_classifier':
        # Classification metrics
        metrics['loss'] = loss # for object detector, this is sum loss (sum from loss_dict)
                                # for object classifier, this is cross entropy loss
        metrics['accuracy'] = accuracy_score(gt_labels_cat, pred_labels_cat)
        metrics['precision'] = precision_score(gt_labels_cat, pred_labels_cat, average='weighted')
        metrics['recall'] = recall_score(gt_labels_cat, pred_labels_cat, average='weighted')
        metrics['f1_score'] = f1_score(gt_labels_cat, pred_labels_cat, average='weighted')
        metrics['confusion_matrix'] = confusion_matrix(gt_labels_cat, pred_labels_cat)
    elif args.model_type == 'object_detector':
        # Object detection metrics
        if preds_for_metric is not None and targets is not None:
            metrics['loss'] = loss # for object detector, this is sum loss (sum from loss_dict)
                                    # for object classifier, this is cross entropy loss
            mean_ap = MeanAveragePrecision(iou_type='bbox')
            mean_ap.update(preds_for_metric, targets)
            metrics['mean_ap'] = mean_ap.compute()

            Iou = IntersectionOverUnion(box_format='xyxy', # Supported formats are [`xyxy`, `xywh`, `cxcywh`].
                                                       iou_threshold=0.5,
                                                       respect_labels=True) # If True, IoU is computed only for boxes with matching labels.
            Iou.update(preds_for_metric, targets)
            metrics['iou'] = Iou.compute()

    return metrics  # metrics['mean_ap'] contains the following keys (from torchmetrics' MeanAveragePrecision):
                    #   - 'map':        Mean Average Precision (mAP) averaged over all classes and IoU thresholds.
                    #   - 'map_50':     mAP at IoU threshold 0.5 (PASCAL VOC metric).
                    #   - 'map_75':     mAP at IoU threshold 0.75 (stricter matching).
                    #   - 'map_small', 'map_medium', 'map_large': mAP for small, medium, and large objects.
                    #   - 'mar_1', 'mar_10', 'mar_100': Mean Average Recall at 1, 10, and 100 detections per image.
                    #   - 'mar_small', 'mar_medium', 'mar_large': MAR for small, medium, and large objects.
                    #   - 'map_per_class', 'mar_100_per_class': Per-class mAP and MAR (may be -1 if not enough data).
                    #   - 'classes':    List of class indices present in the evaluation.
                    #
                    # metrics['iou'] contains:
                    #   - 'iou':        Mean Intersection over Union (IoU) between predicted and ground-truth boxes.
                    #
                    # Values close to 1.0 indicate perfect detection and overlap; values near 0 indicate poor performance.
                    # -1 for per-class metrics means not enough data for those classes.





def log_metrics(
    args, writer, epoch, i, train_loader, metrics, label_map=None
):
    """
    Logs and saves metrics for classification or object detection.
    """
    if args.model_type == 'object_classifier':
        writer.add_scalar('Accuracy/train_batch', metrics['accuracy'], epoch * len(train_loader) + i + 1)
        print(f'F1 Score: {metrics["f1_score"]}')
        print(f'Precision: {metrics["precision"]}')
        print(f'Recall: {metrics["recall"]}')
        print(f'Confusion Matrix: {metrics["confusion_matrix"]}')
        if label_map is not None:
            confusion_matrix_display = ConfusionMatrixDisplay(
                confusion_matrix=metrics["confusion_matrix"],
                display_labels=label_map.keys()
            )
            confusion_matrix_display.plot(cmap='Blues')

    elif args.model_type == 'object_detector':
        if metrics.get('mean_ap') is not None:
            writer.add_scalar('Mean Average Precision/train_batch', metrics['mean_ap']['map'], epoch * len(train_loader) + i + 1)
            print(f'Mean Average Precision: {metrics["mean_ap"]["map"]}')
        if metrics.get('iou') is not None:
            writer.add_scalar('IoU/train_batch', metrics['iou']['iou'], epoch * len(train_loader) + i + 1)
            print(f'IoU: {metrics["iou"]["iou"]}')

        print("Saving metrics to file...")
        try:
            # save metrics to txt file
            with open(f'{args.training_output_txt}', 'a') as f:
                f.write(
                    f'Epoch: {epoch}, Batch: {i + 1}, Loss: {metrics.get("loss", "N/A")}, Mean Average Precision: {metrics.get("mean_ap", {}).get("map", "N/A")}, IoU: {metrics.get("Iou", "N/A")}\n'
                )
            # save metrics to CSV file
            with open(f'{args.training_output_csv}', 'a', newline='') as csvfile:
                writer_csv = csv.writer(csvfile)
                if csvfile.tell() == 0:
                    writer_csv.writerow(['Epoch', 'Batch', 'Mean Average Precision', 'IoU'])
                writer_csv.writerow([
                    epoch,
                    i + 1,
                    metrics.get("loss", "N/A"),
                    metrics.get("mean_ap", {}).get("map", "N/A"),
                    metrics.get("Iou", "N/A")
                ])

        except Exception as e:
            print(f"Error saving metrics: {e}")
        


################################################## CHECKPOINT  ##################################################

def save_checkpoint(epoch, model, optimizer, data_folder, metrics):
    """
    Save model checkpoint.

    :param epoch: epoch number
    :param model: model
    :param optimizer: optimizer
    :param save_dir: directory where the checkpoint will be saved
    """
    data_folder_path = os.path.normpath(data_folder)  # Normalize the path (handles slashes)
    parent_dir = os.path.dirname(data_folder_path)         # Get the parent directory (removes the last folder)
    parent_folder = os.path.basename(parent_dir)           # Get the folder name (gets the last folder name)
    save_dir = f'6_Checkpoint/{parent_folder}_epoch{epoch}_model.pth'
    
    state = {
                'epoch': epoch,
                'model': model,
                'optimizer': optimizer,
                'loss': metrics.get("loss", "N/A"),
                'mean_ap': metrics.get("mean_ap", {}).get("map", "N/A"),
                'IoU': metrics.get("Iou", "N/A"),
                #'avg_loss': avg_loss,
                #'accuracy' : accuracy,
                #'precision' : precision,
                #'recall' : recall,
                #'f1' : f1,
                #'mAP' : mAP,
                #'IoU' : IoU
            }
    
    try:
        torch.save(state, save_dir)
        print(f'Model state dictionary saved as {save_dir}')

    except Exception as e:
        print(f"Error saving model: {e}")


def manage_top_checkpoints(epoch, model, optimizer, metrics, date_of_dataset_used, save_dir):
    """
    Save one checkpoint with the highest score for each of the following metrics:
    avg_loss, accuracy, precision, recall, f1, mAP, IoU.

    :param epoch: epoch number
    :param model: model
    :param optimizer: optimizer
    :param metrics: dictionary containing avg_loss, accuracy, precision, recall, f1, mAP, IoU
    :param date_of_dataset_used: date of the dataset used for training
    :param save_dir: directory where the checkpoint will be saved
    """
    # Save the current checkpoint
    save_checkpoint(epoch, model, optimizer, metrics, date_of_dataset_used, save_dir)

    # Track checkpoints
    checkpoint_files = [f for f in os.listdir(save_dir) if f.endswith('.pth.tar')]
    checkpoint_files = [os.path.join(save_dir, f) for f in checkpoint_files]

    # Initialize dictionaries to store the best checkpoint for each metric
    best_checkpoints = {
        'avg_loss': None,
        'accuracy': None,
        'precision': None,
        'recall': None,
        'f1': None,
        'mAP': None,
        'IoU': None
    }
    best_scores = {
        'avg_loss': float('inf'),  # Lower is better
        'accuracy': float('-inf'),  # Higher is better
        'precision': float('-inf'),
        'recall': float('-inf'),
        'f1': float('-inf'),
        'mAP': float('-inf'),
        'IoU': float('-inf')
    }

    # Evaluate each checkpoint
    for file in checkpoint_files:
        checkpoint = torch.load(file)
        for metric in best_scores:
            score = checkpoint[metric]
            if (metric == 'avg_loss' and score < best_scores[metric]) or (metric != 'avg_loss' and score > best_scores[metric]):
                best_scores[metric] = score
                best_checkpoints[metric] = file

    # Copy the best checkpoints to the 'Top_checkpoints' directory
    top_checkpoints_dir = r'6_Checkpoints\Top_checkpoints'
    os.makedirs(top_checkpoints_dir, exist_ok=True)
    for metric, file in best_checkpoints.items():
        if file:
            destination = os.path.join(top_checkpoints_dir, os.path.basename(file))
            if not os.path.exists(destination):
                shutil.copy(file, destination)

    # Remove all checkpoints except the best ones
    best_files = set(best_checkpoints.values())
    for file in checkpoint_files:
        if file not in best_files:
            os.remove(file)






def calculate_metrics_and_loss(loader, model, loss_fn, device=torch.device("cpu")):
    """
    Calculate accuracy, precision, recall, F1 score, loss, mAP, and IoU for a given data loader.
    Uses separate_preds_targets to support both classification and object detection.
    """
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_outputs = []
    running_loss = 0.0
    iou_scores = []

    with torch.no_grad():
        for batch in loader:
            # Use separate_preds_targets if batch is a tuple/list with images, boxes, labels, image_files
            if isinstance(batch, (tuple, list)) and len(batch) >= 4:
                images, boxes, labels, image_files = batch[:4]
                preds, targets, valid_image_files = separate_preds_targets(images, boxes, labels, image_files, device)
                # For object detection, model expects list of images, returns list of dicts
                outputs = model(preds)
                # Compute loss if possible (may require custom loss for detection)
                if loss_fn is not None:
                    loss = loss_fn(outputs, targets)
                    running_loss += loss.item()
                # Gather predictions and targets for metrics
                for out, tgt in zip(outputs, targets):
                    all_outputs.append(out)
                    all_labels.append(tgt)
                # Optionally, calculate IoU for each prediction-target pair
                for out, tgt in zip(outputs, targets):
                    if "boxes" in out and "boxes" in tgt and out["boxes"].numel() > 0 and tgt["boxes"].numel() > 0:
                        # Compute IoU between predicted and target boxes (mean over all pairs)
                        iou = calculate_iou(out["boxes"][0], tgt["boxes"][0])
                        iou_scores.append(iou)
            else:
                # Classification case
                inputs, labels = batch[:2]
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                running_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_outputs.extend(outputs.cpu().numpy())
                # For classification, IoU is not meaningful

    # Metrics calculation
    if total > 0:
        accuracy = correct / total
        precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        avg_loss = running_loss / len(loader)
        mAP = calculate_mAP(all_labels, all_outputs)
        mean_iou = sum(iou_scores) / len(iou_scores) if iou_scores else 0.0
    else:
        # For detection, accuracy/precision/recall/f1 may not be meaningful, set to 0
        accuracy = precision = recall = f1 = 0.0
        avg_loss = running_loss / len(loader) if len(loader) > 0 else 0.0
        mAP = calculate_mAP(all_labels, all_outputs) if all_labels and all_outputs else 0.0
        mean_iou = sum(iou_scores) / len(iou_scores) if iou_scores else 0.0

    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(f'Average Loss: {avg_loss:.4f}')
    print(f'mAP: {mAP:.4f}')
    print(f'Mean IoU: {mean_iou:.4f}')

    return accuracy, precision, recall, f1, avg_loss, mAP, mean_iou

def precision_score(y_true, y_pred):
    """
    Calculate precision score for binary classification.

    :param y_true: True labels
    :param y_pred: Predicted labels
    :return: Precision score
    """
    return precision_score(y_true, y_pred, average='binary', zero_division=0)

def recall_score(y_true, y_pred):
    """
    Calculate recall score for binary classification.

    :param y_true: True labels
    :param y_pred: Predicted labels
    :return: Recall score
    """
    return recall_score(y_true, y_pred, average='binary', zero_division=0)

def f1_score(y_true, y_pred): 
    """
    Calculate F1 score for binary classification.

    :param y_true: True labels
    :param y_pred: Predicted labels
    :return: F1 score
    """
    return f1_score(y_true, y_pred, average='binary', zero_division=0)

def calculate_iou(pred, label):
    """
    Calculate Intersection over Union (IoU) for a single prediction and label using torchmetrics.

    :param pred: Predicted tensor (class index or binary mask).
    :param label: Ground truth tensor (class index or binary mask).
    :return: IoU value.
    """
    metric = IntersectionOverUnion(num_classes=2)
    # Ensure pred and label are 1D tensors of the same shape
    pred = pred.unsqueeze(0)
    label = label.unsqueeze(0)
    iou = metric(pred, label)
    return iou.item()


def calculate_mAP(all_labels, all_outputs):
    """
    Calculate mAP (mean Average Precision) using torchmetrics' MeanAveragePrecision.

    :param all_labels: List of true labels (as tensors or lists).
    :param all_outputs: List of model outputs (as tensors or lists of dicts with boxes, scores, labels).
    :return: mAP (mean Average Precision).
    """
    # Prepare predictions and targets in the format expected by torchmetrics
    # Each prediction/target is a dict with keys: 'boxes', 'scores', 'labels'
    # For classification, you can use dummy boxes if needed

    metric = MeanAveragePrecision()
    preds = []
    targets = []

    for i in range(len(all_labels)):
        # For classification, use dummy boxes
        preds.append({
            "boxes": torch.tensor([[0, 0, 1, 1]]),  # dummy box
            "scores": torch.tensor([max(all_outputs[i]) if isinstance(all_outputs[i], (list, tuple)) else all_outputs[i].max().item()]),
            "labels": torch.tensor([all_outputs[i].index(max(all_outputs[i])) if isinstance(all_outputs[i], (list, tuple)) else all_outputs[i].argmax().item()])
        })
        targets.append({
            "boxes": torch.tensor([[0, 0, 1, 1]]),  # dummy box
            "labels": torch.tensor([all_labels[i]])
        })

    metric.update(preds, targets)
    results = metric.compute()
    mAP = results["map"].item() if "map" in results else 0.0
    return mAP






################################################### METRICS LOGGING ##################################################

def save_metrics(epoch, total_epochs, 
                    writer, training_output_file,
                    train_metrics, val_metrics):
    """
    Logs and saves training and validation metrics for each epoch.
    Args:
        epoch (int): The current epoch number.
        total_epochs (int): The total number of epochs.
        writer (torch.utils.tensorboard.SummaryWriter): TensorBoard writer for logging metrics.
        training_output_file (str): Path to the file where metrics will be saved as CSV.
        train_metrics (tuple): A tuple containing training metrics in the following order:
            (accuracy, precision, recall, F1 score, loss, mAP, IoU).
        val_metrics (tuple): A tuple containing validation metrics in the following order:
            (accuracy, precision, recall, F1 score, loss, mAP, IoU).
    Prints:
        - Training and validation metrics for the current epoch.
        - Model, loss function, and optimizer details (only during the first epoch).
    Logs:
        - Training and validation metrics to TensorBoard.
    Saves:
        - Training and validation metrics to a CSV file.
    Note:
        The function assumes that `model`, `loss_fn`, and `optimizer` are defined globally.
        Ensure that these variables are accessible within the function.
    """
    model.eval()

    print(f'Epoch {epoch}/{total_epochs}')

    train_accuracy, train_precision, train_recall, train_f1, train_loss, train_mAP, train_iou = train_metrics
    val_accuracy, val_precision, val_recall, val_f1, val_loss, val_mAP, val_iou = val_metrics

    # Print metrics
    print(f'Train Loss: {train_loss}, Val Loss: {val_loss}')
    print(f'Train Accuracy: {train_accuracy}, Val Accuracy: {val_accuracy}')
    print(f'Train Precision: {train_precision}, Val Precision: {val_precision}')
    print(f'Train Recall: {train_recall}, Val Recall: {val_recall}')
    print(f'Train F1 Score: {train_f1}, Val F1 Score: {val_f1}')
    print(f'Train mAP: {train_mAP}, Val mAP: {val_mAP}')
    print(f'Train IoU: {train_iou}, Val IoU: {val_iou}')

    # Print model, loss function, and optimizer details once
    if epoch == 1:
        print(f'Model: {model}')
        print(f'Loss Function: {loss_fn}')
        print(f'Optimizer: {optimizer}')

    # Log metrics to TensorBoard
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Loss/val', val_loss, epoch)
    writer.add_scalar('Accuracy/train', train_accuracy, epoch)
    writer.add_scalar('Accuracy/val', val_accuracy, epoch)
    writer.add_scalar('Precision/train', train_precision, epoch)
    writer.add_scalar('Precision/val', val_precision, epoch)
    writer.add_scalar('Recall/train', train_recall, epoch)
    writer.add_scalar('Recall/val', val_recall, epoch)
    writer.add_scalar('F1_Score/train', train_f1, epoch)
    writer.add_scalar('F1_Score/val', val_f1, epoch)
    writer.add_scalar('mAP/train', train_metrics[5], epoch)
    writer.add_scalar('mAP/val', val_metrics[5], epoch)
    writer.add_scalar('IoU/train', train_metrics[6], epoch)
    writer.add_scalar('IoU/val', val_metrics[6], epoch)

    # Save metrics to csv
    save_metrics_to_csv(epoch, train_metrics, val_metrics, training_output_file)


def save_metrics_to_csv(epoch, train_metrics, val_metrics, training_output_file):
    """
    Save training and validation metrics to a CSV file.
    """
    csv_file = f'5_Results/{training_output_file.replace(".txt", ".csv")}'
    file_exists = os.path.isfile(csv_file)

    with open(csv_file, mode='a', newline='') as file:
        csv_writer = csv.writer(file)
        if not file_exists:
            # Write header if file does not exist
            csv_writer.writerow(['Epoch', 'Train Loss', 'Val Loss', 'Train Accuracy', 'Val Accuracy',
                                 'Train Precision', 'Val Precision', 'Train Recall', 'Val Recall',
                                 'Train F1 Score', 'Val F1 Score', 'Train mAP', 'Val mAP', 'Train IoU', 'Val IoU'])
        
        # Write metrics
        csv_writer.writerow([epoch] + train_metrics + val_metrics)

######################################################### OTHERS ###########################################################



def decimate(tensor, m):
    """
    Decimate a tensor by a factor 'm', i.e. downsample by keeping every 'm'th value.

    This is used when we convert FC layers to equivalent Convolutional layers, BUT of a smaller size.

    :param tensor: tensor to be decimated
    :param m: list of decimation factors for each dimension of the tensor; None if not to be decimated along a dimension
    :return: decimated tensor
    """
    assert tensor.dim() == len(m)
    for d in range(tensor.dim()):
        if m[d] is not None:
            tensor = tensor.index_select(dim=d,
                                         index=torch.arange(start=0, end=tensor.size(d), step=m[d]).long())

    return tensor



"""
def calculate_mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties):

    Calculate the Mean Average Precision (mAP) of detected objects.

    See https://medium.com/@jonathan_hui/map-mean-average-precision-for-object-detection-45c121a31173 for an explanation

    :param det_boxes: list of tensors, one tensor for each image containing detected objects' bounding boxes
    :param det_labels: list of tensors, one tensor for each image containing detected objects' labels
    :param det_scores: list of tensors, one tensor for each image containing detected objects' labels' scores
    :param true_boxes: list of tensors, one tensor for each image containing actual objects' bounding boxes
    :param true_labels: list of tensors, one tensor for each image containing actual objects' labels
    :param true_difficulties: list of tensors, one tensor for each image containing actual objects' difficulty (0 or 1)
    :return: list of average precisions for all classes, mean average precision (mAP)
    
    assert len(det_boxes) == len(det_labels) == len(det_scores) == len(true_boxes) == len(
        true_labels) == len(
        true_difficulties)  # these are all lists of tensors of the same length, i.e. number of images
    n_classes = len(label_map)

    # Store all (true) objects in a single continuous tensor while keeping track of the image it is from
    true_images = list()
    for i in range(len(true_labels)):
        true_images.extend([i] * true_labels[i].size(0))
    true_images = torch.LongTensor(true_images).to(
        device)  # (n_objects), n_objects is the total no. of objects across all images
    true_boxes = torch.cat(true_boxes, dim=0)  # (n_objects, 4)
    true_labels = torch.cat(true_labels, dim=0)  # (n_objects)
    true_difficulties = torch.cat(true_difficulties, dim=0)  # (n_objects)

    assert true_images.size(0) == true_boxes.size(0) == true_labels.size(0)

    # Store all detections in a single continuous tensor while keeping track of the image it is from
    det_images = list()
    for i in range(len(det_labels)):
        det_images.extend([i] * det_labels[i].size(0))
    det_images = torch.LongTensor(det_images).to(device)  # (n_detections)
    det_boxes = torch.cat(det_boxes, dim=0)  # (n_detections, 4)
    det_labels = torch.cat(det_labels, dim=0)  # (n_detections)
    det_scores = torch.cat(det_scores, dim=0)  # (n_detections)

    assert det_images.size(0) == det_boxes.size(0) == det_labels.size(0) == det_scores.size(0)

    # Calculate APs for each class (except background)
    average_precisions = torch.zeros((n_classes - 1), dtype=torch.float)  # (n_classes - 1)
    for c in range(1, n_classes):
        # Extract only objects with this class
        true_class_images = true_images[true_labels == c]  # (n_class_objects)
        true_class_boxes = true_boxes[true_labels == c]  # (n_class_objects, 4)
        true_class_difficulties = true_difficulties[true_labels == c]  # (n_class_objects)
        n_easy_class_objects = (1 - true_class_difficulties).sum().item()  # ignore difficult objects

        # Keep track of which true objects with this class have already been 'detected'
        # So far, none
        true_class_boxes_detected = torch.zeros((true_class_difficulties.size(0)), dtype=torch.uint8).to(
            device)  # (n_class_objects)

        # Extract only detections with this class
        det_class_images = det_images[det_labels == c]  # (n_class_detections)
        det_class_boxes = det_boxes[det_labels == c]  # (n_class_detections, 4)
        det_class_scores = det_scores[det_labels == c]  # (n_class_detections)
        n_class_detections = det_class_boxes.size(0)
        if n_class_detections == 0:
            continue

        # Sort detections in decreasing order of confidence/scores
        det_class_scores, sort_ind = torch.sort(det_class_scores, dim=0, descending=True)  # (n_class_detections)
        det_class_images = det_class_images[sort_ind]  # (n_class_detections)
        det_class_boxes = det_class_boxes[sort_ind]  # (n_class_detections, 4)

        # In the order of decreasing scores, check if true or false positive
        true_positives = torch.zeros((n_class_detections), dtype=torch.float).to(device)  # (n_class_detections)
        false_positives = torch.zeros((n_class_detections), dtype=torch.float).to(device)  # (n_class_detections)
        for d in range(n_class_detections):
            this_detection_box = det_class_boxes[d].unsqueeze(0)  # (1, 4)
            this_image = det_class_images[d]  # (), scalar

            # Find objects in the same image with this class, their difficulties, and whether they have been detected before
            object_boxes = true_class_boxes[true_class_images == this_image]  # (n_class_objects_in_img)
            object_difficulties = true_class_difficulties[true_class_images == this_image]  # (n_class_objects_in_img)
            # If no such object in this image, then the detection is a false positive
            if object_boxes.size(0) == 0:
                false_positives[d] = 1
                continue

            # Find maximum overlap of this detection with objects in this image of this class
            overlaps = find_jaccard_overlap(this_detection_box, object_boxes)  # (1, n_class_objects_in_img)
            max_overlap, ind = torch.max(overlaps.squeeze(0), dim=0)  # (), () - scalars

            # 'ind' is the index of the object in these image-level tensors 'object_boxes', 'object_difficulties'
            # In the original class-level tensors 'true_class_boxes', etc., 'ind' corresponds to object with index...
            original_ind = torch.LongTensor(range(true_class_boxes.size(0)))[true_class_images == this_image][ind]
            # We need 'original_ind' to update 'true_class_boxes_detected'

            # If the maximum overlap is greater than the threshold of 0.5, it's a match
            if max_overlap.item() > 0.5:
                # If the object it matched with is 'difficult', ignore it
                if object_difficulties[ind] == 0:
                    # If this object has already not been detected, it's a true positive
                    if true_class_boxes_detected[original_ind] == 0:
                        true_positives[d] = 1
                        true_class_boxes_detected[original_ind] = 1  # this object has now been detected/accounted for
                    # Otherwise, it's a false positive (since this object is already accounted for)
                    else:
                        false_positives[d] = 1
            # Otherwise, the detection occurs in a different location than the actual object, and is a false positive
            else:
                false_positives[d] = 1

        # Compute cumulative precision and recall at each detection in the order of decreasing scores
        cumul_true_positives = torch.cumsum(true_positives, dim=0)  # (n_class_detections)
        cumul_false_positives = torch.cumsum(false_positives, dim=0)  # (n_class_detections)
        cumul_precision = cumul_true_positives / (
                cumul_true_positives + cumul_false_positives + 1e-10)  # (n_class_detections)
        cumul_recall = cumul_true_positives / n_easy_class_objects  # (n_class_detections)

        # Find the mean of the maximum of the precisions corresponding to recalls above the threshold 't'
        recall_thresholds = torch.arange(start=0, end=1.1, step=.1).tolist()  # (11)
        precisions = torch.zeros((len(recall_thresholds)), dtype=torch.float).to(device)  # (11)
        for i, t in enumerate(recall_thresholds):
            recalls_above_t = cumul_recall >= t
            if recalls_above_t.any():
                precisions[i] = cumul_precision[recalls_above_t].max()
            else:
                precisions[i] = 0.
        average_precisions[c - 1] = precisions.mean()  # c is in [1, n_classes - 1]

    # Calculate Mean Average Precision (mAP)
    mean_average_precision = average_precisions.mean().item()

    # Keep class-wise average precisions in a dictionary
    average_precisions = {rev_label_map[c + 1]: v for c, v in enumerate(average_precisions.tolist())}

    return average_precisions, mean_average_precision
"""

def xy_to_cxcy(xy):
    """
    Convert bounding boxes from boundary coordinates (x_min, y_min, x_max, y_max) to center-size coordinates (c_x, c_y, w, h).

    :param xy: bounding boxes in boundary coordinates, a tensor of size (n_boxes, 4)
    :return: bounding boxes in center-size coordinates, a tensor of size (n_boxes, 4)
    """
    return torch.cat([(xy[:, 2:] + xy[:, :2]) / 2,  # c_x, c_y
                      xy[:, 2:] - xy[:, :2]], 1)  # w, h


def cxcy_to_xy(cxcy):
    """
    Convert bounding boxes from center-size coordinates (c_x, c_y, w, h) to boundary coordinates (x_min, y_min, x_max, y_max).

    :param cxcy: bounding boxes in center-size coordinates, a tensor of size (n_boxes, 4)
    :return: bounding boxes in boundary coordinates, a tensor of size (n_boxes, 4)
    """
    return torch.cat([cxcy[:, :2] - (cxcy[:, 2:] / 2),  # x_min, y_min
                      cxcy[:, :2] + (cxcy[:, 2:] / 2)], 1)  # x_max, y_max


def cxcy_to_gcxgcy(cxcy, priors_cxcy):
    """
    Encode bounding boxes (that are in center-size form) w.r.t. the corresponding prior boxes (that are in center-size form).

    For the center coordinates, find the offset with respect to the prior box, and scale by the size of the prior box.
    For the size coordinates, scale by the size of the prior box, and convert to the log-space.

    In the model, we are predicting bounding box coordinates in this encoded form.

    :param cxcy: bounding boxes in center-size coordinates, a tensor of size (n_priors, 4)
    :param priors_cxcy: prior boxes with respect to which the encoding must be performed, a tensor of size (n_priors, 4)
    :return: encoded bounding boxes, a tensor of size (n_priors, 4)
    """

    # The 10 and 5 below are referred to as 'variances' in the original Caffe repo, completely empirical
    # They are for some sort of numerical conditioning, for 'scaling the localization gradient'
    # See https://github.com/weiliu89/caffe/issues/155
    return torch.cat([(cxcy[:, :2] - priors_cxcy[:, :2]) / (priors_cxcy[:, 2:] / 10),  # g_c_x, g_c_y
                      torch.log(cxcy[:, 2:] / priors_cxcy[:, 2:]) * 5], 1)  # g_w, g_h


def gcxgcy_to_cxcy(gcxgcy, priors_cxcy):
    """
    Decode bounding box coordinates predicted by the model, since they are encoded in the form mentioned above.

    They are decoded into center-size coordinates.

    This is the inverse of the function above.

    :param gcxgcy: encoded bounding boxes, i.e. output of the model, a tensor of size (n_priors, 4)
    :param priors_cxcy: prior boxes with respect to which the encoding is defined, a tensor of size (n_priors, 4)
    :return: decoded bounding boxes in center-size form, a tensor of size (n_priors, 4)
    """

    return torch.cat([gcxgcy[:, :2] * priors_cxcy[:, 2:] / 10 + priors_cxcy[:, :2],  # c_x, c_y
                      torch.exp(gcxgcy[:, 2:] / 5) * priors_cxcy[:, 2:]], 1)  # w, h


def find_intersection(set_1, set_2):
    """
    Find the intersection of every box combination between two sets of boxes that are in boundary coordinates.

    :param set_1: set 1, a tensor of dimensions (n1, 4)
    :param set_2: set 2, a tensor of dimensions (n2, 4)
    :return: intersection of each of the boxes in set 1 with respect to each of the boxes in set 2, a tensor of dimensions (n1, n2)
    """

    # PyTorch auto-broadcasts singleton dimensions
    lower_bounds = torch.max(set_1[:, :2].unsqueeze(1), set_2[:, :2].unsqueeze(0))  # (n1, n2, 2)
    upper_bounds = torch.min(set_1[:, 2:].unsqueeze(1), set_2[:, 2:].unsqueeze(0))  # (n1, n2, 2)
    intersection_dims = torch.clamp(upper_bounds - lower_bounds, min=0)  # (n1, n2, 2)
    return intersection_dims[:, :, 0] * intersection_dims[:, :, 1]  # (n1, n2)


def find_jaccard_overlap(set_1, set_2):
    """
    Find the Jaccard Overlap (IoU) of every box combination between two sets of boxes that are in boundary coordinates.

    :param set_1: set 1, a tensor of dimensions (n1, 4)
    :param set_2: set 2, a tensor of dimensions (n2, 4)
    :return: Jaccard Overlap of each of the boxes in set 1 with respect to each of the boxes in set 2, a tensor of dimensions (n1, n2)
    """

    # Find intersections
    intersection = find_intersection(set_1, set_2)  # (n1, n2)

    # Find areas of each box in both sets
    areas_set_1 = (set_1[:, 2] - set_1[:, 0]) * (set_1[:, 3] - set_1[:, 1])  # (n1)
    areas_set_2 = (set_2[:, 2] - set_2[:, 0]) * (set_2[:, 3] - set_2[:, 1])  # (n2)

    # Find the union
    # PyTorch auto-broadcasts singleton dimensions
    union = areas_set_1.unsqueeze(1) + areas_set_2.unsqueeze(0) - intersection  # (n1, n2)

    return intersection / union  # (n1, n2)


# Some augmentation functions below have been adapted from
# From https://github.com/amdegroot/ssd.pytorch/blob/master/utils/augmentations.py

def expand(image, boxes, filler):
    """
    Perform a zooming out operation by placing the image in a larger canvas of filler material.

    Helps to learn to detect smaller objects.

    :param image: image, a tensor of dimensions (3, original_h, original_w)
    :param boxes: bounding boxes in boundary coordinates, a tensor of dimensions (n_objects, 4)
    :param filler: RBG values of the filler material, a list like [R, G, B]
    :return: expanded image, updated bounding box coordinates
    """
    # Calculate dimensions of proposed expanded (zoomed-out) image
    original_h = image.size(1)
    original_w = image.size(2)
    max_scale = 4
    scale = random.uniform(1, max_scale)
    new_h = int(scale * original_h)
    new_w = int(scale * original_w)

    # Create such an image with the filler
    filler = torch.FloatTensor(filler)  # (3)
    new_image = torch.ones((3, new_h, new_w), dtype=torch.float) * filler.unsqueeze(1).unsqueeze(1)  # (3, new_h, new_w)
    # Note - do not use expand() like new_image = filler.unsqueeze(1).unsqueeze(1).expand(3, new_h, new_w)
    # because all expanded values will share the same memory, so changing one pixel will change all

    # Place the original image at random coordinates in this new image (origin at top-left of image)
    left = random.randint(0, new_w - original_w)
    right = left + original_w
    top = random.randint(0, new_h - original_h)
    bottom = top + original_h
    new_image[:, top:bottom, left:right] = image

    # Adjust bounding boxes' coordinates accordingly
    new_boxes = boxes + torch.FloatTensor([left, top, left, top]).unsqueeze(
        0)  # (n_objects, 4), n_objects is the no. of objects in this image

    return new_image, new_boxes


def random_crop(image, boxes, labels, difficulties):
    """
    Performs a random crop in the manner stated in the paper. Helps to learn to detect larger and partial objects.

    Note that some objects may be cut out entirely.

    Adapted from https://github.com/amdegroot/ssd.pytorch/blob/master/utils/augmentations.py

    :param image: image, a tensor of dimensions (3, original_h, original_w)
    :param boxes: bounding boxes in boundary coordinates, a tensor of dimensions (n_objects, 4)
    :param labels: labels of objects, a tensor of dimensions (n_objects)
    :param difficulties: difficulties of detection of these objects, a tensor of dimensions (n_objects)
    :return: cropped image, updated bounding box coordinates, updated labels, updated difficulties
    """
    original_h = image.size(1)
    original_w = image.size(2)
    # Keep choosing a minimum overlap until a successful crop is made
    while True:
        # Randomly draw the value for minimum overlap
        min_overlap = random.choice([0., .1, .3, .5, .7, .9, None])  # 'None' refers to no cropping

        # If not cropping
        if min_overlap is None:
            return image, boxes, labels, difficulties

        # Try up to 50 times for this choice of minimum overlap
        # This isn't mentioned in the paper, of course, but 50 is chosen in paper authors' original Caffe repo
        max_trials = 50
        for _ in range(max_trials):
            # Crop dimensions must be in [0.3, 1] of original dimensions
            # Note - it's [0.1, 1] in the paper, but actually [0.3, 1] in the authors' repo
            min_scale = 0.3
            scale_h = random.uniform(min_scale, 1)
            scale_w = random.uniform(min_scale, 1)
            new_h = int(scale_h * original_h)
            new_w = int(scale_w * original_w)

            # Aspect ratio has to be in [0.5, 2]
            aspect_ratio = new_h / new_w
            if not 0.5 < aspect_ratio < 2:
                continue

            # Crop coordinates (origin at top-left of image)
            left = random.randint(0, original_w - new_w)
            right = left + new_w
            top = random.randint(0, original_h - new_h)
            bottom = top + new_h
            crop = torch.FloatTensor([left, top, right, bottom])  # (4)

            # Calculate Jaccard overlap between the crop and the bounding boxes
            overlap = find_jaccard_overlap(crop.unsqueeze(0),
                                           boxes)  # (1, n_objects), n_objects is the no. of objects in this image
            overlap = overlap.squeeze(0)  # (n_objects)

            # If not a single bounding box has a Jaccard overlap of greater than the minimum, try again
            if overlap.max().item() < min_overlap:
                continue

            # Crop image
            new_image = image[:, top:bottom, left:right]  # (3, new_h, new_w)

            # Find centers of original bounding boxes
            bb_centers = (boxes[:, :2] + boxes[:, 2:]) / 2.  # (n_objects, 2)

            # Find bounding boxes whose centers are in the crop
            centers_in_crop = (bb_centers[:, 0] > left) * (bb_centers[:, 0] < right) * (bb_centers[:, 1] > top) * (
                    bb_centers[:, 1] < bottom)  # (n_objects), a Torch uInt8/Byte tensor, can be used as a boolean index

            # If not a single bounding box has its center in the crop, try again
            if not centers_in_crop.any():
                continue

            # Discard bounding boxes that don't meet this criterion
            new_boxes = boxes[centers_in_crop, :]
            new_labels = labels[centers_in_crop]
            new_difficulties = difficulties[centers_in_crop]

            # Calculate bounding boxes' new coordinates in the crop
            new_boxes[:, :2] = torch.max(new_boxes[:, :2], crop[:2])  # crop[:2] is [left, top]
            new_boxes[:, :2] -= crop[:2]
            new_boxes[:, 2:] = torch.min(new_boxes[:, 2:], crop[2:])  # crop[2:] is [right, bottom]
            new_boxes[:, 2:] -= crop[:2]

            return new_image, new_boxes, new_labels, new_difficulties


def flip(image, boxes):
    """
    Flip image horizontally.

    :param image: image, a PIL Image
    :param boxes: bounding boxes in boundary coordinates, a tensor of dimensions (n_objects, 4)
    :return: flipped image, updated bounding box coordinates
    """
    # Flip image
    new_image = FT.hflip(image)

    # Flip boxes
    new_boxes = boxes
    new_boxes[:, 0] = image.width - boxes[:, 0] - 1
    new_boxes[:, 2] = image.width - boxes[:, 2] - 1
    new_boxes = new_boxes[:, [2, 1, 0, 3]]

    return new_image, new_boxes


def resize(image, boxes, dims=(300, 300), return_percent_coords=True):
    """
    Resize image. For the SSD300, resize to (300, 300).

    Since percent/fractional coordinates are calculated for the bounding boxes (w.r.t image dimensions) in this process,
    you may choose to retain them.

    :param image: image, a PIL Image
    :param boxes: bounding boxes in boundary coordinates, a tensor of dimensions (n_objects, 4)
    :return: resized image, updated bounding box coordinates (or fractional coordinates, in which case they remain the same)
    """
    # Resize image
    new_image = FT.resize(image, dims)

    # Resize bounding boxes
    old_dims = torch.FloatTensor([image.width, image.height, image.width, image.height]).unsqueeze(0)
    new_boxes = boxes / old_dims  # percent coordinates

    if not return_percent_coords:
        new_dims = torch.FloatTensor([dims[1], dims[0], dims[1], dims[0]]).unsqueeze(0)
        new_boxes = new_boxes * new_dims

    return new_image, new_boxes


def photometric_distort(image):
    """
    Distort brightness, contrast, saturation, and hue, each with a 50% chance, in random order.

    :param image: image, a PIL Image
    :return: distorted image
    """
    new_image = image

    distortions = [FT.adjust_brightness,
                   FT.adjust_contrast,
                   FT.adjust_saturation,
                   FT.adjust_hue]

    random.shuffle(distortions)

    for d in distortions:
        if random.random() < 0.5:
            if d.__name__ is 'adjust_hue':
                # Caffe repo uses a 'hue_delta' of 18 - we divide by 255 because PyTorch needs a normalized value
                adjust_factor = random.uniform(-18 / 255., 18 / 255.)
            else:
                # Caffe repo uses 'lower' and 'upper' values of 0.5 and 1.5 for brightness, contrast, and saturation
                adjust_factor = random.uniform(0.5, 1.5)

            # Apply this distortion
            new_image = d(new_image, adjust_factor)

    return new_image


def transform(image, boxes, labels, difficulties, split):
    """
    Apply the transformations above.

    :param image: image, a PIL Image
    :param boxes: bounding boxes in boundary coordinates, a tensor of dimensions (n_objects, 4)
    :param labels: labels of objects, a tensor of dimensions (n_objects)
    :param difficulties: difficulties of detection of these objects, a tensor of dimensions (n_objects)
    :param split: one of 'TRAIN' or 'TEST', since different sets of transformations are applied
    :return: transformed image, transformed bounding box coordinates, transformed labels, transformed difficulties
    """
    assert split in {'TRAIN', 'TEST'}

    # Mean and standard deviation of ImageNet data that our base VGG from torchvision was trained on
    # see: https://pytorch.org/docs/stable/torchvision/models.html
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    new_image = image
    new_boxes = boxes
    new_labels = labels
    new_difficulties = difficulties
    # Skip the following operations for evaluation/testing
    if split == 'TRAIN':
        # A series of photometric distortions in random order, each with 50% chance of occurrence, as in Caffe repo
        new_image = photometric_distort(new_image)

        # Convert PIL image to Torch tensor
        new_image = FT.to_tensor(new_image)

        # Expand image (zoom out) with a 50% chance - helpful for training detection of small objects
        # Fill surrounding space with the mean of ImageNet data that our base VGG was trained on
        if random.random() < 0.5:
            new_image, new_boxes = expand(new_image, boxes, filler=mean)

        # Randomly crop image (zoom in)
        new_image, new_boxes, new_labels, new_difficulties = random_crop(new_image, new_boxes, new_labels,
                                                                         new_difficulties)

        # Convert Torch tensor to PIL image
        new_image = FT.to_pil_image(new_image)

        # Flip image with a 50% chance
        if random.random() < 0.5:
            new_image, new_boxes = flip(new_image, new_boxes)

    # Resize image to (300, 300) - this also converts absolute boundary coordinates to their fractional form
    new_image, new_boxes = resize(new_image, new_boxes, dims=(300, 300))

    # Convert PIL image to Torch tensor
    new_image = FT.to_tensor(new_image)

    # Normalize by mean and standard deviation of ImageNet data that our base VGG was trained on
    new_image = FT.normalize(new_image, mean=mean, std=std)

    return new_image, new_boxes, new_labels, new_difficulties


def adjust_learning_rate(optimizer, scale):
    """
    Scale learning rate by a specified factor.

    :param optimizer: optimizer whose learning rate must be shrunk.
    :param scale: factor to multiply learning rate with.
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * scale
    print("DECAYING learning rate.\n The new LR is %f\n" % (optimizer.param_groups[1]['lr'],))


def accuracy(scores, targets, k):
    """
    Computes top-k accuracy, from predicted and true labels.

    :param scores: scores from the model
    :param targets: true labels
    :param k: k in top-k accuracy
    :return: top-k accuracy
    """
    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)




class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)






def report_epoch_metrics(epoch, total_epochs, train_metrics, val_metrics):
    """
    Print training and validation metrics and loss for the current epoch.

    Args:
        epoch (int): The current epoch number.
        total_epochs (int): The total number of epochs.
        train_metrics (tuple): Training metrics (accuracy, precision, recall, f1, loss, mAP, IoU).
        val_metrics (tuple): Validation metrics (accuracy, precision, recall, f1, loss, mAP, IoU).
    """
    print(f"Epoch {epoch}/{total_epochs}")
    print(f"Train Loss: {train_metrics[4]:.4f}, Val Loss: {val_metrics[4]:.4f}")
    print(f"Train Accuracy: {train_metrics[0]:.4f}, Val Accuracy: {val_metrics[0]:.4f}")
    print(f"Train Precision: {train_metrics[1]:.4f}, Val Precision: {val_metrics[1]:.4f}")
    print(f"Train Recall: {train_metrics[2]:.4f}, Val Recall: {val_metrics[2]:.4f}")
    print(f"Train F1 Score: {train_metrics[3]:.4f}, Val F1 Score: {val_metrics[3]:.4f}")
    print(f"Train mAP: {train_metrics[5]:.4f}, Val mAP: {val_metrics[5]:.4f}")
    print(f"Train IoU: {train_metrics[6]:.4f}, Val IoU: {val_metrics[6]:.4f}")
    print("-" * 50)



