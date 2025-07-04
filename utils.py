import os
import zipfile
import shutil
import json
import xml.etree.ElementTree as ET
import torch
import random
import torchvision.transforms.functional as FT
from sklearn.model_selection import train_test_split
import pandas as pd
from datetime import datetime
from label_map import label_map as label_map_Classifier
from label_map_OD import label_map as label_map_OD
from hyperparameters import * 


device = torch.device("cpu")

# Color map for bounding boxes of detected objects from https://sashat.me/2017/01/11/list-of-20-simple-distinct-colors/
#distinct_colors = ['#e6194b', '#3cb44b', '#ffe119', '#0082c8']
# label_color_map = {k: distinct_colors[i] for i, k in enumerate(label_map.keys())}

# Dataset prep


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
    Returns:
        tuple: A tuple containing the paths to the created folders:
               (annotations_folder, images_folder, split_folder, date_folder, train_image_folder, train_annotation_folder).
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

    # Define train subfolders if augmented
    split_folder = os.path.join(date_folder, 'Split') if Split else None
    train_image_folder = os.path.join(split_folder, 'train', 'images') if Split else None
    train_annotation_folder = os.path.join(split_folder, 'train', 'annotations') if Split else None
    test_image_folder = os.path.join(split_folder, 'test', 'images') if Split else None
    test_annotation_folder = os.path.join(split_folder, 'test', 'annotations') if Split else None

    # Check if the folder already exists
    if os.path.exists(date_folder):
        print(f'Dataset folder already exists: {date_folder}')
        if Split:
            print(f'Train Images Folder: {train_image_folder}')
            print(f'Train Annotations Folder: {train_annotation_folder}')
            print(f'Test Images Folder: {test_image_folder}')
            print(f'Test Annotations Folder: {test_annotation_folder}')
        return annotations_folder, images_folder, split_folder, date_folder, train_image_folder, train_annotation_folder

    # Create the folders if they don't exist
    os.makedirs(annotations_folder, exist_ok=True)
    os.makedirs(images_folder, exist_ok=True)
    os.makedirs(split_folder, exist_ok=True)

    if Split:
        os.makedirs(train_image_folder, exist_ok=True)
        os.makedirs(train_annotation_folder, exist_ok=True)
        os.makedirs(test_image_folder, exist_ok=True)
        os.makedirs(test_annotation_folder, exist_ok=True)

    print(f'Dataset folder created: {date_folder}')
    if Split:
        print(f'Train Images Folder: {train_image_folder}')
        print(f'Train Annotations Folder: {train_annotation_folder}')
        print(f'Test Images Folder: {test_image_folder}')
        print(f'Test Annotations Folder: {test_annotation_folder}')
    return annotations_folder, images_folder, split_folder, date_folder, train_image_folder, train_annotation_folder


if __name__ == '__main__':
    use_current_date = input("Use current date for folder creation? (yes/no): ").strip().lower()

    if use_current_date == 'yes':
        folder_date = None  # Use current date
    else:
        folder_date = input("Enter the folder date (e.g., 20250318): ").strip()

    # Create the dataset folder
    create_dataset_folder(base_folder='1_GA_Dataset',
                          folder_date=folder_date, # Change this to the correct folder for which files are to be extracted to
                           Augmented=True,  # Set to True if the dataset is augmented
                           Split=True)  # Set to True if the dataset is split into train/test folders









def extract_files(date_of_dataset_used, annotations_folder, images_folder, annotations_src_folder, images_src_folder):   # Define source, annotations, and images folders
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

    if not os.path.exists(annotations_folder):
        os.makedirs(annotations_folder)
    else:
        print(f"Folder {annotations_folder} already exists.")
        

    if not os.path.exists(images_folder):
        os.makedirs(images_folder)
    else:
        print(f"Folder {images_folder} already exists.")

    print("-" * 50)  # Add a separator line for better readability

    # Extract .xml files to annotations folder
    num_annotations = len([f for f in os.listdir(annotations_src_folder) if f.endswith('.xml')]) 
    print(f"Total annotations from {annotations_src_folder}: {num_annotations}")

    counter = len([f for f in os.listdir(annotations_folder) if f.endswith('.xml')]) + 1
    for root, dirs, files in os.walk(annotations_src_folder):
        for file in files:
            if file.endswith('.xml'):
                dst_file = os.path.join(annotations_folder, f"{counter}.xml") 
                if os.path.exists(dst_file):
                    print(f"Error: File {dst_file} already exists. / "
                          f"Source file: {file} / "
                          f"Destination file: {dst_file}")
                    return
                shutil.copy(os.path.join(root, file), dst_file)
                counter += 1

    num_annotations_moved = len([f for f in os.listdir(annotations_folder) if f.endswith('.xml')]) 
    print(f"Files extracted from {date_of_dataset_used} to {annotations_folder}/ "
          f"Total annotations: {num_annotations_moved}")

    # Extract .tif files to images folder
    num_images = len([f for f in os.listdir(images_src_folder) if f.endswith('.tif')])
    print(f"Total images from {images_src_folder}: {num_images}")
    
    counter = len([f for f in os.listdir(images_folder) if f.endswith('.tif')]) + 1
    for root, dirs, files in os.walk(images_src_folder):
        for file in files:
            if file.endswith('.tif'):
                dst_file = os.path.join(images_folder, f"{counter}.tif")
                if os.path.exists(dst_file):
                    print(f"Error: File {dst_file} already exists. / "
                          f"Source file: {file} / "
                          f"Destination file: {dst_file}")
                    return
                shutil.copy(os.path.join(root, file), dst_file)
                counter += 1
            
    num_images_moved = len([f for f in os.listdir(images_folder) if f.endswith('.tif')])
    print(f"Files extracted from {date_of_dataset_used} to {images_folder}/"
          f"Total images: {num_images_moved}")
    print("-" * 50)  # Add a separator line for better readability

    if len(annotations_folder) == len(images_folder):
        print("The number of annotation files and image files are the same.")
        print("Number of annotation files:", len(annotations_folder))
        print("Number of image files:", len(images_folder)) 

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
        print("-" * 50)  # Add a separator line for better readability


        

if __name__ == '__main__':
    extract_files(date_of_dataset_used='Bluff_230724',  # Change this to the correct dataset used, FOR REFERENCE ONLY
                    annotations_folder=r'GA_Dataset\20250219\Annotations',  # Change this to the correct folder for which files were extracted 
                    images_folder=r'GA_Dataset\20250219\Images',  # Change this to the correct folder for which files were extracted 
                    images_src_folder=r'Completed annotations/Bluff_230724/Original_Images_Unlabelled_Bluff_230724',
                    annotations_src_folder=r'Completed annotations\Bluff_230724')  # Change this to your source folder path 
                    



def extract_files_from_multiple_folders(source_folders, annotations_folder, images_folder, include_test=False):
    """
    Extracts files from multiple source folders and organizes them into specified 
    annotations and images folders. Each folder in `source_folders` must contain 
    subdirectories named 'Images' and 'Annotations'. The function processes these 
    subdirectories, optionally including those containing 'test' in their names.
    Args:
        source_folders (list of str): A list of paths to the source folders containing 
            datasets to be processed.
        annotations_folder (str): The destination folder where annotation files will 
            be copied or moved.
        images_folder (str): The destination folder where image files will be copied 
            or moved.
        include_test (bool): Whether to include folders containing 'test' in their names.
    Returns:
        None
    Side Effects:
        - Prints the details of each dataset being processed, including the dataset 
            name, source folder for images, and source folder for annotations.
        - Calls the `extract_files` function for each dataset to handle the actual 
            file extraction and organization.
    Notes:
        - The function assumes that each dataset folder contains subdirectories named 
            'Images' and 'Annotations' for the respective files.
        - This function relies on the `extract_files` function to handle the actual 
            file extraction and organization. Ensure that `extract_files` is defined 
            and accessible in the codebase.
    """
    
    datasets = []

    for source_folder in source_folders:
        for folder_name in os.listdir(source_folder):
            if (include_test and 'test' in folder_name.lower()) or (not include_test and 'test' not in folder_name.lower()):
                folder_path = os.path.join(source_folder, folder_name)
                if os.path.isdir(folder_path):  # Ensure it's a directory
                    datasets.append({
                        "date_of_dataset_used": folder_name,
                        "images_src_folder": os.path.join(folder_path, 'Images'),
                        "annotations_src_folder": os.path.join(folder_path, 'Annotations')
                    })
    print(datasets)
    print("-" * 50)  # Add a separator line for better readability

    # If include_test is True, filter datasets to only include those with 'test' in their names
    if include_test:
        datasets = [dataset for dataset in datasets if 'test' in dataset['date_of_dataset_used'].lower()]

    # Print all source files in individual lines
    for dataset in datasets:
        print(f"Dataset: {dataset['date_of_dataset_used']}")
        print(f"Images Source Folder: {dataset['images_src_folder']}")
        print(f"Annotations Source Folder: {dataset['annotations_src_folder']}")
        print("-" * 50)  # Add a separator line for better readability

    proceed = input("Do you want to proceed with extracting files? (yes/no): ").strip().lower()
    while proceed not in ['yes', 'no']:
        print(f"Invalid input: {proceed}")
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

if __name__ == "__main__":
    # Define the source folders to extract files from
    source_folders = [
        r'1_GA_Dataset\20250318\Split',  # Original dataset, not augmented
        r'2_DataAugmentation\20250318'  # Augmented datasets
    ]

    # Define the destination folders for annotations and images
    annotations_folder = r'3_TrainingData\20250318_Augmented\Split\train\annotations'
    images_folder = r'3_TrainingData\20250318_Augmented\Split\train\images'

    # Call the function to extract files from multiple folders
    extract_files_from_multiple_folders(source_folders, annotations_folder, images_folder, include_test=True)























# Train-test split

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



def create_folders(output_folder):
    """
    Create necessary folders for train and test datasets.
    """

    train_image_folder = os.path.join(output_folder,  'train', 'images')
    test_image_folder = os.path.join(output_folder,  'test', 'images')
    train_annotation_folder = os.path.join(output_folder,  'train', 'annotations')
    test_annotation_folder = os.path.join(output_folder,  'test', 'annotations')

    if not os.path.exists(train_image_folder):
        os.makedirs(train_image_folder)
    else:
        print(f"Folder {train_image_folder} already exists.")

    if not os.path.exists(test_image_folder):
        os.makedirs(test_image_folder)
    else:
        print(f"Folder {test_image_folder} already exists.")

    if not os.path.exists(train_annotation_folder):
        os.makedirs(train_annotation_folder)
    else:
        print(f"Folder {train_annotation_folder} already exists")
    
    if not os.path.exists(test_annotation_folder):
        os.makedirs(test_annotation_folder)
    else:
        print(f"Folder {test_annotation_folder} already exists.")
        

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


def split_and_copy_files(images, annotations, output_folder,  test_size=None, random_state=None):

    if test_size is None:
        test_size = 0.2
    if random_state is None:
        random_state = 42

    if os.path.exists(os.path.join(output_folder,  'train', 'images')) and \
       os.path.exists(os.path.join(output_folder,   'test', 'images')) and \
       os.path.exists(os.path.join(output_folder,   'train', 'annotations')) and \
       os.path.exists(os.path.join(output_folder,   'test', 'annotations')) and \
        all(file.endswith('.tif') for file in os.listdir(os.path.join(output_folder,   'train', 'images'))) and \
        all(file.endswith('.tif') for file in os.listdir(os.path.join(output_folder,   'test', 'images'))) and \
        all(file.endswith('.xml') for file in os.listdir(os.path.join(output_folder,   'train', 'annotations'))) and \
        all(file.endswith('.xml') for file in os.listdir(os.path.join(output_folder,   'test', 'annotations'))) and \
        len(os.listdir(os.path.join(output_folder,   'train', 'images'))) > 0 and \
        len(os.listdir(os.path.join(output_folder,   'test', 'images'))) > 0 and \
        len(os.listdir(os.path.join(output_folder,   'train', 'annotations'))) > 0 and \
        len(os.listdir(os.path.join(output_folder,   'test', 'annotations'))) > 0:

        print("Train and test lists already exist. Dataset has been split and contains relevant files.")

        total_annotations = len(os.listdir(os.path.join(output_folder, 'train', 'annotations'))) + \
                            len(os.listdir(os.path.join(output_folder, 'test', 'annotations')))
        if total_annotations == len(annotations):
            print("The number of XML files in train and test folders matches the total number of XML files in annotations.")
        else:
            print("Mismatch in the number of XML files between train/test folders and annotations. Train" \
            "test split may belong to an old version of the dataset. Try extracting and splitting the dataset again.")
            
        return
    
    else:
        train_images, test_images, train_annotations, test_annotations = train_test_split(
        images, annotations, test_size=test_size, random_state=random_state)

        create_folders(output_folder)
        copy_files(train_images, os.path.join(output_folder,   'train', 'images'), file_extension='.tif')
        copy_files(test_images, os.path.join(output_folder,   'test', 'images'), file_extension= '.tif')
        copy_files(train_annotations, os.path.join(output_folder,   'train', 'annotations'), file_extension='.xml')
        copy_files(test_annotations, os.path.join(output_folder,   'test', 'annotations'),file_extension='.xml')

        print(f"Files have been split and copied to {output_folder}")

if __name__ == '__main__':
    """Change paths and output folder accordingly to your setup"""
    split_and_copy_files(images, annotations, output_folder=r'GA_Dataset\Split', test_size=0.2, random_state=42)



# Datalist creation

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
#annotation_file is created in create_data_lists()

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

    return {'boxes': boxes, 'labels': labels, 'difficulties': difficulties}



def create_data_lists(train_annotation_path, train_image_path, test_annotation_path, test_image_path, 
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












def check_model_trained(checkpoint_dir='6_Checkpoints',
                        date_of_dataset_used='20250318', 
                        augmented=False,):
    """
    Check if a model has already been trained by verifying the presence of a checkpoint file.

    Args:
        date_of_dataset_used (str): The date of the dataset used.
        augmented (bool): Whether the dataset is augmented. Defaults to False.
        checkpoint_dir (str): Directory where checkpoints are stored. Defaults to '6_Checkpoints'.

    Returns:
        bool: True if the model checkpoint exists, False otherwise.
    """
    if augmented:
        date_of_dataset_used += '_Augmented'
    
    model_files = [f for f in os.listdir(checkpoint_dir) if date_of_dataset_used in f]
    if augmented:
        model_files = [f for f in model_files if 'Augmented' in f]
    if model_files:
        model_path = os.path.join(checkpoint_dir, model_files[0])
        print(f'Model may be present. Please check: {model_path}')
    else:
        print(f'Model for date: {date_of_dataset_used} has not been trained or is not present in {checkpoint_dir}')

if __name__ == '__main__':
    check_model_trained(checkpoint_dir='6_Checkpoints',
                        date_of_dataset_used='20250318', 
                        augmented=False)






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



def calculate_mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties):
    """
    Calculate the Mean Average Precision (mAP) of detected objects.

    See https://medium.com/@jonathan_hui/map-mean-average-precision-for-object-detection-45c121a31173 for an explanation

    :param det_boxes: list of tensors, one tensor for each image containing detected objects' bounding boxes
    :param det_labels: list of tensors, one tensor for each image containing detected objects' labels
    :param det_scores: list of tensors, one tensor for each image containing detected objects' labels' scores
    :param true_boxes: list of tensors, one tensor for each image containing actual objects' bounding boxes
    :param true_labels: list of tensors, one tensor for each image containing actual objects' labels
    :param true_difficulties: list of tensors, one tensor for each image containing actual objects' difficulty (0 or 1)
    :return: list of average precisions for all classes, mean average precision (mAP)
    """
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


def manage_training_output_file(results_folder, 
                                date_of_dataset_used,
                                params, 
                                augmented=False):
    """
    This function manages the training output file by creating or appending to it, and writes the training parameters at the top of the file if they do not already exist.

    Args:
        results_folder (str): The folder where the results will be stored.
        date_of_dataset_used (str): The date of the dataset used for training.
        augmentation (bool): Whether the dataset is augmented.

    Returns:
        str: The path to the training output file.
    """

    # Hyperparameters
    params = Hyperparameters

    if augmented:
        date_of_dataset_used += '_Augmented'

    training_output_file = os.path.join(results_folder, 
                                        f'training_results_{date_of_dataset_used}.txt')
    
    if os.path.exists(training_output_file):
        with open(training_output_file, 'r') as read_file:
            content = read_file.read()
        mode = 'a'  # Append mode
    else:
        os.makedirs(results_folder, exist_ok=True)  # Ensure the results folder exists
        content = ''
        mode = 'w'  # Write mode
        
    with open(training_output_file, mode) as f:
        # Write the training parameters at the top of the file if they do not already exist
        if mode == 'a':
            if f'Checkpoint Frequency: {checkpoint_freq}' not in content:
                f.write(f'Checkpoint Frequency: {checkpoint_freq}\n')
            if f'Date of Dataset Used: {date_of_dataset_used}' not in content:
                f.write(f'Date of Dataset Used: {date_of_dataset_used}\n')
            if f'Learning Rate: {lr}' not in content:
                f.write(f'Learning Rate: {lr}\n')
            if f'Iterations: {iterations}' not in content:
                f.write(f'Iterations: {iterations}\n\n')
            
        if mode == 'w':    
            f.write(f'Checkpoint Frequency: {checkpoint_freq}\n')
            f.write(f'Date of Dataset Used: {date_of_dataset_used}\n')
            f.write(f'Learning Rate: {lr}\n')
            f.write(f'Iterations: {iterations}\n')

    return training_output_file


def save_checkpoint(epoch, model, optimizer, date_of_dataset_used, save_dir):
    """
    Save model checkpoint.

    :param epoch: epoch number
    :param model: model
    :param optimizer: optimizer
    :param save_dir: directory where the checkpoint will be saved
    """
    state = {'epoch': epoch,
             'model': model,
             'optimizer': optimizer}
    filename = os.path.join(save_dir, f'{date_of_dataset_used}_checkpoint_{epoch}.pth.tar')
    
    torch.save(state, filename)


def keep_checkpoints(checkpoint_dir, log_file, date_of_dataset_used):
    """
    Keeps only the checkpoints corresponding to the last epoch and the epoch with the lowest loss.
    Args:
        checkpoint_dir (str): Directory where checkpoint files are stored.
        log_file (str): Path to the log file containing epoch loss information.
        date_of_dataset_used (str): Date string used to identify relevant checkpoint files.
    Returns:
        None
    This function reads the log file to find the epoch with the lowest loss and the last epoch.
    It then removes all checkpoint files in the specified directory except for those corresponding
    to the last epoch and the epoch with the lowest loss.
    """
    # Read the log file to find the epoch with the lowest loss
    with open(log_file, 'r') as f:
        lines = f.readlines()[8:]
    
    epoch_losses = {}
    for line in lines:
        if "Epoch" in line and "Loss" in line:
            parts = line.split()
            # Extract the epoch number from the format "Epoch: [52][0/17]"
            epoch_str = parts[1] # Extract the epoch number from the format "Epoch: [52][0/17]"
            epoch_str = epoch_str[:-6]  # Remove the last 6 characters
            epoch = int(epoch_str.strip('[]'))
            # Extract the loss value from the format "Loss: 0.000"
            loss_str = parts[11]
            loss = float(loss_str)
            # Store the loss value for this epoch
            epoch_losses[epoch] = loss 
    
    if not epoch_losses:
        print("No epoch loss information found in log file.")
        return
    
    lowest_loss_epoch = min(epoch_losses, key=epoch_losses.get)
    print(f"Lowest loss epoch: {lowest_loss_epoch} with loss: {epoch_losses[lowest_loss_epoch]}")

    # Get the last epoch
    last_epoch = max(epoch_losses.keys())
    print(f"Last epoch: {last_epoch}")

    # Remove all checkpoints except the last epoch and the lowest loss epoch
    for filename in os.listdir(checkpoint_dir):
        if date_of_dataset_used not in filename:
            continue

        epoch_num = int(filename.split('_')[-1].split('.')[0])

        if epoch_num != last_epoch and epoch_num != lowest_loss_epoch:
            os.remove(os.path.join(checkpoint_dir, filename))
            print(f"Removed checkpoint: {filename}")


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
