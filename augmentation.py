from torchvision.transforms import v2 as T
import PIL
import torch 
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import sys
import xml.etree.ElementTree as ET
import os
import shutil
from utils import parse_annotation
from label_map import label_map
from torchvision import tv_tensors
from torchvision import io, utils
from torchvision.transforms.v2 import functional as F
from itertools import combinations




def generate_all_transformations(transformations_to_include):
    """
    Generate all possible pairings of transformations.
    Parameters:
    transformations_to_include (list): List of transformations to include in the combinations.
    Returns:
    list: List of composed transformations.
    """
    all_transformations = []
    for r in range(1, len(transformations_to_include) + 1):
        for combo in combinations(transformations_to_include, r):
            all_transformations.append(T.Compose(combo))
    return all_transformations

if __name__ == "__main__":
    transformations_to_include = [
    T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.5),
    T.RandomRotation(degrees=45),
    T.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
    T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
    T.RandomGrayscale(p=0.2)
]
# Define possible transformations
# transformations_to_include = [
   # T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
   # T.RandomHorizontalFlip(p=0.5),
   # T.RandomVerticalFlip(p=0.5),
   # T.RandomRotation(degrees=45),
   # T.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
   # T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
   # T.RandomGrayscale(p=0.2)]


def prepare_data_for_augmentation(augmented_dataset_path, 
                                  date_of_dataset_used,
                                  image_dir, annotation_dir, 
                                  transformations):
    """
    Prepares data for augmentation by creating necessary directories, copying
    original images and annotations, and verifying the integrity of the copied files.
    Parameters:
    augmented_dataset_path (str): The path to the augmented GA dataset.
    data_dir (str): The directory where the data is stored.
    date_of_dataset_used (str): The date of the dataset being used.
    image_dir (str): The directory containing the original images.
    annotation_dir (str): The directory containing the original annotations.
    transformations (object): The transformations to be applied to the data.
    Raises:
    SystemExit: If no transformations are provided, if the copied files are different,
                or if the images are different sizes.
    """

    # Define the transformations
    if transformations is None:
        print("No transformations provided. Please provide transformations.")
        sys.exit()

    #Create a new folder to store the augmented data
    if isinstance(transformations, T.Compose):
        transform_name = "_".join([type(t).__name__.lower() for t in transformations.transforms])
            # Check if no transformation is provided, then use the first item on the list (ColourJitter)
    else:
        transform_name = type(transformations).__name__.lower()


    # Create a new directory to store the augmented data
    new_data_dir = os.path.join(augmented_dataset_path,
                                f"{date_of_dataset_used}_{transform_name}")
    if not os.path.exists(new_data_dir):
        os.makedirs(new_data_dir, exist_ok=True)
        print(f"Directory {new_data_dir} created.")
    else:
        print(f"Directory {new_data_dir} already exists.")

    new_image_dir = os.path.join(new_data_dir, "Images")  
    if not os.path.exists(new_image_dir):
        os.makedirs(new_image_dir, exist_ok=True) 
        print(f"Directory {new_image_dir} created.")
    else:
        print(f"Directory {new_image_dir} already exists.")

    new_annotation_dir = os.path.join(new_data_dir, "Annotations")
    if not os.path.exists(new_annotation_dir):
        os.makedirs(new_annotation_dir, exist_ok=True)
        print(f"Directory {new_annotation_dir} created.")
    else:
        print(f"Directory {new_annotation_dir} already exists")

    # Copy original images and annotations to the new directory
    for image in os.listdir(image_dir):
        
        image_path = os.path.join(image_dir, image)
        image = Image.open(image_path, mode='r').convert('RGB')
        image_size = image.size  # Get the size of the image

        new_image_path = shutil.copy(image_path, new_image_dir) # Make a copy of the image
        new_image = Image.open(new_image_path, mode='r').convert('RGB')
        new_image_size = new_image.size

        with open(image_path, 'rb') as f1, open(new_image_path, 'rb') as f2:
            if f1.read() == f2.read():
                continue
            else:
                print("The files are different.")
                sys.exit()

        # Check if the images are the same size
        if image_size == new_image_size:
            continue
        else:
            print("The images are different sizes.")
            sys.exit()

    for annotation in os.listdir(annotation_dir):
        annotation_path = os.path.join(annotation_dir, annotation)

        new_annotation_path = shutil.copy(annotation_path, new_annotation_dir)  # Make a copy of the annotation
        if parse_annotation(annotation_path, label_map) \
            == parse_annotation(new_annotation_path, label_map):
            continue
        else:
            print("The annotations are different.")
            sys.exit()
        


def data_augmentation(augmented_image_dir, augmented_annotation_dir, 
                      transformations):
    """
    Perform data augmentation on images and their corresponding annotations.
    Parameters:
    augmented_image_dir (str): Directory containing the images to be augmented.
    augmented_annotation_dir (str): Directory containing the annotation files corresponding to the images.
    boxes (list): List of bounding boxes for the objects in the images.
    labels (list): List of labels for the objects in the images.
    transformations (list): List of transformation functions to apply to the images and annotations.
    Returns:
    None
    """

    for image in os.listdir(augmented_image_dir):
        image_path = os.path.join(augmented_image_dir, image)
        image = Image.open(image_path, mode='r').convert('RGB')
        image = tv_tensors.Image(image)
        augmented_image = transformations(image)
        augmented_image = T.ToPILImage()(augmented_image)
        augmented_image.save(image_path)

    for annotation in os.listdir(augmented_annotation_dir):
        annotation_path = os.path.join(augmented_annotation_dir, annotation)
        
        parsed_annotations = parse_annotation(annotation_path, label_map)
        # Read objects in this image (bounding boxes, labels, difficulties) from parsed_annotations
        boxes = torch.FloatTensor(parsed_annotations['boxes'])  # (n_objects, 4)
        boxes = tv_tensors.BoundingBoxes(boxes, 
                                       format='XYXY',
                                       canvas_size=(augmented_image.size[1], 
                                                    augmented_image.size[0]))

        augmented_boxes = transformations(boxes)

        # Parse the XML file and get the root element
        tree = ET.parse(annotation_path)
        root = tree.getroot()

        # Save the new bounding boxes to XML file
        new_tree = ET.ElementTree(root)
        for obj, new_bbox in zip(root.findall('object'), augmented_boxes):
            bbox = obj.find('bndbox')
            bbox.find('xmin').text = str(int(new_bbox[0].item()))
            bbox.find('ymin').text = str(int(new_bbox[1].item()))
            bbox.find('xmax').text = str(int(new_bbox[2].item()))
            bbox.find('ymax').text = str(int(new_bbox[3].item()))
        new_tree.write(annotation_path)



def visual_augmentation_check(original_image_dir, original_annotation_dir, 
                                augmented_image_dir, augmented_annotation_dir, 
                                num_pairs=15):
    """
    Visualizes the original and augmented images with bounding boxes for comparison.
    This function loops through a set of original and augmented images along with their corresponding annotations.
    It displays a specified number of pairs of images side by side, with bounding boxes drawn on them to show the regions of interest.
    Args:
        original_image_dir (str): Directory containing the original images.
        original_annotation_dir (str): Directory containing the original annotations.
        augmented_image_dir (str): Directory containing the augmented images.
        augmented_annotation_dir (str): Directory containing the augmented annotations.
        num_pairs (int): Number of image-annotation pairs to display. Default is 15.
    Returns:
        None
    """
    # Loop through images and annotations to show bounding boxes on images
    image_files_original = list(Path(original_image_dir).glob('*'))
    annotation_files_original = list(Path(original_annotation_dir).glob('*'))
    image_files_augmented = list(Path(augmented_image_dir).glob('*'))
    annotation_files_augmented = list(Path(augmented_annotation_dir).glob('*'))

    indices = np.random.choice(len(image_files_original), num_pairs, replace=False)

    for idx, i in enumerate(indices):
        image_file_original = image_files_original[i]
        annotation_file_original = annotation_files_original[i]
        image_file_augmented = image_files_augmented[i]
        annotation_file_augmented = annotation_files_augmented[i]

        # Extract the pair number from the stem of the image or annotation file
        pair_number = image_file_original.stem

        # Show original image with bounding boxes
        image_original = Image.open(image_file_original, mode='r').convert('RGB')
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].imshow(image_original)

        parsed_annotations_original = parse_annotation(annotation_file_original, label_map)
        boxes_original = torch.FloatTensor(parsed_annotations_original['boxes'])

        for bbox in boxes_original:
            xmin, ymin, xmax, ymax = bbox
            width = xmax - xmin
            height = ymax - ymin
            rect = plt.Rectangle((xmin, ymin), width, height, edgecolor='r', facecolor='none')
            ax[0].add_patch(rect)
        ax[0].set_title(f'Original Image - Pair {pair_number}')

        # Show augmented image with bounding boxes
        image_augmented = Image.open(image_file_augmented, mode='r').convert('RGB')
        ax[1].imshow(image_augmented)

        parsed_annotations_augmented = parse_annotation(annotation_file_augmented, label_map)
        boxes_augmented = torch.FloatTensor(parsed_annotations_augmented['boxes'])

        for bbox in boxes_augmented:
            xmin, ymin, xmax, ymax = bbox
            width = xmax - xmin
            height = ymax - ymin
            rect = plt.Rectangle((xmin, ymin), width, height, edgecolor='r', facecolor='none')
            ax[1].add_patch(rect)
        ax[1].set_title(f'Augmented Image - Pair {pair_number}')

        plt.show()


def run_augmentation_pipeline(augmented_dataset_path, date_of_dataset_used, 
                              image_dir, annotation_dir, 
                              transformations, num_pairs=15):
    """
    Runs the entire augmentation pipeline including data preparation, augmentation,
    and visualization of the original and augmented images with bounding boxes.
    Parameters:
    augmented_dataset_path (str): The path to the augmented GA dataset.
    date_of_dataset_used (str): The date of the dataset being used.
    image_dir (str): The directory containing the original images.
    annotation_dir (str): The directory containing the original annotations.
    transformations (object): The transformations to be applied to the data.
    num_pairs (int): Number of image-annotation pairs to display. Default is 15.
    Returns:
    None
    """
    # Prepare data for augmentation
    prepare_data_for_augmentation(augmented_dataset_path, date_of_dataset_used, 
                                  image_dir, annotation_dir, 
                                  transformations)
    print("Data prepared for augmentation.")
    
    # Define new directories for augmented data
    if isinstance(transformations, T.Compose):
        transform_name = "_".join([type(t).__name__.lower() for t in transformations.transforms])
    else:
        transform_name = type(transformations).__name__.lower()
    new_data_dir = os.path.join(augmented_dataset_path, f"{date_of_dataset_used}_{transform_name}")
    new_image_dir = os.path.join(new_data_dir, "Images")
    new_annotation_dir = os.path.join(new_data_dir, "Annotations")
    print(f"New data directory: {new_data_dir}")
    print(f"New image directory: {new_image_dir}")
    print(f"New annotation directory: {new_annotation_dir}")
    
    # Perform data augmentation
    data_augmentation(new_image_dir, new_annotation_dir, transformations)
    print("Data augmentation completed.")
    
    # Visualize the original and augmented images with bounding boxes
    visual_augmentation_check(image_dir, annotation_dir, 
                              new_image_dir, new_annotation_dir, 
                              num_pairs)
    





