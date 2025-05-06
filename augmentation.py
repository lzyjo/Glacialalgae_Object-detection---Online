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
from torchvision import tv_tensors
from torchvision import io, utils
from torchvision.transforms.v2 import functional as F
from itertools import combinations
import random

from label_map import label_map_classifier # This is the label map for the classifier
from label_map import label_map_OD # This is the label map for the object detector



def generate_all_transformations(transformations_to_include):
    """
    Generate all possible unique pairings of transformations without repetition.
    Parameters:
    transformations_to_include (list): List of transformations to include in the combinations.
    Returns:
    list: List of composed transformations.
    """
    all_transformations = []
    seen_combinations = set()
    for r in range(1, len(transformations_to_include) + 1):
        for combo in combinations(transformations_to_include, r):
            combo_types = tuple(sorted(type(t).__name__ for t in combo))
            if combo_types not in seen_combinations:
                seen_combinations.add(combo_types)
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
        T.RandomGrayscale(p=0.1)
    ]


def define_and_generate_transformations(num_random_augmentations=7, 
                                        include_color_jitter=True, 
                                        include_horizontal_flip=True, 
                                        include_vertical_flip=True, 
                                        include_photometric_distort=True,
                                        include_random_rotation=False,
                                        include_random_resized_crop=False,
                                        include_gaussian_blur=False):
    """
    Define a set of transformations, generate all possible unique combinations, 
    and randomly select a specified number of augmentations.

    Parameters:
    num_random_augmentations (int): Number of random augmentations to select from all possible combinations.
    include_color_jitter (bool): Whether to include ColorJitter transformations.
    include_horizontal_flip (bool): Whether to include RandomHorizontalFlip transformation.
    include_vertical_flip (bool): Whether to include RandomVerticalFlip transformation.
    include_photometric_distort (bool): Whether to include RandomPhotometricDistort transformation.
    include_random_rotation (bool): Whether to include RandomRotation transformation.
    include_random_resized_crop (bool): Whether to include RandomResizedCrop transformation.
    include_gaussian_blur (bool): Whether to include GaussianBlur transformation.

    Returns:
    list: A list of randomly selected transformations composed of unique combinations.
    """
    # Define the transformations to be included
    transformations_to_include = []

    if include_color_jitter:
        color_jitter = T.ColorJitter(brightness=(0, 1), contrast=(0, 1), saturation=(0, 1), hue=(0, 0.5))
        ColourJitter_params = T.ColorJitter.get_params(
            color_jitter.brightness, color_jitter.contrast, color_jitter.saturation,
            color_jitter.hue)
        transformations_to_include.append(
            T.ColorJitter(brightness=ColourJitter_params[1],
                          contrast=ColourJitter_params[2],
                          saturation=ColourJitter_params[3],
                          hue=ColourJitter_params[4])
        )

    if include_horizontal_flip:
        transformations_to_include.append(T.RandomHorizontalFlip(p=1))

    if include_vertical_flip:
        transformations_to_include.append(T.RandomVerticalFlip(p=1))

    if include_photometric_distort:
        photometric_distort = T.RandomPhotometricDistort(brightness=0.5, 
                                                         contrast=0.5, 
                                                         saturation=0.5, 
                                                         hue=0.1)
        PhotometricDistort_params = T.RandomPhotometricDistort.get_params(
            photometric_distort.brightness, 
            photometric_distort.contrast, 
            photometric_distort.saturation, 
            photometric_distort.hue)
        transformations_to_include.append(
            T.RandomPhotometricDistort(brightness=PhotometricDistort_params[0],
                                       contrast=PhotometricDistort_params[1],
                                       saturation=PhotometricDistort_params[2],
                                       hue=PhotometricDistort_params[3])
        )
        if include_random_rotation:
            random_rotation = T.RandomRotation(degrees=45)
            RandomRotation_params = T.RandomRotation.get_params(random_rotation.degrees)
            transformations_to_include.append(
                T.RandomRotation(degrees=RandomRotation_params)
            )
        
        if include_random_resized_crop:
            random_resized_crop = T.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0))
            RandomResizedCrop_params = T.RandomResizedCrop.get_params(
                random_resized_crop.size, random_resized_crop.scale, random_resized_crop.ratio
            )
            transformations_to_include.append(
                T.RandomResizedCrop(size=RandomResizedCrop_params[0], scale=RandomResizedCrop_params[1])
            )
        
        if include_gaussian_blur:
            gaussian_blur = T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))
            GaussianBlur_params = T.GaussianBlur.get_params(
                gaussian_blur.kernel_size, gaussian_blur.sigma
            )
            transformations_to_include.append(
                T.GaussianBlur(kernel_size=GaussianBlur_params[0], sigma=GaussianBlur_params[1])
            )

    # Generate all possible pairings of transformations
    all_transformations = generate_all_transformations(transformations_to_include)
    print(f"Total number of transformations: {len(all_transformations)}")

    # Choose random augmentations from all transformations
    random_augmentations = random.sample(all_transformations, 
                                         min(num_random_augmentations, len(all_transformations)))
    
    print(f"Random augmentations: {random_augmentations}")  

    return random_augmentations

if __name__ == "__main__":
    # Example usage
    random_augmentations = define_and_generate_transformations(num_random_augmentations=7, 
                                                                include_color_jitter=True, 
                                                                include_horizontal_flip=True, 
                                                                include_vertical_flip=True, 
                                                                include_photometric_distort=True)
    print(f"Random augmentations: {random_augmentations}")  


##########################################################################################################################


def prepare_data_for_augmentation(augmented_dataset_path, 
                                  date_of_dataset_used,
                                  image_dir, annotation_dir, 
                                  transformations,
                                  label_map):
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
        raise ValueError("The images are different sizes.")

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

    for annotation in [f for f in os.listdir(annotation_dir) if f.endswith('.xml')]:
        annotation_path = os.path.join(annotation_dir, annotation)

        new_annotation_path = shutil.copy(annotation_path, new_annotation_dir)  # Make a copy of the annotation
        original_parsed = parse_annotation(annotation_path, label_map)
        new_parsed = parse_annotation(new_annotation_path, label_map)
        if original_parsed == new_parsed:
            continue
        else:
            print("The annotations are different.")
            sys.exit()
        


def data_augmentation(augmented_image_dir, augmented_annotation_dir, 
                      transformations,
                      label_map):
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
        pil_image = Image.open(image_path, mode='r').convert('RGB')
        image_to_augment = tv_tensors.Image(pil_image)

        augmented_image = transformations(image_to_augment)

        # Convert the augmented image back to PIL format and save it
        augmented_image = F.to_pil_image(augmented_image)
        augmented_image.save(image_path)

        # Process the corresponding annotation for this image
        annotation_name = os.path.splitext(os.path.basename(image_path))[0] + ".xml"
        annotation_path = os.path.join(augmented_annotation_dir, annotation_name)

        if os.path.exists(annotation_path):
            parsed_annotations = parse_annotation(annotation_path, label_map)
            # Read objects in this image (bounding boxes, labels, difficulties) from parsed_annotations
            boxes = torch.FloatTensor(parsed_annotations['boxes'])  # (n_objects, 4)
            boxes = tv_tensors.BoundingBoxes(boxes, 
                                            format="XYXY",
                                            canvas_size=(pil_image.size[1], 
                                                         pil_image.size[0]))

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
        else:
            print(f"Annotation file {annotation_name} not found for image {image}. Skipping.")



def visual_augmentation_check(original_image_dir, original_annotation_dir, 
                                augmented_image_dir, augmented_annotation_dir, 
                                label_map,
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
            rect_0 = plt.Rectangle((xmin, ymin), width, height, edgecolor='r', facecolor='none')
            ax[0].add_patch(rect_0)
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
            rect_1 = plt.Rectangle((xmin, ymin), width, height, edgecolor='r', facecolor='none')
            ax[1].add_patch(rect_1)
        ax[1].set_title(f'Augmented Image - Pair {pair_number}')

        plt.show()


def run_augmentation_pipeline(augmented_dataset_path, date_of_dataset_used, 
                              image_dir, annotation_dir, 
                              augmentation, num_pairs=15, 
                              object_detector=False):
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
    object_detector (bool): Whether to use the object detector label map. Default is False.
    Returns:
    None
    """
    # Select the appropriate label map
    label_map = label_map_OD if object_detector is True else label_map_classifier

    # Create a folder for the date used under 2_DataAugmentation
    augmented_dataset_path = os.path.join(augmented_dataset_path, date_of_dataset_used)

    if not os.path.exists(augmented_dataset_path):
        os.makedirs(augmented_dataset_path, exist_ok=True)
        print(f"Date folder {augmented_dataset_path} created.")
    else:
        print(f"Date folder {augmented_dataset_path} already exists.")

    # Check if transformations are provided
    if not augmentation:
        raise ValueError("No augmentations provided. Please provide augmentations.")

    if not isinstance(augmentation, list):
        raise TypeError("The 'augmentation' parameter must be a list of transformations.")
        
    for i, transformations in enumerate(augmentation):
        if isinstance(transformations, T.Compose):
            transform_name = "_".join([type(t).__name__.lower() for t in transformations.transforms])
        else:
            transform_name = type(transformations).__name__.lower()

        new_data_dir = os.path.join(augmented_dataset_path, f"{date_of_dataset_used}_{transform_name}")
        print(f"New data directory: {new_data_dir}")
        print("-" * 50)  # Add a line for easier visual separation

        # Check if the data directory for this transformation already exists
        if os.path.exists(new_data_dir):
            print(f"Data directory {new_data_dir} already exists. Skipping this transformation.")
            continue

        print(f"Running augmentation pipeline for transformation set {i+1}/{len(augmentation)}: {transform_name}")
        print(f"New data directory: {new_data_dir}")
        print(f"Transformations: {transformations}")

        # Prepare data for augmentation
        prepare_data_for_augmentation(augmented_dataset_path, date_of_dataset_used, 
                                      image_dir, annotation_dir, 
                                      transformations, label_map)
        print("Data prepared for augmentation.")
        
        # Define new directories for augmented data
        new_image_dir = os.path.join(new_data_dir, "Images")
        new_annotation_dir = os.path.join(new_data_dir, "Annotations")

        # Perform data augmentation
        data_augmentation(new_image_dir, new_annotation_dir, 
                          transformations, 
                          label_map)
        print("Data augmentation completed.")
        
        # Visualize the original and augmented images with bounding boxes
        visual_augmentation_check(image_dir, annotation_dir, 
                                  new_image_dir, new_annotation_dir, 
                                  label_map,
                                  num_pairs)
        
    print("All augmentations in the provided augmentation list have been completed.")


"""   if os.path.exists(os.path.join(augmented_dataset_path, f"{date_of_dataset_used}_{transform_name}")):
        print(f"New data directory already exists: {os.path.join(augmented_dataset_path, f'{date_of_dataset_used}_{transform_name}')}. "
              "Check to see if data augmentation has already been carried out. "
              "Stopping execution.")
        return"""



##########################################################################################################################


def test_test_prepare_data_for_augmentation(augmented_dataset_path, 
                                    date_of_dataset_used,
                                    image_dir, annotation_dir, 
                                    transformations,
                                    label_map,
                                    augmentation):
    """
    Prepares data for augmentation by creating necessary directories, copying
    original images and annotations, and verifying the integrity of the copied files.
    Also creates a folder for the specified date under the augmented dataset path and checks for existing transformations.

    Parameters:
    augmented_dataset_path (str): The base path where the augmented dataset will be stored.
    date_of_dataset_used (str): The date of the dataset being used, used to create a unique folder.
    image_dir (str): The directory containing the original images.
    annotation_dir (str): The directory containing the original annotations.
    transformations (object): The specific transformations to be applied to the data.
    label_map (dict): The label map for the dataset, used for parsing annotations.
    augmentation (list): A list of transformation sets, where each set can be a single transformation 
                            or a composition of multiple transformations. This differs from the 
                            `transformations` parameter, which refers to the specific transformations 
                            being applied to the data in the current context.

    Returns:
    list: A list of transformations that do not already have corresponding directories.
    
    Raises:
    SystemExit: If no transformations are provided, if the copied files are different,
                or if the images are different sizes.
    """

    # Create a folder for the specified date under the augmented dataset path
    augmented_dataset_path = os.path.join(augmented_dataset_path, date_of_dataset_used)

    if not os.path.exists(augmented_dataset_path):
        os.makedirs(augmented_dataset_path, exist_ok=True)
        print(f"Date folder {augmented_dataset_path} created.")
    else:
        print(f"Date folder {augmented_dataset_path} already exists.")

    # Iterate through each transformation set in the augmentation list
    for i, transformations in enumerate(augmentation):

        if transformations is None:
            print("No transformations provided. Please provide transformations.")
            sys.exit()

        # Generate a unique name for the transformation set
        if isinstance(transformations, T.Compose):
            transform_name = "_".join([type(t).__name__.lower() for t in transformations.transforms])
        else:
            transform_name = type(transformations).__name__.lower()

        # Define the directory for the current transformation set
        new_data_dir = os.path.join(augmented_dataset_path, f"{date_of_dataset_used}_{transform_name}")
        print(f"Data directory for transformation: {new_data_dir}")

        # Check if the directory already exists; if so, skip this transformation
        if os.path.exists(new_data_dir):
            print(f"Data directory {new_data_dir} already exists. Skipping this transformation.")
            print("-" * 50)  # Add a line for easier visual separation
            continue
        else:
            os.makedirs(new_data_dir, exist_ok=True)
            print(f"Directory {new_data_dir} created.")

        # Create subdirectories for images and annotations within the transformation directory
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

        print("-" * 50)  # Add a line for easier visual separation

        # Copy original images to the new directory and verify their integrity
        for image in os.listdir(image_dir):
            image_path = os.path.join(image_dir, image)
            image = Image.open(image_path, mode='r').convert('RGB')
            image_size = image.size  # Get the size of the image

            new_image_path = shutil.copy(image_path, new_image_dir)  # Make a copy of the image
            new_image = Image.open(new_image_path, mode='r').convert('RGB')
            new_image_size = new_image.size

            # Verify that the copied file is identical to the original
            with open(image_path, 'rb') as f1, open(new_image_path, 'rb') as f2:
                if f1.read() == f2.read():
                    continue
                else:
                    print("The files are different.")
                    sys.exit()

            # Check if the image dimensions are the same
            if image_size == new_image_size:
                continue
            else:
                print("The images are different sizes.")
                sys.exit()
        
        # Copy original annotations to the new directory and verify their integrity
        for annotation in os.listdir(annotation_dir):
            annotation_path = os.path.join(annotation_dir, annotation)

            new_annotation_path = shutil.copy(annotation_path, new_annotation_dir)  # Make a copy of the annotation
            if parse_annotation(annotation_path, label_map) \
                == parse_annotation(new_annotation_path, label_map):
                continue
            else:
                raise ValueError("The annotations are different.")
                sys.exit()

        print("Data prepared for augmentation.")