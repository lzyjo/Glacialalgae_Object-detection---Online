import numpy as np
import torch
from torch.utils.data import Dataset
import json
import os
from PIL import Image
from utils import transform
import xml.etree.ElementTree as ET
from utils import parse_annotation
from label_map import label_map_Classifier
from label_map import  label_map_OD


"""
“PyTorch Dataset is all about. It is an object that is required to implement two methods:
 __len__ and __getitem__. The former should return the number of items in the dataset; 
 the latter should return the item, consisting of a sample and its corresponding label (an integer index).2” 
 ([Stevens et al., 2020, p. 166])
"""



class PC_Dataset(Dataset):
    """
    PyTorch Dataset for phase contrast images with object detection annotations.

    Expected folder structure:
        dataset_folder/
            train/
                annotations/
                images/
            test/
                annotations/
                images/
            val/
                annotations/
                images/

    Each sample consists of an image and its corresponding annotation file.
    """

    def __init__(self, 
                 dataset_folder, 
                 split='train',
                 transform=None,
                 keep_difficult=False):
        """
        Initialize the PC_Dataset.

        Args:
            dataset_folder (str): Path to the root dataset directory. This directory should contain subfolders for each split
                      ('train', 'test', 'val'), each with 'images' and 'annotations' subdirectories.
            split (str, optional): Dataset split to use. Must be one of 'train', 'test', or 'val'. Default is 'train'.
            transform (callable, optional): Optional transformation function to apply to each sample (image, boxes, labels, difficulties).
            keep_difficult (bool, optional): Whether to keep objects marked as 'difficult' in the annotations. Default is False.

        Raises:
            AssertionError: If the split is not one of 'train', 'test', or 'val'.
            FileNotFoundError: If the required image or annotation directories do not exist.
            AssertionError: If the number of images and annotation files do not match.

        The dataset expects the following directory structure:
            dataset_folder/
            train/
                images/
                annotations/
            test/
                images/
                annotations/
            val/
                images/
                annotations/
        """
        
        self.dataset_folder = dataset_folder  # Root folder of the dataset
        self.keep_difficult = keep_difficult  # Whether to keep difficult objects in the dataset

        self.split = split.lower() 

        assert split in ['train', 'test', 'val'], "split must be 'train', 'test', or 'val'"

        # Read image and annotation file names from folders (not using .json files)
        annotation_folder = os.path.join(dataset_folder, split, "annotations")
        image_folder = os.path.join(dataset_folder, split, "images")
        if not (os.path.isdir(annotation_folder) and os.path.isdir(image_folder)):
            raise FileNotFoundError(f"Missing directory: {annotation_folder} or {image_folder}")

        # List all image and annotation files
        self.image_files = sorted([
            f for f in os.listdir(image_folder)
            if os.path.isfile(os.path.join(image_folder, f))
        ])
        self.annotation_files = sorted([
            f for f in os.listdir(annotation_folder)
            if os.path.isfile(os.path.join(annotation_folder, f))
        ])

        self.annotation_folder = annotation_folder
        self.image_folder = image_folder
        self.transform = transform

        assert len(self.image_files) == len(self.annotation_files), "Number of images and annotations must match"

    def __len__(self):
        return len(self.image_files) #can just return because previously checked that image_files and annotation_files have the same length

    def __getitem__(self, idx):
        """
        Retrieve a sample from the dataset at the given index `idx`.

        This method performs the following steps:
            1. Resolves the file paths for the image and its annotation based on the index.
            2. Loads the image from disk, converts it to RGB, and transforms it into a normalized PyTorch tensor.
            3. Parses the corresponding annotation file (e.g., Pascal VOC XML) to extract bounding boxes, labels, and difficulty flags.
            4. Converts the annotation data into PyTorch tensors.
            5. Optionally filters out objects marked as 'difficult' if `keep_difficult` is False.
            6. Applies any provided data transformations (e.g., augmentation, normalization).
            7. Returns a tuple containing the image tensor, bounding boxes, labels, and difficulties.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: (image, boxes, labels, difficulties)
            - image (Tensor): Image tensor of shape (3, H, W), normalized to [0, 1].
            - boxes (Tensor): Bounding boxes, shape (n_objects, 4).
            - labels (Tensor): Class labels for each object.
            - difficulties (Tensor): Difficulty flags for each object.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_path = os.path.join(self.image_folder, self.image_files[idx])
        annotation_path = os.path.join(self.annotation_folder, self.annotation_files[idx])

        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Missing image file: {image_path}")
        if not os.path.isfile(annotation_path):
            raise FileNotFoundError(f"Missing annotation file: {annotation_path}")

        # Load image and convert to tensor
        # image = read_image(image_path) frm torchvion.io cannot handle .tiff
        image = Image.open(image_path).convert('RGB')
        image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0  # (C, H, W), normalized
                                                                        # Normalised because:
                                                                        # RGB have a range of [0, 255],
                                                                        # and we want to convert it to [0, 1] for PyTorch
                                                                        # so that the model will not learn in an imbalanced way:
                                                                        # pixel values or feature values (input) can differ largely
                                                                        # we want to keep the input values in a similar range


        # Read objects in this image (bounding boxes, labels, difficulties)
        boxes, labels, difficulties = parse_annotation(annotation_path, label_map=label_map_OD)

        # Parse annotation to get boxes, labels, difficulties
            # annotation_file is a file path (usually to an XML file), not a dictionary.
            # The function parse_annotation reads and parses the XML file, extracting the bounding boxes, labels, and difficulties.
            # The line below ensures all coordinates are floats, which is important if the XML parsing returns strings or integers.
            # The lists are then converted to PyTorch tensors.
            # This approach is used when working directly with raw annotation files (e.g., Pascal VOC XMLs) and not with preprocessed JSON data.
        # boxes = []
        # labels = []
        # difficulties = []
        
        # parsed = parse_annotation(annotation_file, 
        #                            label_map=label_map_OD)
        # boxes.append(parsed[1]) #(n_objects, 4)
        # labels.append(parsed[2]) # (n_objects)
        # difficulties.append(parsed[3]) # (n_objects)
        
        # # Convert lists to tensors
        boxes = torch.FloatTensor(boxes)
        labels = torch.LongTensor(labels)
        difficulties = torch.ByteTensor(difficulties)

        # Optionally filter out difficult objects
        if not self.keep_difficult:
            boxes = boxes[1 - difficulties]
            labels = labels[1 - difficulties]
            difficulties = difficulties[1 - difficulties]

        # Apply transforms if provided
        if self.transform:
            image, boxes, labels, difficulties = self.transform(image, boxes, labels, difficulties, split=self.split)

            # # Print a couple of sample indexes for debugging
            # if idx in [0, 3]:
            #     print(f"Index {idx} - Image tensor shape: {image.shape}")
            #     print("Image tensor:", image)
            #     print(f"Index {idx} - Boxes tensor: {boxes}")
            #     print("Boxes tensor:", boxes)
            #     print(f"Index {idx} - Labels tensor: {labels}")
            #     print("Labels tensor:", labels)
            #     print(f"Index {idx} - Difficulties tensor: {difficulties}")
            #     print("Difficulties tensor:", difficulties)

        return image, boxes, labels, difficulties

    def collate_fn(self, batch):
        """
        Custom collate function for batching data samples in a DataLoader.

        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
        """
        images = []
        boxes = []
        labels = []
        difficulties = []

        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            labels.append(b[2])
            difficulties.append(b[3])

        images = torch.stack(images, dim=0) 

        # # Print a couple of sample batches for debugging
        # if len(images) in [1, 4]:
        #     print("Batch Images tensor:", images)
        #     print("Batch Boxes tensor:", boxes)
        #     print("Batch Labels tensor:", labels)
        #     print("Batch Difficulties tensor:", difficulties)
            
        return images, boxes, labels, difficulties  # tensor (N, 3, H, W), 3 lists of N tensors each



class GA_Dataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, split, keep_difficult=False):
        """
        :param data_folder: folder where data files are stored
        :param split: split, one of 'TRAIN', 'TEST', or 'VAL'
        :param keep_difficult: keep or discard objects that are considered difficult to detect?
        """
        self.split = split.upper()

        assert self.split in {'TRAIN', 'TEST', 'VAL'}

        self.data_folder = data_folder
        self.keep_difficult = keep_difficult

        # Read data files
        with open(os.path.join(data_folder, self.split + '_images.json'), 'r') as j:
            self.images = json.load(j)
        with open(os.path.join(data_folder, self.split + '_objects.json'), 'r') as j:
            self.objects = json.load(j)

        assert len(self.images) == len(self.objects)

    def __getitem__(self, i):
        # Read image
        image = Image.open(self.images[i], mode='r')
        image = image.convert('RGB')

        # Read objects in this image (bounding boxes, labels, difficulties)
            # This code assumes that annotation_file is already a dictionary (not a file path), with keys 'boxes', 'labels', and 'difficulties'.
            # Each key directly provides a list of values (e.g., a list of bounding boxes).
            # This is typical when loading preprocessed data from a JSON file, where the annotation information is already structured and ready to use.
            # The conversion to PyTorch tensors is straightforward and efficient, as no further parsing or type conversion is needed.
        objects = self.objects[i]
        boxes = torch.FloatTensor(objects['boxes'])  # (n_objects, 4)
        labels = torch.LongTensor(objects['labels'])  # (n_objects)
        difficulties = torch.ByteTensor(objects['difficulties'])  # (n_objects)

        # Discard difficult objects, if desired
        if not self.keep_difficult:
            boxes = boxes[1 - difficulties]
            labels = labels[1 - difficulties]
            difficulties = difficulties[1 - difficulties]

        # Apply transformations
        image, boxes, labels, difficulties = transform(image, boxes, labels, difficulties, split=self.split)
        # Determine split based on the filename
        

        # image, boxes, labels, difficulties = transform(image, boxes, labels, difficulties, split=self.split)


        return image, boxes, labels, difficulties

    def __len__(self):
        return len(self.images)

    def collate_fn(self, batch):
        """
        Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).

        This describes how to combine these tensors of different sizes. We use lists.

        Note: this need not be defined in this Class, can be standalone.

        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
        """

        images = list()
        boxes = list()
        labels = list()
        difficulties = list()

        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            labels.append(b[2])
            difficulties.append(b[3])

        images = torch.stack(images, dim=0)

        return images, boxes, labels, difficulties  # tensor (N, 3, 300, 300), 3 lists of N tensors each
