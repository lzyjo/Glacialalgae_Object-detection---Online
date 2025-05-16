import torch
from torch.utils.data import Dataset
import json
import os
from PIL import Image
from utils import transform
import xml.etree.ElementTree as ET
from utils import parse_annotation


"""
“PyTorch Dataset is all about. It is an object that is required to implement two methods:
 __len__ and __getitem__. The former should return the number of items in the dataset; 
 the latter should return the item, consisting of a sample and its corresponding label (an integer index).2” 
 ([Stevens et al., 2020, p. 166])
"""




class PC_Dataset(Dataset):
    """
    Phase contrast dataset.

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
    """

    def __init__(self, 
                 dataset_folder, 
                 root_dir, transform=None): #transforms needed if samples are not of the same size
                                        # however, in this case, all images are of the same size
                                        # something to think about in terms of usability for other datasets
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        
        for split in ['train', 'test', 'val']: 
            annotation_folder = os.path.join(dataset_folder, split, "annotations")
            image_folder = os.path.join(dataset_folder, split, "images")
            
            if os.path.isdir(annotation_folder) and os.path.isdir(image_folder):
                self.datasets[split] = {
                    "annotations": annotation_folder,
                    "images": image_folder
                }
            else:
                raise FileNotFoundError(f"Missing directory: {annotation_folder} or {image_folder}")

        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        # Return the number of images in the training set by default
        # You may want to change 'train' to another split if needed
        return len(os.listdir(self.datasets['train']['images']))
    
    
    def __getitem__(self, idx,
                    image_folder, annotations_folder):
        """
        Retrieve an image and its corresponding annotation data by index.
                Args:
                    idx (int or torch.Tensor): Index of the item to retrieve. If a torch.Tensor is provided, it will be converted to a list or integer.
                    image_folder (str): Path to the folder containing image files.
                    annotations_folder (str): Path to the folder containing annotation files.
                Returns:
                    tuple: A tuple containing:
                        - image (PIL.Image.Image): The loaded image in RGB format.
                        - boxes (np.ndarray or torch.Tensor): Bounding box coordinates for objects in the image.
                        - labels (np.ndarray or torch.Tensor): Class labels for each object.
                        - difficulties (np.ndarray or torch.Tensor): Difficulty flags for each object.
                Raises:
                    FileNotFoundError: If the image or annotation file for the given index does not exist.
                Notes:
                    - If the attribute 'keep_difficult' is set to False, objects marked as difficult will be filtered out.
                    - If a transform is specified, it will be applied to the image and annotation data before returning.
                    - The function expects the image and annotation files to be sorted and aligned by filename.
                    - No iteration/for loop is used in this function; it retrieves a single item based on the provided index. 
                        This is because the function is designed to work with PyTorch's DataLoader,
                        which handles batching and iteration separately.
        """

        
        if torch.is_tensor(idx): # convert to list 
            idx = idx.tolist() 

        """
        When using PyTorch’s DataLoader, the indices passed to __getitem__ 
        can sometimes be PyTorch tensors instead of plain integers. This 
        often happens when you use certain DataLoader settings or batch samplers. 
        For example, if you use advanced indexing or certain transforms, 
        PyTorch may wrap the index in a tensor.
        """

        # Get image and annotation file paths for the given idx
        image_files = sorted(os.listdir(image_folder))
        annotation_files = sorted(os.listdir(annotations_folder))

        image_file = os.path.join(image_folder, image_files[idx])
        annotation_file = os.path.join(annotations_folder, annotation_files[idx])

        if not os.path.isfile(image_file):
            raise FileNotFoundError(f"Missing image file: {image_file}")
        if not os.path.isfile(annotation_file):
            raise FileNotFoundError(f"Missing annotation file: {annotation_file}")

        # Load image
        image = Image.open(image_file).convert('RGB')

        # Parse annotation
        boxes, labels, difficulties = parse_annotation(annotation_file)

        # Discard difficult objects, if desired
        if hasattr(self, 'keep_difficult') and not self.keep_difficult:
            boxes = boxes[1 - difficulties]
            labels = labels[1 - difficulties]
            difficulties = difficulties[1 - difficulties]


        # Apply transformations
        if self.transform:
            image, boxes, labels, difficulties = self.transform(image, boxes, labels, difficulties, split=self.split)

        return image, boxes, labels, difficulties

    # Pytorch dataloader requires a collate function to handle batches of data
    # Pytorch has a default collate function, but it may not work well with variable-length sequences
    # Therefore, we need to define a custom collate function if the default one does not work:
    # def collate_fn(self, batch):
    # default_collate() function did not work with the dataset: 


    def collate_fn(self, batch):
        """
        In Object detection, each image has a different number of objects, unlike in classification 
        (where each image has the same number of classes). Since each image may have a different number of objects,
        we need a collate function (to be passed to the DataLoader).
        The default PyTorch collate function tries to stack everything into tensors, which fails if the data has 
        variable lengths. This custom function solves that problem.

        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
        """

        images = list()
        boxes = list()
        labels = list()
        difficulties = list()

        for b in batch: # b is a tuple of (image, boxes, labels, difficulties)
            # b[0] is the image, b[1] is the boxes, b[2] is the labels, b[3] is the difficulties
            images.append(b[0])
            boxes.append(b[1])
            labels.append(b[2])
            difficulties.append(b[3])

        images = torch.stack(images, dim=0)

        return images, boxes, labels, difficulties  # tensor (N, 3, 300, 300), 3 lists of N tensors each







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
