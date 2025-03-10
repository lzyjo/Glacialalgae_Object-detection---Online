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


from augmentation import prepare_data_for_augmentation

GA_dataset_path = r'GA_Dataset'
data_dir = r'GA_Dataset\20250221'
date_of_dataset_used = '20250221'
image_dir = r'GA_Dataset\20250221\Images'
annotation_dir = r'GA_Dataset\20250221\Annotations'

transformations = T.RandomHorizontalFlip(p=1.0) # Always flip horizontally

prepare_data_for_augmentation(GA_dataset_path= GA_dataset_path,
                                data_dir= data_dir,
                                date_of_dataset_used= date_of_dataset_used,
                                image_dir= image_dir,
                                annotation_dir= annotation_dir,
                                transformations= transformations)

from augmentation import data_augmentation

augmented_image_dir = r'GA_Dataset\20250221_randomhorizontalflip\Images'
augmented_annotation_dir = r'GA_Dataset\20250221_randomhorizontalflip\Annotations'

data_augmentation(augmented_image_dir= augmented_image_dir,
                augmented_annotation_dir = augmented_annotation_dir,
                transform= transformations)



from augmentation import visual_augmentation_check

original_image_dir = r'GA_Dataset\20250221\Images'
original_annotation_dir = r'GA_Dataset\20250221\Annotations'
augmented_image_dir = r'GA_Dataset\20250221_randomhorizontalflip\Images'
augmented_annotation_dir = r'GA_Dataset\20250221_randomhorizontalflip\Annotations'

visual_augmentation_check(original_annotation_dir= original_annotation_dir,
                        original_image_dir= original_image_dir,
                        augmented_image_dir= augmented_image_dir,
                        augmented_annotation_dir= augmented_annotation_dir,
                        num_pairs= 3)

















# Loop through images and annotations to show bounding boxes on images, stop after 10 images have been shown
image_dir_original = Path(r'GA_Dataset\20250221\Images')
annotation_dir_original = Path(r'GA_Dataset\20250221\Annotations')
image_dir_augmented = Path(r'GA_Dataset\20250221_randomhorizontalflip\Images')
annotation_dir_augmented = Path(r'GA_Dataset\20250221_randomhorizontalflip\Annotations')

image_files_original = sorted(image_dir_original.glob('*.tif'))
annotation_files_original = sorted(annotation_dir_original.glob('*.xml'))
image_files_augmented = sorted(image_dir_augmented.glob('*.tif'))
annotation_files_augmented = sorted(annotation_dir_augmented.glob('*.xml'))

for i, (image_file_original, annotation_file_original, image_file_augmented, annotation_file_augmented) in enumerate(zip(image_files_original, annotation_files_original, image_files_augmented, annotation_files_augmented)):
    if i >= 10:
        break

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
    ax[0].set_title('Original Image')

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
    ax[1].set_title('Augmented Image')

    plt.show()




from torchvision import tv_tensors  # we'll describe this a bit later, bare with us

boxes = tv_tensors.BoundingBoxes(
    [
        [15, 10, 370, 510],
        [275, 340, 510, 510],
        [130, 345, 210, 425]
    ],
    format="XYXY", canvas_size=img.shape[-2:])

transforms = v2.Compose([
    v2.RandomResizedCrop(size=(224, 224), antialias=True),
    v2.RandomPhotometricDistort(p=1),
    v2.RandomHorizontalFlip(p=1),
])
out_img, out_boxes = transforms(img, boxes)
print(type(boxes), type(out_boxes))









# Check the original image and annotation
original_image = r'GA_Dataset\20250221_randomhorizontalflip\Images\1.tif'
original_image = Image.open(original_image, mode='r').convert('RGB')
original_annotation = r'GA_Dataset\20250221_randomhorizontalflip\Annotations\1.xml'
original_boxes = torch.FloatTensor(
    parse_annotation(original_annotation, label_map)['boxes'])

plt.imshow(original_image)
plt.show()

#origibal image
image_path = r'GA_Dataset\20250221_randomhorizontalflip\Images\1.tif'
image = Image.open(image_path, mode='r').convert('RGB') #image
plt.imshow(image)
plt.show()

fig, ax = plt.subplots(1)
ax.imshow(augmented_image)

annotation_path = r'GA_Dataset\20250221_randomhorizontalflip\Annotations\1.xml'
parsed_annotations = parse_annotation(annotation_path, label_map)
# Plot the bounding boxes
for bbox in augmented_boxes:
    xmin, ymin, xmax, ymax = bbox
    width = xmax - xmin
    height = ymax - ymin
    rect = plt.Rectangle((xmin, ymin), width, height, edgecolor='r', facecolor='none')
    ax.add_patch(rect)

plt.show()





from augmentation import data_augmentation

image_path = r'GA_Dataset\20250221_randomhorizontalflip\Images\1.tif'
image = Image.open(image_path, mode='r').convert('RGB') #image
plt.imshow(image)
plt.show()
image = tv_tensors.Image(image) #tensor
augmented_image = transformations(image) #tensor
augmented_image = T.ToPILImage()(augmented_image) #image
plt.imshow(augmented_image)
plt.show()
print(f"Canvas size: {augmented_image.size[1]}, {augmented_image.size[0]}")


annotation_path = r'GA_Dataset\20250221_randomhorizontalflip\Annotations\1.xml'
parsed_annotations = parse_annotation(annotation_path, label_map)
# Read objects in this image (bounding boxes, labels, difficulties) from parsed_annotations
boxes = torch.FloatTensor(parsed_annotations['boxes'])  # (n_objects, 4)
boxes = tv_tensors.BoundingBoxes(boxes, 
                                format='XYXY',
                                canvas_size=(augmented_image.size[1], 
                                             augmented_image.size[0]))

augmented_boxes = transformations(boxes) #augmented_boxes = [apply_transform_to_bbox(bbox, transform, image_size) for bbox in boxes]

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

parsed_annotations = parse_annotation(annotation_path, label_map)
checking_boxes = torch.FloatTensor(parsed_annotations['boxes'])  # (n_objects, 4)

#augmented_image
plt.imshow(augmented_image)
plt.show()

fig, ax = plt.subplots(1)
ax.imshow(augmented_image)

# Plot the bounding boxes
for bbox in augmented_boxes:
    xmin, ymin, xmax, ymax = bbox
    width = xmax - xmin
    height = ymax - ymin
    rect = plt.Rectangle((xmin, ymin), width, height, edgecolor='r', facecolor='none')
    ax.add_patch(rect)

plt.show()







