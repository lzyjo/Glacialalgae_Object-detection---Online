"""
Test if create_data_lists(train_annotation_path, train_image_path, test_annotation_path, test_image_path, output_folder) function is working
Problem found in parse_annotations 
"""



import os      
import json
import pandas as pd
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


from BCCD_utils import parse_annotation


label_classes_path = os.path.abspath(r"label_classes.csv") # Load label classes from CSV
label_classes_df = pd.read_csv(label_classes_path) # read csv
label_classes = tuple(label_classes_df.iloc[:, 0].tolist())  # Derive labels from the first column of the CSV
label_map = {k: v + 1 for v, k in enumerate(label_classes)}
rev_label_map = {v: k for k, v in label_map.items()}  # Inverse mapping


image_path= r'GA_Dataset/Split/train/images'
image_extension='.tif'
annotation_path=r'GA_Dataset/Split/train/annotations'


train_images = list()
train_objects = list()
n_objects = 0


image_files = [file for file in os.listdir(image_path) if file.endswith(image_extension)]
ids = [os.path.splitext(file)[0] for file in image_files]

print(f"Found {len(ids)} images.")

annotation_file = os.path.join(annotation_path, '1' + '.xml')

if not os.path.isfile(annotation_file):
    print(f"Annotation file {annotation_file} does not exist, skipping.")


objects = parse_annotation(annotation_file) # problem here 

if len(objects['boxes']) == 0:
    print(f"No objects found in {annotation_file}, skipping.")

n_objects += len(objects['boxes'])
objects_list.append(objects)
images_list.append(os.path.join(image_path, id + image_extension))

print(f"Processed {annotation_file}, found {len(objects['boxes'])} objects.")


assert len(objects_list) == len(images_list)
# return objects_list, images_list, n_object



tree = ET.parse(annotation_file)
root = tree.getroot()

boxes = list()
labels = list()
difficulties = list()

for obj in root.iter('object'):

    difficult = int(obj.find('difficult').text == '1')

    label = obj.find('name').text.strip()
    if label not in label_map:
        print (f"Label '{label}' not present in label_map") # when running this length of code, this line is not printed.. 
                                                            # but when running as pase annotation function() it is returned?
        continue

    bbox = obj.find('bndbox')
    xmin = int(bbox.find('xmin').text) - 1
    ymin = int(bbox.find('ymin').text) - 1
    xmax = int(bbox.find('xmax').text) - 1
    ymax = int(bbox.find('ymax').text) - 1

    boxes.append([xmin, ymin, xmax, ymax])
    labels.append(label_map[label])
    difficulties.append(difficult)

print({'boxes': boxes, 'labels': labels, 'difficulties': difficulties})

