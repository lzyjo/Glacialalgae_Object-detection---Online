


        process_annotations(annotation_path=test_annotation_path, 
                            image_path=test_image_path, 
                            image_extension='.tif', 
                            objects_list=test_objects, 
                            images_list=test_images, 
                            n_objects=n_objects)

    create_data_lists(train_annotation_path=r'GA_Dataset/Split/train/annotations',
                      train_image_path=r'GA_Dataset/Split/train/images',
                      test_annotation_path=r'GA_Dataset/Split/test/annotations',
                      test_image_path=r'GA_Dataset/Split/test/images',
                      output_folder=r'GA_Dataset/Output')

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

objects = parse_annotation(annotation_file)

if len(objects['boxes']) == 0:
    print(f"No objects found in {annotation_file}, skipping.")

n_objects += len(objects['boxes'])
objects_list.append(objects)
images_list.append(os.path.join(image_path, id + image_extension))

print(f"Processed {annotation_file}, found {len(objects['boxes'])} objects.")


for id in ids:
    annotation_file = os.path.join(annotation_path, id + '.xml')

    if not os.path.isfile(annotation_file):
        print(f"Annotation file {annotation_file} does not exist, skipping.")
        continue

    objects = parse_annotation(annotation_file)
    if len(objects['boxes']) == 0:
        print(f"No objects found in {annotation_file}, skipping.")
        continue

    n_objects += len(objects['boxes'])
    objects_list.append(objects)
    images_list.append(os.path.join(image_path, id + image_extension))

    print(f"Processed {annotation_file}, found {len(objects['boxes'])} objects.")

assert len(objects_list) == len(images_list)
return objects_list, images_list, n_object