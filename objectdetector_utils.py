import os
import shutil
import json
import xml.etree.ElementTree as ET


def convert_labels_to_cell(annotations_folder):
    for filename in os.listdir(annotations_folder):
        if filename.endswith('.xml'):
            file_path = os.path.join(annotations_folder, filename)
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            for obj in root.findall('object'):
                label = obj.find('name').text
                if label != 'UNKNOWN':
                    obj.find('name').text = 'cell'
            
            tree.write(file_path)

if __name__ == '__main__':
    annotations_folder = r'GA_Dataset\20250219\Annotations'
    convert_labels_to_cell(annotations_folder)