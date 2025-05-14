import os
import shutil
import json
import xml.etree.ElementTree as ET


def convert_labels_to_cell(annotations_folder):
    """
    Updates the labels in XML annotation files within a specified folder.
    This function iterates through all XML files in the given folder, and for each
    file, it parses the XML structure to find all object elements. If the label
    ('name' tag) of an object is not 'UNKNOWN', it updates the label to 'cell'.
    The modified XML file is then saved back to its original location.
    Args:
        annotations_folder (str): The path to the folder containing XML annotation files.
    Returns:
        None
    """
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
    
    print(f"Labels in {annotations_folder} have been updated to 'cell' for all objects except 'UNKNOWN'.")

if __name__ == '__main__':
    annotations_folder = r'GA_Dataset\20250219\Annotations'
    convert_labels_to_cell(annotations_folder)


def remove_unknowns_from_labels(annotations_folder):
    """
    Removes objects with the label 'UNKNOWN' from XML annotation files in the specified folder.
    This function iterates through all XML files in the given folder, parses each file,
    and removes any <object> elements where the <name> sub-element has the text 'UNKNOWN'.
    The modified XML files are then saved back to their original locations.
    Args:
        annotations_folder (str): The path to the folder containing XML annotation files.
    Returns:
        None
    """
    for filename in os.listdir(annotations_folder):
        if filename.endswith('.xml'):
            file_path = os.path.join(annotations_folder, filename)
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            for obj in root.findall('object'):
                label = obj.find('name').text
                if label == 'UNKNOWN':
                    root.remove(obj)
                
            tree.write(file_path)
    
    print(f"Removed 'UNKNOWN' labels from {annotations_folder}.")

if __name__ == '__main__':
    annotations_folder = r'GA_Dataset\20250219\Annotations'
    remove_unknowns_from_labels(annotations_folder)

def object_count_for_cells(annotations_folder):
    count = 0
    for filename in os.listdir(annotations_folder):
        if filename.endswith('.xml'):
            file_path = os.path.join(annotations_folder, filename)
            tree = ET.parse(file_path)
            root = tree.getroot()
            for obj in root.findall('object'):
                name = obj.find('name').text
                if name != 'UNKNOWN':
                    count += 1
    return count

if __name__ == '__main__':
    annotations_folder = r'Training_GA_Dataset\20250221\Annotations' # Change this to the correct folder for which files are to be extracted to
    number_of_objects = object_count_for_cells(annotations_folder)
    print(f"Number of objects that are cells: {number_of_objects}")