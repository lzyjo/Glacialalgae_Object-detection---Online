import os
import xml.etree.ElementTree as ET

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