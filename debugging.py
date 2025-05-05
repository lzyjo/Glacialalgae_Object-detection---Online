
import shutil
import os
annotations_src_folder =r'1_GA_Dataset\20250318\Split\train\annotations'
annotations_folder = r'3_TrainingData\20250318_Augmented\Split\train\annotations' # Change this to the correct folder for which files are to be extracted to


# Extract .xml files to annotations folder
src_file_number = len([f for f in os.listdir(annotations_src_folder) if f.endswith('.xml')])
counter = len([f for f in os.listdir(annotations_folder) if f.endswith('.xml')]) + 1


for root, dirs, files in os.walk(annotations_src_folder):
for file in files:
if file.endswith('.xml'):
    dst_file = os.path.join(annotations_folder, f"{counter}.xml") 
    if os.path.exists(dst_file):
        print(f"Error: File {dst_file} already exists. / "
                f"Source file: {file} / "
                f"Destination file: {dst_file}")
        return
    shutil.copy(os.path.join(root, file), dst_file)
    counter += 1