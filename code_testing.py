import os




# Define the paths
annotations_folder = r'Completed annotations/Bluff_230724'
images_folder = r'Completed annotations/Bluff_230724/Original_Images_Unlabelled_Bluff_230724'

# Get the list of .xml files in the annotations folder including subfolders
xml_files = []
for root, dirs, files in os.walk(annotations_folder):
    for file in files:
        if file.endswith('.xml'):
            xml_files.append(file)
xml_files_no_ext = [os.path.splitext(file)[0] for file in xml_files]

# Get the list of .tif files in the images folder including subfolders
tif_files = []
for root, dirs, files in os.walk(images_folder):
    for file in files:
        if file.endswith('.tif'):
            tif_files.append(file)
tif_files_no_ext = [os.path.splitext(file)[0] for file in tif_files]

# Find unmatched files
unmatched_annotations = set(xml_files_no_ext) - set(tif_files_no_ext)
unmatched_images = set(tif_files_no_ext) - set(xml_files_no_ext)

# Print the results
print("The files in the annotations and images folders do not match.")
print("Unmatched annotation files:", unmatched_annotations)
print("Unmatched image files:", unmatched_images)
