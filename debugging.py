import os
import shutil
from datetime import datetime



source_folders = [
    r'1_GA_Dataset\20250513\Split',  # Original dataset, not augmented
    r'2_DataAugmentation\20250513'  # Augmented datasets
]





annotations_src_folder = r'1_GA_Dataset\20250513\Split\test\Annotations'
images_src_folder = r'1_GA_Dataset\20250513\Split\test\Images'
annotations_folder = r'3_TrainingData\20250513_Augmented\Split\test\annotations' # Change this to the correct folder for which files are to be extracted to
images_folder = r'3_TrainingData\20250513_Augmented\Split\test\images' # Change this to the correct folder for which files are to be extracted to


# Extract .xml files to annotations folder
num_annotations_start = 0
xml_files = [f for f in os.listdir(annotations_folder) if f.endswith('.xml')]
num_annotations_start += len(xml_files)
print(f"Total annotations from {annotations_folder}: {num_annotations_start}")

counter = len([f for f in os.listdir(annotations_folder) if f.endswith('.xml')]) + 1
if isinstance(annotations_src_folder, str):
    annotations_src_folders = [annotations_src_folder]
else:
    annotations_src_folders = annotations_src_folder

for folder in annotations_src_folders:
    for file in os.listdir(folder):
        if file.endswith('.xml'):
            src_file = os.path.join(folder, file)
            dst_file = os.path.join(annotations_folder, f"{counter}.xml")
            if os.path.exists(dst_file):
                print(f"Error: File {dst_file} already exists. / "
                        f"Source file: {src_file} / "
                        f"Destination file: {dst_file}")
                raise RuntimeError(f"File {dst_file} already exists. Stopping execution.")
            shutil.copy(src_file, dst_file)
        counter += 1






    # Create destination folders if they do not exist
    if not os.path.exists(annotations_folder):
        os.makedirs(annotations_folder)
    else:
        print(f"Folder {annotations_folder} already exists.")
        

    if not os.path.exists(images_folder):
        os.makedirs(images_folder)
    else:
        print(f"Folder {images_folder} already exists.")

    print("-" * 20)  # Add a separator line for better readability

    # Extract .xml files to annotations folder
    num_annotations_start = 0
    xml_files = [f for f in os.listdir(annotations_folder) if f.endswith('.xml')]
    num_annotations_start += len(xml_files)
    print(f"Total annotations from {annotations_folder}: {num_annotations_start}")

    counter = len([f for f in os.listdir(annotations_folder) if f.endswith('.xml')]) + 1
    for folder in annotations_src_folder:
        for file in os.listdir(folder):
            if file.endswith('.xml'):
               src_file = os.path.join(folder, file)
               dst_file = os.path.join(annotations_folder, f"{counter}.xml")
               if os.path.exists(dst_file):
                   print(f"Error: File {dst_file} already exists. / "
                         f"Source file: {src_file} / "
                         f"Destination file: {dst_file}")
                   return
               shutil.copy(src_file, dst_file)
            counter += 1

    # Count the number of annotations at the end
    num_annotations_end = len([f for f in os.listdir(annotations_folder) if f.endswith('.xml')])
    
    # Calculate the number of annotations moved
    num_annotations_moved = num_annotations_end - num_annotations_start

    # Verify if the counts add up
    if num_annotations_moved == (counter -1):  
        print(f"Files extracted from {date_of_dataset_used} to {annotations_folder}/ "
              f"Start: {num_annotations_start}\n"
              f"Moved: {num_annotations_moved}\n"
              f"End: {num_annotations_end}\n")
    else:
        print(f"Error: Number of annotations in {annotations_folder} does not match the expected count.\n"
              f"Start: {num_annotations_start}\n"
              f"Moved: {num_annotations_moved}\n"
              f"End: {num_annotations_end}\n"
              f"Expected Moved: {counter - 1}")

    print("-" * 20)  # Add a separator line for better readability

    # Extract .tif files to images folder
    num_images_start = 0
    tif_files = [f for f in os.listdir(images_folder) if f.endswith('.tif')]
    num_images_start += len(tif_files)
    print(f"Total images from {images_folder}: {num_images_start}")


    counter = len([f for f in os.listdir(images_folder) if f.endswith('.tif')]) + 1
    for folder in images_src_folder:
        for file in os.listdir(folder):
            if file.endswith('.tif'):
               src_file = os.path.join(folder, file)
               dst_file = os.path.join(images_folder, f"{counter}.tif")
               if os.path.exists(dst_file):
                   print(f"Error: File {dst_file} already exists. / "
                         f"Source file: {src_file} / "
                         f"Destination file: {dst_file}")
                   return
               shutil.copy(src_file, dst_file)
            counter += 1
        
    # Count the number of images at the end
    num_images_end = len([f for f in os.listdir(images_folder) if f.endswith('.tif')])
    
    # Calculate the number of images moved
    num_images_moved = num_images_end - num_images_start

    # Verify if the counts add up
    if num_images_end == (counter - 1):
        print(f"Files extracted from {date_of_dataset_used} to {images_folder}/ ")
        print(f"Start: {num_images_start}")
        print(f"Moved: {num_images_moved}")
        print(f"End: {num_images_end}")
    else:
        print(f"Error: Number of images in {images_folder} does not match the expected count. ")
        print(f"Start: {num_images_start}")
        print(f"Moved: {num_images_moved}")
        print(f"End: {num_images_end}")
        print(f"Expected Moved: {counter - 1}")
    
    print("-" * 50)  # Add a separator line for better readability


    if len(annotations_folder) == len(images_folder):
        print("The number of annotation files and image files are the same.")
        print("Number of annotation files:", len(annotations_folder))
        print("Number of image files:", len(images_folder)) 

    def check_file_matching(annotations_folder, images_folder, annotations_src_folder, images_src_folder):
        """
        Check if the files in the annotations and images folders have matching file names (different extensions).
        If not, identify unmatched files in both folders and their source folders.

        Args:
            annotations_folder (str): Path to the folder containing annotation (.xml) files.
            images_folder (str): Path to the folder containing image (.tif) files.
            annotations_src_folder (str): Path to the source folder containing annotation (.xml) files.
            images_src_folder (str): Path to the source folder containing image (.tif) files.

        Returns:
            None
        """
        # Check if the files in these folders have the same file name (just different extension)
        files_in_annotations_folder = [f for f in os.listdir(annotations_folder) if f.endswith('.xml')]
        files_in_images_folder = [f for f in os.listdir(images_folder) if f.endswith('.tif')]
        files_in_annotations_folder_no_ext = [os.path.splitext(f)[0] for f in files_in_annotations_folder]
        files_in_images_folders_no_ext = [os.path.splitext(f)[0] for f in files_in_images_folder]

        if set(files_in_annotations_folder_no_ext) == set(files_in_images_folders_no_ext):
            print("The files in the annotations and images folders have the same file names (just different extensions).")
            print("-" * 50)  # Add a separator line for better readability
        else:
            unmatched_annotations = set(files_in_annotations_folder_no_ext) - set(files_in_images_folders_no_ext)
            unmatched_images = set(files_in_images_folders_no_ext) - set(files_in_annotations_folder_no_ext)

            print("The files in the annotations and images folders do not match.")
            print("Unmatched annotation files:", unmatched_annotations)
            print("Unmatched image files:", unmatched_images)

            # Get the list of .xml files in the annotations folder including subfolders
            xml_files = []
            for root, dirs, files in os.walk(annotations_src_folder):
                for file in files:
                    if file.endswith('.xml'):
                        xml_files.append(file)
            xml_files_no_ext = [os.path.splitext(file)[0] for file in xml_files]

            # Get the list of .tif files in the images folder including subfolders
            tif_files = []
            for root, dirs, files in os.walk(images_src_folder):
                for file in files:
                    if file.endswith('.tif'):
                        tif_files.append(file)
            tif_files_no_ext = [os.path.splitext(file)[0] for file in tif_files]

            # Find unmatched files
            unmatched_annotations = set(xml_files_no_ext) - set(tif_files_no_ext)
            unmatched_images = set(tif_files_no_ext) - set(xml_files_no_ext)

            # Print the results
            print(f"The files in the {annotations_src_folder} and {images_src_folder} do not match.")
            print("Unmatched annotation files:", unmatched_annotations)
            print("Unmatched image files:", unmatched_images)
            print("-" * 50)  # Add a separator line for better readability)

    check_file_matching(annotations_folder, images_folder, annotations_src_folder, images_src_folder)
        
if __name__ == '__main__':
    extract_files(date_of_dataset_used=date_of_dataset_used,  # Change this to the correct dataset used, FOR REFERENCE ONLY
                    annotations_folder=r'GA_Dataset\20250219\Annotations',  # Change this to the correct folder for which files were extracted 
                    images_folder=r'GA_Dataset\20250219\Images',  # Change this to the correct folder for which files were extracted 
                    images_src_folder=r'Completed annotations/Bluff_230724/Original_Images_Unlabelled_Bluff_230724',
                    annotations_src_folder=r'Completed annotations\Bluff_230724')  # Change this to your source folder path 
                    