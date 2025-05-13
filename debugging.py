import os
import subprocess












def run_training_process(data_folder, 
                         date_of_dataset_used, 
                         training_output_file, 
                         save_dir=r'6_Checkpoints'):
    """
    Run the training process and save the output to a file.

    Args:
        data_folder (str): Path to the data folder.
        date_of_dataset_used (str): Date of the dataset used for training.
        training_output_file (str): Path to the file where training output will be saved.
        save_dir (str): Directory to save checkpoints.
    """
    with open(training_output_file, 'a') as f:
        try:
            result = subprocess.run(['python', 'train.py', 
                        '--data_folder', data_folder,
                        '--date_of_dataset_used', date_of_dataset_used,
                        '--object_detector', 'yes',
                        '--save_dir', save_dir],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True)
            if result.stdout:
                f.write(result.stdout)
            if result.stderr:
                f.write("Standard Error:\n")
                f.write(result.stderr)
        except subprocess.CalledProcessError as e:
            print(f"Error occurred during training: {e}")
            f.write(f"Error occurred during training: {e}\n")
            if e.stdout:
                f.write("Standard Output:\n")
                f.write(e.stdout + '\n')
            if e.stderr:
                f.write("Standard Error:\n")
                f.write(e.stderr + '\n')

if __name__ == "__main__":
    # Run the training process and save the output
    run_training_process(data_folder=data_folder,
                         date_of_dataset_used=date_of_dataset_used,
                         training_output_file=training_output_file,
                         save_dir=r'6_Checkpoints')
    


# TRAIN MODEL: Run the training process and save the output
data_folder = r'4_JSON_folder\20250318_Augmented'
date_of_dataset_used = '20250318'  # Date of dataset used for training
training_output_file = r'5_Results\training_results_20250318_Augmented.txt'

run_training_process(data_folder=data_folder,
                        date_of_dataset_used=date_of_dataset_used,
                        training_output_file=training_output_file,
                        save_dir=r'6_Checkpoints')


# TRAIN MODEL: Run the training process and save the output
data_folder = r'4_JSON_folder\20250318_Augmented'
date_of_dataset_used = '20250318'  # Date of dataset used for training
training_output_file = r'5_Results\training_results_20250318_Augmented.txt'
with open(training_output_file, 'a') as f:
    try:
        result = subprocess.run(['python', 'train.py', 
                    '--data_folder', data_folder,
                    '--date_of_dataset_used', date_of_dataset_used,
                    '--object_detector', 'yes',
                    '--save_dir', r'6_Checkpoints'],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE)
        if result.stdout:
            f.write(result.stdout)
        if result.stderr:
            f.write("Standard Error:\n")
            f.write(result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred during training: {e}")
        f.write(f"Error occurred during training: {e}\n")
        if e.stdout:
            f.write("Standard Output:\n")
            f.write(e.stdout + '\n')
        if e.stderr:
            f.write("Standard Error:\n")
            f.write(e.stderr + '\n')    