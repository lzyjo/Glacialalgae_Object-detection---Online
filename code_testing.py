import os
import subprocess


training_output_file = r'Results\training_results_20250221.txt'


def manage_training_output_file(results_folder, date_of_dataset_used, checkpoint_frequency, lr, iterations):
    """
    This function manages the training output file by creating or appending to it, and writes the training parameters at the top of the file if they do not already exist.

    Args:
        results_folder (str): The folder where the results will be stored.
        date_of_dataset_used (str): The date of the dataset used for training.
        checkpoint_frequency (int): The frequency of checkpoints during training.
        lr (float): The learning rate for training.
        iterations (int): The number of iterations for training.

    Returns:
        str: The path to the training output file.
    """

    training_output_file = os.path.join(results_folder, 
                                        f'training_results_{date_of_dataset_used}.txt')
    
    if os.path.exists(training_output_file):
        with open(training_output_file, 'r') as read_file:
            content = read_file.read()
        mode = 'a'  # Append mode
    else:
        os.makedirs(results_folder, exist_ok=True)  # Ensure the results folder exists
        content = ''
        mode = 'w'  # Write mode
        
    with open(training_output_file, mode) as f:
        # Write the training parameters at the top of the file if they do not already exist
        if mode == 'a':
            if f'Checkpoint Frequency: {checkpoint_frequency}' not in content:
                f.write(f'Checkpoint Frequency: {checkpoint_frequency}\n')
            if f'Date of Dataset Used: {date_of_dataset_used}' not in content:
                f.write(f'Date of Dataset Used: {date_of_dataset_used}\n')
            if f'Learning Rate: {lr}' not in content:
                f.write(f'Learning Rate: {lr}\n')
            if f'Iterations: {iterations}' not in content:
                f.write(f'Iterations: {iterations}\n\n')
            
        if mode == 'w':    
            f.write(f'Checkpoint Frequency: {checkpoint_frequency}\n')
            f.write(f'Date of Dataset Used: {date_of_dataset_used}\n')
            f.write(f'Learning Rate: {lr}\n')

# Define the parameters for the manage_training_output_file function
results_folder = 'Results'
date_of_dataset_used = '20250221'
checkpoint_frequency = 5
lr = 0.001
iterations = 10000

# Get the training output file path
training_output_file = manage_training_output_file(results_folder, date_of_dataset_used, checkpoint_frequency, lr, iterations)

# Define additional parameters for the training process
data_folder = 'data'
checkpoint = 'checkpoint.pth'

# Run the training process and save the output
with open(training_output_file, 'a') as f:
    process = subprocess.run(['python', 'train.py', 
            '--data_folder', data_folder,
            '--date_of_dataset_used', date_of_dataset_used,
            '--save_dir', r'Checkpoints',
            '--checkpoint', checkpoint,
            '--checkpoint_frequency', checkpoint_frequency,
            '--lr', lr,
            '--iterations', iterations], stdout=subprocess.PIPE, text=True)
    
    for line in process.stdout.splitlines():
        if line.startswith('Epoch:'):
            f.write(line + '\n')


# Return the relative file path of the training output file
print(f"Training output file saved at: {os.path.relpath(training_output_file)}")

