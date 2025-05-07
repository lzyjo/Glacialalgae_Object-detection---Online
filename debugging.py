    print("File extraction completed.")
    # Print a list of all the date_of_dataset_used at the end
    print("List of all augmentations used:")
    augmentations_used = [dataset["date_of_dataset_used"] for dataset in datasets]
    print(augmentations_used)

    # Save the list to a .txt file in the training_data_folder
    output_file = os.path.join(training_data_folder, "augmentations_used.txt")
    os.makedirs(training_data_folder, exist_ok=True)
    with open(output_file, "w") as f:
        for augmentation in augmentations_used:
            f.write(augmentation + "\n")
    print(f"List of augmentations saved to {output_file}")


