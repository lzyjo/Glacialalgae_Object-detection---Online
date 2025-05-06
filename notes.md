
06/05/2025

1. dataset.py and GA_Dataset Class
In my attempt to create custom code, for this I didn't cahnge much, if not anything, from the original code as this seems to work exactly as i want it to.... 
considerations: 
- code piracy
- using class for data_augmentation instead of a whole pipeline for it?

2. train_test_split() skicit-learn
Works with arrays: 
*arrays
sequence of indexables with same length / shape[0]
Allowed inputs are lists, numpy arrays, scipy-sparse matrices or pandas dataframes

So technically could directly split images (list) and annotations (numpy arrays) if formatted as such? 
Is it beneficial to do so? Or keep as is, is also fine and efficient? 