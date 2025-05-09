
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

3. extraction_pipeline()
Capacity in the future to make it more modular in terms of image type (.tif vs .jpeg) or annotation type (/xml vs .jason) or even folder names/branch terms to look out for (ROI, VOC, Overlays etc)



4. python debug console
read u aout it... may be able to use... may be able to save lots of time 
<<<<<<< HEAD
>>>>>>> Stashed changes


5. include vs not include images with no objects?
=======
>>>>>>> 7b68930347bc2397d62d611e2dd6818ef7cc3c1b
