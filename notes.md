
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
currently 3 sections of code for train test spit extraction.. want to find a way in future commit/funciton creation or clean up, to make this less convoluted and more simple/modular 



4. python debug console
read u aout it... may be able to use... may be able to save lots of time 

5. FasterRCNN class
in order to train amodel with the faster rcnn architecture (not-pretrained), am i able to simply do so by using model = fasterrcnn_resnet50_fpn(pretrained=False, num_classes=2) as compared to  model = fasterrcnn_resnet50_fpn(pretrained=True, num_classes=2)? 
Or do i have to create a model.py file essentally, and use class FasterRCNN to define the layers (among others), and if so, how do I go about doing that? It was very confusing. 
Most of the code in current model.py file (FasterRCNN class etc) is from pytorch docs

6. to use tensorboard from torch.utils, you may NOT have a .py file called tensorboard.py or the file
will try to import from there causing problems...

7. now need to work on custom dataloaders and datasets


TO-DO-LIST

- [ ] Include integration with TensorBoard for  visualization and monitoring and model progress tracking.
- [ ] Include optmiser and loss criterion in model training.txt file 
