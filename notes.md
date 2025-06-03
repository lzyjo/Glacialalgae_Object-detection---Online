
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

8. class PC_Dataset(Dataset) 
'transforms' argument currently not tested or modified.. as current dataset does not need any data
pre-processing before using in the dataloader
however might be a consideration when distributing OD/OC model as an open-source tool? 
TBC.. ! (https://docs.pytorch.org/tutorials/beginner/data_loading_tutorial.html)

9. Extraction pipeline problems:
- when printing datasets, Do you want to proceed with extracting files? (yes/no): has to be asked twice, because if only 1, then it returns a blank space as an answer automatically for some reason
- repeated datasets returned in terminal (seems to be 'surface' level, it is returned but dataset is nmot extracted twice even if shown twice for example)

10. datalaoder creation location
currently in train_custom.py... but should consider doing it in runfile? where is the best place to put it?

11. importing custom collate_fn()
e 'collate_fn'  does not need to be imported (ImportError: cannot import name 'collate_fn' from 'dataset' (E:\JOEY\Glacialalgae_Object-detection---Server\dataset.py)) but why?

12. Separation for main train function and training set up functions (including dataloaders and hyperparams)

13. why do files never save unless i literally click ctrl + s? crt +s simple shifts whatever changes to the change list (not staged)

TO-DO-LIST

- [ ] Include integration with TensorBoard for  visualization and monitoring and model progress tracking.
- [ ] Include optmiser and loss criterion in model training.txt file 
