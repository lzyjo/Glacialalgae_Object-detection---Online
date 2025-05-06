import json
import os
import pandas as pd

label_classes_path = os.path.abspath(r"label_classes.csv") # Load label classes from CSV
label_classes_df = pd.read_csv(label_classes_path) # read csv
label_classes_Classifier = tuple(label_classes_df.iloc[:, 0].tolist())  # Derive labels from the first column of the CSV
label_map_Classifier = {k: v + 1 for v, k in enumerate(label_classes_Classifier)}
label_map_Classifier['background'] = 0  # Background is the first class
rev_label_map_Classifier = {v: k for k, v in label_map_Classifier.items()}  # Inverse mapping

label_classes_OD = ['cell']  # Label classes
label_map_OD = {k: v + 1 for v, k in enumerate(label_classes_OD)}
label_map_OD['background'] = 0  # Background is the first class
rev_label_map_OD = {v: k for k, v in label_map_OD.items()}  # Inverse mapping


# Color map for bounding boxes of detected objects from https://sashat.me/2017/01/11/list-of-20-simple-distinct-colors/
#distinct_colors = ['#e6194b', '#3cb44b', '#ffe119', '#0082c8']
# label_color_map = {k: distinct_colors[i] for i, k in enumerate(label_map.keys())}
