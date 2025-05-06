import json
import os
import pandas as pd

label_classes_path = os.path.abspath(r"label_classes.csv") # Load label classes from CSV
label_classes_df = pd.read_csv(label_classes_path) # read csv
label_classes_Classifier = tuple(label_classes_df.iloc[:, 0].tolist())  # Derive labels from the first column of the CSV
label_map = {k: v + 1 for v, k in enumerate(label_classes_Classifier)}
label_map['background'] = 0  # Background is the first class
rev_label_map = {v: k for k, v in label_map.items()}  # Inverse mapping

label_classes_OD = ['cell']  # Label classes
label_map_OD = {k: v + 1 for v, k in enumerate(label_classes_OD)}
label_map_OD['background'] = 0  # Background is the first class
rev_label_map_OD = {v: k for k, v in label_map_OD.items()}  # Inverse mapping