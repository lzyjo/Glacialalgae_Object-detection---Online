import json
import os
import pandas as pd

label_classes_path = os.path.abspath(r"label_classes.csv") # Load label classes from CSV
label_classes_df = pd.read_csv(label_classes_path) # read csv
label_classes = tuple(label_classes_df.iloc[:, 0].tolist())  # Derive labels from the first column of the CSV
label_map = {k: v + 1 for v, k in enumerate(label_classes)}
rev_label_map = {v: k for k, v in label_map.items()}  # Inverse mapping