import json
import os
import pandas as pd


label_classes = ['cell', 'UNKNOWN']  # Label classes
label_map = {k: v + 1 for v, k in enumerate(label_classes)}
label_map['background'] = 0  # Background is the first class
rev_label_map = {v: k for k, v in label_map.items()}  # Inverse mapping