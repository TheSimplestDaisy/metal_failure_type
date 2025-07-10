# -*- coding: utf-8 -*-
"""
Created on Fri Jun 27 21:32:43 2025

@author: zzulk
"""

import os
import shutil
from sklearn.model_selection import train_test_split

# Lokasi asal dataset
base_dir = r"C:\Users\zzulk\Downloads\Metal_Type_Fracture"
# Lokasi untuk folder baru split
output_dir = r"C:\Users\zzulk\Downloads\Metal_Type_Fracture_Split"
os.makedirs(output_dir, exist_ok=True)

# Senarai semua kelas
labels = os.listdir(base_dir)

for label in labels:
    label_path = os.path.join(base_dir, label)
    if os.path.isdir(label_path):
        images = [f for f in os.listdir(label_path) if f.lower().endswith(('.jpg', '.png'))]
        train_files, test_files = train_test_split(images, test_size=0.2, random_state=42)

        for split_type, split_files in [('train', train_files), ('test', test_files)]:
            split_folder = os.path.join(output_dir, split_type, label)
            os.makedirs(split_folder, exist_ok=True)
            for img_file in split_files:
                src_path = os.path.join(label_path, img_file)
                dst_path = os.path.join(split_folder, img_file)
                shutil.copy2(src_path, dst_path)

print("✅ Selesai split train/test untuk semua folder kelas.")
