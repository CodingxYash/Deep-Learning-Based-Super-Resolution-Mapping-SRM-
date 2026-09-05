import os
import tifffile
import numpy as np

lr_dir = "train_lr/train_lr"
hr_dir = "train_hr_all"

lr_files = set(os.listdir(lr_dir))
hr_files = set(os.listdir(hr_dir))

common_files = sorted(lr_files & hr_files)  # sirf common filenames

print("Total LR files:", len(lr_files))
print("Total HR files:", len(hr_files))
print("Common (matched) files:", len(common_files))
print("First common filename:", common_files[0])

lr_img = tifffile.imread(os.path.join(lr_dir, common_files[0]))
hr_img = tifffile.imread(os.path.join(hr_dir, common_files[0]))

print("LR shape:", lr_img.shape, "dtype:", lr_img.dtype)
print("HR shape:", hr_img.shape, "dtype:", hr_img.dtype)
print("LR min/max:", lr_img.min(), lr_img.max())
print("HR min/max:", hr_img.min(), hr_img.max())