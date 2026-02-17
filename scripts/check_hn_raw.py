import SimpleITK as sitk
import glob
import numpy as np
import os

# Path to the RAW labels you processed
label_path = "/datasets/work/hb-iphd-sct/source/datasets/synthrad2025_HN/nnsyn_origin/synthrad2025_task1_mri2ct_HN/LABELS/*.mha"

for f in glob.glob(label_path):
    img = sitk.ReadImage(f)
    nda = sitk.GetArrayFromImage(img)
    if nda.min() < 0 or nda.max() > 31:
        print(f"!!! CORRUPT FILE FOUND: {f}")
        print(f"    Min: {nda.min()}, Max: {nda.max()}")
