import os
import SimpleITK as sitk
import numpy as np

# Configuration
LABEL_DIR = '/datasets/work/hb-iphd-sct/source/datasets/synthrad2025_AB/nnUNet_raw/Dataset961_SEG_synthrad2025_task1_mri2ct_AB/labelsTr'
MAX_ALLOWED_LABEL = 61  # Model has 62 classes (0-61)

def scan_labels():
    print(f"Scanning files in: {LABEL_DIR}")
    print(f"Checking for labels > {MAX_ALLOWED_LABEL}...")
    
    files = [f for f in os.listdir(LABEL_DIR) if f.endswith('.mha')]
    corrupt_files = 0

    for filename in files:
        filepath = os.path.join(LABEL_DIR, filename)
        
        # Load image and get unique values
        image = sitk.ReadImage(filepath)
        data = sitk.GetArrayFromImage(image)
        unique_labels = np.unique(data)
        
        # Check against limit
        max_val = unique_labels.max()
        if max_val > MAX_ALLOWED_LABEL:
            print(f"❌ VIOLATION: {filename} | Max label: {max_val} | All labels: {unique_labels}")
            corrupt_files += 1
        else:
            # Optional: Print progress for clean files
            print(f"✅ OK: {filename}")
            pass

    if corrupt_files == 0:
        print("\nAll files are valid! No labels found above 61.")
    else:
        print(f"\nFound {corrupt_files} files with invalid labels.")

if __name__ == "__main__":
    scan_labels()
