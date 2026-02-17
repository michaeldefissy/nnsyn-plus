import os
import glob
import numpy as np
from tqdm import tqdm

def check_seg_files(directory, min_val=0, max_val=61):
    # Search for all files ending in _seg.npy
    search_pattern = os.path.join(directory, "*_seg.npy")
    files = glob.glob(search_pattern, recursive=True)
    
    if not files:
        print(f"No _seg.npy files found in {directory}")
        return

    print(f"Checking {len(files)} files...")
    corrupt_files = []

    for f in tqdm(files):
        try:
            # nnU-Net v2 preprocessed files are usually standard numpy arrays
            data = np.load(f)
            
            # Check for values outside the specified range
            # Note: nnU-Net labels are usually integers
            found_min = data.min()
            found_max = data.max()
            
            if found_min < min_val or found_max > max_val:
                unique_vals = np.unique(data)
                corrupt_files.append({
                    'file': f,
                    'min': found_min,
                    'max': found_max,
                    'out_of_bounds': unique_vals[(unique_vals < min_val) | (unique_vals > max_val)]
                })
        except Exception as e:
            print(f"Error loading {f}: {e}")

    # Report results
    if corrupt_files:
        print(f"\nFound {len(corrupt_files)} files with invalid label values:")
        for entry in corrupt_files:
            print(f"\nFile: {entry['file']}")
            print(f"  Range: [{entry['min']}, {entry['max']}]")
            print(f"  Invalid values found: {entry['out_of_bounds']}")
    else:
        print(f"\nAll files passed! No values outside [{min_val}, {max_val}] were found.")

if __name__ == "__main__":
    target_dir = "/datasets/work/hb-iphd-sct/source/datasets/synthrad2025_AB/nnUNet_preprocessed/Dataset961_SEG_synthrad2025_task1_mri2ct_AB/nnUNetResEncUNetLPlans_Dataset960_nnUNetPlans_3d_fullres"
    check_seg_files(target_dir)
