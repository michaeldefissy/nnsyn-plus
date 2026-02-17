import torch
import os

# REPLACE THIS PATH with the actual path to your checkpoint file
checkpoint_path = "/datasets/work/hb-iphd-sct/source/datasets/synthrad2025_AB/nnUNet_results/Dataset960_synthrad2025_task1_mri2ct_AB/nnUNetTrainer_nnsyn_loss_map__nnUNetResEncUNetLPlans__3d_fullres/fold_0/checkpoint_latest.pth"

if os.path.exists(checkpoint_path):
    print(f"Loading: {checkpoint_path}...")
    try:
        # Load the checkpoint on CPU to avoid needing a GPU for just checking
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        # Check for the current_epoch key (nnUNet usually stores it as 'current_epoch')
        if 'current_epoch' in checkpoint:
            print(f"-----------------------------------")
            print(f"Saved Epoch: {checkpoint['current_epoch']}")
            print(f"-----------------------------------")
        else:
            print("The 'current_epoch' key was not found. Here are the available keys:")
            print(checkpoint.keys())
            
    except Exception as e:
        print(f"Error loading file: {e}")
else:
    print("File not found! Check your path.")