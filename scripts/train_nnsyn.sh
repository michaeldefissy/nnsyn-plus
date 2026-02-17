#!/bin/bash

#SBATCH --account=OD-214219
#SBATCH --job-name=nnsyn_960debug_train
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --mem=80gb
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --signal=USR1@360
#SBATCH --output=logs/R-%x.%j.out
#SBATCH --open-mode=append

# Application specific commands:
module load cuda

source /datasets/work/hb-iphd-sct/source/nnsynenv/bin/activate
cd /datasets/work/hb-iphd-sct/source/nnsyn

export nnsyn_origin_dataset="/datasets/work/hb-iphd-sct/source/datasets/synthrad2025_AB/nnsyn_origin/synthrad2025_task1_mri2ct_AB"
export nnUNet_raw="/datasets/work/hb-iphd-sct/source/datasets/synthrad2025_AB/nnUNet_raw"
export nnUNet_preprocessed="/datasets/work/hb-iphd-sct/source/datasets/synthrad2025_AB/nnUNet_preprocessed"
export nnUNet_results="/datasets/work/hb-iphd-sct/source/datasets/synthrad2025_AB/nnUNet_results"

#export nnsyn_origin_dataset="/datasets/work/hb-iphd-sct/source/datasets/synthrad2025_HN/nnsyn_origin/synthrad2025_task1_mri2ct_HN"
#export nnUNet_raw="/datasets/work/hb-iphd-sct/source/datasets/synthrad2025_HN/nnUNet_raw"
#export nnUNet_preprocessed="/datasets/work/hb-iphd-sct/source/datasets/synthrad2025_HN/nnUNet_preprocessed"
#export nnUNet_results="/datasets/work/hb-iphd-sct/source/datasets/synthrad2025_HN/nnUNet_results"
export CUDA_LAUNCH_BLOCKING=1

#git switch nnunetv2
#srun nnUNetv2_train 961 3d_fullres 0 -tr nnUNetTrainer -p nnUNetResEncUNetLPlans_Dataset960 --c
srun nnsyn_train 960 3d_fullres 0 -tr nnUNetTrainer_nnsyn_loss_map -p nnUNetResEncUNetLPlans --c

# segmentation loss
# srun nnUNetv2_train 250 3d_fullres 0 -tr nnUNetTrainerMRCT_loss_masked -p nnUNetResEncUNetLPlans -pretrained_weights /datasets/work/hb-synthrad2023/work/synthrad2025/bw_workplace/data/nnunet_struct/results/Dataset260_synthrad2025_task1_MR_AB_pre_v2r_stitched_masked/nnUNetTrainerMRCT_loss_masked__nnUNetResEncUNetLPlans__3d_fullres/fold_0/checkpoint_final.pth