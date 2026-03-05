#!/bin/bash
#SBATCH --account=OD-214219
#SBATCH --job-name=nnsyn_train_static
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=logs/train_%j.log
#SBATCH --error=logs/train_%j.err


export nnsyn_origin_dataset="/datasets/work/hb-iphd-sct/source/datasets/synthrad2025_AB/nnsyn_origin/synthrad2025_task1_mri2ct_AB"
export nnUNet_raw="/datasets/work/hb-iphd-sct/source/datasets/synthrad2025_AB/nnUNet_raw"
export nnUNet_preprocessed="/datasets/work/hb-iphd-sct/source/datasets/synthrad2025_AB/nnUNet_preprocessed"
export nnUNet_results="/datasets/work/hb-iphd-sct/source/datasets/synthrad2025_AB/nnUNet_results"
export USE_DYNAMIC_BALANCING="False"

module load cuda

source /datasets/work/hb-iphd-sct/source/nnsynenv/bin/activate
cd /datasets/work/hb-iphd-sct/source/nnsyn

# Train segmentation network
#nnUNetv2_train 141 3d_fullres 0 \
#    -tr nnUNetTrainer \
#    -p nnUNetResEncUNetLPlans_Dataset140 \
#    --c

# It includes GDL, FFL, and ReLoBRaLo (Dynamic Balancing).
# Train synthesis network
nnsyn_train 140 3d_fullres 0 \
    -tr nnUNetTrainer_nnsyn_loss_map \
    -p nnUNetResEncUNetLPlans \
    --c

echo "Synthesis Training Complete."