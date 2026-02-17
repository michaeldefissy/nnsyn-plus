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

# Change these in accordance with the specific input MR image you want to predict sCT from.
export INPUT_PATH=""
export OUTPUT_PATH=""
export MASK_PATH="" 

#export nnsyn_origin_dataset="/datasets/work/hb-iphd-sct/source/datasets/synthrad2025_HN/nnsyn_origin/synthrad2025_task1_mri2ct_HN"
#export nnUNet_raw="/datasets/work/hb-iphd-sct/source/datasets/synthrad2025_HN/nnUNet_raw"
#export nnUNet_preprocessed="/datasets/work/hb-iphd-sct/source/datasets/synthrad2025_HN/nnUNet_preprocessed"
#export nnUNet_results="/datasets/work/hb-iphd-sct/source/datasets/synthrad2025_HN/nnUNet_results"

nnsyn_predict -d 960 -i $INPUT_PATH \
    -o $OUTPUT_PATH \
    -m $MASK_PATH \
    -c 3d_fullres \
    -p nnUNetResEncUNetLPlans \
    -tr nnUNetTrainer_nnsyn_loss_map \
    -f 0