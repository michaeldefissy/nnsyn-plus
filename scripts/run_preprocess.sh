#!/bin/bash
#SBATCH --account=OD-214219
#SBATCH --job-name=nnsyn_prep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=logs/prep_%j.log
#SBATCH --error=logs/prep_%j.err

export nnsyn_origin_dataset="/datasets/work/hb-iphd-sct/source/datasets/synthrad2025_AB/nnsyn_origin/synthrad2025_task1_mri2ct_AB"
export nnUNet_raw="/datasets/work/hb-iphd-sct/source/datasets/synthrad2025_AB/nnUNet_raw"
export nnUNet_preprocessed="/datasets/work/hb-iphd-sct/source/datasets/synthrad2025_AB/nnUNet_preprocessed"
export nnUNet_results="/datasets/work/hb-iphd-sct/source/datasets/synthrad2025_AB/nnUNet_results"
export USE_DYNAMIC_BALANCING="False"

source /datasets/work/hb-iphd-sct/source/nnsynenv/bin/activate
cd /datasets/work/hb-iphd-sct/source/nnsyn

nnsyn_plan_and_preprocess \
    -d 140 \
    -c 3d_fullres \
    -pl nnUNetPlannerResEncL \
    -p nnUNetResEncUNetLPlans \
    --preprocessing_input MR \
    --preprocessing_target CT 

nnsyn_plan_and_preprocess_seg \
    -d 140 \
    -dseg 141 \
    -c 3d_fullres \
    -p nnUNetResEncUNetLPlans