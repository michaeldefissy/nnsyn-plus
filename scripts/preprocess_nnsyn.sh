sinteractive -g gpu:1 -m 20gb -n 1 -c 4 -t 1:59:00 -A OD-214219

module load miniconda3/
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

nnsyn_plan_and_preprocess -d 960 -c 3d_fullres -pl nnUNetPlannerResEncL -p nnUNetResEncUNetLPlans  --preprocessing_input MR --preprocessing_target CT 
nnsyn_plan_and_preprocess_seg -d 960 -dseg 961 -c 3d_fullres -p nnUNetResEncUNetLPlans