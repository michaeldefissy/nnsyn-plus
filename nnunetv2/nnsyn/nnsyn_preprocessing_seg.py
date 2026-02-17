import glob
import os
import SimpleITK as sitk
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import json
import shutil
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name


def copy_dataset_json(raw_path, preprocessed_path, dataset_name):
    src_path = os.path.join(raw_path, dataset_name, 'dataset.json')
    dst_path = os.path.join(preprocessed_path, dataset_name, 'dataset.json')
    if not os.path.exists(dst_path):
        shutil.copy(src_path, dst_path)
    else:
        print(f"Dataset JSON already exists at {dst_path}, skipping copy.")

def create_dataset_json(num_train, preprocessing, dataset_data_path, label_to_use): 
    labels = {str(label): i + 1 for i, label in enumerate(label_to_use)}
    labels["background"] = 0

    data_dataset_json = {
        "labels": labels,
        "channel_names": {
            "0": preprocessing,
            
        },
        "numTraining": num_train,
        "file_ending": ".mha"
    }
    dump_data_datasets_path = os.path.join(dataset_data_path, 'dataset.json')
    with open(dump_data_datasets_path, 'w') as f:
        json.dump(data_dataset_json, f)

def makedirs_raw_dataset(dataset_data_path):
    
    os.makedirs(dataset_data_path, exist_ok = True)
    os.makedirs(os.path.join(dataset_data_path, 'imagesTr'), exist_ok=True)
    os.makedirs(os.path.join(dataset_data_path, 'labelsTr'), exist_ok = True)

def process_file(data_path, dataset_path, modality_suffix="_0000"):
    curr_img = sitk.ReadImage(data_path)
    filename = os.path.basename(data_path)
    if not filename.endswith(f'{modality_suffix}.mha'):
        filename = filename + f'{modality_suffix}.mha'
    sitk.WriteImage(curr_img, os.path.join(dataset_path, f'imagesTr/{filename}'))

def process_file_labels(data_path, dataset_path, max_label):
    """
    data_path: path to the raw .mha label
    dataset_path: path to the nnUNet_raw/DatasetXXX folder
    max_label: the highest integer label index allowed (e.g., 31)
    """
    curr_img = sitk.ReadImage(data_path)
    nda = sitk.GetArrayFromImage(curr_img)
    
    # 1. Force the range to [0, max_label]
    # Any value below 0 becomes 0 (background)
    # Any value above max_label becomes 0 (background/ignore) 
    # Or you can clip to max_label: np.clip(nda, 0, max_label)
    invalid_mask = (nda < 0) | (nda > max_label)
    if np.any(invalid_mask):
        print(f"WARNING: Cleaning {os.path.basename(data_path)}. Found values outside [0, {max_label}]")
        nda[invalid_mask] = 0 
    
    # 2. Ensure it is an integer type (critical for nnU-Net)
    nda = nda.astype(np.uint8)
    
    # 3. Rebuild and Save
    new_img = sitk.GetImageFromArray(nda)
    new_img.CopyInformation(curr_img)
    
    filename = os.path.basename(data_path)
    sitk.WriteImage(new_img, os.path.join(dataset_path, f'labelsTr/{filename}'))


def nnsyn_plan_and_preprocess_seg(dataset_id_syn: int,  dataset_id_seg: int,
                        configuration: str = '3d_fullres', plan: str = 'nnUNetPlans', 
                        dataset_name: str = None):
    data_origin_path = os.environ.get('nnsyn_origin_dataset', None)
    list_data_ct = sorted(glob.glob(os.path.join(data_origin_path, 'TARGET_IMAGES','*.mha'), recursive=True))
    list_data_labels = sorted(glob.glob(os.path.join(data_origin_path, 'LABELS','*.mha'), recursive=True))
    print("target ---", len(list_data_ct), list_data_ct[:2])
    print("labels ---", len(list_data_labels), list_data_labels[:2])

    # 1. Determine max_label from the dataset.json
    json_path = os.path.join(data_origin_path, 'LABELS', 'dataset.json')
    with open(json_path, 'r') as f:
        dj = json.load(f)
    
    # Count labels: if "labels" is a dict, max index is usually len - 1
    # Example: {"background": 0, "liver": 1} -> max_label is 1
    max_label = max([int(v) for v in dj['labels'].values()])
    print(f"Dynamic Check: Max allowed label is {max_label}")

    if dataset_name is None:
        dataset_name = 'SEG_' + os.path.basename(data_origin_path)


    # copy data from orign to nnUNet_raw
    dataset_data_path = os.path.join(os.environ['nnUNet_raw'], f'Dataset{dataset_id_seg:03d}_{dataset_name}') 
    makedirs_raw_dataset(dataset_data_path)

    with ThreadPoolExecutor() as executor:
        list(tqdm(executor.map(lambda data_path: process_file(data_path, dataset_data_path, "_0000"), list_data_ct), total=len(list_data_ct)))

    with ThreadPoolExecutor() as executor:
        list(tqdm(executor.map(lambda target_path: process_file_labels(target_path, dataset_data_path, max_label), list_data_labels), total=len(list_data_labels)))

    
    if os.path.exists(os.path.join(data_origin_path,'LABELS', 'dataset.json')):
        print(f"Segmentation dataset.json found in {data_origin_path}, copying it to {dataset_data_path}")
        shutil.copy(os.path.join(data_origin_path, 'LABELS', 'dataset.json'), dataset_data_path)
    # create dataset.json -> change it for general segmentator using dataset.json in ORIGIN
    # num_train = len(list_data_ct)
    # assert len(list_data_labels) == len(list_data_ct)
    # labels_to_use = [
    #             2, # kidney right
    #             3, # kidney left
    #             5, # liver
    #             6, # stomach
    #             *range(10, 14+1), #lungs
    #             *range(26, 50+1), #vertebrae
    #             51, #heart
    #             79, # spinal cord
    #             *range(92, 115+1), # ribs
    #             116 #sternum
    #         ]
    # create_dataset_json(num_train, preprocessing_target, dataset_data_path, labels_to_use)
    
    SOURCE_PLAN_IDENTIFIER = plan
    TARGET_PLAN_IDENTIFIER = plan + f'_Dataset{dataset_id_syn}'

    os.system(f'nnUNetv2_extract_fingerprint -d {dataset_id_seg} --verify_dataset_integrity')
    os.system(f'nnUNetv2_move_plans_between_datasets -s {dataset_id_syn} -t {dataset_id_seg} -sp {SOURCE_PLAN_IDENTIFIER} -tp {TARGET_PLAN_IDENTIFIER}')
    copy_dataset_json(os.environ['nnUNet_raw'], os.environ['nnUNet_preprocessed'], f'Dataset{dataset_id_seg:03d}_{dataset_name}')
    os.system(f'nnUNetv2_preprocess -d {dataset_id_seg} -c {configuration} -plans_name {TARGET_PLAN_IDENTIFIER} -np 4')

    if dataset_id_seg is not None:
        # append dataset_id_seg to plans.json
        plans_file = os.path.join(os.environ['nnUNet_preprocessed'], f'Dataset{dataset_id_syn:03d}_{os.path.basename(data_origin_path)}', f'{plan}.json')
        with open(plans_file, 'r') as f:
            plans = json.load(f)
        plans['dataset_name_seg'] =  maybe_convert_to_dataset_name(dataset_id_seg)
        with open(plans_file, 'w') as f:
            json.dump(plans, f, indent=4)
        print(f"Saved dataset_name_seg {dataset_id_seg} to {plans_file}")

if __name__ == '__main__':
    # example usage:
    # python -m nnsyn_preprocessing_entry -d 982 --data_origin_path '/datasets/work/hb-synthrad2023/work/synthrad2025/bw_workplace/data/nnunet_struct/ORIGIN/Synthrad2025_MRI2CT_AB' --preprocessing_target CT --dataset_id_syn 960
    # nnsyn_plan_and_preprocess_seg -d 961 -ds 960 -c 3d_fullres -p nnUNetResEncUNetLPlans --data_origin_path '/datasets/work/hb-synthrad2023/work/synthrad2025/bw_workplace/data/nnunet_struct/ORIGIN/synthrad2025_task1_mri2ct_AB' --preprocessing_target CT
    nnsyn_plan_and_preprocess_seg(data_origin_path='/datasets/work/hb-synthrad2023/work/synthrad2025/bw_workplace/data/nnunet_struct/ORIGIN/synthrad2025_task1_mri2ct_AB', 
                              dataset_id_seg=961, dataset_id_syn=960,
                              configuration='3d_fullres', plan='nnUNetResEncUNetLPlans')