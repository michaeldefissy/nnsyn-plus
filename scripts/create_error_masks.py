import SimpleITK as sitk
import os
import glob

def process_single_case(syn_path, gt_path, mask_path, output_dir, patient_id):
    try:
        # --- 1. Load Images & Cast Types ---
        syn_img = sitk.ReadImage(syn_path)
        gt_img = sitk.ReadImage(gt_path)
        mask_img = sitk.ReadImage(mask_path)
        
        # Cast to Float32 to fix the subtraction error
        gt_img = sitk.Cast(gt_img, sitk.sitkFloat32)
        syn_img = sitk.Cast(syn_img, sitk.sitkFloat32)
        mask_img = sitk.Cast(mask_img, sitk.sitkUInt8)

        # --- 2. Mask Synthetic CT ---
        air_value = -1024.0
        masked_syn_img = sitk.Mask(syn_img, mask_img, outsideValue=air_value)

        # --- 3a. Compute Raw Error Map ---
        diff_img = sitk.Subtract(masked_syn_img, gt_img)
        abs_error_map = sitk.Abs(diff_img)
        final_error_map = sitk.Mask(abs_error_map, mask_img, outsideValue=0)

        # --- 3b. Create Colored Error Heatmap (RGB) ---
        # Step A: Clamp high errors. 
        # Any error > 200 HU will be "Max Red". Without this, small errors become invisible.
        # You can adjust 'upperBound' to your needs (e.g., 100, 200, 500).
        clamped_error = sitk.Clamp(final_error_map, lowerBound=0, upperBound=200)

        # Step B: Rescale to 0-255 for the colormap filter
        rescaled_error = sitk.RescaleIntensity(clamped_error, 0, 255)
        rescaled_error = sitk.Cast(rescaled_error, sitk.sitkUInt8)

        # Step C: Apply "Jet" Colormap (Blue=Low, Green=Med, Red=High)
        colormap_filter = sitk.ScalarToRGBColormapImageFilter()
        colormap_filter.SetColormap(sitk.ScalarToRGBColormapImageFilter.Jet)
        rgb_error_map = colormap_filter.Execute(rescaled_error)
        
        # Make background transparent-ish (Black) by applying mask again to RGB
        # (This ensures the area outside the body is black, not blue)
        rgb_error_map = sitk.Mask(rgb_error_map, mask_img)

        # --- 4. Overlap Visualization ---
        checkerboard = sitk.CheckerBoard(masked_syn_img, gt_img, checkerPattern=[4, 4, 4])

        # --- 5. Save Outputs ---
        patient_out_dir = os.path.join(output_dir, patient_id)
        if not os.path.exists(patient_out_dir):
            os.makedirs(patient_out_dir)

        # Save standard files
        sitk.WriteImage(masked_syn_img, os.path.join(patient_out_dir, f"{patient_id}_sCT_Masked.mha"))
        sitk.WriteImage(final_error_map, os.path.join(patient_out_dir, f"{patient_id}_ErrorMap.mha"))
        sitk.WriteImage(checkerboard, os.path.join(patient_out_dir, f"{patient_id}_Overlap.mha"))
        
        # Save the new Colored Heatmap
        sitk.WriteImage(rgb_error_map, os.path.join(patient_out_dir, f"{patient_id}_ErrorMap_Heatmap.mha"))
        
        print(f"[Success] Processed: {patient_id}")

    except Exception as e:
        print(f"[Error] Failed to process {patient_id}: {e}")

def batch_process(syn_dir, gt_dir, mask_dir, output_dir):
    """
    Iterates through synthetic CTs, finds matching GT and Mask files, and processes them.
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get all .mha files in the synthetic folder
    sct_files = glob.glob(os.path.join(syn_dir, "*.mha"))

    if not sct_files:
        print("No .mha files found in the Synthetic CT directory.")
        return

    print(f"Found {len(sct_files)} synthetic CTs. Starting batch processing...")

    for sct_path in sct_files:
        filename = os.path.basename(sct_path)
        patient_id = os.path.splitext(filename)[0]  # Extract patient ID from filename

        gt_path = os.path.join(gt_dir, f"{patient_id}_0000.mha")
        mask_path = os.path.join(mask_dir, filename)

        # Verify files exist before processing
        if not os.path.exists(gt_path):
            print(f"[Skipping] Missing Ground Truth for {patient_id} at: {gt_path}")
            continue
        
        if not os.path.exists(mask_path):
            print(f"[Skipping] Missing Mask for {patient_id} at: {mask_path}")
            continue

        # Process
        process_single_case(sct_path, gt_path, mask_path, output_dir, patient_id)

    print("Batch processing complete.")


if __name__ == "__main__":

    synthetic_dir = "/datasets/work/hb-iphd-sct/source/datasets/synthrad2025_AB/nnUNet_results/Dataset960_synthrad2025_task1_mri2ct_AB/nnUNetTrainer_nnsyn_loss_map__nnUNetResEncUNetLPlans__3d_fullres/fold_0/validation"      # Folder containing your generated sCTs
    ground_truth_dir = "/datasets/work/hb-iphd-sct/source/datasets/synthrad2025_AB/nnUNet_preprocessed/Dataset960_synthrad2025_task1_mri2ct_AB/gt_target"                                                                       # Folder containing ALL ground truths
    masks_dir = "/datasets/work/hb-iphd-sct/source/datasets/synthrad2025_AB/nnUNet_preprocessed/Dataset960_synthrad2025_task1_mri2ct_AB/masks"                                                                                  # Folder containing ALL masks
    results_dir = "/datasets/work/hb-iphd-sct/source/datasets/synthrad2025_AB/nnUNet_results/Dataset960_synthrad2025_task1_mri2ct_AB/nnUNetTrainer_nnsyn_loss_map__nnUNetResEncUNetLPlans__3d_fullres/fold_0/error_results"     # Where you want the files saved

    batch_process(synthetic_dir, ground_truth_dir, masks_dir, results_dir)