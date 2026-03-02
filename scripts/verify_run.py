import torch
import matplotlib
matplotlib.use('Agg') # Ensures headless plotting works on SLURM
import matplotlib.pyplot as plt

# Point this to your recent fold_0/checkpoint_latest.pth or checkpoint_final.pth
checkpoint_path = "/datasets/work/hb-iphd-sct/source/datasets/synthrad2025_AB/nnUNet_results/Dataset140_synthrad2025_task1_mri2ct_AB/nnUNetTrainer_nnsyn_loss_map__nnUNetResEncUNetLPlans__3d_fullres/fold_0/checkpoint_final.pth"

print(f"Loading checkpoint: {checkpoint_path}")
ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

if 'logging' not in ckpt:
    print("Error: No logging dictionary found in checkpoint.")
    exit()

logs = ckpt['logging']

# 1. Print verification metrics for the last 5 epochs
gdl = logs.get('train_gdl_loss', [])
ffl = logs.get('train_ffl_loss', [])

print(f"\n--- Model Verification (Last 5 Epochs) ---")
print(f"GDL Loss Active: {len(gdl) > 0} | Latest values: {[round(x, 4) for x in gdl[-5:]]}")
print(f"FFL Loss Active: {len(ffl) > 0} | Latest values: {[round(x, 4) for x in ffl[-5:]]}")

# 2. Plot Dynamic Weights if they exist
if 'train_weight_perc' in logs and len(logs['train_weight_perc']) > 0:
    print("\nDynamic balancing data found. Generating plot...")
    
    epochs = range(len(logs['train_weight_perc']))
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, logs['train_weight_perc'], label='Perception')
    plt.plot(epochs, logs['train_weight_mse'], label='MSE')
    plt.plot(epochs, logs['train_weight_gdl'], label='GDL')
    plt.plot(epochs, logs['train_weight_ffl'], label='FFL')
    
    plt.title('Dynamic Loss Balancing (ReLoBRaLo) Weights over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Weight Value')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save in the same folder as the checkpoint
    out_img = checkpoint_path.replace(checkpoint_path.split('/')[-1], 'dynamic_weights_history.png')
    plt.savefig(out_img)
    print(f"Plot saved successfully to: {out_img}")
else:
    print("\nNo dynamic balancing weights found in this run (Static weights were used).")