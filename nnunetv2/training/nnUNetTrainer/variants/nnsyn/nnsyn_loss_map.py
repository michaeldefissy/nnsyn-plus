import glob
import os
import torch
from torch import nn
import numpy as np
from typing import Union


from nnunetv2.training.loss.dice import SoftDiceLoss, MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss, DC_and_BCE_loss
from nnunetv2.utilities.helpers import softmax_helper_dim1
from batchgenerators.utilities.file_and_folder_operations import load_json,join
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from nnunetv2.training.nnUNetTrainer.nnUNetTSTrainer import nnUNetTSTrainer
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from acvl_utils.cropping_and_padding.padding import pad_nd_image
from nnunetv2.training.loss.mse import myMSE, myMaskedMSE
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans_and_class
from nnunetv2.training.loss.unet import ResidualEncoderUNet
# from nnunetv2.training.loss.ssim_losses import SSIMLoss
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name



# https://github.com/richzhang/PerceptualSimilarity/blob/master/lpips/lpips.py
# https://github.com/Project-MONAI/GenerativeModels/blob/main/generative/losses/perceptual.py


class GradientDifferenceLoss(nn.Module):
    def __init__(self, alpha=2):
        super(GradientDifferenceLoss, self).__init__()
        self.alpha = alpha

    def forward(self, pred, target, mask=None):
        # 1. Use torch.diff to get directional gradients
        dz_pred, dy_pred, dx_pred = pred.diff(dim=-3), pred.diff(dim=-2), pred.diff(dim=-1)
        dz_target, dy_target, dx_target = target.diff(dim=-3), target.diff(dim=-2), target.diff(dim=-1)

        # 2. Compute the difference between directional gradients
        loss_z = torch.abs(dz_pred - dz_target) ** self.alpha
        loss_y = torch.abs(dy_pred - dy_target) ** self.alpha
        loss_x = torch.abs(dx_pred - dx_target) ** self.alpha

        if mask is not None:
            # Shift masks to match the .diff() dimensionality reduction
            mask_z = mask[:, :, 1:, :, :] * mask[:, :, :-1, :, :]
            mask_y = mask[:, :, :, 1:, :] * mask[:, :, :, :-1, :]
            mask_x = mask[:, :, :, :, 1:] * mask[:, :, :, :, :-1]
            
            loss_z = loss_z * mask_z
            loss_y = loss_y * mask_y
            loss_x = loss_x * mask_x

            return (loss_z.sum() + loss_y.sum() + loss_x.sum()) / pred.numel()

        return (loss_z.sum() + loss_y.sum() + loss_x.sum()) / pred.numel()

class FocalFrequencyLoss(nn.Module):
    def __init__(self, loss_weight=1.0, alpha=1.0):
        super(FocalFrequencyLoss, self).__init__()
        self.loss_weight = loss_weight
        self.alpha = alpha

    def forward(self, pred, target, mask=None):
        
        # 1. 3D Fast Fourier Transform (Complex values)
        pred_fft = torch.fft.fftn(pred, dim=(-3, -2, -1), norm='ortho')
        target_fft = torch.fft.fftn(target, dim=(-3, -2, -1), norm='ortho')

        # 2. Complex difference (preserves both Amplitude and Phase)
        diff = pred_fft - target_fft
        freq_distance = torch.abs(diff) ** 2 # Squared Euclidean distance

        # 3. THE FOCAL MECHANISM: Dynamic Spectrum Weighting
        weight_matrix = torch.abs(diff) ** self.alpha

        # Match official logic: Normalize weight matrix by the maximum distance 
        # We use amax to find the max across the 3D spatial dimensions (D, H, W)
        max_vals = weight_matrix.amax(dim=(-3, -2, -1), keepdim=True)
        max_vals = torch.clamp(max_vals, min=1e-8) # Safe division
        
        weight_matrix = weight_matrix / max_vals

        # Detach the matrix from the computation graph (we only optimize the distance)
        weight_matrix = torch.clamp(weight_matrix, min=0.0, max=1.0).detach()

        # 4. Apply the focal weights (Hadamard product)
        loss = weight_matrix * freq_distance
        
        return self.loss_weight * loss.mean()

class ReLoBRaLo:
    """
    Relative Loss Balancing with Random Lookback.
    Dynamically adjusts weights based on how much a loss has dropped 
    compared to the previous step (short-term) or initial step (long-term).
    """
    def __init__(self, num_losses, initial_weights=None, alpha=0.99, temperature=1.0, rho=0.5):
        self.num_losses = num_losses
        self.alpha = alpha  # Smoothing factor for history
        self.temperature = temperature # Softmax temperature
        self.rho = rho # Probability of looking back at previous step vs initial
        
        # If no initial weights provided, start equal
        if initial_weights is None:
            self.current_weights = np.ones(num_losses)
        else:
            self.current_weights = np.array(initial_weights)
            
        self.initial_losses = None
        self.prev_losses = None
        self.loss_history = [] 

    def update(self, current_losses: list):
        # Convert tensor losses to numpy/float
        current_loss_vals = np.array([l.item() if isinstance(l, torch.Tensor) else l for l in current_losses])
        
        # Initialization step
        if self.initial_losses is None:
            self.initial_losses = current_loss_vals
            self.prev_losses = current_loss_vals
            return torch.tensor(self.current_weights, dtype=torch.float32, device='cuda')

        # 1. Short-term lookback
        T_stat = current_loss_vals / (self.prev_losses * self.temperature + 1e-12)
        exp_T_stat = np.exp(T_stat - np.max(T_stat)) # safe softmax
        lambs_hat = self.num_losses * (exp_T_stat / np.sum(exp_T_stat))

        # 2. Initial lookback
        T_init = current_loss_vals / (self.initial_losses * self.temperature + 1e-12)
        exp_T_init = np.exp(T_init - np.max(T_init)) # safe softmax
        lambs0_hat = self.num_losses * (exp_T_init / np.sum(exp_T_init))

        # 3. Deterministic combination
        self.current_weights = (self.rho * self.alpha * self.current_weights) + \
                               ((1 - self.rho) * self.alpha * lambs0_hat) + \
                               ((1 - self.alpha) * lambs_hat)
        
        # Update history
        self.prev_losses = current_loss_vals
        
        return torch.tensor(self.current_weights, dtype=torch.float32, device='cuda')
    
class MaskedAnatomicalPerceptionLoss(nn.Module):
    def __init__(self, dataset_name_seg: int, 
                 image_loss_weight: float = 0.5, 
                 perception_masked=False,
                 gdl_weight: float = 0.0,
                 ffl_weight: float = 0.0,
                 dynamic_balancing: bool = False): # Toggle switch
        """
        Initializes the loss module with optional Dynamic Balancing.
        """
        super(MaskedAnatomicalPerceptionLoss, self).__init__()

        # Load Segmentor
        self.seg_model, self.seg_model_info = self._load_trained_segmentor(dataset_name_seg)
        self.seg_model.eval()
        for param in self.seg_model.parameters(): 
            param.requires_grad = False
        self.seg_model.to(device='cuda', dtype=torch.float16)

        # Standard Losses
        self.L1 = nn.L1Loss()
        self.perception_masked = perception_masked
        self.image_loss = myMaskedMSE()
        
        # Configuration
        self.dynamic_balancing = dynamic_balancing
        
        # Initialize Weights (Static defaults)
        # Order: [Perception, MSE, GDL, FFL]
        self.static_weights = [1.0 - image_loss_weight, image_loss_weight, gdl_weight, ffl_weight]
        
        # Initialize Sub-losses
        self.gdl = GradientDifferenceLoss()
        self.ffl = FocalFrequencyLoss()
        
        # Initialize Dynamic Balancer if enabled
        if self.dynamic_balancing:
            # We have 4 potential loss components
            self.relobralo = ReLoBRaLo(num_losses=4, initial_weights=self.static_weights)

        # Logging placeholders
        self.cur_weights = self.static_weights # To log the current weights used
        self.cur_seg_loss = 0.0
        self.cur_img_loss = 0.0
        self.cur_gdl_loss = 0.0
        self.cur_ffl_loss = 0.0

    def _load_trained_segmentor(self, dataset_name_seg: Union[int, str]):
        # segmentor_training_output_dir = {
        #     "1":
        #         {"AB": "/datasets/work/hb-synthrad2023/work/synthrad2025/bw_workplace/data/nnunet_struct/results/Dataset800_SEGMENTATION_synthrad2025_task1_CT_AB_aligned_to_Dataset261/nnUNetTrainer__nnUNetResEncUNetLPlans_Dataset261__3d_fullres",
        #         "HN": "/datasets/work/hb-synthrad2023/work/synthrad2025/bw_workplace/data/nnunet_struct/results/Dataset801_SEGMENTATION_synthrad2025_task1_CT_HN_aligned_to_Dataset263/nnUNetTrainer__nnUNetResEncUNetLPlans_Dataset263__3d_fullres",
        #         "TH": "/datasets/work/hb-synthrad2023/work/synthrad2025/bw_workplace/data/nnunet_struct/results/Dataset802_SEGMENTATION_synthrad2025_task1_CT_TH_aligned_to_Dataset265/nnUNetTrainer__nnUNetResEncUNetLPlans_Dataset265__3d_fullres"},
        #     "2":
        #         {"AB": "/datasets/work/hb-synthrad2023/work/synthrad2025/bw_workplace/data/nnunet_struct/results/Dataset810_SEGMENTATION_synthrad2025_task2_CT_AB_aligned_to_Dataset541/nnUNetTrainer__nnUNetResEncUNetLPlans_Dataset541__3d_fullres",
        #         "HN": "/datasets/work/hb-synthrad2023/work/synthrad2025/bw_workplace/data/nnunet_struct/results/Dataset811_SEGMENTATION_synthrad2025_task2_CT_HN_aligned_to_Dataset543/nnUNetTrainer__nnUNetResEncUNetLPlans_Dataset543__3d_fullres",
        #         "TH": "/datasets/work/hb-synthrad2023/work/synthrad2025/bw_workplace/data/nnunet_struct/results/Dataset812_SEGMENTATION_synthrad2025_task2_CT_TH_aligned_to_Dataset545/nnUNetTrainer__nnUNetResEncUNetLPlans_Dataset545__3d_fullres"},
        # }
        # model_training_output_dir = '/datasets/work/hb-synthrad2023/work/synthrad2025/bw_workplace/ref/evaluation/.totalsegmentator/nnunet/results/Dataset297_TotalSegmentator_total_3mm_1559subj/nnUNetTrainer_4000epochs_NoMirroring__nnUNetPlans__3d_fullres'
        # model_training_output_dir = segmentor_training_output_dir[task][region]

        # /datasets/work/hb-synthrad2023/work/synthrad2025/bw_workplace/data/nnunet_struct/results/Dataset961_SEG_synthrad2025_task1_mri2ct_AB/nnUNetTrainer__nnUNetResEncUNetLPlans_Dataset960__3d_fullres
        # /datasets/work/hb-synthrad2023/work/synthrad2025/bw_workplace/data/nnunet_struct/results/Dataset960_synthrad2025_task1_mri2ct_AB/nnUNetTrainer_nnsyn_loss_masked_perception_masked_track__nnUNetResEncUNetLPlans__3d_fullres
        dataset_name_seg = maybe_convert_to_dataset_name(dataset_name_seg)
        segmentation_output_dir = glob.glob(os.path.join(os.environ['nnUNet_results'], dataset_name_seg, '*'))[0]
        assert len(glob.glob(os.path.join(os.environ['nnUNet_results'], dataset_name_seg, '*'))) == 1, f"Segmentation model output dir is more than one or empty: {segmentation_output_dir}"
        model_training_output_dir = segmentation_output_dir
        checkpoint_name = 'checkpoint_final.pth'
        if not os.path.exists(join(model_training_output_dir, f'fold_0', checkpoint_name)):
            checkpoint_name = 'checkpoint_best.pth'
            print('checkpoint_final.pth not found, using checkpoint_best.pth instead')
        dataset_json = load_json(join(model_training_output_dir, 'dataset.json'))
        plans = load_json(join(model_training_output_dir, 'plans.json'))
        plans_manager = PlansManager(plans)

        with torch.serialization.safe_globals([np.core.multiarray.scalar, np.dtype, np.dtypes.Float64DType,np.dtypes.Float32DType]):
            checkpoint = torch.load(join(model_training_output_dir, f'fold_0', checkpoint_name),
                            map_location=torch.device('cpu'), weights_only=False)
        configuration_name = checkpoint['init_args']['configuration']
        trainer_name = checkpoint['trainer_name']
        
        configuration_manager = plans_manager.get_configuration(configuration_name)

        num_input_channels = determine_num_input_channels(plans_manager, configuration_manager, dataset_json)
        # trainer_class = recursive_find_python_class(join(nnunetv2.__path__[0], "training", "nnUNetTrainer"),
        #                                             "nnUNetTSTrainer", 'nnunetv2.training.nnUNetTSTrainer.nnUNetTSTrainer')
        trainer_class = nnUNetTSTrainer

        network = get_network_from_plans_and_class(
            ResidualEncoderUNet,
            configuration_manager.network_arch_init_kwargs,
            configuration_manager.network_arch_init_kwargs_req_import,
            num_input_channels,
            plans_manager.get_label_manager(dataset_json).num_segmentation_heads,
        )
        network.load_state_dict(checkpoint['network_weights'])

        network_info = {
            "num_classes": plans_manager.get_label_manager(dataset_json).num_segmentation_heads, 
            "patch_size": configuration_manager.patch_size,
            "n_stages": configuration_manager.network_arch_init_kwargs['n_stages'],
        }

        return network, network_info

    def forward(self, output: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # 1. Perception Loss
        if self.perception_masked:
            out_seg_input = output * mask + (1 - mask) * -1
            tgt_seg_input = target * mask + (1 - mask) * -1
        else:
            out_seg_input = output
            tgt_seg_input = target

        pred_outputs = self.seg_model(out_seg_input)
        pred_gt_outputs = self.seg_model(tgt_seg_input)

        perception_loss = 0
        for i in range(self.seg_model_info['n_stages']):
            perception_loss += self.L1(self._normalize_tensor(pred_outputs[i]), 
                                     self._normalize_tensor(pred_gt_outputs[i].detach()))

        # 2. Image Loss (MSE)
        img_loss = self.image_loss(output, target, mask=mask)

        # 3. GDL
        gdl_loss = self.gdl(output, target, mask)

        # 4. FFL
        ffl_loss = self.ffl(output, target, mask)

        # --- Weighting Strategy ---
        if self.dynamic_balancing:
            # Update and get new weights [Perception, MSE, GDL, FFL]
            losses_list = [perception_loss, img_loss, gdl_loss, ffl_loss]
            weights = self.relobralo.update(losses_list)
            
            # Store for logging
            self.cur_weights = weights.detach().cpu().numpy().tolist()
            
            total_loss = (weights[0] * perception_loss) + \
                         (weights[1] * img_loss) + \
                         (weights[2] * gdl_loss) + \
                         (weights[3] * ffl_loss)
        else:
            # Static Weighting
            w = self.static_weights
            total_loss = (w[0] * perception_loss) + \
                         (w[1] * img_loss) + \
                         (w[2] * gdl_loss) + \
                         (w[3] * ffl_loss)

        # Logging
        self.cur_seg_loss = perception_loss.detach().cpu().numpy()
        self.cur_img_loss = img_loss.detach().cpu().numpy()
        self.cur_gdl_loss = gdl_loss.detach().cpu().numpy()
        self.cur_ffl_loss = ffl_loss.detach().cpu().numpy()
        
        return total_loss

    def _normalize_tensor(self, in_feat, eps=1e-10):
        norm_factor = torch.sqrt(torch.sum(in_feat**2, dim=1, keepdim=True))
        return in_feat/(norm_factor+eps)
    

if __name__ == "__main__":
    # Example usage
    region = "AB"  # or "HN", "TH"
    syn_perception_loss = MaskedAnatomicalPerceptionLoss(dataset_name_seg=800, image_loss_weight=0.5, perception_masked=True)

    # # Dummy data for testing
    output = torch.randn(2, 1, 40, 192, 192)  # Example output from a model
    target = torch.randn(2, 1, 40, 192, 192)  # Example ground truth mask
    mask = torch.randint(0, 2, (2, 1, 40, 192, 192))  # Example mask
    output = output.to(device='cuda', dtype=torch.float16)  # Move to GPU if available
    target = target.to(device='cuda', dtype=torch.float16)  # Move to GPU if available
    mask = mask.to(device='cuda', dtype=torch.float16)  # Move to GPU if available

    loss_value = syn_perception_loss(output, target, mask)
    print(loss_value)
    print(f"Loss value: {loss_value.item()}")