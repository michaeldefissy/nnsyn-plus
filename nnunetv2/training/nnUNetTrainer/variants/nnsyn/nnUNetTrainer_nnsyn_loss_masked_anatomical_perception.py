import torch
import os
from typing import Union, Tuple, List
import numpy as np
from nnunetv2.training.loss.syn_perception_loss import SynPerceptionLoss, SynPerceptionLoss_L2
from nnunetv2.utilities.collate_outputs import collate_outputs
from nnunetv2.training.nnUNetTrainer.variants.nnsyn.nnUNetTrainer_nnsyn_loss_masked import nnUNetTrainer_nnsyn_loss_masked
from nnunetv2.training.nnUNetTrainer.variants.nnsyn.nnsyn_loss_map import MaskedAnatomicalPerceptionLoss


class nnUNetTrainer_nnsyn_loss_map(nnUNetTrainer_nnsyn_loss_masked):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True, device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)

        dataset_name = plans.get('dataset_name', '')
        try:
            # Splits "Dataset960_Name" -> "Dataset960" -> "960"
            current_id = int(dataset_name.split('_')[0].replace('Dataset', ''))
            self.dataset_name_seg = current_id + 1
            print(f"nnsyn_loss_map: Automatically detected synthesis task {current_id}. "
                  f"Setting segmentation target task to {self.dataset_name_seg}.")
        except Exception as e:
            # Fallback if naming convention isn't standard
            self.dataset_name_seg = 961 
            print(f"nnsyn_loss_map: WARNING - Could not parse dataset ID from '{dataset_name}'. "
                  f"Defaulting segmentation target to {self.dataset_name_seg}. Error: {e}")
        
        # --- CONFIGURATION TOGGLES ---
        self.image_loss_weight = 0.5
        self.perception_masked = True
        
        # Enable GDL and FFL with static base weights
        self.gdl_weight = 0.1 
        self.ffl_weight = 0.1
        
        # TOGGLE: Set to True to enable ReLoBRaLo dynamic balancing
        self.dynamic_balancing = os.getenv('USE_DYNAMIC_BALANCING', 'True') == 'True'
        # -----------------------------

        # Initialize logging for new metrics
        for k in ['train_gdl_loss', 'train_ffl_loss', 'val_gdl_loss', 'val_ffl_loss', 
                  'train_weight_perc', 'train_weight_mse', 'train_weight_gdl', 'train_weight_ffl']:
            self.logger.my_fantastic_logging[k] = list()

    def _build_loss(self):
        # Pass the dynamic_balancing flag to the loss class
        loss = MaskedAnatomicalPerceptionLoss(
            dataset_name_seg=self.dataset_name_seg, 
            image_loss_weight=self.image_loss_weight, 
            perception_masked=self.perception_masked,
            gdl_weight=self.gdl_weight,
            ffl_weight=self.ffl_weight,
            dynamic_balancing=self.dynamic_balancing # <--- Flag passed here
        )
        return loss

    def train_step(self, batch: dict) -> dict:
        outputs = super().train_step(batch)
        # Log losses
        outputs['train_seg_loss'] = self.loss.cur_seg_loss
        outputs['train_img_loss'] = self.loss.cur_img_loss
        outputs['train_gdl_loss'] = self.loss.cur_gdl_loss
        outputs['train_ffl_loss'] = self.loss.cur_ffl_loss
        
        # Log Dynamic Weights (to see if ReLoBRaLo is working)
        if hasattr(self.loss, 'cur_weights'):
            outputs['train_weight_perc'] = self.loss.cur_weights[0]
            outputs['train_weight_mse'] = self.loss.cur_weights[1]
            outputs['train_weight_gdl'] = self.loss.cur_weights[2]
            outputs['train_weight_ffl'] = self.loss.cur_weights[3]
        return outputs
    
    def on_train_epoch_end(self, train_outputs: List[dict]):
        outputs = collate_outputs(train_outputs)
        super().on_train_epoch_end(train_outputs) # Logs standard stuff
        
        # Log custom metrics
        self.logger.log('train_gdl_loss', np.mean(outputs['train_gdl_loss']), self.current_epoch)
        self.logger.log('train_ffl_loss', np.mean(outputs['train_ffl_loss']), self.current_epoch)
        
        if self.dynamic_balancing:
            self.logger.log('train_weight_perc', np.mean(outputs['train_weight_perc']), self.current_epoch)
            self.logger.log('train_weight_mse', np.mean(outputs['train_weight_mse']), self.current_epoch)
            self.logger.log('train_weight_gdl', np.mean(outputs['train_weight_gdl']), self.current_epoch)
            self.logger.log('train_weight_ffl', np.mean(outputs['train_weight_ffl']), self.current_epoch)



