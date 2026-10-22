"""Load an MAE anomaly detector checkpoint for inference."""

from pathlib import Path
import torch

from . import models_mae


class AnomalyDetector:
    def __init__(self, weights_path, device='cuda:0', model_type='mae_vit_base_patch16',
                 mask_ratio=0.0, no_mask=False, anomaly_mask=False, input_size=224):
        if model_type != 'mae_vit_base_patch16':
            raise ValueError("AnomNOVIC supports only mae_vit_base_patch16")
        if not weights_path or not Path(weights_path).is_file():
            raise FileNotFoundError(f"MAE checkpoint not found: {weights_path}")
        self.device = torch.device(device)
        self.mask_ratio = mask_ratio
        self.no_mask = no_mask
        self.anomaly_mask = anomaly_mask
        self.input_size = input_size
        self.model = models_mae.mae_vit_base_patch16(
            img_size=input_size, no_mask=no_mask, anomaly_mask=anomaly_mask).to(self.device)
        checkpoint = torch.load(weights_path, map_location='cpu', weights_only=True)
        self.model.load_state_dict(checkpoint.get('model', checkpoint), strict=True)
        self.model.eval()
        print(f"Loaded weights from {weights_path}")
