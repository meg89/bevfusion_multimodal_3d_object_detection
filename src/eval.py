
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional
import pickle
from pathlib import Path
from tqdm import tqdm
import yaml
import os
from PIL import Image
import torchvision.transforms as T
import torch.nn.functional as F

from utils_v2 import compute_metrics, save_and_print_metrics
from fusion_detection import decode_centernet_predictions, DetectionLoss
import sys
from fusion import (
    load_config,
    create_detector
)

from train_detect import NuScenesDataset, collate_fn

@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    use_camera: bool = True,
    use_lidar: bool = True,
    use_radar: bool = True
) -> Dict[str, float]:
    """Evaluate model on validation set"""
    model.eval()
    
    all_predictions = []
    all_ground_truths = []

    # Get point cloud range from dataloader if not provided
    #if pc_range is None:
    #    pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]  # Default NuScenes range
    
    
    for batch in tqdm(dataloader, desc="Evaluating"):
        camera_imgs = batch['camera_imgs'].to(device) if use_camera else None
        lidar_points = batch['lidar_points'].to(device) if use_lidar else None
        radar_points = [r.to(device) for r in batch['radar_points']] if use_radar else None
        
        # Forward pass
        predictions = model(camera_imgs, lidar_points, radar_points)
 
        # Decode predictions (if CenterNet-style)
        if 'heatmap' in predictions:
            #K=100
            decoded = decode_centernet_predictions(
                predictions,
                score_thresh=0.0, #0.3  # Don't filter yet
                max_detections=100  # Max detections per image
            )
            all_predictions.extend(decoded)
        else:
            batch_size = predictions['cls'].size(0) if 'cls' in predictions else len(batch['gt_boxes'])
            
            for i in range(batch_size):
                if 'cls' in predictions and 'box' in predictions:
                    # MLP predictions
                    cls_scores = F.softmax(predictions['cls'][i], dim=-1)
                    scores, labels = cls_scores.max(dim=-1)
                    boxes = predictions['box'][i]
                    
                    decoded_pred = {
                        'boxes': boxes.cpu().numpy(),
                        'scores': scores.cpu().numpy(),
                        'labels': labels.cpu().numpy()
                    }
                else:
                    # Unknown format - create empty prediction
                    import numpy as np
                    decoded_pred = {
                        'boxes': np.zeros((0, 7)),
                        'scores': np.zeros(0),
                        'labels': np.zeros(0, dtype=np.int64)
                    }
                
                all_predictions.append(decoded_pred)
        # Store ground truths
        for i in range(len(batch['gt_boxes'])):
            gt_dict = {
                'boxes': batch['gt_boxes'][i].cpu().numpy(),
                'labels': batch['gt_labels'][i].cpu().numpy()
            }
            all_ground_truths.append(gt_dict)
        
        #all_ground_truths.append(batch)

    print(f"predictions: ", all_predictions[:1] )
    print(f"all_ground_truths: ", all_ground_truths[:1] )
    # Print the result
    #for entry in all_predictions[:1]:
    #    print("prediction: ", entry)
    
    # Compute metrics (AP, mAP, etc.)
    # Prepare modality info
    #for entry in all_ground_truths[:1]:
    #    print("prediction: ", entry)
    metrics = compute_metrics(all_predictions, all_ground_truths)
    
    return metrics  


def main(config_path: Optional[str] = None):
    """
    Main training function
    Can load configuration from config.yaml or use defaults
    
    Args:
        config_path: Path to config.yaml file (optional)
    """
    model_path = './checkpoints/best_model.pth'
    #model_path = './checkpoints/checkpoint_epoch_1.pth'
    # Load configuration
    if config_path is not None:
        print(f"Loading configuration from {config_path}")
        cfg = load_config(config_path)
        
        # Extract training configuration
        data_cfg = cfg.get('dataset', {})
        train_cfg = cfg.get('train', {})
        model_cfg = cfg.get('model', {})
        #dataset_cfg = cfg.get('dataset', {})
        
        config = {
            'data_root': data_cfg.get('data_root', './data/nuscenes'),
            'batch_size': train_cfg.get('batch_size', 4),
            'num_epochs': train_cfg.get('num_epochs', 50),
            'lr': train_cfg.get('learning_rate', 1e-4),
            'weight_decay': train_cfg.get('weight_decay', 0.01),
            'num_workers': data_cfg.get('num_workers', 4),
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'fusion_type': model_cfg.get('fusion_type', 'attention'),
            'detection_head': model_cfg.get('detection_head', 'mlp'),
            'num_classes': data_cfg.get('num_classes', 10),
            'checkpoint_dir': train_cfg.get('checkpoint_dir', './checkpoints'),
            'log_interval': train_cfg.get('log_interval', 1),
            'save_interval': train_cfg.get('save_interval', 1),
            # Modality configuration (defaults)
            'use_camera': model_cfg.get('use_camera', True) ,
            'use_lidar': model_cfg.get('use_lidar', True),
            'use_radar': model_cfg.get('use_radar', True) 
        }
    else:
        # Default configuration
        config = {
            'data_root': './data/nuscenes',
            'batch_size': 4,
            'num_epochs': 1,
            'lr': 1e-4,
            'weight_decay': 0.01,
            'num_workers': 4,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'fusion_type': 'attention',
            'detection_head': 'mlp',
            'num_classes': 10,
            'checkpoint_dir': './checkpoints',
            'log_interval': 10,
            'save_interval': 5
        }
    
    print("="*80)
    print("Multi-Modal 3D Object Detection Evaluation")
    print("="*80)
    print(f"\nConfiguration:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    print()
    
    # Device
    device = torch.device(config['device'])
    print(f"Using device: {device}")
    
    # Create datasets
    if config_path is not None:
        val_dataset = NuScenesDataset(
            split='val',
            config_path=config_path
        )
    else:
        val_dataset = NuScenesDataset(
            data_root=config['data_root'],
            split='val'
        )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # Create model
    #if config_path is not None:
        
    checkpoint = torch.load(model_path, map_location=device)
    model = create_detector(config_path='configs/base.yaml')
    model.load_state_dict(checkpoint['model_state_dict'],strict=False)
    model = model.to(device)
    model.eval()
    print(f"Loaded model from {model_path}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    
    print("\nRunning validation...")
    save_dir = "eval_results"
    os.makedirs(save_dir, exist_ok=True)
    val_metrics = evaluate(model, val_loader, device)
    file_path = os.path.join(save_dir, "eval_metrics_output.txt")
    save_and_print_metrics(val_metrics, file_path )
             
    print("\n" + "="*80)
    print("Training Complete!")
    print("="*80)

if __name__ == '__main__':
    config_path = sys.argv[2] if len(sys.argv) > 2 else None
    main(config_path)   
            
