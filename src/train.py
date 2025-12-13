"""
Complete Training and Inference Pipeline for Multi-Modal 3D Detection
Includes data loading, training loop, evaluation, and visualization
Now supports loading configuration from config.yaml
"""

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
from PIL import Image
import sys
#import os
import torchvision.transforms as T

from fusion_detection import decode_centernet_predictions
from fusion_detection import MultiModal3DDetector, DetectionLoss

def load_config(config_path: str = 'config.yaml') -> Dict:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to config.yaml file
    
    Returns:
        Configuration dictionary
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


# ============================================================================
# 1. DATASET
# ============================================================================

class NuScenesDataset(Dataset):
    """
    NuScenes dataset for multi-modal 3D detection
    Loads camera, LiDAR, and radar data from converted pickle files
    Can be initialized from config.yaml or with direct parameters
    """
    
    def __init__(
        self,
        data_root: Optional[str] = None,
        split: Optional[str] = None,
        max_points: Optional[int] = None,
        max_radar_points: Optional[int] = None,
        config: Optional[Dict] = None,
        config_path: Optional[str] = None
    ):
        super(NuScenesDataset, self).__init__()
        
        # Load from config if provided
        if config is not None or config_path is not None:
            if config is None:
                config = load_config(config_path)
            
            dataset_cfg = config.get('dataset', {})
            data_cfg = config.get('data', {})
            
            self.data_root = Path(data_cfg.get('data_root', './data/nuscenes')) if data_root is None else Path(data_root)
            self.split = split if split is not None else 'train'
            self.max_points = dataset_cfg.get('max_lidar_points', 35000) if max_points is None else max_points
            self.max_radar_points = dataset_cfg.get('max_radar_points', 125) if max_radar_points is None else max_radar_points
        else:
            # Use direct parameters
            self.data_root = Path(data_root) if data_root is not None else Path('./data/nuscenes')
            self.split = split if split is not None else 'train'
            self.max_points = max_points if max_points is not None else 35000
            self.max_radar_points = max_radar_points if max_radar_points is not None else 125
        
        # Load pickle file
        pkl_path = self.data_root / f'nuscenes_infos_{self.split}.pkl'
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        self.infos = data['infos']
        self.classes = data['metadata']['classes']
        
        print(f"Loaded {len(self.infos)} samples for {self.split} split")
    
    def __len__(self) -> int:
        return len(self.infos)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single sample
        
        Returns:
            Dict with:
                - camera_imgs: (6, 3, H, W)
                - lidar_points: (N, 5)
                - radar_points: List of 5 × (N_i, 7)
                - gt_boxes: (M, 7)
                - gt_labels: (M,)
                - gt_velocities: (M, 2)
        """
        info = self.infos[idx]
        
        # Load camera images
        camera_imgs = self._load_camera_images(info)
        
        # Load LiDAR points
        lidar_points = self._load_lidar_points(info)
        
        # Load radar points
        radar_points = self._load_radar_points(info)
        
        # Get ground truth
        gt_boxes = torch.from_numpy(info['gt_boxes']).float()
        gt_labels = self._encode_labels(info['gt_names'])
        gt_velocities = torch.from_numpy(info['gt_velocity']).float()
        
        return {
            'camera_imgs': camera_imgs,
            'lidar_points': lidar_points,
            'radar_points': radar_points,
            'gt_boxes': gt_boxes,
            'gt_labels': gt_labels,
            'gt_velocities': gt_velocities,
            'token': info['token']
        }
    
    def _load_camera_images(self, info: Dict) -> torch.Tensor:
        """Load and preprocess camera images"""
        
        
        transform = T.Compose([
            T.Resize((448, 800)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        cameras = []
        cam_order = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
                     'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
        
        for cam_name in cam_order:
            cam_path = self.data_root / info['cams'][cam_name]['filename']
            img = Image.open(cam_path).convert('RGB')
            img_tensor = transform(img)
            cameras.append(img_tensor)
        
        return torch.stack(cameras)  # (6, 3, 448, 800)
    
    def _load_lidar_points(self, info: Dict) -> torch.Tensor:
        """Load and preprocess LiDAR points"""
        lidar_path = info['lidar_path']
        points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)
        
        # Filter points in range
        mask = (points[:, 0] > -51.2) & (points[:, 0] < 51.2) & \
               (points[:, 1] > -51.2) & (points[:, 1] < 51.2) & \
               (points[:, 2] > -5.0) & (points[:, 2] < 3.0)
        points = points[mask]
        
        # Pad or subsample
        points = self._pad_or_subsample(points, self.max_points)
        
        return torch.from_numpy(points).float()
    
    def _load_radar_points(self, info: Dict) -> List[torch.Tensor]:
        """Load and preprocess radar points"""
        radar_order = ['RADAR_FRONT', 'RADAR_FRONT_LEFT', 'RADAR_FRONT_RIGHT',
                       'RADAR_BACK_LEFT', 'RADAR_BACK_RIGHT']
        
        radar_list = []
        for radar_name in radar_order:
            radar_path = self.data_root / info['radars'][radar_name]['filename']
            
            # Load radar points (this is simplified - actual loading depends on format)
            # For this example, we'll create dummy data
            points = np.random.randn(self.max_radar_points, 7).astype(np.float32)
            
            radar_list.append(torch.from_numpy(points).float())
        
        return radar_list
    
    def _pad_or_subsample(self, points: np.ndarray, max_points: int) -> np.ndarray:
        """Pad or subsample points to fixed size"""
        N = points.shape[0]
        if N >= max_points:
            indices = np.random.choice(N, max_points, replace=False)
            return points[indices]
        else:
            padding = np.zeros((max_points - N, points.shape[1]), dtype=points.dtype)
            return np.concatenate([points, padding], axis=0)
    
    def _encode_labels(self, names: np.ndarray) -> torch.Tensor:
        """Encode class names to indices"""
        label_map = {name: i for i, name in enumerate(self.classes)}
        labels = [label_map.get(name, -1) for name in names]
        return torch.tensor(labels, dtype=torch.long)


def collate_fn(batch: List[Dict]) -> Dict:
    """Custom collate function for batching"""
    camera_imgs = torch.stack([item['camera_imgs'] for item in batch])
    lidar_points = torch.stack([item['lidar_points'] for item in batch])
    
    # Radar points - list of lists
    radar_points = [
        torch.stack([batch[b]['radar_points'][r] for b in range(len(batch))])
        for r in range(5)
    ]
    
    # Ground truth (pad to max objects in batch)
    max_objs = max(len(item['gt_boxes']) for item in batch)
    
    gt_boxes = []
    gt_labels = []
    gt_velocities = []
    
    for item in batch:
        n_objs = len(item['gt_boxes'])
        if n_objs < max_objs:
            pad_size = max_objs - n_objs
            boxes = torch.cat([item['gt_boxes'], 
                              torch.zeros(pad_size, 7)], dim=0)
            labels = torch.cat([item['gt_labels'],
                               torch.full((pad_size,), -1, dtype=torch.long)], dim=0)
            vels = torch.cat([item['gt_velocities'],
                             torch.zeros(pad_size, 2)], dim=0)
        else:
            boxes = item['gt_boxes']
            labels = item['gt_labels']
            vels = item['gt_velocities']
        
        gt_boxes.append(boxes)
        gt_labels.append(labels)
        gt_velocities.append(vels)
    
    return {
        'camera_imgs': camera_imgs,
        'lidar_points': lidar_points,
        'radar_points': radar_points,
        'gt_boxes': torch.stack(gt_boxes),
        'gt_labels': torch.stack(gt_labels),
        'gt_velocities': torch.stack(gt_velocities),
        'tokens': [item['token'] for item in batch]
    }


# ============================================================================
# 2. TRAINING
# ============================================================================

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int
) -> Dict[str, float]:
    """Train for one epoch"""
    model.train()
    
    total_loss = 0.0
    loss_dict = {}
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(pbar):
        # Move to device
        camera_imgs = batch['camera_imgs'].to(device)
        lidar_points = batch['lidar_points'].to(device)
        radar_points = [r.to(device) for r in batch['radar_points']]
        
        # Forward pass
        predictions = model(camera_imgs, lidar_points, radar_points)
        
        # Prepare targets (this is simplified)
        targets = prepare_targets(batch, device)
        
        # Compute loss
        losses = criterion(predictions, targets)
        loss = losses['total_loss']
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        
        optimizer.step()
        
        # Track losses
        total_loss += loss.item()
        for k, v in losses.items():
            if k not in loss_dict:
                loss_dict[k] = 0.0
            loss_dict[k] += v.item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': loss.item(),
            'avg_loss': total_loss / (batch_idx + 1)
        })
    
    # Average losses
    avg_loss_dict = {k: v / len(dataloader) for k, v in loss_dict.items()}
    
    return avg_loss_dict


def prepare_targets(batch: Dict, device: torch.device) -> Dict[str, torch.Tensor]:
    """
    Prepare ground truth targets for loss computation
    This is a simplified version - actual implementation depends on detection head
    """
    # For CenterNet-style head, we need to create heatmaps, offsets, etc.
    # This is a placeholder
    
    B = batch['camera_imgs'].shape[0]
    H, W = 200, 200  # BEV size
    num_classes = 10
    
    heatmap = torch.zeros(B, num_classes, H, W, device=device)
    offset = torch.zeros(B, 2, H, W, device=device)
    size = torch.zeros(B, 3, H, W, device=device)
    rot = torch.zeros(B, 2, H, W, device=device)
    vel = torch.zeros(B, 2, H, W, device=device)
    mask = torch.zeros(B, 1, H, W, device=device)
    
    # TODO: Implement actual target generation from gt_boxes
    # This would involve projecting 3D boxes to BEV and creating Gaussian heatmaps
    
    return {
        'heatmap': heatmap,
        'offset': offset,
        'size': size,
        'rot': rot,
        'vel': vel,
        'mask': mask
    }


# ============================================================================
# 3. EVALUATION
# ============================================================================

@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> Dict[str, float]:
    """Evaluate model on validation set"""
    model.eval()
    
    all_predictions = []
    all_ground_truths = []
    
    for batch in tqdm(dataloader, desc="Evaluating"):
        camera_imgs = batch['camera_imgs'].to(device)
        lidar_points = batch['lidar_points'].to(device)
        radar_points = [r.to(device) for r in batch['radar_points']]
        
        # Forward pass
        predictions = model(camera_imgs, lidar_points, radar_points)
        
        # Decode predictions (if using CenterNet)
        
        decoded = decode_centernet_predictions(predictions)
        
        all_predictions.extend(decoded)
        all_ground_truths.append(batch)
    
    # Compute metrics (AP, mAP, etc.)
    metrics = compute_metrics(all_predictions, all_ground_truths)
    
    return metrics


def compute_metrics(predictions: List, ground_truths: List) -> Dict[str, float]:
    """
    Compute detection metrics (AP, mAP)
    This is a placeholder - actual implementation requires NuScenes evaluation code
    """
    # TODO: Implement actual metric computation
    # This would use NuScenes devkit for official evaluation
    
    return {
        'mAP': 0.0,
        'NDS': 0.0,
        'mATE': 0.0,
        'mASE': 0.0,
        'mAOE': 0.0
    }


# ============================================================================
# 4. MAIN TRAINING SCRIPT
# ============================================================================

def main(config_path: Optional[str] = None):
    """
    Main training function
    Can load configuration from config.yaml or use defaults
    
    Args:
        config_path: Path to config.yaml file (optional)
    """
    
    # Load configuration
    if config_path is not None:
        print(f"Loading configuration from {config_path}")
        cfg = load_config(config_path)
        
        # Extract training configuration
        data_cfg = cfg.get('data', {})
        train_cfg = cfg.get('training', {})
        model_cfg = cfg.get('model', {})
        dataset_cfg = cfg.get('dataset', {})
        
        config = {
            'data_root': data_cfg.get('data_root', './data/nuscenes'),
            'batch_size': train_cfg.get('batch_size', 4),
            'num_epochs': train_cfg.get('num_epochs', 50),
            'lr': train_cfg.get('learning_rate', 1e-4),
            'weight_decay': train_cfg.get('weight_decay', 0.01),
            'num_workers': data_cfg.get('num_workers', 4),
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'fusion_type': model_cfg.get('fusion_type', 'bev'),
            'detection_head': model_cfg.get('detection_head', 'centernet'),
            'num_classes': dataset_cfg.get('num_classes', 10),
            'checkpoint_dir': train_cfg.get('checkpoint_dir', './checkpoints'),
            'log_interval': train_cfg.get('log_interval', 10),
            'save_interval': train_cfg.get('save_interval', 5)
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
            'detection_head': 'centernet',
            'num_classes': 10,
            'checkpoint_dir': './checkpoints',
            'log_interval': 10,
            'save_interval': 5
        }
    
    print("="*80)
    print("Multi-Modal 3D Object Detection Training")
    print("="*80)
    print(f"\nConfiguration:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    print()
    
    # Create checkpoint directory
    Path(config['checkpoint_dir']).mkdir(parents=True, exist_ok=True)
    
    # Device
    device = torch.device(config['device'])
    print(f"Using device: {device}")
    
    # Create datasets
    if config_path is not None:
        train_dataset = NuScenesDataset(
            split='train',
            config_path=config_path
        )
        
        val_dataset = NuScenesDataset(
            split='val',
            config_path=config_path
        )
    else:
        train_dataset = NuScenesDataset(
            data_root=config['data_root'],
            split='train'
        )
        
        val_dataset = NuScenesDataset(
            data_root=config['data_root'],
            split='val'
        )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        collate_fn=collate_fn,
        pin_memory=True
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
    
    
    if config_path is not None:
        model = MultiModal3DDetector(config_path=config_path).to(device)
    else:
        model = MultiModal3DDetector(
            num_classes=config['num_classes'],
            fusion_type=config['fusion_type'],
            detection_head=config['detection_head']
        ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['num_epochs'],
        eta_min=1e-6
    )
    
    # Loss function
    criterion = DetectionLoss()
    
    # Training loop
    best_map = 0.0
    
    print("\n" + "="*80)
    print("Starting Training")
    print("="*80 + "\n")
    
    for epoch in range(1, config['num_epochs'] + 1):
        print(f"\nEpoch {epoch}/{config['num_epochs']}")
        print("-" * 80)
        
        # Train
        train_losses = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )
        
        print(f"\nTraining losses:")
        for k, v in train_losses.items():
            print(f"  {k}: {v:.4f}")
        
        # Validate
        if epoch % config['log_interval'] == 0:
            print("\nRunning validation...")
            val_metrics = evaluate(model, val_loader, device)
            
            print(f"\nValidation metrics:")
            for k, v in val_metrics.items():
                print(f"  {k}: {v:.4f}")
            
            # Save best model
            if val_metrics['mAP'] > best_map:
                best_map = val_metrics['mAP']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_map': best_map,
                    'config': config
                }, Path(config['checkpoint_dir']) / 'best_model.pth')
                print(f"\n✓ Saved best model (mAP: {best_map:.4f})")
        
        # Save checkpoint
        if epoch % config['save_interval'] == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config
            }, Path(config['checkpoint_dir']) / f'checkpoint_epoch_{epoch}.pth')
        
        # Update learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f"\nLearning rate: {current_lr:.6f}")
    
    print("\n" + "="*80)
    print("Training Complete!")
    print("="*80)
    print(f"\nBest mAP: {best_map:.4f}")


# ============================================================================
# 5. INFERENCE
# ============================================================================

@torch.no_grad()
def inference(
    model_path: str,
    data_root: str,
    sample_idx: int = 0,
    device: str = 'cuda'
):
    """Run inference on a single sample"""
    
    print("="*80)
    print("Running Inference")
    print("="*80)
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']
    
    
    model = MultiModal3DDetector(
        num_classes=config['num_classes'],
        fusion_type=config['fusion_type'],
        detection_head=config['detection_head']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Loaded model from {model_path}")
    print(f"Epoch: {checkpoint['epoch']}")
    
    # Load dataset
    dataset = NuScenesDataset(data_root=data_root, split='val')
    sample = dataset[sample_idx]
    
    # Prepare inputs
    camera_imgs = sample['camera_imgs'].unsqueeze(0).to(device)
    lidar_points = sample['lidar_points'].unsqueeze(0).to(device)
    radar_points = [r.unsqueeze(0).to(device) for r in sample['radar_points']]
    
    # Forward pass
    predictions = model(camera_imgs, lidar_points, radar_points)
    
    # Decode predictions
    
    detections = decode_centernet_predictions(predictions, score_thresh=0.3)[0]
    
    print(f"\nDetected {len(detections['boxes'])} objects:")
    for i, (box, score, label) in enumerate(zip(
        detections['boxes'], 
        detections['scores'],
        detections['labels']
    )):
        class_name = dataset.classes[label.item()]
        print(f"\n  Object {i+1}:")
        print(f"    Class: {class_name}")
        print(f"    Score: {score:.3f}")
        print(f"    Box: {box.cpu().numpy()}")
    
    return detections


if __name__ == '__main__':
    
    
    if len(sys.argv) > 1 and sys.argv[1] == 'train':
        # Check if config path is provided
        config_path = sys.argv[2] if len(sys.argv) > 2 else None
        main(config_path)
    elif len(sys.argv) > 1 and sys.argv[1] == 'infer':
        model_path = sys.argv[2] if len(sys.argv) > 2 else './checkpoints/best_model.pth'
        inference(model_path, './data/nuscenes')
    else:
        print("Usage:")
        print("  python train_detection.py train                 # Train with default config")
        print("  python train_detection.py train config.yaml     # Train with config file")
        print("  python train_detection.py infer [path]          # Run inference")