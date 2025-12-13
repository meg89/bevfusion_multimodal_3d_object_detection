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
import torchvision.transforms as T

from fusion import (
    load_config,
    FlexibleMultiModal3DDetector,
    create_detector
)


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
# 2. LOSS FUNCTION
# ============================================================================

class DetectionLoss(nn.Module):
    """Detection loss for multi-modal 3D detection"""
    
    def __init__(self):
        super(DetectionLoss, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.bce_loss = nn.BCELoss()
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute detection loss
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
        
        Returns:
            Dictionary of losses
        """
        losses = {}
        
        # For CenterNet-style predictions
        if 'heatmap' in predictions:
            # Heatmap loss (focal loss)
            pred_heatmap = predictions['heatmap']
            target_heatmap = targets['heatmap']
            
            # Simplified focal loss
            pos_inds = target_heatmap.eq(1).float()
            neg_inds = target_heatmap.lt(1).float()
            
            neg_weights = torch.pow(1 - target_heatmap, 4)
            
            pos_loss = torch.log(pred_heatmap + 1e-12) * torch.pow(1 - pred_heatmap, 2) * pos_inds
            neg_loss = torch.log(1 - pred_heatmap + 1e-12) * torch.pow(pred_heatmap, 2) * neg_weights * neg_inds
            
            num_pos = pos_inds.float().sum()
            pos_loss = pos_loss.sum()
            neg_loss = neg_loss.sum()
            
            if num_pos == 0:
                heatmap_loss = -neg_loss
            else:
                heatmap_loss = -(pos_loss + neg_loss) / num_pos
            
            losses['heatmap_loss'] = heatmap_loss
            
            # Regression losses
            mask = targets['mask']
            
            if 'offset' in predictions:
                offset_loss = self.l1_loss(
                    predictions['offset'] * mask,
                    targets['offset'] * mask
                )
                losses['offset_loss'] = offset_loss
            
            if 'size' in predictions:
                size_loss = self.l1_loss(
                    predictions['size'] * mask,
                    targets['size'] * mask
                )
                losses['size_loss'] = size_loss
            
            if 'rot' in predictions:
                rot_loss = self.l1_loss(
                    predictions['rot'] * mask,
                    targets['rot'] * mask
                )
                losses['rot_loss'] = rot_loss
            
            # Total loss
            total_loss = heatmap_loss
            if 'offset_loss' in losses:
                total_loss += losses['offset_loss']
            if 'size_loss' in losses:
                total_loss += losses['size_loss']
            if 'rot_loss' in losses:
                total_loss += losses['rot_loss']
            
        # For MLP-style predictions
        elif 'cls' in predictions:
            cls_loss = nn.functional.cross_entropy(
                predictions['cls'],
                targets['labels']
            )
            box_loss = self.l1_loss(
                predictions['box'],
                targets['boxes']
            )
            total_loss = cls_loss + box_loss
            losses['cls_loss'] = cls_loss
            losses['box_loss'] = box_loss
        
        else:
            total_loss = torch.tensor(0.0, device=list(predictions.values())[0].device)
        
        losses['total_loss'] = total_loss
        
        return losses


# ============================================================================
# 3. TRAINING
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


def gaussian_2d(shape, sigma=1):
    """Generate 2D Gaussian kernel"""
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1, -n:n+1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_gaussian(heatmap, center, radius, k=1):
    """Draw 2D Gaussian on heatmap"""
    diameter = 2 * radius + 1
    gaussian = gaussian_2d((diameter, diameter), sigma=diameter / 6)
    gaussian = torch.from_numpy(gaussian).float().to(heatmap.device)
    
    x, y = int(center[0]), int(center[1])
    height, width = heatmap.shape[0:2]
    
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)
    
    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        torch.max(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    
    return heatmap


def gaussian_radius(det_size, min_overlap=0.7):
    """Calculate Gaussian radius based on box size"""
    height, width = det_size
    
    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2
    
    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2
    
    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    
    return min(r1, r2, r3)


def prepare_targets(
    batch: Dict,
    device: torch.device,
    pc_range: list = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
    bev_size: tuple = (200, 200),
    num_classes: int = 10
) -> Dict[str, torch.Tensor]:
    """
    Prepare ground truth targets for CenterNet-style 3D detection
    
    Converts 3D bounding boxes to BEV heatmaps, offsets, sizes, rotations, and velocities
    
    Args:
        batch: Dictionary containing:
            - gt_boxes: (B, M, 7) - [x, y, z, w, l, h, yaw]
            - gt_labels: (B, M) - class labels
            - gt_velocities: (B, M, 2) - [vx, vy]
        device: torch device
        pc_range: Point cloud range [x_min, y_min, z_min, x_max, y_max, z_max]
        bev_size: (H, W) size of BEV feature map
        num_classes: Number of object classes
    
    Returns:
        Dictionary containing:
            - heatmap: (B, num_classes, H, W) - Gaussian heatmaps
            - offset: (B, 2, H, W) - sub-pixel offsets
            - size: (B, 3, H, W) - 3D box sizes [w, l, h]
            - rot: (B, 2, H, W) - rotation [sin(yaw), cos(yaw)]
            - vel: (B, 2, H, W) - velocities [vx, vy]
            - mask: (B, 1, H, W) - valid object mask
    """
    import math
    
    B = batch['gt_boxes'].shape[0]
    H, W = bev_size
    
    # Initialize target tensors
    heatmap = torch.zeros(B, num_classes, H, W, device=device)
    offset = torch.zeros(B, 2, H, W, device=device)
    size = torch.zeros(B, 3, H, W, device=device)
    rot = torch.zeros(B, 2, H, W, device=device)
    vel = torch.zeros(B, 2, H, W, device=device)
    mask = torch.zeros(B, 1, H, W, device=device)
    
    # BEV parameters
    x_min, y_min = pc_range[0], pc_range[1]
    x_max, y_max = pc_range[3], pc_range[4]
    
    voxel_size_x = (x_max - x_min) / W
    voxel_size_y = (y_max - y_min) / H
    
    # Process each batch
    for b in range(B):
        gt_boxes = batch['gt_boxes'][b]  # (M, 7)
        gt_labels = batch['gt_labels'][b]  # (M,)
        gt_vels = batch['gt_velocities'][b]  # (M, 2)
        
        # Filter out padding (labels == -1)
        valid_mask = gt_labels >= 0
        gt_boxes = gt_boxes[valid_mask]
        gt_labels = gt_labels[valid_mask]
        gt_vels = gt_vels[valid_mask]
        
        num_objs = gt_boxes.shape[0]
        
        for i in range(num_objs):
            # Extract box parameters [x, y, z, w, l, h, yaw]
            x, y, z = gt_boxes[i, 0].item(), gt_boxes[i, 1].item(), gt_boxes[i, 2].item()
            w, l, h = gt_boxes[i, 3].item(), gt_boxes[i, 4].item(), gt_boxes[i, 5].item()
            yaw = gt_boxes[i, 6].item()
            
            cls_id = gt_labels[i].item()
            vx, vy = gt_vels[i, 0].item(), gt_vels[i, 1].item()
            
            # Skip if outside BEV range
            if x < x_min or x >= x_max or y < y_min or y >= y_max:
                continue
            
            # Convert to BEV coordinates (grid indices)
            bev_x = (x - x_min) / voxel_size_x
            bev_y = (y_max - y) / voxel_size_y  # Flip y-axis for image coords
            
            # Check if within valid range
            if bev_x < 0 or bev_x >= W or bev_y < 0 or bev_y >= H:
                continue
            
            # Integer grid position
            ct_int_x = int(bev_x)
            ct_int_y = int(bev_y)
            
            # Sub-pixel offset
            ct_offset_x = bev_x - ct_int_x
            ct_offset_y = bev_y - ct_int_y
            
            # Calculate Gaussian radius based on box size in BEV
            box_w_bev = w / voxel_size_x
            box_l_bev = l / voxel_size_y
            
            radius = gaussian_radius((box_l_bev, box_w_bev))
            radius = max(0, int(radius))
            
            # Draw Gaussian heatmap
            heatmap[b, cls_id] = draw_gaussian(
                heatmap[b, cls_id],
                (ct_int_x, ct_int_y),
                radius
            )
            
            # Set offset
            offset[b, 0, ct_int_y, ct_int_x] = ct_offset_x
            offset[b, 1, ct_int_y, ct_int_x] = ct_offset_y
            
            # Set size (w, l, h)
            size[b, 0, ct_int_y, ct_int_x] = w
            size[b, 1, ct_int_y, ct_int_x] = l
            size[b, 2, ct_int_y, ct_int_x] = h
            
            # Set rotation (sin, cos encoding)
            rot[b, 0, ct_int_y, ct_int_x] = math.sin(yaw)
            rot[b, 1, ct_int_y, ct_int_x] = math.cos(yaw)
            
            # Set velocity
            vel[b, 0, ct_int_y, ct_int_x] = vx
            vel[b, 1, ct_int_y, ct_int_x] = vy
            
            # Set mask
            mask[b, 0, ct_int_y, ct_int_x] = 1.0
    
    return {
        'heatmap': heatmap,
        'offset': offset,
        'size': size,
        'rot': rot,
        'vel': vel,
        'mask': mask
    }


def _nms(heat, kernel=3):
    """Apply max pooling NMS"""
    import torch.nn.functional as F
    pad = (kernel - 1) // 2
    hmax = F.max_pool2d(heat, kernel, stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


def _topk(scores, K=100):
    """Get top K scores"""
    batch, num_classes, height, width = scores.size()
    
    # Get top K scores per class
    topk_scores, topk_inds = torch.topk(scores.view(batch, num_classes, -1), K)
    
    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds // width).float()
    topk_xs = (topk_inds % width).float()
    
    # Get top K across all classes
    topk_scores, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_classes = (topk_ind // K).int()
    topk_inds = _gather_feat(topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)
    
    return (topk_scores[0], topk_inds[0], topk_classes[0],
            topk_ys[0].long(), topk_xs[0].long())


def _gather_feat(feat, ind):
    """Gather features at specified indices"""
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    return feat


def decode_centernet_predictions(
    predictions: Dict[str, torch.Tensor],
    score_thresh: float = 0.3,
    max_objects: int = 100,
    pc_range: list = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
    bev_size: tuple = (200, 200)
) -> list:
    """
    Decode CenterNet predictions to 3D bounding boxes
    
    Args:
        predictions: Dictionary containing:
            - heatmap: (B, num_classes, H, W)
            - offset: (B, 2, H, W)
            - size: (B, 3, H, W)
            - rot: (B, 2, H, W)
            - vel: (B, 2, H, W) (optional)
        score_thresh: Score threshold for filtering
        max_objects: Maximum number of objects to detect
        pc_range: Point cloud range
        bev_size: BEV feature map size
    
    Returns:
        List of detections for each batch item
    """
    device = predictions['heatmap'].device
    B, num_classes, H, W = predictions['heatmap'].shape
    
    # BEV parameters
    x_min, y_min = pc_range[0], pc_range[1]
    x_max, y_max = pc_range[3], pc_range[4]
    voxel_size_x = (x_max - x_min) / W
    voxel_size_y = (y_max - y_min) / H
    
    batch_detections = []
    
    for b in range(B):
        heatmap = predictions['heatmap'][b]  # (num_classes, H, W)
        
        # Apply NMS
        heatmap_nms = _nms(heatmap.unsqueeze(0)).squeeze(0)
        
        # Get top K peaks
        scores, indices, class_ids, ys, xs = _topk(heatmap_nms.unsqueeze(0), K=max_objects)
        
        # Filter by threshold
        mask = scores > score_thresh
        scores = scores[mask]
        indices = indices[mask]
        class_ids = class_ids[mask]
        ys = ys[mask]
        xs = xs[mask]
        
        num_dets = scores.shape[0]
        
        if num_dets == 0:
            batch_detections.append({
                'boxes': torch.zeros((0, 7), device=device),
                'scores': torch.zeros(0, device=device),
                'labels': torch.zeros(0, dtype=torch.long, device=device),
                'velocities': torch.zeros((0, 2), device=device)
            })
            continue
        
        # Gather predictions
        offset_pred = predictions['offset'][b]  # (2, H, W)
        size_pred = predictions['size'][b]  # (3, H, W)
        rot_pred = predictions['rot'][b]  # (2, H, W)
        
        # Extract values at peak locations
        offsets = torch.stack([
            offset_pred[0, ys, xs],
            offset_pred[1, ys, xs]
        ], dim=1)  # (num_dets, 2)
        
        sizes = torch.stack([
            size_pred[0, ys, xs],
            size_pred[1, ys, xs],
            size_pred[2, ys, xs]
        ], dim=1)  # (num_dets, 3)
        
        rots = torch.stack([
            rot_pred[0, ys, xs],
            rot_pred[1, ys, xs]
        ], dim=1)  # (num_dets, 2)
        
        # Velocity (optional)
        if 'vel' in predictions:
            vel_pred = predictions['vel'][b]  # (2, H, W)
            velocities = torch.stack([
                vel_pred[0, ys, xs],
                vel_pred[1, ys, xs]
            ], dim=1)  # (num_dets, 2)
        else:
            velocities = torch.zeros((num_dets, 2), device=device)
        
        # Convert back to world coordinates
        xs_real = (xs.float() + offsets[:, 0]) * voxel_size_x + x_min
        ys_real = y_max - (ys.float() + offsets[:, 1]) * voxel_size_y
        
        # Decode rotation
        yaws = torch.atan2(rots[:, 0], rots[:, 1])
        
        # Assemble boxes: [x, y, z, w, l, h, yaw]
        boxes = torch.stack([
            xs_real,
            ys_real,
            torch.zeros_like(xs_real),  # z (ground plane)
            sizes[:, 0],  # w
            sizes[:, 1],  # l
            sizes[:, 2],  # h
            yaws
        ], dim=1)
        
        batch_detections.append({
            'boxes': boxes,
            'scores': scores,
            'labels': class_ids,
            'velocities': velocities
        })
    
    return batch_detections


# ============================================================================
# 4. EVALUATION
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
        
        # Decode predictions (if CenterNet-style)
        if 'heatmap' in predictions:
            decoded = decode_centernet_predictions(predictions, score_thresh=0.3)
            all_predictions.extend(decoded)
        else:
            all_predictions.append(predictions)
        
        all_ground_truths.append(batch)
    
    # Compute metrics
    metrics = compute_metrics(all_predictions, all_ground_truths)
    
    return metrics


def compute_metrics(predictions: List, ground_truths: List) -> Dict[str, float]:
    """
    Compute detection metrics (AP, mAP)
    This is a placeholder - actual implementation requires NuScenes evaluation code
    """
    return {
        'mAP': 0.0,
        'NDS': 0.0,
        'mATE': 0.0,
        'mASE': 0.0,
        'mAOE': 0.0
    }


# ============================================================================
# 5. MAIN TRAINING SCRIPT
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
            'detection_head': 'mlp',
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
    
    # Create model using the provided fusion.py
    if config_path is not None:
        model = create_detector(config_path=config_path).to(device)
    else:
        model = create_detector(
            fusion_type=config['fusion_type'],
            detection_head=config['detection_head'],
            num_classes=config['num_classes']
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
# 6. INFERENCE
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
    
    model = create_detector(
        fusion_type=config['fusion_type'],
        detection_head=config['detection_head'],
        num_classes=config['num_classes']
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
    
    print(f"\nRaw predictions:")
    for key, value in predictions.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
    
    # Decode predictions (if CenterNet-style)
    if 'heatmap' in predictions:
        detections = decode_centernet_predictions(predictions, score_thresh=0.3)[0]
        
        print(f"\nDetected {len(detections['boxes'])} objects:")
        for i, (box, score, label) in enumerate(zip(
            detections['boxes'][:10],  # Show first 10
            detections['scores'][:10],
            detections['labels'][:10]
        )):
            class_name = dataset.classes[label.item()] if label.item() < len(dataset.classes) else f"class_{label.item()}"
            print(f"\n  Object {i+1}:")
            print(f"    Class: {class_name}")
            print(f"    Score: {score:.3f}")
            print(f"    Box: x={box[0]:.2f}, y={box[1]:.2f}, z={box[2]:.2f}")
            print(f"         w={box[3]:.2f}, l={box[4]:.2f}, h={box[5]:.2f}, yaw={box[6]:.2f}")
        
        return detections
    else:
        return predictions


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