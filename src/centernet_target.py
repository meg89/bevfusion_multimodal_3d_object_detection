"""
Complete CenterNet Target Preparation for 3D Object Detection
Generates heatmaps, offsets, sizes, rotations, and velocities from 3D ground truth boxes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple
import math

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
    
def gaussian_2d(shape: Tuple[int, int], sigma: float = 1.0) -> np.ndarray:
    """Generate 2D Gaussian kernel"""
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1, -n:n+1]
    
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def gaussian_radius(det_size: Tuple[float, float], min_overlap: float = 0.7) -> float:
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

def draw_gaussian(heatmap: np.ndarray, center: Tuple[int, int], radius: float, k: float = 1.0):
    """Draw 2D Gaussian on heatmap"""
    diameter = 2 * radius + 1
    gaussian = gaussian_2d((diameter, diameter), sigma=diameter / 6)
    
    x, y = int(center[0]), int(center[1])
    
    height, width = heatmap.shape[0:2]
    
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)
    
    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)

def prepare_centernet_targets(
    batch: Dict,
    device: torch.device,
    pc_range: List[float] = None,
    bev_size: Tuple[int, int] = (50, 50), #(200, 200)   # full BEV grid
    num_classes: int = 10,
    max_objects: int = 500,
    gaussian_overlap: float = 0.7,
    min_radius: int = 2
) -> Dict[str, torch.Tensor]:
    """
    CenterNet-style target generator (Stride=4).
    Output heatmap size = bev_size / 4 = (50, 50) for 200x200 input.

    ✔ Keeps same interface
    ✔ Keeps same return values
    ✔ Fixes heatmap resolution
    ✔ Fixes target indexing
    ✔ Fixes Gaussian center mapping
    """

    if pc_range is None:
        pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

    # -------------------------------
    # Compute output heatmap size
    # -------------------------------
    #H, W = bev_size
    H, W = bev_size
    B = len(batch['gt_boxes'])

    # -------------------------------
    # Allocate target tensors
    # -------------------------------
    heatmap = torch.zeros(B, num_classes, H, W, device=device)
    offset  = torch.zeros(B, 2, H, W, device=device)
    size    = torch.zeros(B, 3, H, W, device=device)
    rot     = torch.zeros(B, 2, H, W, device=device)
    vel     = torch.zeros(B, 2, H, W, device=device)

    ind      = torch.zeros(B, max_objects, dtype=torch.long, device=device)
    mask     = torch.zeros(B, max_objects, dtype=torch.uint8, device=device)
    reg_mask = torch.zeros(B, max_objects, dtype=torch.uint8, device=device)

    target_offset = torch.zeros(B, max_objects, 2, device=device)
    target_size   = torch.zeros(B, max_objects, 3, device=device)
    target_rot    = torch.zeros(B, max_objects, 2, device=device)
    target_vel    = torch.zeros(B, max_objects, 2, device=device)

    # -------------------------------
    # Compute voxel size in meters
    # -------------------------------
    x_min, y_min, _, x_max, y_max, _ = pc_range
    voxel_x = (x_max - x_min) / W
    voxel_y = (y_max - y_min) / H

    # -------------------------------
    # Process each item in batch
    # -------------------------------
    for b in range(B):
        gt_boxes = batch['gt_boxes'][b]
        gt_labels = batch['gt_labels'][b]

        if isinstance(gt_boxes, torch.Tensor):
            gt_boxes = gt_boxes.cpu().numpy()
        if isinstance(gt_labels, torch.Tensor):
            gt_labels = gt_labels.cpu().numpy()

        num_objs = min(len(gt_boxes), max_objects)

        for k in range(num_objs):
            cls_id = int(gt_labels[k])
            if cls_id < 0 or cls_id >= num_classes:
                continue

            x, y, z, w, l, h, yaw = gt_boxes[k][:7]

            # ----------------------------------------------
            # Convert to BEV pixel coordinates (full 200x200)
            # ----------------------------------------------
            px = (x - x_min) / voxel_x
            py = (y - y_min) / voxel_y

            # Skip if outside BEV
            if px < 0 or px >= W or py < 0 or py >= H:
                continue

            ct_int = np.array([int(px), int(py)], dtype=np.int32)

            # fractional offset
            ct_offset = np.array([px - ct_int[0], py - ct_int[1]])

            if ct_int[0] < 0 or ct_int[0] >= W or ct_int[1] < 0 or ct_int[1] >= H:
                continue

            # ---------------------------
            # Compute Gaussian radius
            # ---------------------------
            # Convert box size into heatmap pixels
            box_w = w / voxel_x 
            box_l = l / voxel_y 

            radius = gaussian_radius((box_l, box_w), gaussian_overlap)
            radius = max(min_radius, int(radius))

            # ---------------------------
            # Draw Gaussian peak
            # ---------------------------
            heat_np = heatmap[b, cls_id].cpu().numpy()
            draw_gaussian(heat_np, ct_int, radius)
            heatmap[b, cls_id] = torch.from_numpy(heat_np).to(device)

            # ---------------------------
            # Regression targets
            # ---------------------------
            flat_index = ct_int[1] * W + ct_int[0]  # 50 × 50 indexing
            ind[b, k] = flat_index
            reg_mask[b, k] = 1
            mask[b, k] = 1

            # Offset
            target_offset[b, k] = torch.tensor(ct_offset, device=device)
            offset[b, 0, ct_int[1], ct_int[0]] = ct_offset[0]
            offset[b, 1, ct_int[1], ct_int[0]] = ct_offset[1]

            # Size (in meters)
            target_size[b, k] = torch.tensor([w, l, h], device=device)
            size[b, :, ct_int[1], ct_int[0]] = torch.tensor([w, l, h], device=device)

            # Rotation
            sin_yaw = np.sin(yaw)
            cos_yaw = np.cos(yaw)
            target_rot[b, k] = torch.tensor([sin_yaw, cos_yaw], device=device)
            rot[b, :, ct_int[1], ct_int[0]] = torch.tensor([sin_yaw, cos_yaw], device=device)

            # Velocity (if present)
            if gt_boxes.shape[1] > 7:
                vx, vy = gt_boxes[k][7:9]
                target_vel[b, k] = torch.tensor([vx, vy], device=device)
                vel[b, :, ct_int[1], ct_int[0]] = torch.tensor([vx, vy], device=device)

    return {
        'heatmap': heatmap,
        'offset': offset,
        'size': size,
        'rot': rot,
        'vel': vel,
        'mask': mask,
        'ind': ind,
        'reg_mask': reg_mask,
        'target_offset': target_offset,
        'target_size': target_size,
        'target_rot': target_rot,
        'target_vel': target_vel
    }

def decode_centernet_predictions(
    predictions: Dict[str, torch.Tensor],
    score_thresh: float = 0.3,
    max_detections: int = 100
) -> List[Dict[str, torch.Tensor]]:
    """
    Decode CenterNet predictions to 3D boxes
    
    Args:
        predictions: Dict with heatmap, offset, size, rot, vel
        score_thresh: Score threshold for detections
        max_detections: Maximum number of detections
    
    Returns:
        List of detection dicts per batch
    """
    heatmap = predictions['heatmap']  # (B, num_classes, H, W)
    offset = predictions['offset']    # (B, 2, H, W)
    size = predictions['size']        # (B, 3, H, W)
    rot = predictions['rot']          # (B, 2, H, W)
    vel = predictions['vel']          # (B, 2, H, W)
    
    B, C, H, W = heatmap.shape
    
    # Non-maximum suppression on heatmap
    heatmap = _nms(heatmap, kernel=3)
    
    # Get top K detections
    scores, indices, classes, ys, xs = _topk(heatmap, K=max_detections)
    
    batch_detections = []
    
    for b in range(B):
        # Filter by score
        mask = scores[b] > score_thresh
        
        if mask.sum() == 0:
            batch_detections.append({
                'boxes': torch.zeros(0, 7),
                'scores': torch.zeros(0),
                'labels': torch.zeros(0, dtype=torch.long),
                'velocities': torch.zeros(0, 2)
            })
            continue
        
        batch_scores = scores[b][mask]
        batch_classes = classes[b][mask]
        batch_ys = ys[b][mask]
        batch_xs = xs[b][mask]
        
        # Get offset, size, rotation, velocity at peak locations
        batch_offset = offset[b, :, batch_ys, batch_xs].T  # (N, 2)
        batch_size = size[b, :, batch_ys, batch_xs].T      # (N, 3)
        batch_rot = rot[b, :, batch_ys, batch_xs].T        # (N, 2)
        batch_vel = vel[b, :, batch_ys, batch_xs].T        # (N, 2)
        
        # Compute centers with offset
        centers_x = batch_xs.float() + batch_offset[:, 0]
        centers_y = batch_ys.float() + batch_offset[:, 1]
        
        # Convert to world coordinates (assuming BEV grid)
        # This depends on your BEV configuration
        #voxel_size = 0.512  # meters per pixel #200x200 grid
        voxel_size = 2.048  # meters per pixel 50x50 grid
        pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
        
        world_x = centers_x * voxel_size + pc_range[0]
        world_y = centers_y * voxel_size + pc_range[1]
        world_z = torch.zeros_like(world_x) - 1.0  # Ground plane
        
        # Compute yaw from rotation
        yaw = torch.atan2(batch_rot[:, 0], batch_rot[:, 1])
        
        # Assemble boxes [x, y, z, w, l, h, yaw]
        boxes = torch.stack([
            world_x, world_y, world_z,
            batch_size[:, 0], batch_size[:, 1], batch_size[:, 2],
            yaw
        ], dim=1)
        
        batch_detections.append({
            'boxes': boxes,
            'scores': batch_scores,
            'labels': batch_classes,
            'velocities': batch_vel
        })
    
    return batch_detections


def _nms(heat: torch.Tensor, kernel: int = 3) -> torch.Tensor:
    """Apply max pooling for NMS"""
    pad = (kernel - 1) // 2
    hmax = F.max_pool2d(heat, kernel, stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


def _topk(scores: torch.Tensor, K: int = 100) -> Tuple:
    """Get top K scores and their locations"""
    B, C, H, W = scores.shape
    
    # Flatten spatial dimensions
    scores_flat = scores.view(B, C, -1)  # (B, C, H*W)
    
    # Get top K
    topk_scores, topk_indices = torch.topk(scores_flat, K, dim=2)
    
    topk_classes = topk_indices // (H * W)
    topk_indices = topk_indices % (H * W)
    topk_ys = topk_indices // W
    topk_xs = topk_indices % W
    
    # Get overall top K across classes
    topk_scores = topk_scores.view(B, -1)
    topk_score, topk_ind = torch.topk(topk_scores, K, dim=1)
    
    topk_classes_flat = topk_classes.view(B, -1)
    topk_classes = torch.gather(topk_classes_flat, 1, topk_ind)
    
    topk_ys_flat = topk_ys.view(B, -1)
    topk_ys = torch.gather(topk_ys_flat, 1, topk_ind)
    
    topk_xs_flat = topk_xs.view(B, -1)
    topk_xs = torch.gather(topk_xs_flat, 1, topk_ind)
    
    return topk_score, topk_ind, topk_classes, topk_ys, topk_xs


class CenterNetLoss(nn.Module):
    """
    Complete CenterNet Loss for 3D Object Detection
    """
    
    def __init__(
        self,
        heatmap_weight: float = 1.0,
        offset_weight: float = 1.0,
        size_weight: float = 1.0,
        rot_weight: float = 1.0,
        vel_weight: float = 0.1
    ):
        super().__init__()
        
        self.heatmap_weight = heatmap_weight
        self.offset_weight = offset_weight
        self.size_weight = size_weight
        self.rot_weight = rot_weight
        self.vel_weight = vel_weight
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute CenterNet loss
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
        
        Returns:
            Dictionary of losses
        """
        # Focal loss for heatmap
        heatmap_loss = self._focal_loss(
            predictions['heatmap'],
            targets['heatmap']
        )
        
        # Regression losses (only at object centers)
        offset_loss = self._regression_loss(
            predictions['offset'],
            targets['target_offset'],
            targets['ind'],
            targets['reg_mask']
        )
        
        size_loss = self._regression_loss(
            predictions['size'],
            targets['target_size'],
            targets['ind'],
            targets['reg_mask']
        )
        
        rot_loss = self._regression_loss(
            predictions['rot'],
            targets['target_rot'],
            targets['ind'],
            targets['reg_mask']
        )
        
        vel_loss = self._regression_loss(
            predictions['vel'],
            targets['target_vel'],
            targets['ind'],
            targets['reg_mask']
        )
        
        # Total loss
        total_loss = (
            self.heatmap_weight * heatmap_loss +
            self.offset_weight * offset_loss +
            self.size_weight * size_loss +
            self.rot_weight * rot_loss +
            self.vel_weight * vel_loss
        )
        
        return {
            'total_loss': total_loss,
            'heatmap_loss': heatmap_loss,
            'offset_loss': offset_loss,
            'size_loss': size_loss,
            'rot_loss': rot_loss,
            'vel_loss': vel_loss
        }
    
    def _focal_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        alpha: float = 2.0,
        beta: float = 4.0
    ) -> torch.Tensor:
        """
        Focal loss for heatmap
        
        Args:
            pred: Predicted heatmap (B, C, H, W)
            target: Target heatmap (B, C, H, W)
            alpha: Focal loss alpha parameter
            beta: Focal loss beta parameter
        
        Returns:
            Focal loss value
        """
        pred = torch.clamp(torch.sigmoid(pred), min=1e-4, max=1-1e-4)
        
        pos_inds = target.eq(1).float()
        neg_inds = target.lt(1).float()
        
        neg_weights = torch.pow(1 - target, beta)
        
        pos_loss = torch.log(pred) * torch.pow(1 - pred, alpha) * pos_inds
        neg_loss = torch.log(1 - pred) * torch.pow(pred, alpha) * neg_weights * neg_inds
        
        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()
        
        if num_pos == 0:
            loss = -neg_loss
        else:
            loss = -(pos_loss + neg_loss) / num_pos
        
        return loss
    
    def _regression_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        ind: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        L1 regression loss at object centers
        
        Args:
            pred: Predicted values (B, C, H, W)
            target: Target values (B, max_objects, C)
            ind: Object center indices (B, max_objects)
            mask: Valid object mask (B, max_objects)
        
        Returns:
            Regression loss value
        """
        B, C, H, W = pred.shape
        max_objects = target.shape[1]
        
        # Gather predictions at object centers
        pred = pred.view(B, C, -1)  # (B, C, H*W)
        pred = pred.permute(0, 2, 1)  # (B, H*W, C)
        
        # Expand indices
        ind = ind.unsqueeze(2).expand(B, max_objects, C)  # (B, max_objects, C)
        
        # Gather
        pred = pred.gather(1, ind)  # (B, max_objects, C)
        
        # Compute L1 loss
        mask = mask.unsqueeze(2).expand_as(target).float()
        
        loss = torch.abs(pred - target) * mask
        loss = loss.sum() / (mask.sum() + 1e-4)
        
        return loss


# Usage example
def example_usage():
    """Example of how to use the target preparation"""
    
    # Simulate batch
    batch = {
        'camera_imgs': torch.randn(2, 6, 3, 448, 800),
        'gt_boxes': [
            torch.tensor([
                [10.5, 20.3, -0.5, 1.8, 4.5, 1.6, 0.5],
                [-5.2, -15.7, -0.8, 2.0, 4.8, 1.7, -1.2],
            ]),
            torch.tensor([
                [8.1, 12.4, -0.6, 1.9, 4.6, 1.65, 0.8],
                [15.3, -8.9, -0.7, 1.85, 4.55, 1.62, -0.5],
                [-12.7, 25.6, -0.55, 1.95, 4.7, 1.68, 1.1],
            ])
        ],
        'gt_labels': [
            torch.tensor([0, 0]),
            torch.tensor([0, 1, 0])
        ]
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Prepare targets
    targets = prepare_centernet_targets(
        batch=batch,
        device=device,
        pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        bev_size=(200, 200),
        num_classes=10
    )
    
    print("Target shapes:")
    for key, value in targets.items():
        print(f"  {key}: {value.shape}")
    
    # Simulate predictions
    predictions = {
        'heatmap': torch.randn(2, 10, 200, 200, device=device),
        'offset': torch.randn(2, 2, 200, 200, device=device),
        'size': torch.randn(2, 3, 200, 200, device=device),
        'rot': torch.randn(2, 2, 200, 200, device=device),
        'vel': torch.randn(2, 2, 200, 200, device=device)
    }
    
    # Compute loss
    loss_fn = CenterNetLoss()
    losses = loss_fn(predictions, targets)
    
    print("\nLosses:")
    for key, value in losses.items():
        print(f"  {key}: {value.item():.4f}")


if __name__ == '__main__':
    example_usage()