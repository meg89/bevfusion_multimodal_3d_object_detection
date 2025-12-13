"""
Multi-Modal Fusion Techniques for 3D Object Detection
Combines Camera, LiDAR, and Radar features using various fusion strategies
Includes complete detection heads for 3D bounding box prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import math


# ============================================================================
# 1. BIRD'S EYE VIEW (BEV) FUSION
# ============================================================================

class BEVFusion(nn.Module):
    """
    Bird's Eye View fusion for multi-modal features
    Projects all modalities to a common BEV representation
    
    Input:
        Camera: (B, 6, 2048, 14, 25)
        LiDAR:  (B, 1024)
        Radar:  (B, 256)
    
    Output:
        BEV features: (B, bev_channels, H_bev, W_bev)
        Typical: (B, 256, 200, 200)
    """
    
    def __init__(
        self,
        camera_channels: int = 2048,
        lidar_channels: int = 1024,
        radar_channels: int = 256,
        bev_h: int = 200,
        bev_w: int = 200,
        bev_channels: int = 256,
        pc_range: List[float] = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
    ):
        super(BEVFusion, self).__init__()
        
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.bev_channels = bev_channels
        self.pc_range = pc_range
        
        # Camera to BEV projection
        self.camera_proj = nn.Sequential(
            nn.Conv2d(camera_channels, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, bev_channels, 1),
            nn.BatchNorm2d(bev_channels),
            nn.ReLU(inplace=True)
        )
        
        # LiDAR to BEV
        self.lidar_proj = nn.Sequential(
            nn.Linear(lidar_channels, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, bev_channels * bev_h * bev_w)
        )
        
        # Radar to BEV
        self.radar_proj = nn.Sequential(
            nn.Linear(radar_channels, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, bev_channels * bev_h * bev_w)
        )
        
        # BEV fusion network
        self.bev_fusion = nn.Sequential(
            nn.Conv2d(bev_channels * 3, bev_channels * 2, 3, padding=1),
            nn.BatchNorm2d(bev_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(bev_channels * 2, bev_channels, 3, padding=1),
            nn.BatchNorm2d(bev_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(
        self, 
        camera_features: torch.Tensor,
        lidar_features: torch.Tensor,
        radar_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            camera_features: (B, 6, 2048, H', W')
            lidar_features: (B, 1024)
            radar_features: (B, 256)
        
        Returns:
            bev_features: (B, bev_channels, H_bev, W_bev)
        """
        B = camera_features.shape[0]
        
        # Process camera: pool over cameras and project to BEV
        cam_bev = camera_features.mean(dim=1)  # (B, 2048, H', W')
        cam_bev = self.camera_proj(cam_bev)    # (B, bev_channels, H', W')
        cam_bev = F.interpolate(cam_bev, size=(self.bev_h, self.bev_w), 
                                mode='bilinear', align_corners=False)
        
        # Process LiDAR: project to BEV grid
        lidar_bev = self.lidar_proj(lidar_features)  # (B, bev_channels*H*W)
        lidar_bev = lidar_bev.view(B, self.bev_channels, self.bev_h, self.bev_w)
        
        # Process Radar: project to BEV grid
        radar_bev = self.radar_proj(radar_features)  # (B, bev_channels*H*W)
        radar_bev = radar_bev.view(B, self.bev_channels, self.bev_h, self.bev_w)
        
        # Concatenate all BEV features
        bev_concat = torch.cat([cam_bev, lidar_bev, radar_bev], dim=1)
        
        # Fuse BEV features
        bev_fused = self.bev_fusion(bev_concat)
        
        return bev_fused


# ============================================================================
# 2. ATTENTION-BASED FUSION
# ============================================================================

class CrossModalAttention(nn.Module):
    """
    Cross-modal attention for feature fusion
    Learns to attend to relevant features from different modalities
    """
    
    def __init__(
        self,
        query_dim: int,
        key_dim: int,
        value_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super(CrossModalAttention, self).__init__()
        
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        
        self.query = nn.Linear(query_dim, query_dim)
        self.key = nn.Linear(key_dim, query_dim)
        self.value = nn.Linear(value_dim, query_dim)
        
        self.out = nn.Linear(query_dim, query_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            query: (B, N_q, D_q)
            key: (B, N_k, D_k)
            value: (B, N_v, D_v)
        
        Returns:
            output: (B, N_q, D_q)
        """
        B, N_q, _ = query.shape
        
        # Linear projections
        Q = self.query(query)  # (B, N_q, D)
        K = self.key(key)      # (B, N_k, D)
        V = self.value(value)  # (B, N_v, D)
        
        # Reshape for multi-head attention
        Q = Q.view(B, N_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = torch.matmul(attn, V)  # (B, num_heads, N_q, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, N_q, -1)
        
        return self.out(out)


class AttentionFusion(nn.Module):
    """
    Attention-based multi-modal fusion
    
    Input:
        Camera: (B, 6, 2048, 14, 25)
        LiDAR:  (B, 1024)
        Radar:  (B, 256)
    
    Output:
        Fused features: (B, feat_dim)
        Typical: (B, 512)
    """
    
    def __init__(
        self,
        camera_channels: int = 2048,
        lidar_channels: int = 1024,
        radar_channels: int = 256,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 2
    ):
        super(AttentionFusion, self).__init__()
        
        self.hidden_dim = hidden_dim
        
        # Project all modalities to same dimension
        self.camera_proj = nn.Linear(camera_channels, hidden_dim)
        self.lidar_proj = nn.Linear(lidar_channels, hidden_dim)
        self.radar_proj = nn.Linear(radar_channels, hidden_dim)
        
        # Positional embeddings for different modalities
        self.cam_pos_embed = nn.Parameter(torch.randn(1, 6 * 14 * 25, hidden_dim))
        self.lidar_pos_embed = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.radar_pos_embed = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
        # Cross-modal attention layers
        self.cross_attention_layers = nn.ModuleList([
            nn.ModuleDict({
                'cam_to_lidar': CrossModalAttention(hidden_dim, hidden_dim, hidden_dim, num_heads),
                'cam_to_radar': CrossModalAttention(hidden_dim, hidden_dim, hidden_dim, num_heads),
                'lidar_to_cam': CrossModalAttention(hidden_dim, hidden_dim, hidden_dim, num_heads),
                'radar_to_cam': CrossModalAttention(hidden_dim, hidden_dim, hidden_dim, num_heads),
                'self_attn': CrossModalAttention(hidden_dim, hidden_dim, hidden_dim, num_heads),
            })
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(
        self,
        camera_features: torch.Tensor,
        lidar_features: torch.Tensor,
        radar_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            camera_features: (B, 6, 2048, H', W')
            lidar_features: (B, 1024)
            radar_features: (B, 256)
        
        Returns:
            fused_features: (B, hidden_dim)
        """
        B = camera_features.shape[0]
        
        # Flatten and project camera features
        cam_flat = camera_features.flatten(2)  # (B, 6*2048, H'*W')
        cam_flat = cam_flat.transpose(1, 2)    # (B, H'*W'*6, 2048)
        cam_flat = cam_flat.reshape(B, -1, camera_features.shape[1])  # Simplified
        
        # For simplicity, pool camera spatial dimensions
        cam_pooled = camera_features.mean(dim=[3, 4])  # (B, 6, 2048)
        cam_pooled = cam_pooled.flatten(1, 2)  # (B, 6*2048)
        cam_pooled = cam_pooled.view(B, -1, camera_features.shape[1])
        
        cam_tokens = self.camera_proj(cam_pooled.mean(dim=1, keepdim=True))  # (B, 1, hidden_dim)
        lidar_tokens = self.lidar_proj(lidar_features).unsqueeze(1)  # (B, 1, hidden_dim)
        radar_tokens = self.radar_proj(radar_features).unsqueeze(1)  # (B, 1, hidden_dim)
        
        # Add positional embeddings
        cam_tokens = cam_tokens + self.cam_pos_embed[:, :1, :]
        lidar_tokens = lidar_tokens + self.lidar_pos_embed
        radar_tokens = radar_tokens + self.radar_pos_embed
        
        # Concatenate all tokens
        all_tokens = torch.cat([cam_tokens, lidar_tokens, radar_tokens], dim=1)  # (B, 3, hidden_dim)
        
        # Apply cross-modal attention
        for layer in self.cross_attention_layers:
            # Self-attention
            all_tokens = all_tokens + layer['self_attn'](all_tokens, all_tokens, all_tokens)
        
        # Global average pooling
        fused = all_tokens.mean(dim=1)  # (B, hidden_dim)
        
        # Output projection
        fused = self.output_proj(fused)
        
        return fused


# ============================================================================
# 3. LATE FUSION (SIMPLE CONCATENATION)
# ============================================================================

class LateFusion(nn.Module):
    """
    Simple late fusion by concatenating features
    
    Input:
        Camera: (B, 6, 2048, 14, 25)
        LiDAR:  (B, 1024)
        Radar:  (B, 256)
    
    Output:
        Fused features: (B, feat_dim)
    """
    
    def __init__(
        self,
        camera_channels: int = 2048,
        lidar_channels: int = 1024,
        radar_channels: int = 256,
        output_dim: int = 512
    ):
        super(LateFusion, self).__init__()
        
        total_dim = camera_channels + lidar_channels + radar_channels
        
        self.fusion_mlp = nn.Sequential(
            nn.Linear(total_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        
    def forward(
        self,
        camera_features: torch.Tensor,
        lidar_features: torch.Tensor,
        radar_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            camera_features: (B, 6, 2048, H', W')
            lidar_features: (B, 1024)
            radar_features: (B, 256)
        
        Returns:
            fused_features: (B, output_dim)
        """
        # Pool camera features
        cam_global = camera_features.mean(dim=[1, 3, 4])  # (B, 2048)
        
        # Concatenate all features
        fused = torch.cat([cam_global, lidar_features, radar_features], dim=1)
        
        # MLP fusion
        fused = self.fusion_mlp(fused)
        
        return fused


# ============================================================================
# 4. 3D DETECTION HEADS
# ============================================================================

class CenterNetHead(nn.Module):
    """
    CenterNet-style detection head for 3D object detection
    Predicts heatmap, offset, size, and rotation
    
    Input: BEV features (B, C, H, W)
    Output: Detection predictions
    """
    
    def __init__(
        self,
        in_channels: int = 256,
        num_classes: int = 10,
        head_conv: int = 64
    ):
        super(CenterNetHead, self).__init__()
        
        self.num_classes = num_classes
        
        # Heatmap head (class probabilities)
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(in_channels, head_conv, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, num_classes, 1, bias=True)
        )
        
        # Offset head (sub-pixel location)
        self.offset_head = nn.Sequential(
            nn.Conv2d(in_channels, head_conv, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, 2, 1, bias=True)
        )
        
        # Size head (w, l, h)
        self.size_head = nn.Sequential(
            nn.Conv2d(in_channels, head_conv, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, 3, 1, bias=True)
        )
        
        # Rotation head (sin, cos of yaw)
        self.rot_head = nn.Sequential(
            nn.Conv2d(in_channels, head_conv, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, 2, 1, bias=True)
        )
        
        # Velocity head (vx, vy)
        self.vel_head = nn.Sequential(
            nn.Conv2d(in_channels, head_conv, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, 2, 1, bias=True)
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights with special treatment for heatmap"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # Special initialization for heatmap (focal loss)
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.heatmap_head[-1].bias, bias_value)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: BEV features (B, C, H, W)
        
        Returns:
            predictions: Dict with keys:
                - heatmap: (B, num_classes, H, W)
                - offset: (B, 2, H, W)
                - size: (B, 3, H, W)
                - rot: (B, 2, H, W)
                - vel: (B, 2, H, W)
        """
        heatmap = self.heatmap_head(x)
        heatmap = torch.sigmoid(heatmap)
        
        offset = self.offset_head(x)
        size = self.size_head(x)
        rot = self.rot_head(x)
        vel = self.vel_head(x)
        
        return {
            'heatmap': heatmap,
            'offset': offset,
            'size': size,
            'rot': rot,
            'vel': vel
        }


class AnchorBasedHead(nn.Module):
    """
    Anchor-based detection head (similar to SECOND, PointPillars)
    
    Input: BEV features (B, C, H, W)
    Output: Detection predictions per anchor
    """
    
    def __init__(
        self,
        in_channels: int = 256,
        num_classes: int = 10,
        num_anchors: int = 2,  # 2 orientations per location
        head_conv: int = 256
    ):
        super(AnchorBasedHead, self).__init__()
        
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        # Shared convolutions
        self.shared_conv = nn.Sequential(
            nn.Conv2d(in_channels, head_conv, 3, padding=1, bias=True),
            nn.BatchNorm2d(head_conv),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, head_conv, 3, padding=1, bias=True),
            nn.BatchNorm2d(head_conv),
            nn.ReLU(inplace=True)
        )
        
        # Classification head
        self.cls_head = nn.Conv2d(
            head_conv, 
            num_anchors * num_classes, 
            1
        )
        
        # Regression head (7 params: x, y, z, w, l, h, yaw)
        self.reg_head = nn.Conv2d(
            head_conv,
            num_anchors * 7,
            1
        )
        
        # Direction classification (for angle ambiguity)
        self.dir_head = nn.Conv2d(
            head_conv,
            num_anchors * 2,  # 2 direction bins
            1
        )
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # Special init for classification
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.cls_head.bias, bias_value)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: BEV features (B, C, H, W)
        
        Returns:
            predictions: Dict with keys:
                - cls: (B, num_anchors*num_classes, H, W)
                - reg: (B, num_anchors*7, H, W)
                - dir: (B, num_anchors*2, H, W)
        """
        x = self.shared_conv(x)
        
        cls = self.cls_head(x)
        reg = self.reg_head(x)
        dir_cls = self.dir_head(x)
        
        return {
            'cls': cls,
            'reg': reg,
            'dir': dir_cls
        }


# ============================================================================
# 5. COMPLETE MULTI-MODAL 3D DETECTOR
# ============================================================================

class MultiModal3DDetector(nn.Module):
    """
    Complete multi-modal 3D object detector
    Combines camera, LiDAR, and radar with fusion and detection head
    
    Architecture:
        Encoders → Fusion → Detection Head → 3D Boxes
    """
    
    def __init__(
        self,
        num_classes: int = 10,
        fusion_type: str = 'bev',  # 'bev', 'attention', 'late'
        detection_head: str = 'centernet',  # 'centernet', 'anchor'
        bev_h: int = 200,
        bev_w: int = 200
    ):
        super(MultiModal3DDetector, self).__init__()
        
        self.fusion_type = fusion_type
        self.detection_head_type = detection_head
        
        # Import encoders (assuming they're available)
        from multimodal_encoders import (
            ResNetCameraEncoder,
            PointNetLiDAREncoder,
            MultiRadarEncoder
        )
        
        # Encoders
        self.camera_encoder = ResNetCameraEncoder(backbone='resnet50', pretrained=True)
        self.lidar_encoder = PointNetLiDAREncoder(input_channels=5, feat_dim=1024)
        self.radar_encoder = MultiRadarEncoder(input_channels=7, feat_dim=256, num_radars=5)
        
        # Fusion module
        if fusion_type == 'bev':
            self.fusion = BEVFusion(
                camera_channels=2048,
                lidar_channels=1024,
                radar_channels=256,
                bev_h=bev_h,
                bev_w=bev_w,
                bev_channels=256
            )
            fusion_out_channels = 256
            is_spatial = True
        elif fusion_type == 'attention':
            self.fusion = AttentionFusion(
                camera_channels=2048,
                lidar_channels=1024,
                radar_channels=256,
                hidden_dim=512
            )
            fusion_out_channels = 512
            is_spatial = False
        elif fusion_type == 'late':
            self.fusion = LateFusion(
                camera_channels=2048,
                lidar_channels=1024,
                radar_channels=256,
                output_dim=512
            )
            fusion_out_channels = 512
            is_spatial = False
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
        
        # Detection head
        if is_spatial:
            if detection_head == 'centernet':
                self.det_head = CenterNetHead(
                    in_channels=fusion_out_channels,
                    num_classes=num_classes
                )
            elif detection_head == 'anchor':
                self.det_head = AnchorBasedHead(
                    in_channels=fusion_out_channels,
                    num_classes=num_classes
                )
            else:
                raise ValueError(f"Unknown detection head: {detection_head}")
        else:
            # For non-spatial fusion, use MLP head
            self.det_head = nn.Sequential(
                nn.Linear(fusion_out_channels, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, num_classes + 7)  # class + box
            )
    
    def forward(
        self,
        camera_imgs: torch.Tensor,
        lidar_points: torch.Tensor,
        radar_points: List[torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            camera_imgs: (B, 6, 3, H, W)
            lidar_points: (B, N, 5)
            radar_points: List of 5 tensors, each (B, N_i, 7)
        
        Returns:
            predictions: Dict of detection outputs
        """
        # Extract features
        cam_feat = self.camera_encoder(camera_imgs)      # (B, 6, 2048, 14, 25)
        lidar_feat = self.lidar_encoder(lidar_points)    # (B, 1024)
        radar_feat = self.radar_encoder(radar_points)    # (B, 256)
        
        # Fuse features
        fused_feat = self.fusion(cam_feat, lidar_feat, radar_feat)
        
        # Detection
        predictions = self.det_head(fused_feat)
        
        return predictions


# ============================================================================
# 6. POST-PROCESSING & NMS
# ============================================================================

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
        voxel_size = 0.512  # meters per pixel
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


# ============================================================================
# 7. LOSS FUNCTIONS
# ============================================================================

class DetectionLoss(nn.Module):
    """
    Combined loss for 3D object detection
    """
    
    def __init__(
        self,
        hm_weight: float = 1.0,
        reg_weight: float = 1.0,
        size_weight: float = 0.1,
        rot_weight: float = 0.1,
        vel_weight: float = 0.1
    ):
        super(DetectionLoss, self).__init__()
        
        self.hm_weight = hm_weight
        self.reg_weight = reg_weight
        self.size_weight = size_weight
        self.rot_weight = rot_weight
        self.vel_weight = vel_weight
        
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute losses
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
        
        Returns:
            Dict of losses
        """
        # Heatmap loss (focal loss)
        hm_loss = self._focal_loss(
            predictions['heatmap'],
            targets['heatmap']
        )
        
        #hm_loss = None
        # Regression losses (only at positive locations)
        mask = targets['mask']
        
        offset_loss = F.l1_loss(
            predictions['offset'] * mask,
            targets['offset'] * mask,
            reduction='sum'
        ) / (mask.sum() + 1e-4)
        
        size_loss = F.l1_loss(
            predictions['size'] * mask,
            targets['size'] * mask,
            reduction='sum'
        ) / (mask.sum() + 1e-4)
        
        rot_loss = F.l1_loss(
            predictions['rot'] * mask,
            targets['rot'] * mask,
            reduction='sum'
        ) / (mask.sum() + 1e-4)
        
        vel_loss = F.l1_loss(
            predictions['vel'] * mask,
            targets['vel'] * mask,
            reduction='sum'
        ) / (mask.sum() + 1e-4)
        
        # Total loss
        total_loss = (
            #self.hm_weight * hm_loss +
            self.reg_weight * offset_loss +
            self.size_weight * size_loss +
            self.rot_weight * rot_loss +
            self.vel_weight * vel_loss
        )
        
        return {
            'total_loss': total_loss,
            'hm_loss': hm_loss,
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
        """Focal loss for heatmap"""
        pos_mask = target.eq(1).float()
        neg_mask = target.lt(1).float()
        
        neg_weights = torch.pow(1 - target, beta)
        
        pos_loss = torch.log(pred + 1e-12) * torch.pow(1 - pred, alpha) * pos_mask
        neg_loss = torch.log(1 - pred + 1e-12) * torch.pow(pred, alpha) * neg_weights * neg_mask
        
        num_pos = pos_mask.sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()
        
        if num_pos == 0:
            loss = -neg_loss
        else:
            loss = -(pos_loss + neg_loss) / num_pos
        
        return loss

if __name__ == '__main__':
    print("Multi-Modal 3D Detection with Fusion - Module loaded successfully!")
    print("\nAvailable fusion methods:")
    print("  1. BEVFusion - Bird's Eye View projection")
    print("  2. AttentionFusion - Cross-modal attention")
    print("  3. LateFusion - Simple concatenation")
    print("\nAvailable detection heads:")
    print("  1. CenterNetHead - Anchor-free detection")
    print("  2. AnchorBasedHead - Anchor-based detection")