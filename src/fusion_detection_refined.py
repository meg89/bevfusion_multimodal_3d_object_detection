"""
Refined Multi-Modal Fusion for 3D Object Detection
Supports flexible modality selection:
- Single modality: camera_only, lidar_only, radar_only
- Dual modality: camera+lidar, camera+radar, lidar+radar
- Triple modality: camera+lidar+radar

All three fusion methods (BEV, Attention, Late) support all combinations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import math


# ============================================================================
# 1. FLEXIBLE BEV FUSION
# ============================================================================

class FlexibleBEVFusion(nn.Module):
    """
    BEV Fusion with flexible modality selection
    Automatically adapts to available modalities
    
    Supported combinations:
    - camera_only
    - lidar_only
    - radar_only
    - camera+lidar
    - camera+radar
    - lidar+radar
    - camera+lidar+radar
    """
    
    def __init__(
        self,
        use_camera: bool = True,
        use_lidar: bool = True,
        use_radar: bool = True,
        camera_channels: int = 2048,
        lidar_channels: int = 1024,
        radar_channels: int = 256,
        bev_h: int = 200,
        bev_w: int = 200,
        bev_channels: int = 256,
        pc_range: List[float] = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
    ):
        super(FlexibleBEVFusion, self).__init__()
        
        self.use_camera = use_camera
        self.use_lidar = use_lidar
        self.use_radar = use_radar
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.bev_channels = bev_channels
        self.pc_range = pc_range
        
        # Count active modalities
        self.num_modalities = sum([use_camera, use_lidar, use_radar])
        assert self.num_modalities > 0, "At least one modality must be enabled"
        
        # Camera to BEV projection
        if use_camera:
            self.camera_proj = nn.Sequential(
                nn.Conv2d(camera_channels, 512, 3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, bev_channels, 1),
                nn.BatchNorm2d(bev_channels),
                nn.ReLU(inplace=True)
            )
        
        # LiDAR to BEV
        if use_lidar:
            self.lidar_proj = nn.Sequential(
                nn.Linear(lidar_channels, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, bev_channels * bev_h * bev_w)
            )
        
        # Radar to BEV
        if use_radar:
            self.radar_proj = nn.Sequential(
                nn.Linear(radar_channels, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, bev_channels * bev_h * bev_w)
            )
        
        # BEV fusion network (adapts to number of modalities)
        fusion_in_channels = bev_channels * self.num_modalities
        self.bev_fusion = nn.Sequential(
            nn.Conv2d(fusion_in_channels, bev_channels * 2, 3, padding=1),
            nn.BatchNorm2d(bev_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(bev_channels * 2, bev_channels, 3, padding=1),
            nn.BatchNorm2d(bev_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(
        self, 
        camera_features: Optional[torch.Tensor] = None,
        lidar_features: Optional[torch.Tensor] = None,
        radar_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            camera_features: (B, 6, 2048, H', W') or None
            lidar_features: (B, 1024) or None
            radar_features: (B, 256) or None
        
        Returns:
            bev_features: (B, bev_channels, H_bev, W_bev)
        """
        bev_features = []
        
        if self.use_camera and camera_features is not None:
            B = camera_features.shape[0]
            # Process camera: pool over cameras and project to BEV
            cam_bev = camera_features.mean(dim=1)  # (B, 2048, H', W')
            cam_bev = self.camera_proj(cam_bev)    # (B, bev_channels, H', W')
            cam_bev = F.interpolate(cam_bev, size=(self.bev_h, self.bev_w), 
                                    mode='bilinear', align_corners=False)
            bev_features.append(cam_bev)
        
        if self.use_lidar and lidar_features is not None:
            B = lidar_features.shape[0]
            # Process LiDAR: project to BEV grid
            lidar_bev = self.lidar_proj(lidar_features)  # (B, bev_channels*H*W)
            lidar_bev = lidar_bev.view(B, self.bev_channels, self.bev_h, self.bev_w)
            bev_features.append(lidar_bev)
        
        if self.use_radar and radar_features is not None:
            B = radar_features.shape[0]
            # Process Radar: project to BEV grid
            radar_bev = self.radar_proj(radar_features)  # (B, bev_channels*H*W)
            radar_bev = radar_bev.view(B, self.bev_channels, self.bev_h, self.bev_w)
            bev_features.append(radar_bev)
        
        # Concatenate all available BEV features
        if len(bev_features) == 0:
            raise ValueError("No modality features provided")
        
        bev_concat = torch.cat(bev_features, dim=1)
        
        # Fuse BEV features
        bev_fused = self.bev_fusion(bev_concat)
        
        return bev_fused
    
    def get_config_str(self) -> str:
        """Get configuration string"""
        modalities = []
        if self.use_camera: modalities.append("camera")
        if self.use_lidar: modalities.append("lidar")
        if self.use_radar: modalities.append("radar")
        return "+".join(modalities)


# ============================================================================
# 2. FLEXIBLE ATTENTION FUSION
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


class FlexibleAttentionFusion(nn.Module):
    """
    Attention-based fusion with flexible modality selection
    
    Supported combinations:
    - Single: camera_only, lidar_only, radar_only
    - Dual: camera+lidar, camera+radar, lidar+radar
    - Triple: camera+lidar+radar
    """
    
    def __init__(
        self,
        use_camera: bool = True,
        use_lidar: bool = True,
        use_radar: bool = True,
        camera_channels: int = 2048,
        lidar_channels: int = 1024,
        radar_channels: int = 256,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 2
    ):
        super(FlexibleAttentionFusion, self).__init__()
        
        self.use_camera = use_camera
        self.use_lidar = use_lidar
        self.use_radar = use_radar
        self.hidden_dim = hidden_dim
        
        # Count active modalities
        self.num_modalities = sum([use_camera, use_lidar, use_radar])
        assert self.num_modalities > 0, "At least one modality must be enabled"
        
        # Project all modalities to same dimension
        if use_camera:
            self.camera_proj = nn.Linear(camera_channels, hidden_dim)
            self.cam_pos_embed = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
        if use_lidar:
            self.lidar_proj = nn.Linear(lidar_channels, hidden_dim)
            self.lidar_pos_embed = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
        if use_radar:
            self.radar_proj = nn.Linear(radar_channels, hidden_dim)
            self.radar_pos_embed = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
        # Self-attention layers (within combined features)
        self.self_attention_layers = nn.ModuleList([
            nn.ModuleDict({
                'self_attn': CrossModalAttention(
                    hidden_dim, hidden_dim, hidden_dim, num_heads
                ),
                'ffn': nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 4),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_dim * 4, hidden_dim),
                    nn.Dropout(0.1)
                ),
                'norm1': nn.LayerNorm(hidden_dim),
                'norm2': nn.LayerNorm(hidden_dim)
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
        camera_features: Optional[torch.Tensor] = None,
        lidar_features: Optional[torch.Tensor] = None,
        radar_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            camera_features: (B, 6, 2048, H', W') or None
            lidar_features: (B, 1024) or None
            radar_features: (B, 256) or None
        
        Returns:
            fused_features: (B, hidden_dim)
        """
        tokens = []
        
        # Process camera
        if self.use_camera and camera_features is not None:
            B = camera_features.shape[0]
            # Pool spatial and camera dimensions
            cam_pooled = camera_features.mean(dim=[1, 3, 4])  # (B, 2048)
            cam_token = self.camera_proj(cam_pooled).unsqueeze(1)  # (B, 1, hidden_dim)
            cam_token = cam_token + self.cam_pos_embed
            tokens.append(cam_token)
        
        # Process LiDAR
        if self.use_lidar and lidar_features is not None:
            B = lidar_features.shape[0]
            lidar_token = self.lidar_proj(lidar_features).unsqueeze(1)  # (B, 1, hidden_dim)
            lidar_token = lidar_token + self.lidar_pos_embed
            tokens.append(lidar_token)
        
        # Process Radar
        if self.use_radar and radar_features is not None:
            B = radar_features.shape[0]
            radar_token = self.radar_proj(radar_features).unsqueeze(1)  # (B, 1, hidden_dim)
            radar_token = radar_token + self.radar_pos_embed
            tokens.append(radar_token)
        
        if len(tokens) == 0:
            raise ValueError("No modality features provided")
        
        # Concatenate all tokens
        all_tokens = torch.cat(tokens, dim=1)  # (B, num_modalities, hidden_dim)
        
        # Apply self-attention layers
        for layer in self.self_attention_layers:
            # Self-attention with residual
            attn_out = layer['self_attn'](all_tokens, all_tokens, all_tokens)
            all_tokens = layer['norm1'](all_tokens + attn_out)
            
            # FFN with residual
            ffn_out = layer['ffn'](all_tokens)
            all_tokens = layer['norm2'](all_tokens + ffn_out)
        
        # Global pooling
        fused = all_tokens.mean(dim=1)  # (B, hidden_dim)
        
        # Output projection
        fused = self.output_proj(fused)
        
        return fused
    
    def get_config_str(self) -> str:
        """Get configuration string"""
        modalities = []
        if self.use_camera: modalities.append("camera")
        if self.use_lidar: modalities.append("lidar")
        if self.use_radar: modalities.append("radar")
        return "+".join(modalities)


# ============================================================================
# 3. FLEXIBLE LATE FUSION
# ============================================================================

class FlexibleLateFusion(nn.Module):
    """
    Late fusion with flexible modality selection
    Simple concatenation approach
    
    Supported combinations:
    - Single: camera_only, lidar_only, radar_only
    - Dual: camera+lidar, camera+radar, lidar+radar
    - Triple: camera+lidar+radar
    """
    
    def __init__(
        self,
        use_camera: bool = True,
        use_lidar: bool = True,
        use_radar: bool = True,
        camera_channels: int = 2048,
        lidar_channels: int = 1024,
        radar_channels: int = 256,
        output_dim: int = 512
    ):
        super(FlexibleLateFusion, self).__init__()
        
        self.use_camera = use_camera
        self.use_lidar = use_lidar
        self.use_radar = use_radar
        
        # Count active modalities
        self.num_modalities = sum([use_camera, use_lidar, use_radar])
        assert self.num_modalities > 0, "At least one modality must be enabled"
        
        # Calculate total input dimension
        total_dim = 0
        if use_camera:
            total_dim += camera_channels
        if use_lidar:
            total_dim += lidar_channels
        if use_radar:
            total_dim += radar_channels
        
        # Fusion MLP (adapts to total dimension)
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
        camera_features: Optional[torch.Tensor] = None,
        lidar_features: Optional[torch.Tensor] = None,
        radar_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            camera_features: (B, 6, 2048, H', W') or None
            lidar_features: (B, 1024) or None
            radar_features: (B, 256) or None
        
        Returns:
            fused_features: (B, output_dim)
        """
        features = []
        
        # Pool camera features
        if self.use_camera and camera_features is not None:
            cam_global = camera_features.mean(dim=[1, 3, 4])  # (B, 2048)
            features.append(cam_global)
        
        # LiDAR features
        if self.use_lidar and lidar_features is not None:
            features.append(lidar_features)
        
        # Radar features
        if self.use_radar and radar_features is not None:
            features.append(radar_features)
        
        if len(features) == 0:
            raise ValueError("No modality features provided")
        
        # Concatenate all features
        fused = torch.cat(features, dim=1)
        
        # MLP fusion
        fused = self.fusion_mlp(fused)
        
        return fused
    
    def get_config_str(self) -> str:
        """Get configuration string"""
        modalities = []
        if self.use_camera: modalities.append("camera")
        if self.use_lidar: modalities.append("lidar")
        if self.use_radar: modalities.append("radar")
        return "+".join(modalities)


# ============================================================================
# 4. DETECTION HEADS (unchanged from previous)
# ============================================================================

class CenterNetHead(nn.Module):
    """CenterNet-style detection head for 3D object detection"""
    
    def __init__(
        self,
        in_channels: int = 256,
        num_classes: int = 10,
        head_conv: int = 64
    ):
        super(CenterNetHead, self).__init__()
        
        self.num_classes = num_classes
        
        # Heatmap head
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(in_channels, head_conv, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, num_classes, 1, bias=True)
        )
        
        # Offset head
        self.offset_head = nn.Sequential(
            nn.Conv2d(in_channels, head_conv, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, 2, 1, bias=True)
        )
        
        # Size head
        self.size_head = nn.Sequential(
            nn.Conv2d(in_channels, head_conv, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, 3, 1, bias=True)
        )
        
        # Rotation head
        self.rot_head = nn.Sequential(
            nn.Conv2d(in_channels, head_conv, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, 2, 1, bias=True)
        )
        
        # Velocity head
        self.vel_head = nn.Sequential(
            nn.Conv2d(in_channels, head_conv, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, 2, 1, bias=True)
        )
        
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.heatmap_head[-1].bias, bias_value)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
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


class MLPDetectionHead(nn.Module):
    """
    Simple MLP head for non-spatial fusion methods
    Outputs single detection prediction
    """
    
    def __init__(
        self,
        in_channels: int = 512,
        num_classes: int = 10
    ):
        super(MLPDetectionHead, self).__init__()
        
        self.num_classes = num_classes
        
        self.head = nn.Sequential(
            nn.Linear(in_channels, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes + 7)  # class + box (x,y,z,w,l,h,yaw)
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (B, in_channels)
        
        Returns:
            predictions with 'cls' and 'box' keys
        """
        output = self.head(x)  # (B, num_classes + 7)
        
        return {
            'cls': output[:, :self.num_classes],  # (B, num_classes)
            'box': output[:, self.num_classes:]   # (B, 7)
        }


# ============================================================================
# 5. FLEXIBLE MULTI-MODAL 3D DETECTOR
# ============================================================================

class FlexibleMultiModal3DDetector(nn.Module):
    """
    Flexible multi-modal 3D object detector
    Supports any combination of camera, lidar, and radar
    with all three fusion methods
    
    Modality Configurations:
    - Single: camera_only, lidar_only, radar_only
    - Dual: camera+lidar, camera+radar, lidar+radar
    - Triple: camera+lidar+radar (all)
    
    Fusion Methods:
    - BEV: Bird's Eye View fusion
    - Attention: Cross-modal attention
    - Late: Simple concatenation
    """
    
    def __init__(
        self,
        # Modality selection
        use_camera: bool = True,
        use_lidar: bool = True,
        use_radar: bool = True,
        # Architecture
        num_classes: int = 10,
        fusion_type: str = 'bev',  # 'bev', 'attention', 'late'
        detection_head: str = 'centernet',  # 'centernet', 'mlp'
        # BEV parameters
        bev_h: int = 200,
        bev_w: int = 200
    ):
        super(FlexibleMultiModal3DDetector, self).__init__()
        
        self.use_camera = use_camera
        self.use_lidar = use_lidar
        self.use_radar = use_radar
        self.fusion_type = fusion_type
        self.detection_head_type = detection_head
        
        # Validate at least one modality
        num_modalities = sum([use_camera, use_lidar, use_radar])
        assert num_modalities > 0, "At least one modality must be enabled"
        
        # Import encoders
        from multimodal_encoders import (
            ResNetCameraEncoder,
            PointNetLiDAREncoder,
            MultiRadarEncoder
        )
        
        # Initialize encoders (only for enabled modalities)
        if use_camera:
            self.camera_encoder = ResNetCameraEncoder(
                backbone='resnet50', 
                pretrained=True
            )
        
        if use_lidar:
            self.lidar_encoder = PointNetLiDAREncoder(
                input_channels=5, 
                feat_dim=1024
            )
        
        if use_radar:
            self.radar_encoder = MultiRadarEncoder(
                input_channels=7, 
                feat_dim=256, 
                num_radars=5
            )
        
        # Fusion module
        if fusion_type == 'bev':
            self.fusion = FlexibleBEVFusion(
                use_camera=use_camera,
                use_lidar=use_lidar,
                use_radar=use_radar,
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
            self.fusion = FlexibleAttentionFusion(
                use_camera=use_camera,
                use_lidar=use_lidar,
                use_radar=use_radar,
                camera_channels=2048,
                lidar_channels=1024,
                radar_channels=256,
                hidden_dim=512
            )
            fusion_out_channels = 512
            is_spatial = False
            
        elif fusion_type == 'late':
            self.fusion = FlexibleLateFusion(
                use_camera=use_camera,
                use_lidar=use_lidar,
                use_radar=use_radar,
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
        if is_spatial and detection_head == 'centernet':
            self.det_head = CenterNetHead(
                in_channels=fusion_out_channels,
                num_classes=num_classes
            )
        else:
            # Use MLP head for non-spatial features
            self.det_head = MLPDetectionHead(
                in_channels=fusion_out_channels,
                num_classes=num_classes
            )
    
    def forward(
        self,
        camera_imgs: Optional[torch.Tensor] = None,
        lidar_points: Optional[torch.Tensor] = None,
        radar_points: Optional[List[torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass - automatically handles available modalities
        
        Args:
            camera_imgs: (B, 6, 3, H, W) or None
            lidar_points: (B, N, 5) or None
            radar_points: List of 5 tensors, each (B, N_i, 7) or None
        
        Returns:
            predictions: Dict of detection outputs
        """
        # Extract features (only for enabled modalities)
        cam_feat = None
        if self.use_camera and camera_imgs is not None:
            cam_feat = self.camera_encoder(camera_imgs)
        
        lidar_feat = None
        if self.use_lidar and lidar_points is not None:
            lidar_feat = self.lidar_encoder(lidar_points)
        
        radar_feat = None
        if self.use_radar and radar_points is not None:
            radar_feat = self.radar_encoder(radar_points)
        
        # Fuse features
        fused_feat = self.fusion(
            camera_features=cam_feat,
            lidar_features=lidar_feat,
            radar_features=radar_feat
        )
        
        # Detection
        predictions = self.det_head(fused_feat)
        
        return predictions
    
    def get_config_str(self) -> str:
        """Get model configuration string"""
        return f"{self.fusion.get_config_str()}_{self.fusion_type}_{self.detection_head_type}"


# ============================================================================
# 6. MODEL FACTORY
# ============================================================================

def create_detector(
    modality_config: str,
    fusion_type: str = 'bev',
    detection_head: str = 'centernet',
    num_classes: int = 10,
    **kwargs
) -> FlexibleMultiModal3DDetector:
    """
    Factory function to create detector with specified modality configuration
    
    Args:
        modality_config: One of:
            - 'camera_only', 'lidar_only', 'radar_only'
            - 'camera+lidar', 'camera+radar', 'lidar+radar'
            - 'camera+lidar+radar' or 'all'
        fusion_type: 'bev', 'attention', or 'late'
        detection_head: 'centernet' or 'mlp'
        num_classes: Number of detection classes
    
    Returns:
        Configured detector model
    
    Examples:
        >>> # Camera only with BEV fusion
        >>> model = create_detector('camera_only', 'bev')
        
        >>> # LiDAR + Radar with attention fusion
        >>> model = create_detector('lidar+radar', 'attention')
        
        >>> # All modalities with late fusion
        >>> model = create_detector('all', 'late')
    """
    # Parse modality configuration
    modality_config = modality_config.lower().replace(' ', '')
    
    use_camera = 'camera' in modality_config or modality_config == 'all'
    use_lidar = 'lidar' in modality_config or modality_config == 'all'
    use_radar = 'radar' in modality_config or modality_config == 'all'
    
    # Create model
    model = FlexibleMultiModal3DDetector(
        use_camera=use_camera,
        use_lidar=use_lidar,
        use_radar=use_radar,
        num_classes=num_classes,
        fusion_type=fusion_type,
        detection_head=detection_head,
        **kwargs
    )
    
    return model


# ============================================================================
# 7. TESTING AND EXAMPLES
# ============================================================================

def test_all_configurations():
    """Test all modality combinations with all fusion types"""
    
    print("="*80)
    print("TESTING ALL MODALITY CONFIGURATIONS")
    print("="*80)
    
    # All modality configurations
    modality_configs = [
        'camera_only',
        'lidar_only',
        'radar_only',
        'camera+lidar',
        'camera+radar',
        'lidar+radar',
        'camera+lidar+radar'
    ]
    
    fusion_types = ['bev', 'attention', 'late']
    
    batch_size = 2
    
    # Prepare dummy data
    camera_imgs = torch.randn(batch_size, 6, 3, 448, 800)
    lidar_points = torch.randn(batch_size, 34720, 5)
    radar_points = [torch.randn(batch_size, 125, 7) for _ in range(5)]
    
    results = []
    
    for modality_config in modality_configs:
        for fusion_type in fusion_types:
            try:
                # Create model
                model = create_detector(
                    modality_config=modality_config,
                    fusion_type=fusion_type,
                    detection_head='centernet' if fusion_type == 'bev' else 'mlp',
                    num_classes=10
                )
                
                model.eval()
                
                # Prepare inputs based on modality
                inputs = {}
                if 'camera' in modality_config:
                    inputs['camera_imgs'] = camera_imgs
                if 'lidar' in modality_config:
                    inputs['lidar_points'] = lidar_points
                if 'radar' in modality_config:
                    inputs['radar_points'] = radar_points
                
                # Forward pass
                with torch.no_grad():
                    predictions = model(**inputs)
                
                # Count parameters
                params = sum(p.numel() for p in model.parameters())
                
                config_str = model.get_config_str()
                status = "✓ PASS"
                
                results.append({
                    'config': modality_config,
                    'fusion': fusion_type,
                    'config_str': config_str,
                    'params': params,
                    'status': status
                })
                
                print(f"\n{status} {modality_config:<20} + {fusion_type:<10}")
                print(f"   Config: {config_str}")
                print(f"   Parameters: {params:,}")
                
                # Print output shapes
                for key, value in predictions.items():
                    if isinstance(value, torch.Tensor):
                        print(f"   {key}: {value.shape}")
                
            except Exception as e:
                status = "✗ FAIL"
                results.append({
                    'config': modality_config,
                    'fusion': fusion_type,
                    'status': status,
                    'error': str(e)
                })
                print(f"\n{status} {modality_config:<20} + {fusion_type:<10}")
                print(f"   Error: {e}")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    passed = sum(1 for r in results if r['status'] == "✓ PASS")
    total = len(results)
    
    print(f"\nTotal tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    
    return results


if __name__ == '__main__':
    print("\n" + "="*80)
    print("FLEXIBLE MULTI-MODAL 3D DETECTION")
    print("="*80)
    
    print("\nSupported Configurations:")
    print("  Modalities: camera_only, lidar_only, radar_only,")
    print("             camera+lidar, camera+radar, lidar+radar,")
    print("             camera+lidar+radar (all)")
    print("  Fusion: bev, attention, late")
    print("  Detection: centernet, mlp")
    
    print("\n" + "="*80)
    print("EXAMPLE USAGE")
    print("="*80)
    
    examples = [
        ("Camera only with BEV", "camera_only", "bev"),
        ("LiDAR only with Attention", "lidar_only", "attention"),
        ("Camera+LiDAR with Late", "camera+lidar", "late"),
        ("All modalities with BEV", "all", "bev"),
    ]
    
    for name, modality, fusion in examples:
        print(f"\n{name}:")
        print(f"  model = create_detector('{modality}', '{fusion}')")
    
    print("\n" + "="*80)
    print("RUNNING TESTS")
    print("="*80)
    
    test_all_configurations()
    
    print("\n" + "="*80)
    print("ALL TESTS COMPLETE!")
    print("="*80)