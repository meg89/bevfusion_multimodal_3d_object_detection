"""
Refined Multi-Modal Fusion for 3D Object Detection
Supports flexible modality selection with config.yaml support:
- Single modality: camera_only, lidar_only, radar_only
- Dual modality: camera+lidar, camera+radar, lidar+radar
- Triple modality: camera+lidar+radar

All three fusion methods (BEV, Attention, Late) support all combinations
Now supports initialization from config.yaml
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import math
import yaml
from pathlib import Path
from encoders import ResNetCameraEncoder, PointNetLiDAREncoder, MultiRadarEncoder


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
# 1. FLEXIBLE BEV FUSION
# ============================================================================

class FlexibleBEVFusion(nn.Module):
    """
    BEV Fusion with flexible modality selection
    Automatically adapts to available modalities
    Can be initialized from config.yaml or with direct parameters
    
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
        use_camera: Optional[bool] = None,
        use_lidar: Optional[bool] = None,
        use_radar: Optional[bool] = None,
        camera_channels: Optional[int] = None,
        lidar_channels: Optional[int] = None,
        radar_channels: Optional[int] = None,
        bev_h: Optional[int] = None,
        bev_w: Optional[int] = None,
        bev_channels: Optional[int] = None,
        pc_range: Optional[List[float]] = None,
        config: Optional[Dict] = None,
        config_path: Optional[str] = None
    ):
        super(FlexibleBEVFusion, self).__init__()
        
        # Load from config if provided
        if config is not None or config_path is not None:
            if config is None:
                config = load_config(config_path)
            
            # Extract configuration
            model_cfg = config.get('model', {})
            bev_cfg = model_cfg.get('bev_fusion', {})
            camera_enc_cfg = model_cfg.get('camera_encoder', {})
            lidar_enc_cfg = model_cfg.get('lidar_encoder', {})
            radar_enc_cfg = model_cfg.get('radar_encoder', {})
            dataset_cfg = config.get('dataset', {})
            
            # Modality flags
            self.use_camera = model_cfg.get('use_camera', True) if use_camera is None else use_camera
            self.use_lidar = model_cfg.get('use_lidar', True) if use_lidar is None else use_lidar
            self.use_radar = model_cfg.get('use_radar', True) if use_radar is None else use_radar
            
            # Encoder output channels
            camera_channels = camera_enc_cfg.get('output_channels', 512) if camera_channels is None else camera_channels
            lidar_channels = lidar_enc_cfg.get('feature_dim', 1024) if lidar_channels is None else lidar_channels
            radar_channels = radar_enc_cfg.get('feature_dim', 256) if radar_channels is None else radar_channels
            
            # BEV parameters
            self.bev_h = bev_cfg.get('bev_h', dataset_cfg.get('bev_h', 200)) if bev_h is None else bev_h
            self.bev_w = bev_cfg.get('bev_w', dataset_cfg.get('bev_w', 200)) if bev_w is None else bev_w
            self.bev_channels = bev_cfg.get('bev_channels', 256) if bev_channels is None else bev_channels
            self.pc_range = dataset_cfg.get('point_cloud_range', [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]) if pc_range is None else pc_range
        else:
            # Use direct parameters
            self.use_camera = use_camera if use_camera is not None else True
            self.use_lidar = use_lidar if use_lidar is not None else True
            self.use_radar = use_radar if use_radar is not None else True
            camera_channels = camera_channels if camera_channels is not None else 512
            lidar_channels = lidar_channels if lidar_channels is not None else 1024
            radar_channels = radar_channels if radar_channels is not None else 256
            self.bev_h = bev_h if bev_h is not None else 200
            self.bev_w = bev_w if bev_w is not None else 200
            self.bev_channels = bev_channels if bev_channels is not None else 256
            self.pc_range = pc_range if pc_range is not None else [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
        
        # Count active modalities
        self.num_modalities = sum([self.use_camera, self.use_lidar, self.use_radar])
        assert self.num_modalities > 0, "At least one modality must be enabled"
        
        # Camera to BEV projection
        if self.use_camera:
            self.camera_proj = nn.Sequential(
                nn.Conv2d(camera_channels, 512, 3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, self.bev_channels, 1),
                nn.BatchNorm2d(self.bev_channels),
                nn.ReLU(inplace=True)
            )
        
        # =====================================================================
        # LIDAR TO BEV (EFFICIENT - Progressive Upsampling)
        # Use: Small initial + progressive upsample = ~1M params
        # =====================================================================
        if self.use_lidar:
            hidden_dim = 128
            start_size = 25  # Start with 25x25 instead of 200x200
            
            # Small initial projection: 1024 -> 128*25*25 = only 81K entries!
            self.lidar_init = nn.Sequential(
                nn.Linear(lidar_channels, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, hidden_dim * start_size * start_size)
            )
            
            # Progressive upsampling: 25→50→100→200
            self.lidar_upsample = nn.Sequential(
                # 25→50
                nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                
                # 50→100
                #nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
                #nn.BatchNorm2d(hidden_dim),
                #nn.ReLU(inplace=True),
                #nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                
                # 100→200
                #nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
                #nn.BatchNorm2d(hidden_dim),
                #nn.ReLU(inplace=True),
                #nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                
                # Final projection to bev_channels
                nn.Conv2d(hidden_dim, self.bev_channels, 3, padding=1),
                nn.BatchNorm2d(self.bev_channels),
                nn.ReLU(inplace=True)
            )
            
            self.lidar_start_size = start_size
        
        # =====================================================================
        # RADAR TO BEV (EFFICIENT - Even Smaller)
        # =====================================================================
        if self.use_radar:
            # Simple projection then broadcast
            self.radar_proj = nn.Sequential(
                nn.Linear(radar_channels, self.bev_channels),
                nn.ReLU(inplace=True)
            )
            
            # Refine broadcasted features
            self.radar_refine = nn.Sequential(
                nn.Conv2d(self.bev_channels, self.bev_channels, 3, padding=1),
                nn.BatchNorm2d(self.bev_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.bev_channels, self.bev_channels, 3, padding=1),
                nn.BatchNorm2d(self.bev_channels),
                nn.ReLU(inplace=True)
            )
        
        # BEV fusion network (adapts to number of modalities)
        fusion_in_channels = self.bev_channels * self.num_modalities
        self.bev_fusion = nn.Sequential(
            nn.Conv2d(fusion_in_channels, self.bev_channels * 2, 3, padding=1),
            nn.BatchNorm2d(self.bev_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.bev_channels * 2, self.bev_channels, 3, padding=1),
            nn.BatchNorm2d(self.bev_channels),
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
        B= None
        # =====================================================================
        # Process Camera (already spatial)
        # =====================================================================
        if self.use_camera and camera_features is not None:
            B = camera_features.shape[0]
            
            # Handle multi-camera input
            if len(camera_features.shape) == 5:  # (B, num_cams, C, H, W)
                cam_bev = camera_features.mean(dim=1)  # Pool over cameras
            else:
                cam_bev = camera_features
            
            # Project to BEV channels
            cam_bev = self.camera_proj(cam_bev)  # (B, bev_channels, H', W')
            
            # Resize to BEV size
            cam_bev = F.interpolate(
                cam_bev, 
                size=(self.bev_h, self.bev_w),
                mode='bilinear', 
                align_corners=False
            )
            bev_features.append(cam_bev)
        
        # =====================================================================
        # Process LiDAR (global → spatial via progressive upsampling)
        # =====================================================================
        if self.use_lidar and lidar_features is not None:
            if B is None:
                B = lidar_features.shape[0]
            
            # Initial projection to small spatial size
            lidar_flat = self.lidar_init(lidar_features)  # (B, 128*25*25)
            lidar_bev = lidar_flat.view(B, 128, self.lidar_start_size, self.lidar_start_size)
            
            # Progressive upsampling to full BEV size
            lidar_bev = self.lidar_upsample(lidar_bev)  # (B, bev_channels, 200, 200)
            
            bev_features.append(lidar_bev)
        
        # =====================================================================
        # Process Radar (global → spatial via broadcast)
        # =====================================================================
        if self.use_radar and radar_features is not None:
            if B is None:
                B = radar_features.shape[0]
            
            # Project to BEV channels
            radar_proj = self.radar_proj(radar_features)  # (B, bev_channels)
            
            # Broadcast to spatial dimensions
            radar_bev = radar_proj.view(B, self.bev_channels, 1, 1)
            radar_bev = radar_bev.expand(B, self.bev_channels, self.bev_h, self.bev_w)
            
            # Refine with conv
            radar_bev = self.radar_refine(radar_bev)  # (B, bev_channels, bev_h, bev_w)
            
            bev_features.append(radar_bev)
        
        # =====================================================================
        # Fuse all modalities
        # =====================================================================
        if len(bev_features) == 0:
            raise ValueError("No modality features provided")
        
        # Concatenate
        bev_concat = torch.cat(bev_features, dim=1)  # (B, bev_channels*num_modalities, H, W)
        #print("[DEBUG] bev_concat shape: ", len(bev_concat.shape) )
        # Fuse
        bev_fused = self.bev_fusion(bev_concat)  # (B, bev_channels, H, W)
        #print("[DEBUG] bev_fused shape: ", len(bev_fused.shape) )
        return bev_fused
    
    def get_config_str(self) -> str:
        """Get configuration string"""
        modalities = []
        if self.use_camera: modalities.append("camera")
        if self.use_lidar: modalities.append("lidar")
        if self.use_radar: modalities.append("radar")
        return "+".join(modalities)

    def count_parameters(self) -> Dict[str, int]:
        """Count parameters by component"""
        counts = {}
        
        if self.use_camera:
            counts['camera_proj'] = sum(p.numel() for p in self.camera_proj.parameters())
        
        if self.use_lidar:
            counts['lidar_init'] = sum(p.numel() for p in self.lidar_init.parameters())
            counts['lidar_upsample'] = sum(p.numel() for p in self.lidar_upsample.parameters())
            counts['lidar_total'] = counts['lidar_init'] + counts['lidar_upsample']
        
        if self.use_radar:
            counts['radar_proj'] = sum(p.numel() for p in self.radar_proj.parameters())
            counts['radar_refine'] = sum(p.numel() for p in self.radar_refine.parameters())
            counts['radar_total'] = counts['radar_proj'] + counts['radar_refine']
        
        counts['bev_fusion'] = sum(p.numel() for p in self.bev_fusion.parameters())
        counts['total'] = sum(p.numel() for p in self.parameters())
        
        return counts

# ============================================================================
# 2. FLEXIBLE ATTENTION FUSION
# ============================================================================

class SpatialReshaper(nn.Module):
    """
    Lightweight spatial reshaper using broadcast method
    Only ~50K parameters - very efficient!
    """
    
    def __init__(
        self,
        input_channels: int = 512,
        output_channels: int = 512,
        bev_size: tuple = (200, 200)
    ):
        super().__init__()
        
        self.bev_h, self.bev_w = bev_size
        
        # Simple channel projection
        self.channel_proj = nn.Linear(input_channels, output_channels)
        
        # Refine broadcasted features with conv
        self.refine = nn.Sequential(
            nn.Conv2d(output_channels, output_channels, 3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, 3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert global features to spatial
        
        Args:
            x: (B, C) global features or (B, C, H, W) spatial features
        
        Returns:
            (B, C, H, W) spatial features
        """
        # If already spatial, pass through
        if len(x.shape) == 4:
            return x
        
        B = x.size(0)
        
        # Project channels
        x = self.channel_proj(x)  # (B, output_channels)
        
        # Broadcast to spatial dimensions
        x = x.view(B, -1, 1, 1)  # (B, C, 1, 1)
        x = x.expand(B, -1, self.bev_h, self.bev_w)  # (B, C, H, W)
        
        # Refine with conv
        x = self.refine(x)  # (B, C, H, W)
        
        return x

    
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
    Can be initialized from config.yaml or with direct parameters
    
    Supported combinations:
    - Single: camera_only, lidar_only, radar_only
    - Dual: camera+lidar, camera+radar, lidar+radar
    - Triple: camera+lidar+radar
    """
    
    def __init__(
        self,
        use_camera: Optional[bool] = None,
        use_lidar: Optional[bool] = None,
        use_radar: Optional[bool] = None,
        camera_channels: Optional[int] = None,
        lidar_channels: Optional[int] = None,
        radar_channels: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        num_heads: Optional[int] = None,
        num_layers: Optional[int] = None,
        config: Optional[Dict] = None,
        config_path: Optional[str] = None
    ):
        super(FlexibleAttentionFusion, self).__init__()
        
        # Load from config if provided
        if config is not None or config_path is not None:
            if config is None:
                config = load_config(config_path)
            
            # Extract configuration
            model_cfg = config.get('model', {})
            attn_cfg = model_cfg.get('attention_fusion', {})
            camera_enc_cfg = model_cfg.get('camera_encoder', {})
            lidar_enc_cfg = model_cfg.get('lidar_encoder', {})
            radar_enc_cfg = model_cfg.get('radar_encoder', {})
            
            # Modality flags
            self.use_camera = model_cfg.get('use_camera', True) if use_camera is None else use_camera
            self.use_lidar = model_cfg.get('use_lidar', True) if use_lidar is None else use_lidar
            self.use_radar = model_cfg.get('use_radar', True) if use_radar is None else use_radar
            
            # Encoder output channels
            camera_channels = camera_enc_cfg.get('output_channels', 512) if camera_channels is None else camera_channels
            lidar_channels = lidar_enc_cfg.get('feature_dim', 1024) if lidar_channels is None else lidar_channels
            radar_channels = radar_enc_cfg.get('feature_dim', 256) if radar_channels is None else radar_channels
            
            # Attention parameters
            self.hidden_dim = attn_cfg.get('hidden_dim', 512) if hidden_dim is None else hidden_dim
            num_heads = attn_cfg.get('num_heads', 8) if num_heads is None else num_heads
            num_layers = attn_cfg.get('num_layers', 2) if num_layers is None else num_layers
            dropout = attn_cfg.get('dropout', 0.1)
        else:
            # Use direct parameters
            self.use_camera = use_camera if use_camera is not None else True
            self.use_lidar = use_lidar if use_lidar is not None else True
            self.use_radar = use_radar if use_radar is not None else True
            camera_channels = camera_channels if camera_channels is not None else 512
            lidar_channels = lidar_channels if lidar_channels is not None else 1024
            radar_channels = radar_channels if radar_channels is not None else 256
            self.hidden_dim = hidden_dim if hidden_dim is not None else 512
            num_heads = num_heads if num_heads is not None else 8
            num_layers = num_layers if num_layers is not None else 2
            dropout = 0.1
        
        # Count active modalities
        self.num_modalities = sum([self.use_camera, self.use_lidar, self.use_radar])
        assert self.num_modalities > 0, "At least one modality must be enabled"
        
        # Project all modalities to same dimension
        if self.use_camera:
            self.camera_proj = nn.Linear(camera_channels, self.hidden_dim)
            self.cam_pos_embed = nn.Parameter(torch.randn(1, 1, self.hidden_dim))
        
        if self.use_lidar:
            self.lidar_proj = nn.Linear(lidar_channels, self.hidden_dim)
            self.lidar_pos_embed = nn.Parameter(torch.randn(1, 1, self.hidden_dim))
        
        if self.use_radar:
            self.radar_proj = nn.Linear(radar_channels, self.hidden_dim)
            self.radar_pos_embed = nn.Parameter(torch.randn(1, 1, self.hidden_dim))
        
        # Self-attention layers (within combined features)
        self.self_attention_layers = nn.ModuleList([
            nn.ModuleDict({
                'self_attn': CrossModalAttention(
                    self.hidden_dim, self.hidden_dim, self.hidden_dim, num_heads
                ),
                'ffn': nn.Sequential(
                    nn.Linear(self.hidden_dim, self.hidden_dim * 4),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(self.hidden_dim * 4, self.hidden_dim),
                    nn.Dropout(dropout)
                ),
                'norm1': nn.LayerNorm(self.hidden_dim),
                'norm2': nn.LayerNorm(self.hidden_dim)
            })
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim)
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
    Can be initialized from config.yaml or with direct parameters
    
    Supported combinations:
    - Single: camera_only, lidar_only, radar_only
    - Dual: camera+lidar, camera+radar, lidar+radar
    - Triple: camera+lidar+radar
    """
    
    def __init__(
        self,
        use_camera: Optional[bool] = None,
        use_lidar: Optional[bool] = None,
        use_radar: Optional[bool] = None,
        camera_channels: Optional[int] = None,
        lidar_channels: Optional[int] = None,
        radar_channels: Optional[int] = None,
        output_dim: Optional[int] = None,
        config: Optional[Dict] = None,
        config_path: Optional[str] = None
    ):
        super(FlexibleLateFusion, self).__init__()
        
        # Load from config if provided
        if config is not None or config_path is not None:
            if config is None:
                config = load_config(config_path)
            
            # Extract configuration
            model_cfg = config.get('model', {})
            late_cfg = model_cfg.get('late_fusion', {})
            camera_enc_cfg = model_cfg.get('camera_encoder', {})
            lidar_enc_cfg = model_cfg.get('lidar_encoder', {})
            radar_enc_cfg = model_cfg.get('radar_encoder', {})
            
            # Modality flags
            self.use_camera = model_cfg.get('use_camera', True) if use_camera is None else use_camera
            self.use_lidar = model_cfg.get('use_lidar', True) if use_lidar is None else use_lidar
            self.use_radar = model_cfg.get('use_radar', True) if use_radar is None else use_radar
            
            # Encoder output channels
            camera_channels = camera_enc_cfg.get('output_channels', 512) if camera_channels is None else camera_channels
            lidar_channels = lidar_enc_cfg.get('feature_dim', 1024) if lidar_channels is None else lidar_channels
            radar_channels = radar_enc_cfg.get('feature_dim', 256) if radar_channels is None else radar_channels
            
            # Late fusion parameters
            output_dim = late_cfg.get('output_dim', 512) if output_dim is None else output_dim
            dropout = late_cfg.get('dropout', 0.3)
        else:
            # Use direct parameters
            self.use_camera = use_camera if use_camera is not None else True
            self.use_lidar = use_lidar if use_lidar is not None else True
            self.use_radar = use_radar if use_radar is not None else True
            camera_channels = camera_channels if camera_channels is not None else 512
            lidar_channels = lidar_channels if lidar_channels is not None else 1024
            radar_channels = radar_channels if radar_channels is not None else 256
            output_dim = output_dim if output_dim is not None else 512
            dropout = 0.3
        
        # Count active modalities
        self.num_modalities = sum([self.use_camera, self.use_lidar, self.use_radar])
        assert self.num_modalities > 0, "At least one modality must be enabled"
        
        # Calculate total input dimension
        total_dim = 0
        if self.use_camera:
            total_dim += camera_channels
        if self.use_lidar:
            total_dim += lidar_channels
        if self.use_radar:
            total_dim += radar_channels
        
        # Fusion MLP (adapts to total dimension)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(total_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
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
# 4. DETECTION HEADS
# ============================================================================

class CenterNetHead(nn.Module):
    """
    CenterNet-style detection head for 3D object detection
    Can be initialized from config.yaml or with direct parameters
    """
    
    def __init__(
        self,
        in_channels: Optional[int] = None,
        num_classes: Optional[int] = None,
        head_conv: Optional[int] = None,
        config: Optional[Dict] = None,
        config_path: Optional[str] = None
    ):
        super(CenterNetHead, self).__init__()
        
        # Load from config if provided
        if config is not None or config_path is not None:
            if config is None:
                config = load_config(config_path)
            
            model_cfg = config.get('model', {})
            centernet_cfg = model_cfg.get('centernet_head', {})
            dataset_cfg = config.get('dataset', {})
            
            in_channels = centernet_cfg.get('in_channels', 256) if in_channels is None else in_channels
            self.num_classes = dataset_cfg.get('num_classes', 10) if num_classes is None else num_classes
            head_conv = centernet_cfg.get('head_conv', 64) if head_conv is None else head_conv
        else:
            in_channels = in_channels if in_channels is not None else 256
            self.num_classes = num_classes if num_classes is not None else 10
            head_conv = head_conv if head_conv is not None else 64
        
        # Heatmap head
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(in_channels, head_conv, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, self.num_classes, 1, bias=True)
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
    Can be initialized from config.yaml or with direct parameters
    """
    
    def __init__(
        self,
        in_channels: Optional[int] = None,
        num_classes: Optional[int] = None,
        config: Optional[Dict] = None,
        config_path: Optional[str] = None
    ):
        super(MLPDetectionHead, self).__init__()
        
        # Load from config if provided
        if config is not None or config_path is not None:
            if config is None:
                config = load_config(config_path)
            
            model_cfg = config.get('model', {})
            mlp_cfg = model_cfg.get('mlp_head', {})
            dataset_cfg = config.get('dataset', {})
            
            in_channels = mlp_cfg.get('in_channels', 512) if in_channels is None else in_channels
            self.num_classes = dataset_cfg.get('num_classes', 10) if num_classes is None else num_classes
            dropout = mlp_cfg.get('dropout', 0.1)
        else:
            in_channels = in_channels if in_channels is not None else 512
            self.num_classes = num_classes if num_classes is not None else 10
            dropout = 0.1
        
        self.head = nn.Sequential(
            nn.Linear(in_channels, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, self.num_classes + 7)  # class + box (x,y,z,w,l,h,yaw)
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
    Can be initialized from config.yaml or with direct parameters
    
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
        use_camera: Optional[bool] = None,
        use_lidar: Optional[bool] = None,
        use_radar: Optional[bool] = None,
        # Architecture
        num_classes: Optional[int] = None,
        fusion_type: Optional[str] = None,
        detection_head: Optional[str] = None,
        # BEV parameters
        bev_h: Optional[int] = None,
        bev_w: Optional[int] = None,
        # Config
        config: Optional[Dict] = None,
        config_path: Optional[str] = None
    ):
        super(FlexibleMultiModal3DDetector, self).__init__()
        
        # Load from config if provided
        if config is not None or config_path is not None:
            if config is None:
                config = load_config(config_path)
            
            model_cfg = config.get('model', {})
            dataset_cfg = config.get('dataset', {})
            
            # Extract parameters from config
            self.use_camera = model_cfg.get('use_camera', True) if use_camera is None else use_camera
            self.use_lidar = model_cfg.get('use_lidar', True) if use_lidar is None else use_lidar
            self.use_radar = model_cfg.get('use_radar', True) if use_radar is None else use_radar
            num_classes = dataset_cfg.get('num_classes', 10) if num_classes is None else num_classes
            self.fusion_type = model_cfg.get('fusion_type', 'bev') if fusion_type is None else fusion_type
            self.detection_head_type = model_cfg.get('detection_head', 'centernet') if detection_head is None else detection_head
            bev_h = dataset_cfg.get('bev_h', 50) if bev_h is None else bev_h
            bev_w = dataset_cfg.get('bev_w', 50) if bev_w is None else bev_w
        else:
            # Use direct parameters
            self.use_camera = use_camera if use_camera is not None else True
            self.use_lidar = use_lidar if use_lidar is not None else True
            self.use_radar = use_radar if use_radar is not None else True
            num_classes = num_classes if num_classes is not None else 10
            self.fusion_type = fusion_type if fusion_type is not None else 'bev'
            self.detection_head_type = detection_head if detection_head is not None else 'centernet'
            bev_h = bev_h if bev_h is not None else 50
            bev_w = bev_w if bev_w is not None else 50
        
        # Validate at least one modality
        num_modalities = sum([self.use_camera, self.use_lidar, self.use_radar])
        assert num_modalities > 0, "At least one modality must be enabled"
        
        # Initialize encoders (only for enabled modalities)
        if self.use_camera:
            if config is not None or config_path is not None:
                self.camera_encoder = ResNetCameraEncoder(config=config, config_path=config_path)
            else:
                self.camera_encoder = ResNetCameraEncoder(backbone='resnet18', pretrained=True)
        
        if self.use_lidar:
            if config is not None or config_path is not None:
                self.lidar_encoder = PointNetLiDAREncoder(config=config, config_path=config_path)
            else:
                self.lidar_encoder = PointNetLiDAREncoder(input_channels=4, feat_dim=1024)
        
        if self.use_radar:
            if config is not None or config_path is not None:
                self.radar_encoder = MultiRadarEncoder(config=config, config_path=config_path)
                print ("[DEBUG]: Initializing MultiRadarEncoder encoder")
            else:
                self.radar_encoder = MultiRadarEncoder(input_channels=7, feat_dim=256, num_radars=5)
        
        # Fusion module
        if self.fusion_type == 'bev':
            self.fusion = FlexibleBEVFusion(
                use_camera=self.use_camera,
                use_lidar=self.use_lidar,
                use_radar=self.use_radar,
                bev_h=bev_h,
                bev_w=bev_w,
                config=config,
                config_path=config_path
            )
            fusion_out_channels = self.fusion.bev_channels
            is_spatial = True
            
        elif self.fusion_type == 'attention':
            self.fusion = FlexibleAttentionFusion(
                use_camera=self.use_camera,
                use_lidar=self.use_lidar,
                use_radar=self.use_radar,
                config=config,
                config_path=config_path
            )
            fusion_out_channels = self.fusion.hidden_dim
            is_spatial = False
            
        elif self.fusion_type == 'late':
            self.fusion = FlexibleLateFusion(
                use_camera=self.use_camera,
                use_lidar=self.use_lidar,
                use_radar=self.use_radar,
                config=config,
                config_path=config_path
            )
            fusion_out_channels = 512  # Default output_dim
            is_spatial = False
        else:
            raise ValueError(f"Unknown fusion type: {self.fusion_type}")
        
        # Detection head
        if is_spatial and self.detection_head_type == 'centernet':
            self.det_head = CenterNetHead(
                in_channels=fusion_out_channels,
                num_classes=num_classes,
                config=config,
                config_path=config_path
            )
        else:
            # Use MLP head for non-spatial features
            self.det_head = MLPDetectionHead(
                in_channels=fusion_out_channels,
                num_classes=num_classes,
                config=config,
                config_path=config_path
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
        #print("fused_feat shape: ", fused_feat.shape )
        # Detection
        # ✅ ADD THIS
        #if len(fused_feat.shape) == 2:
        #    print("using spatial reshaper")
        #    if not hasattr(self, 'spatial_reshaper'):
        #        self.spatial_reshaper = SpatialReshaper(512, 512, (50, 50)).to(fused_feat.device)
        #    fused_feat = self.spatial_reshaper(fused_feat)

        predictions = self.det_head(fused_feat)
        
        return predictions
    
    def get_config_str(self) -> str:
        """Get model configuration string"""
        return f"{self.fusion.get_config_str()}_{self.fusion_type}_{self.detection_head_type}"


# ============================================================================
# 6. MODEL FACTORY
# ============================================================================

def create_detector(
    modality_config: Optional[str] = None,
    fusion_type: Optional[str] = None,
    detection_head: Optional[str] = None,
    num_classes: Optional[int] = None,
    config: Optional[Dict] = None,
    config_path: Optional[str] = None,
    **kwargs
) -> FlexibleMultiModal3DDetector:
    """
    Factory function to create detector with specified modality configuration
    Can load from config.yaml or use direct parameters
    
    Args:
        modality_config: One of:
            - 'camera_only', 'lidar_only', 'radar_only'
            - 'camera+lidar', 'camera+radar', 'lidar+radar'
            - 'camera+lidar+radar' or 'all'
        fusion_type: 'bev', 'attention', or 'late'
        detection_head: 'centernet' or 'mlp'
        num_classes: Number of detection classes
        config: Configuration dictionary
        config_path: Path to config.yaml
    
    Returns:
        Configured detector model
    
    Examples:
        >>> # Method 1: Direct parameters
        >>> model = create_detector('camera_only', 'bev')
        
        >>> # Method 2: From config file
        >>> model = create_detector(config_path='config.yaml')
        
        >>> # Method 3: Hybrid (config + override)
        >>> model = create_detector(config_path='config.yaml', fusion_type='attention')
    """
    # Load from config if provided
    if config is not None or config_path is not None:
        if config is None:
            config = load_config(config_path)
        
        model_cfg = config.get('model', {})
        
        # Get modality config from config file if not provided
        if modality_config is None:
            modality_config = model_cfg.get('modality_config', 'all')
    
    # Parse modality configuration if provided
    if modality_config is not None:
        modality_config = modality_config.lower().replace(' ', '')
        
        use_camera = 'camera' in modality_config or modality_config == 'all'
        use_lidar = 'lidar' in modality_config or modality_config == 'all'
        use_radar = 'radar' in modality_config or modality_config == 'all'
    else:
        use_camera = None
        use_lidar = None
        use_radar = None
    
    # Create model
    model = FlexibleMultiModal3DDetector(
        use_camera=use_camera,
        use_lidar=use_lidar,
        use_radar=use_radar,
        num_classes=num_classes,
        fusion_type=fusion_type,
        detection_head=detection_head,
        config=config,
        config_path=config_path,
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
        #'camera_only',
        #'lidar_only',
        #'radar_only',
        'camera+lidar',
        #'camera+radar',
        #'lidar+radar',
        'camera+lidar+radar'
    ]
    
    fusion_types = ['bev', 'attention', 'late']
    #fusion_types = ['attention']
    #fusion_types = ['attention', 'late']
    batch_size = 2
    
    # Prepare dummy data
    camera_imgs = torch.randn(batch_size, 3, 3, 448, 800)
    lidar_points = torch.randn(batch_size, 34720, 4)
    radar_points = [torch.randn(batch_size, 125, 7) for _ in range(3)]
    
    results = []
    
    for modality_config in modality_configs:
        for fusion_type in fusion_types:
            try:
                # Create model
                model = create_detector(
                    modality_config=modality_config,
                    fusion_type=fusion_type,
                    detection_head='centernet' if fusion_type == 'bev' or 'attention' else 'mlp',
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
    print("FLEXIBLE MULTI-MODAL 3D DETECTION WITH CONFIG SUPPORT")
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
    
    print("\n1. Direct Parameters (Original Method):")
    print("  model = create_detector('camera_only', 'bev')")
    
    print("\n2. From Config File (NEW):")
    print("  model = create_detector(config_path='config.yaml')")
    
    print("\n3. Hybrid (Config + Override):")
    print("  model = create_detector(config_path='config.yaml', fusion_type='attention')")
    
    print("\n" + "="*80)
    print("RUNNING TESTS")
    print("="*80)
    
    test_all_configurations()
    
    print("\n" + "="*80)
    print("Testing with config.yaml (if available)...")
    print("="*80)
    
    try:
        model_from_config = create_detector(config_path='configs/base.yaml')
        print("✓ Successfully created model from configs/base.yaml")
        print(f"  Config: {model_from_config.get_config_str()}")
    except FileNotFoundError:
        print("⚠ config.yaml not found (this is OK for testing)")
    
    print("\n" + "="*80)
    print("ALL TESTS COMPLETE!")
    print("="*80)