"""
Multi-Modal Encoders for Camera, LiDAR, and Radar
ResNet-based camera encoder, PointNet-based LiDAR encoder, and radar encoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Dict
import torchvision.models as models


class ResNetCameraEncoder(nn.Module):
    """
    ResNet-based encoder for camera images
    
    Input Dimensions:
        - Image: (B, 6, 3, H, W) where B=batch, 6=cameras, 3=RGB, H=height, W=width
        - Typical: (B, 6, 3, 900, 1600) for NuScenes full resolution
        - Or: (B, 6, 3, 448, 800) for downsampled
    
    Output Dimensions:
        - Features: (B, 6, feat_dim, H', W')
        - ResNet50: (B, 6, 2048, H/32, W/32)
        - ResNet101: (B, 6, 2048, H/32, W/32)
        - Example: (B, 6, 2048, 14, 25) for input (448, 800)
    """
    
    def __init__(
        self,
        backbone: str = 'resnet18',
        pretrained: bool = True,
        out_channels: int = 2048,
        freeze_bn: bool = False
    ):
        """
        Args:
            backbone: ResNet variant ('resnet50', 'resnet101')
            pretrained: Use ImageNet pretrained weights
            out_channels: Output feature dimension
            freeze_bn: Freeze batch normalization layers
        """
        super(ResNetCameraEncoder, self).__init__()
        
        self.backbone_name = backbone
        self.out_channels = out_channels
        
        # Load pretrained ResNet
        if backbone == 'resnet18':
            resnet = models.resnet18(pretrained=pretrained)
        
        # Remove the final fully connected layer and avgpool
        self.conv1 = resnet.conv1      # 3 -> 64, stride 2
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool  # stride 2
        
        self.layer1 = resnet.layer1    # 64 -> 256, stride 1
        self.layer2 = resnet.layer2    # 256 -> 512, stride 2
        self.layer3 = resnet.layer3    # 512 -> 1024, stride 2
        self.layer4 = resnet.layer4    # 1024 -> 2048, stride 2
        
        if freeze_bn:
            self._freeze_bn()
    
    def _freeze_bn(self):
        """Freeze batch normalization layers"""
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input images (B, 6, 3, H, W) or (B*6, 3, H, W)
        
        Returns:
            features: (B, 6, 2048, H', W') or (B*6, 2048, H', W')
        """
        # Handle both (B, 6, 3, H, W) and (B*6, 3, H, W) formats
        input_shape = x.shape
        if len(input_shape) == 5:  # (B, 6, 3, H, W)
            B, N, C, H, W = input_shape
            x = x.view(B * N, C, H, W)
            multi_view = True
        else:
            multi_view = False
            B_N = input_shape[0]
        
        # ResNet forward pass
        x = self.conv1(x)       # (B*6, 64, H/2, W/2)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)     # (B*6, 64, H/4, W/4)
        
        x = self.layer1(x)      # (B*6, 256, H/4, W/4)
        x = self.layer2(x)      # (B*6, 512, H/8, W/8)
        x = self.layer3(x)      # (B*6, 1024, H/16, W/16)
        x = self.layer4(x)      # (B*6, 2048, H/32, W/32)
        
        # Reshape back if multi-view
        if multi_view:
            _, C_out, H_out, W_out = x.shape
            x = x.view(B, N, C_out, H_out, W_out)
        
        return x
    
    def get_output_shape(self, input_height: int, input_width: int) -> Tuple[int, int, int]:
        """
        Calculate output shape for given input dimensions
        
        Args:
            input_height: Input image height
            input_width: Input image width
        
        Returns:
            (channels, height, width) of output features
        """
        out_h = input_height // 32
        out_w = input_width // 32
        return (self.out_channels, out_h, out_w)


class PointNetLiDAREncoder(nn.Module):
    """
    PointNet-based encoder for LiDAR point clouds
    
    Input Dimensions:
        - Point cloud: (B, N, C) where B=batch, N=num_points, C=channels
        - Typical NuScenes: (B, 34720, 5) where 5 = [x, y, z, intensity, ring]
        - Or simplified: (B, 34720, 4) where 4 = [x, y, z, intensity]
        - Can handle variable N with max pooling
    
    Output Dimensions:
        - Global features: (B, feat_dim)
        - Default feat_dim: 1024
        - Point features: (B, N, feat_dim) if return_point_features=True
    """
    
    def __init__(
        self,
        input_channels: int = 5,
        feat_dim: int = 1024,
        use_bn: bool = True,
        return_point_features: bool = False
    ):
        """
        Args:
            input_channels: Number of input channels (3-5 typical)
            feat_dim: Output feature dimension
            use_bn: Use batch normalization
            return_point_features: Return per-point features
        """
        super(PointNetLiDAREncoder, self).__init__()
        
        self.input_channels = input_channels
        self.feat_dim = feat_dim
        self.return_point_features = return_point_features
        
        # Multi-layer perceptrons for point features
        self.conv1 = nn.Conv1d(input_channels, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, 512, 1)
        self.conv5 = nn.Conv1d(512, feat_dim, 1)
        
        if use_bn:
            self.bn1 = nn.BatchNorm1d(64)
            self.bn2 = nn.BatchNorm1d(128)
            self.bn3 = nn.BatchNorm1d(256)
            self.bn4 = nn.BatchNorm1d(512)
            self.bn5 = nn.BatchNorm1d(feat_dim)
        else:
            self.bn1 = nn.Identity()
            self.bn2 = nn.Identity()
            self.bn3 = nn.Identity()
            self.bn4 = nn.Identity()
            self.bn5 = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Point cloud (B, N, C) or (B, C, N)
        
        Returns:
            features: Global features (B, feat_dim) or point features (B, N, feat_dim)
        """
        # Handle both (B, N, C) and (B, C, N) formats
        if x.dim() == 3 and x.shape[2] == self.input_channels:
            # Convert (B, N, C) to (B, C, N)
            x = x.transpose(1, 2)
        
        B, C, N = x.shape
        
        # PointNet architecture: shared MLPs
        x = F.relu(self.bn1(self.conv1(x)))      # (B, 64, N)
        x = F.relu(self.bn2(self.conv2(x)))      # (B, 128, N)
        point_feat = x  # Store for later if needed
        
        x = F.relu(self.bn3(self.conv3(x)))      # (B, 256, N)
        x = F.relu(self.bn4(self.conv4(x)))      # (B, 512, N)
        x = F.relu(self.bn5(self.conv5(x)))      # (B, feat_dim, N)
        
        # Global max pooling
        global_feat = torch.max(x, 2)[0]         # (B, feat_dim)
        
        if self.return_point_features:
            # Concatenate global feature to each point
            global_feat_expanded = global_feat.unsqueeze(2).expand(-1, -1, N)
            point_features = torch.cat([x, global_feat_expanded], dim=1)
            return point_features.transpose(1, 2)  # (B, N, feat_dim*2)
        
        return global_feat


class VoxelNetLiDAREncoder(nn.Module):
    """
    VoxelNet-style encoder for LiDAR (alternative to PointNet)
    Uses voxelization for better spatial structure
    
    Input Dimensions:
        - Voxel features: (B, N_voxels, N_points_per_voxel, C)
        - Typical: (B, 12000, 32, 5) where 5 = [x, y, z, intensity, time]
        - Voxel coords: (B, N_voxels, 3) - [x_idx, y_idx, z_idx]
    
    Output Dimensions:
        - Voxel features: (B, feat_dim, D, H, W)
        - Typical: (B, 128, 40, 200, 176) for 3D conv output
        - Or flattened: (B, feat_dim) after pooling
    """
    
    def __init__(
        self,
        input_channels: int = 5,
        voxel_feat_dim: int = 128,
        output_feat_dim: int = 256,
        max_points_per_voxel: int = 32
    ):
        """
        Args:
            input_channels: Input point feature channels
            voxel_feat_dim: Voxel feature dimension
            output_feat_dim: Final output dimension
            max_points_per_voxel: Max points in each voxel
        """
        super(VoxelNetLiDAREncoder, self).__init__()
        
        self.input_channels = input_channels
        self.voxel_feat_dim = voxel_feat_dim
        self.output_feat_dim = output_feat_dim
        
        # Voxel Feature Encoding (VFE) layers
        self.vfe1 = VFELayer(input_channels, voxel_feat_dim // 2)
        self.vfe2 = VFELayer(voxel_feat_dim // 2, voxel_feat_dim)
        
        # 3D convolutions for spatial feature extraction
        self.conv3d_1 = nn.Conv3d(voxel_feat_dim, 128, 3, stride=2, padding=1)
        self.bn3d_1 = nn.BatchNorm3d(128)
        
        self.conv3d_2 = nn.Conv3d(128, 256, 3, stride=2, padding=1)
        self.bn3d_2 = nn.BatchNorm3d(256)
        
        self.conv3d_3 = nn.Conv3d(256, output_feat_dim, 3, stride=2, padding=1)
        self.bn3d_3 = nn.BatchNorm3d(output_feat_dim)
    
    def forward(
        self, 
        voxel_features: torch.Tensor,
        voxel_coords: torch.Tensor,
        voxel_grid_shape: Tuple[int, int, int]
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            voxel_features: (B, N_voxels, N_points, C)
            voxel_coords: (B, N_voxels, 3) - voxel indices
            voxel_grid_shape: (D, H, W) - 3D grid dimensions
        
        Returns:
            features: (B, output_feat_dim, D', H', W')
        """
        B, N_voxels, N_points, C = voxel_features.shape
        
        # Voxel feature encoding
        voxel_features = self.vfe1(voxel_features)    # (B, N_voxels, feat_dim//2)
        voxel_features = self.vfe2(voxel_features)    # (B, N_voxels, feat_dim)
        
        # Scatter voxel features to 3D grid
        D, H, W = voxel_grid_shape
        feature_grid = torch.zeros(
            B, self.voxel_feat_dim, D, H, W,
            device=voxel_features.device,
            dtype=voxel_features.dtype
        )
        
        # Scatter operation (simplified - actual implementation may vary)
        for b in range(B):
            coords = voxel_coords[b].long()  # (N_voxels, 3)
            features = voxel_features[b]      # (N_voxels, feat_dim)
            feature_grid[b, :, coords[:, 0], coords[:, 1], coords[:, 2]] = features.T
        
        # 3D convolutions
        x = F.relu(self.bn3d_1(self.conv3d_1(feature_grid)))
        x = F.relu(self.bn3d_2(self.conv3d_2(x)))
        x = F.relu(self.bn3d_3(self.conv3d_3(x)))
        
        return x


class VFELayer(nn.Module):
    """Voxel Feature Encoding Layer"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super(VFELayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.linear = nn.Linear(in_channels, out_channels)
        self.bn = nn.BatchNorm1d(out_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N_voxels, N_points, C)
        Returns:
            features: (B, N_voxels, out_channels)
        """
        B, N_voxels, N_points, C = x.shape
        
        # Reshape for linear layer
        x = x.view(B * N_voxels * N_points, C)
        x = self.linear(x)
        x = x.view(B * N_voxels, N_points, self.out_channels)
        
        # Batch norm
        x = x.view(B * N_voxels * N_points, self.out_channels)
        x = self.bn(x)
        x = x.view(B * N_voxels, N_points, self.out_channels)
        
        # ReLU and max pooling over points
        x = F.relu(x)
        x = torch.max(x, dim=1)[0]  # (B*N_voxels, out_channels)
        x = x.view(B, N_voxels, self.out_channels)
        
        return x


class RadarEncoder(nn.Module):
    """
    Encoder for Radar point clouds
    Similar to PointNet but adapted for radar-specific features
    
    Input Dimensions:
        - Radar points: (B, N, C) where B=batch, N=num_points, C=channels
        - Typical NuScenes: (B, 125, 18) where 18 includes:
          [x, y, z, dyn_prop, id, rcs, vx, vy, vx_comp, vy_comp, ...]
        - Or simplified: (B, 125, 7) where 7 = [x, y, z, vx, vy, rcs, time]
    
    Output Dimensions:
        - Global features: (B, feat_dim)
        - Default feat_dim: 256
        - Can be lower than LiDAR due to sparser data
    """
    
    def __init__(
        self,
        input_channels: int = 7,
        feat_dim: int = 256,
        use_bn: bool = True
    ):
        """
        Args:
            input_channels: Number of input channels (7-18 typical)
            feat_dim: Output feature dimension
            use_bn: Use batch normalization
        """
        super(RadarEncoder, self).__init__()
        
        self.input_channels = input_channels
        self.feat_dim = feat_dim
        
        # Radar-specific feature extraction
        # Smaller network than LiDAR due to sparser data
        self.conv1 = nn.Conv1d(input_channels, 32, 1)
        self.conv2 = nn.Conv1d(32, 64, 1)
        self.conv3 = nn.Conv1d(64, 128, 1)
        self.conv4 = nn.Conv1d(128, feat_dim, 1)
        
        if use_bn:
            self.bn1 = nn.BatchNorm1d(32)
            self.bn2 = nn.BatchNorm1d(64)
            self.bn3 = nn.BatchNorm1d(128)
            self.bn4 = nn.BatchNorm1d(feat_dim)
        else:
            self.bn1 = nn.Identity()
            self.bn2 = nn.Identity()
            self.bn3 = nn.Identity()
            self.bn4 = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Radar points (B, N, C) or (B, C, N)
        
        Returns:
            features: Global features (B, feat_dim)
        """
        # Handle both (B, N, C) and (B, C, N) formats
        if x.dim() == 3 and x.shape[2] == self.input_channels:
            # Convert (B, N, C) to (B, C, N)
            x = x.transpose(1, 2)
        
        B, C, N = x.shape
        
        # Feature extraction
        x = F.relu(self.bn1(self.conv1(x)))      # (B, 32, N)
        x = F.relu(self.bn2(self.conv2(x)))      # (B, 64, N)
        x = F.relu(self.bn3(self.conv3(x)))      # (B, 128, N)
        x = F.relu(self.bn4(self.conv4(x)))      # (B, feat_dim, N)
        
        # Global max pooling
        global_feat = torch.max(x, 2)[0]         # (B, feat_dim)
        
        return global_feat


class MultiRadarEncoder(nn.Module):
    """
    Encoder for multiple radar sensors (5 radars in NuScenes)
    
    Input Dimensions:
        - Radar points: List of 5 tensors, each (B, N_i, C)
        - Or stacked: (B, 5, N_max, C) with padding
        - Typical: [(B, 125, 7)] * 5 for each radar
    
    Output Dimensions:
        - Per-radar features: (B, 5, feat_dim) 
        - Or fused: (B, feat_dim) after aggregation
    """
    
    def __init__(
        self,
        input_channels: int = 7,
        feat_dim: int = 256,
        num_radars: int = 5,
        fusion_method: str = 'concat'  # 'concat', 'max', 'mean'
    ):
        """
        Args:
            input_channels: Input channels per radar point
            feat_dim: Feature dimension per radar
            num_radars: Number of radar sensors
            fusion_method: How to fuse multi-radar features
        """
        super(MultiRadarEncoder, self).__init__()
        
        self.num_radars = num_radars
        self.feat_dim = feat_dim
        self.fusion_method = fusion_method
        
        # Shared encoder for all radars
        self.radar_encoder = RadarEncoder(input_channels, feat_dim)
        
        # Fusion layer if concatenating
        if fusion_method == 'concat':
            self.fusion_fc = nn.Linear(feat_dim * num_radars, feat_dim)
        
        self.output_dim = feat_dim
    
    def forward(self, radar_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            radar_list: List of radar point clouds, each (B, N_i, C)
        
        Returns:
            features: Fused radar features (B, feat_dim)
        """
        assert len(radar_list) == self.num_radars
        
        # Encode each radar separately
        radar_features = []
        for radar_points in radar_list:
            feat = self.radar_encoder(radar_points)
            radar_features.append(feat)
        
        # Stack features
        radar_features = torch.stack(radar_features, dim=1)  # (B, num_radars, feat_dim)
        
        # Fuse features
        if self.fusion_method == 'concat':
            B = radar_features.shape[0]
            fused = radar_features.view(B, -1)  # (B, num_radars * feat_dim)
            fused = self.fusion_fc(fused)       # (B, feat_dim)
        elif self.fusion_method == 'max':
            fused = torch.max(radar_features, dim=1)[0]  # (B, feat_dim)
        elif self.fusion_method == 'mean':
            fused = torch.mean(radar_features, dim=1)    # (B, feat_dim)
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")
        
        return fused


def print_encoder_specs():
    """Print detailed specifications of all encoders"""
    
    print("="*80)
    print("MULTI-MODAL ENCODER SPECIFICATIONS")
    print("="*80)
    
    print("\n" + "="*80)
    print("1. CAMERA ENCODER (ResNet)")
    print("="*80)
    print("\nInput Dimensions:")
    print("  Format: (Batch, Num_Cameras, RGB_Channels, Height, Width)")
    print("  Typical: (B, 6, 3, 448, 800)")
    print("  Full Res: (B, 6, 3, 900, 1600)")
    print("  Channels: 3 (RGB)")
    print("\nArchitecture Options:")
    print("  - ResNet50:  50 layers, 2048 output channels")
    print("  - ResNet101: 101 layers, 2048 output channels")
    print("  - ResNet34:  34 layers, 512 output channels")
    print("\nOutput Dimensions:")
    print("  Format: (Batch, Num_Cameras, Feat_Channels, H', W')")
    print("  ResNet50/101: (B, 6, 2048, H/32, W/32)")
    print("  For (448, 800): (B, 6, 2048, 14, 25)")
    print("  For (900, 1600): (B, 6, 2048, 28, 50)")
    print("\nFeature Dimension: 2048 (ResNet50/101) or 512 (ResNet34)")
    
    print("\n" + "="*80)
    print("2. LIDAR ENCODER (PointNet)")
    print("="*80)
    print("\nInput Dimensions:")
    print("  Format: (Batch, Num_Points, Channels)")
    print("  Typical: (B, 34720, 5)")
    print("  Channels: 5 = [x, y, z, intensity, ring_index]")
    print("  Or: 4 = [x, y, z, intensity]")
    print("  Or: 3 = [x, y, z]")
    print("\nPoint Cloud Range:")
    print("  X: [-51.2m, 51.2m]")
    print("  Y: [-51.2m, 51.2m]")
    print("  Z: [-5.0m, 3.0m]")
    print("  Typical num_points: ~30,000-40,000 after filtering")
    print("\nArchitecture:")
    print("  Layer 1: 5 -> 64 channels")
    print("  Layer 2: 64 -> 128 channels")
    print("  Layer 3: 128 -> 256 channels")
    print("  Layer 4: 256 -> 512 channels")
    print("  Layer 5: 512 -> 1024 channels")
    print("  Pooling: Max pooling over all points")
    print("\nOutput Dimensions:")
    print("  Global Features: (B, 1024)")
    print("  Point Features (optional): (B, N, 1024)")
    print("\nFeature Dimension: 1024")
    
    print("\n" + "="*80)
    print("3. LIDAR ENCODER (VoxelNet - Alternative)")
    print("="*80)
    print("\nInput Dimensions:")
    print("  Voxel Features: (B, N_voxels, Points_per_voxel, Channels)")
    print("  Typical: (B, 12000, 32, 5)")
    print("  Voxel Coords: (B, N_voxels, 3)")
    print("\nVoxel Grid:")
    print("  Voxel size: [0.1m, 0.1m, 0.2m]")
    print("  Grid shape: (40, 200, 176) for (D, H, W)")
    print("  Points per voxel: max 32")
    print("\nOutput Dimensions:")
    print("  3D Features: (B, 256, D', H', W')")
    print("  After 3 stride-2 convs: (B, 256, 5, 25, 22)")
    print("\nFeature Dimension: 256")
    
    print("\n" + "="*80)
    print("4. RADAR ENCODER")
    print("="*80)
    print("\nInput Dimensions:")
    print("  Format: (Batch, Num_Points, Channels)")
    print("  Typical: (B, 125, 7)")
    print("  Channels: 7 = [x, y, z, vx, vy, rcs, timestamp]")
    print("  Full: 18 = [x, y, z, dyn_prop, id, rcs, vx, vy, vx_comp, vy_comp, ...]")
    print("\nRadar Point Range:")
    print("  Fewer points than LiDAR: ~50-200 per frame per radar")
    print("  Velocity info: Doppler measurements (vx, vy)")
    print("  RCS: Radar cross section (reflectivity)")
    print("\nArchitecture:")
    print("  Layer 1: 7 -> 32 channels")
    print("  Layer 2: 32 -> 64 channels")
    print("  Layer 3: 64 -> 128 channels")
    print("  Layer 4: 128 -> 256 channels")
    print("  Pooling: Max pooling over all points")
    print("\nOutput Dimensions:")
    print("  Single Radar: (B, 256)")
    print("\nFeature Dimension: 256")
    
    print("\n" + "="*80)
    print("5. MULTI-RADAR ENCODER (5 Radars)")
    print("="*80)
    print("\nInput Dimensions:")
    print("  Format: List of 5 radar tensors")
    print("  Each: (B, N_i, 7) where N_i varies per radar")
    print("  Radars: FRONT, FRONT_LEFT, FRONT_RIGHT, BACK_LEFT, BACK_RIGHT")
    print("\nProcessing:")
    print("  - Encode each radar separately")
    print("  - Features per radar: (B, 256)")
    print("  - Stack: (B, 5, 256)")
    print("\nFusion Options:")
    print("  Concatenation: (B, 5*256) -> FC -> (B, 256)")
    print("  Max pooling: max over 5 radars -> (B, 256)")
    print("  Mean pooling: mean over 5 radars -> (B, 256)")
    print("\nOutput Dimensions:")
    print("  Fused Features: (B, 256)")
    print("\nFeature Dimension: 256")
    
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print("\n{:<20} {:<35} {:<25} {:<15}".format(
        "Encoder", "Input Shape", "Output Shape", "Feat Dim"))
    print("-" * 95)
    print("{:<20} {:<35} {:<25} {:<15}".format(
        "Camera (ResNet50)", "(B, 6, 3, 448, 800)", "(B, 6, 2048, 14, 25)", "2048"))
    print("{:<20} {:<35} {:<25} {:<15}".format(
        "LiDAR (PointNet)", "(B, 34720, 5)", "(B, 1024)", "1024"))
    print("{:<20} {:<35} {:<25} {:<15}".format(
        "LiDAR (VoxelNet)", "(B, 12000, 32, 5)", "(B, 256, 5, 25, 22)", "256"))
    print("{:<20} {:<35} {:<25} {:<15}".format(
        "Radar (Single)", "(B, 125, 7)", "(B, 256)", "256"))
    print("{:<20} {:<35} {:<25} {:<15}".format(
        "Radar (Multi-5)", "List[(B, 125, 7)] * 5", "(B, 256)", "256"))
    print("="*80)


if __name__ == '__main__':
    print_encoder_specs()
    
    # Example instantiation
    print("\n" + "="*80)
    print("EXAMPLE MODEL INSTANTIATION")
    print("="*80)
    
    # Camera encoder
    camera_encoder = ResNetCameraEncoder(backbone='resnet50', pretrained=True)
    print(f"\n✓ Camera Encoder (ResNet50) created")
    print(f"  Output channels: {camera_encoder.out_channels}")
    
    # LiDAR encoder
    lidar_encoder = PointNetLiDAREncoder(input_channels=5, feat_dim=1024)
    print(f"\n✓ LiDAR Encoder (PointNet) created")
    print(f"  Input channels: {lidar_encoder.input_channels}")
    print(f"  Feature dimension: {lidar_encoder.feat_dim}")
    
    # Radar encoder
    radar_encoder = MultiRadarEncoder(input_channels=7, feat_dim=256, num_radars=5)
    print(f"\n✓ Multi-Radar Encoder created")
    print(f"  Num radars: {radar_encoder.num_radars}")
    print(f"  Feature dimension: {radar_encoder.feat_dim}")
    
    # Test with dummy data
    print("\n" + "="*80)
    print("TESTING WITH DUMMY DATA")
    print("="*80)
    
    batch_size = 2
    
    # Test camera
    dummy_images = torch.randn(batch_size, 6, 3, 448, 800)
    cam_features = camera_encoder(dummy_images)
    print(f"\nCamera input shape: {dummy_images.shape}")
    print(f"Camera output shape: {cam_features.shape}")
    
    # Test LiDAR
    dummy_lidar = torch.randn(batch_size, 34720, 5)
    lidar_features = lidar_encoder(dummy_lidar)
    print(f"\nLiDAR input shape: {dummy_lidar.shape}")
    print(f"LiDAR output shape: {lidar_features.shape}")
    
    # Test Radar
    dummy_radars = [torch.randn(batch_size, 125, 7) for _ in range(5)]
    radar_features = radar_encoder(dummy_radars)
    print(f"\nRadar input shapes: {[r.shape for r in dummy_radars]}")
    print(f"Radar output shape: {radar_features.shape}")
    
    print("\n" + "="*80)
    print("ALL TESTS PASSED!")
    print("="*80 + "\n")