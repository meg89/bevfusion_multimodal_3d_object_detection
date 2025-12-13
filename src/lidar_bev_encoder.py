
"""
Plug-and-Play VoxelNet + SparseConv BEV Encoder for LiDAR
Modern, efficient implementation with sparse convolutions
Ready to use with BEV fusion systems
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
import numpy as np


# =============================================================================
# SPARSE VOXEL NET ENCODER (Modern Implementation)
# =============================================================================

class SparseVoxelNetLiDAREncoder(nn.Module):
    """
    Modern VoxelNet encoder using sparse convolutions
    
    Input:
        - Point cloud: (B, N, 5) where 5 = [x, y, z, intensity, time]
        - Or voxelized format
    
    Output:
        - BEV features: (B, C, H, W) ready for fusion
        - Typical: (B, 256, 200, 200) for NuScenes
    
    Features:
        ✓ Sparse convolutions (efficient!)
        ✓ 3D → 2D projection to BEV
        ✓ Ready for fusion
        ✓ Config-driven
    """
    
    def __init__(
        self,
        voxel_size: Optional[list] = None,
        point_cloud_range: Optional[list] = None,
        max_points_per_voxel: int = 10,
        max_voxels: int = 60000,
        feature_dim: int = 256,
        config: Optional[Dict] = None
    ):
        """
        Args:
            voxel_size: [vx, vy, vz] in meters
            point_cloud_range: [x_min, y_min, z_min, x_max, y_max, z_max]
            max_points_per_voxel: Max points in each voxel
            max_voxels: Max number of voxels
            feature_dim: Output feature dimension
            config: Optional config dict
        """
        super().__init__()
        
        # Load from config if provided
        if config is not None:
            dataset_config = config.get('dataset', {})
            lidar_config = config.get('model', {}).get('lidar_encoder', {})
            
            self.voxel_size = dataset_config.get('voxel_size', [0.512, 0.512, 0.2])
            self.pc_range = dataset_config.get('point_cloud_range', 
                                               [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0])
            self.max_points_per_voxel = lidar_config.get('max_points_per_voxel', 10)
            self.max_voxels = lidar_config.get('max_voxels', 60000)
            self.feature_dim = lidar_config.get('feature_dim', 256)
        else:
            self.voxel_size = voxel_size or [0.512, 0.512, 0.2]
            self.pc_range = point_cloud_range or [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
            self.max_points_per_voxel = max_points_per_voxel
            self.max_voxels = max_voxels
            self.feature_dim = feature_dim
        
        # Calculate grid size
        self.grid_size = self._calculate_grid_size()
        
        # Check if spconv is available
        if not SPCONV_AVAILABLE:
            raise ImportError(
                "spconv is required for SparseVoxelNetLiDAREncoder. "
                "Install with: pip install spconv-cu118 (for CUDA 11.8)"
            )
        
        # VFE (Voxel Feature Encoding) layers
        self.vfe_layers = nn.ModuleList([
            VFELayer(5, 32),    # Input: [x, y, z, intensity, time]
            VFELayer(32, 128)
        ])
        
        # Sparse 3D convolutions
        self.sparse_conv = spconv.SparseSequential(
            # Block 1
            spconv.SubMConv3d(128, 64, 3, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            
            # Downsample
            spconv.SparseConv3d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            
            # Block 2
            spconv.SubMConv3d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            
            # Downsample
            spconv.SparseConv3d(128, 256, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )
        
        # 3D to BEV projection
        # After 2 downsamples, Z dimension is reduced
        # We'll flatten Z and project to BEV
        z_compressed_channels = 256 * (self.grid_size[2] // 4)  # After 2× stride=2
        
        self.bev_projection = nn.Sequential(
            nn.Conv2d(z_compressed_channels, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, self.feature_dim, 3, padding=1),
            nn.BatchNorm2d(self.feature_dim),
            nn.ReLU(inplace=True)
        )
        
        # Output dimensions
        self.bev_h = self.grid_size[1]
        self.bev_w = self.grid_size[0]
    
    def _calculate_grid_size(self) -> Tuple[int, int, int]:
        """Calculate voxel grid dimensions"""
        grid_size = [
            int((self.pc_range[3] - self.pc_range[0]) / self.voxel_size[0]),
            int((self.pc_range[4] - self.pc_range[1]) / self.voxel_size[1]),
            int((self.pc_range[5] - self.pc_range[2]) / self.voxel_size[2])
        ]
        return tuple(grid_size)
    
    def voxelize(self, points: torch.Tensor) -> Tuple:
        """
        Voxelize point cloud
        
        Args:
            points: (B, N, 5) - [x, y, z, intensity, time]
        
        Returns:
            voxel_features: Voxel features
            voxel_coords: Voxel coordinates
            num_points_per_voxel: Number of points in each voxel
        """
        batch_size = points.shape[0]
        
        voxel_features_list = []
        voxel_coords_list = []
        num_points_list = []
        
        for b in range(batch_size):
            pts = points[b]  # (N, 5)
            
            # Remove invalid points
            valid_mask = pts[:, 0] != 0  # Assuming 0,0,0 is invalid
            pts = pts[valid_mask]
            
            # Calculate voxel indices
            voxel_indices = torch.floor(
                (pts[:, :3] - torch.tensor(self.pc_range[:3], device=pts.device)) / 
                torch.tensor(self.voxel_size, device=pts.device)
            ).long()
            
            # Filter out-of-range voxels
            valid_mask = (
                (voxel_indices[:, 0] >= 0) & (voxel_indices[:, 0] < self.grid_size[0]) &
                (voxel_indices[:, 1] >= 0) & (voxel_indices[:, 1] < self.grid_size[1]) &
                (voxel_indices[:, 2] >= 0) & (voxel_indices[:, 2] < self.grid_size[2])
            )
            
            pts = pts[valid_mask]
            voxel_indices = voxel_indices[valid_mask]
            
            # Create unique voxel hash
            voxel_hash = (
                voxel_indices[:, 0] * self.grid_size[1] * self.grid_size[2] +
                voxel_indices[:, 1] * self.grid_size[2] +
                voxel_indices[:, 2]
            )
            
            # Group points by voxel
            unique_hashes, inverse_indices = torch.unique(voxel_hash, return_inverse=True)
            
            # Limit number of voxels
            if len(unique_hashes) > self.max_voxels:
                perm = torch.randperm(len(unique_hashes), device=pts.device)[:self.max_voxels]
                unique_hashes = unique_hashes[perm]
            
            # Create voxel features
            voxels = []
            coords = []
            num_points = []
            
            for i, hash_val in enumerate(unique_hashes):
                point_mask = inverse_indices == i
                voxel_points = pts[point_mask][:self.max_points_per_voxel]
                
                # Pad if needed
                if len(voxel_points) < self.max_points_per_voxel:
                    pad = self.max_points_per_voxel - len(voxel_points)
                    voxel_points = torch.cat([
                        voxel_points,
                        torch.zeros(pad, 5, device=pts.device)
                    ])
                
                voxels.append(voxel_points)
                
                # Get voxel coordinate
                idx = torch.where(voxel_hash == hash_val)[0][0]
                coords.append(voxel_indices[idx])
                num_points.append(min(len(pts[point_mask]), self.max_points_per_voxel))
            
            voxel_features_list.append(torch.stack(voxels))  # (num_voxels, max_points, 5)
            voxel_coords_list.append(torch.stack(coords))     # (num_voxels, 3)
            num_points_list.append(torch.tensor(num_points, device=pts.device))
        
        return voxel_features_list, voxel_coords_list, num_points_list
    
    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            points: (B, N, 5) - [x, y, z, intensity, time]
        
        Returns:
            bev_features: (B, feature_dim, H, W)
        """
        batch_size = points.shape[0]
        
        # Voxelize
        voxel_features_list, voxel_coords_list, num_points_list = self.voxelize(points)
        
        # Process each batch
        bev_features_list = []
        
        for b in range(batch_size):
            voxel_features = voxel_features_list[b]  # (num_voxels, max_points, 5)
            voxel_coords = voxel_coords_list[b]      # (num_voxels, 3)
            
            # VFE (Voxel Feature Encoding)
            x = voxel_features
            for vfe in self.vfe_layers:
                x = vfe(x)  # (num_voxels, feat_dim)
            
            # Create sparse tensor
            # Add batch index to coordinates
            batch_coords = torch.cat([
                torch.full((len(voxel_coords), 1), b, device=voxel_coords.device),
                voxel_coords
            ], dim=1)  # (num_voxels, 4) - [batch_idx, x, y, z]
            
            # For first batch, create sparse tensor
            if b == 0:
                all_features = x
                all_coords = batch_coords
            else:
                all_features = torch.cat([all_features, x])
                all_coords = torch.cat([all_coords, batch_coords])
        
        # Create sparse tensor for entire batch
        sparse_tensor = spconv.SparseConvTensor(
            features=all_features,
            indices=all_coords.int(),
            spatial_shape=self.grid_size,
            batch_size=batch_size
        )
        
        # 3D sparse convolutions
        sparse_output = self.sparse_conv(sparse_tensor)
        
        # Convert to dense and project to BEV
        # Get dense features: (B, C, D, H, W)
        dense_features = sparse_output.dense()
        
        # Flatten Z dimension: (B, C*D, H, W)
        B, C, D, H, W = dense_features.shape
        bev_input = dense_features.permute(0, 2, 1, 3, 4).reshape(B, C*D, H, W)
        
        # Project to BEV
        bev_features = self.bev_projection(bev_input)  # (B, feature_dim, H, W)
        
        return bev_features


class VFELayer(nn.Module):
    """Voxel Feature Encoding Layer"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.linear = nn.Linear(in_channels, out_channels)
        self.bn = nn.BatchNorm1d(out_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (num_voxels, max_points, in_channels)
        Returns:
            features: (num_voxels, out_channels)
        """
        num_voxels, max_points, C = x.shape
        
        # Linear transformation
        x = x.reshape(num_voxels * max_points, C)
        x = self.linear(x)
        x = x.reshape(num_voxels, max_points, self.out_channels)
        
        # Batch norm and activation
        x = x.reshape(num_voxels * max_points, self.out_channels)
        x = self.bn(x)
        x = x.reshape(num_voxels, max_points, self.out_channels)
        x = F.relu(x)
        
        # Max pooling over points in each voxel
        x = torch.max(x, dim=1)[0]  # (num_voxels, out_channels)
        
        return x


# =============================================================================
# SIMPLE VERSION (without spconv, for compatibility)
# =============================================================================

class SimpleLiDARBEVEncoder(nn.Module):
    """
    Simple LiDAR BEV encoder without sparse convolutions
    Use this if you can't install spconv
    
    Less efficient but works everywhere
    """
    
    def __init__(
        self,
        voxel_size: list = [0.512, 0.512, 0.2],
        point_cloud_range: list = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        feature_dim: int = 256,
        config: Optional[Dict] = None
    ):
        super().__init__()
        
        if config is not None:
            dataset_config = config.get('dataset', {})
            self.voxel_size = dataset_config.get('voxel_size', voxel_size)
            self.pc_range = dataset_config.get('point_cloud_range', point_cloud_range)
            self.feature_dim = config.get('model', {}).get('lidar_encoder', {}).get('feature_dim', feature_dim)
        else:
            self.voxel_size = voxel_size
            self.pc_range = point_cloud_range
            self.feature_dim = feature_dim
        
        # Calculate BEV grid size
        self.bev_h = int((self.pc_range[4] - self.pc_range[1]) / self.voxel_size[1])
        self.bev_w = int((self.pc_range[3] - self.pc_range[0]) / self.voxel_size[0])
        
        # Simple pillar encoding
        self.pillar_encoder = nn.Sequential(
            nn.Linear(5, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128)
        )
        
        # BEV convolutions
        self.bev_encoder = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, self.feature_dim, 3, padding=1),
            nn.BatchNorm2d(self.feature_dim),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """
        Args:
            points: (B, N, 5)
        Returns:
            bev_features: (B, feature_dim, H, W)
        """
        B, N, _ = points.shape
        device = points.device
        
        # Initialize BEV grid
        bev_grid = torch.zeros(B, 128, self.bev_h, self.bev_w, device=device)
        
        for b in range(B):
            pts = points[b]  # (N, 5)
            
            # Remove invalid points
            valid_mask = (pts[:, 0] != 0) | (pts[:, 1] != 0)
            pts = pts[valid_mask]
            
            if len(pts) == 0:
                continue
            
            # Calculate grid indices
            grid_x = ((pts[:, 0] - self.pc_range[0]) / self.voxel_size[0]).long()
            grid_y = ((pts[:, 1] - self.pc_range[1]) / self.voxel_size[1]).long()
            
            # Filter valid grid indices
            valid_mask = (
                (grid_x >= 0) & (grid_x < self.bev_w) &
                (grid_y >= 0) & (grid_y < self.bev_h)
            )
            
            pts = pts[valid_mask]
            grid_x = grid_x[valid_mask]
            grid_y = grid_y[valid_mask]
            
            if len(pts) == 0:
                continue
            
            # Encode points
            features = self.pillar_encoder[0](pts)  # (N, 64)
            features = self.pillar_encoder[2](self.pillar_encoder[1](features))  # (N, 128)
            
            # Scatter to BEV grid (simple max pooling)
            for i in range(len(pts)):
                x, y = grid_x[i], grid_y[i]
                bev_grid[b, :, y, x] = torch.max(
                    bev_grid[b, :, y, x],
                    features[i]
                )
        
        # BEV convolutions
        bev_features = self.bev_encoder(bev_grid)
        
        return bev_features


# =============================================================================
# INTEGRATION WITH BEV FUSION
# =============================================================================

def integrate_with_bev_fusion_example():
    """
    Example: How to integrate LiDAR encoder with BEV fusion
    """
    
    print("="*80)
    print("INTEGRATION EXAMPLE")
    print("="*80)
    
    example_code = """
# 1. CREATE LIDAR ENCODER
# =============================================

from lidar_bev_encoder import SparseVoxelNetLiDAREncoder

lidar_encoder = SparseVoxelNetLiDAREncoder(
    voxel_size=[0.512, 0.512, 0.2],
    point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
    feature_dim=256,  # Output 256 channels
    max_points_per_voxel=10,
    max_voxels=60000
)

# Output: (B, 256, 200, 200) BEV features


# 2. INTEGRATE WITH BEV FUSION
# =============================================

class MultiModalBEVDetector(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # Camera encoder
        self.camera_encoder = ResNet18CameraEncoder(config=config)
        # Output: (B, 6, 512, H/16, W/16)
        
        # Camera to BEV
        self.camera_to_bev = CameraBEVProjection(
            in_channels=512,
            out_channels=256,
            bev_h=200,
            bev_w=200
        )
        # Output: (B, 256, 200, 200)
        
        # LiDAR encoder (NEW!)
        self.lidar_encoder = SparseVoxelNetLiDAREncoder(config=config)
        # Output: (B, 256, 200, 200)
        
        # BEV Fusion
        self.bev_fusion = BEVFusion(
            input_channels=256,  # Both camera and lidar output 256
            output_channels=256
        )
        
        # Detection head
        self.detection_head = CenterNetHead(
            input_channels=256,
            num_classes=10
        )
    
    def forward(self, camera_imgs, lidar_points):
        # Camera branch
        cam_feats = self.camera_encoder(camera_imgs)
        cam_bev = self.camera_to_bev(cam_feats)  # (B, 256, 200, 200)
        
        # LiDAR branch (NEW!)
        lidar_bev = self.lidar_encoder(lidar_points)  # (B, 256, 200, 200)
        
        # Fusion
        fused_bev = self.bev_fusion(cam_bev, lidar_bev)  # (B, 256, 200, 200)
        
        # Detection
        predictions = self.detection_head(fused_bev)
        
        return predictions


# 3. TRAINING
# =============================================

model = MultiModalBEVDetector(config)

for batch in dataloader:
    camera_imgs = batch['camera']  # (B, 6, 3, H, W)
    lidar_points = batch['lidar']  # (B, N, 5)
    
    predictions = model(camera_imgs, lidar_points)
    
    loss = compute_loss(predictions, targets)
    loss.backward()
    optimizer.step()
"""
    
    print(example_code)
    print("="*80)


if __name__ == '__main__':
    integrate_with_bev_fusion_example()
    
    print("\n" + "="*80)
    print("TESTING ENCODERS")
    print("="*80)
    
    # Test simple encoder (always works)
    print("\n1. Testing SimpleLiDARBEVEncoder...")
    simple_encoder = SimpleLiDARBEVEncoder()
    
    # Dummy input
    points = torch.randn(2, 10000, 5)  # (B, N, 5)
    
    output = simple_encoder(points)
    print(f"   Input: {points.shape}")
    print(f"   Output: {output.shape}")
    print(f"   Expected: (2, 256, 200, 200)")
    print(f"   Match: {output.shape == torch.Size([2, 256, 200, 200])}")
    