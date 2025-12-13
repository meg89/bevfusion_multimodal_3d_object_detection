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
import torchvision.transforms as T

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