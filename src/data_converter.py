"""
NuScenes to MMDetection3D Converter (Config-Driven)
Converts NuScenes mini dataset to MMDetection3D format pickle files
Reads all configuration from config.yaml
Supports camera, radar, and lidar data
"""

import os
import pickle
import numpy as np
import yaml
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points, transform_matrix
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud, Box

class ConfigDrivenNuScenesConverter:
    """Convert NuScenes dataset to MMDetection3D format using config.yaml"""
    
    def __init__(self, config_path: str = 'config.yaml'):
        """
        Args:
            config_path: Path to YAML configuration file
        """
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Extract dataset configuration
        dataset_config = self.config['dataset']
        
        # Initialize NuScenes
        self.nusc = NuScenes(
            version=dataset_config['version'],
            dataroot=dataset_config['data_root'],
            verbose=True
        )
        
        # Output directory
        self.out_dir = Path(dataset_config['data_root'])
        self.out_dir.mkdir(parents=True, exist_ok=True)
        
        # Classes from config
        self.classes = dataset_config['classes']
        
        # Camera and radar names from config
        self.camera_types = dataset_config['cameras']['names']
        self.radar_types = dataset_config['radars']['names']
        
        # Point cloud range
        self.pc_range = dataset_config['point_cloud_range']
        
        # Maximum points
        self.max_lidar_points = dataset_config['max_points']['lidar']
        self.max_radar_points = dataset_config['max_points']['radar_per_sensor']
        
        # Split configuration
        self.split_config = dataset_config.get('splits', {
            'train': 'train',
            'val': 'val',
            'test': 'test'
        })
        
        # Split ratios (if using custom splits)
        self.split_ratios = dataset_config.get('split_ratios', {
            'train': 0.7,
            'val': 0.2,
            'test': 0.1
        })
        
        # Augmentation settings
        self.augmentation = dataset_config.get('augmentation', {})
        
        print("Configuration loaded successfully!")
        print(f"Dataset: {dataset_config['name']} {dataset_config['version']}")
        print(f"Classes: {len(self.classes)} classes")
        print(f"Cameras: {len(self.camera_types)}")
        print(f"Radars: {len(self.radar_types)}")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def get_sample_data(self, sample_token: str) -> Dict:
        """Extract all data for a single sample"""
        sample = self.nusc.get('sample', sample_token)
        
        # Get lidar data
        lidar_token = sample['data']['LIDAR_TOP']
        lidar_data = self.nusc.get('sample_data', lidar_token)
        lidar_path = os.path.join(self.nusc.dataroot, lidar_data['filename'])
        
        # Get ego pose and calibration
        lidar_pose = self.nusc.get('ego_pose', lidar_data['ego_pose_token'])
        lidar_calib = self.nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
        
        # Camera data
        cam_infos = {}
        for cam_type in self.camera_types:
            if cam_type not in sample['data']:
                continue
                
            cam_token = sample['data'][cam_type]
            cam_data = self.nusc.get('sample_data', cam_token)
            cam_calib = self.nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
            
            cam_infos[cam_type] = {
                'filename': cam_data['filename'],
                'calibrated_sensor': {
                    'translation': cam_calib['translation'],
                    'rotation': cam_calib['rotation'],
                    'camera_intrinsic': cam_calib['camera_intrinsic']
                }
            }
        
        # Radar data
        radar_infos = {}
        for radar_type in self.radar_types:
            if radar_type not in sample['data']:
                continue
                
            radar_token = sample['data'][radar_type]
            radar_data = self.nusc.get('sample_data', radar_token)
            radar_calib = self.nusc.get('calibrated_sensor', radar_data['calibrated_sensor_token'])
            
            radar_infos[radar_type] = {
                'filename': radar_data['filename'],
                'calibrated_sensor': {
                    'translation': radar_calib['translation'],
                    'rotation': radar_calib['rotation']
                }
            }
        
        # Get annotations
        annotations = self._get_annotations(sample, lidar_pose, lidar_calib)
        
        info = {
            'token': sample_token,
            'timestamp': sample['timestamp'],
            'scene_token': sample['scene_token'],
            'lidar_path': lidar_path,
            'lidar_pose': {
                'translation': lidar_pose['translation'],
                'rotation': lidar_pose['rotation']
            },
            'lidar_calibrated_sensor': {
                'translation': lidar_calib['translation'],
                'rotation': lidar_calib['rotation']
            },
            'cams': cam_infos,
            'radars': radar_infos,
            'gt_boxes': annotations['gt_boxes'],
            'gt_names': annotations['gt_names'],
            'gt_velocity': annotations['gt_velocity'],
            'num_lidar_pts': annotations['num_lidar_pts'],
            'num_radar_pts': annotations['num_radar_pts'],
            'valid_flag': annotations['valid_flag']
        }
        
        return info
    
    def _get_annotations(self, sample: Dict, ego_pose: Dict, calib: Dict) -> Dict:
        """Extract ground truth annotations"""
        annotations = []
        
        for ann_token in sample['anns']:
            ann = self.nusc.get('sample_annotation', ann_token)
            category = ann['category_name']
            
            # Filter by class
            if not self._filter_category(category):
                continue
            
            # Get box in global coordinates
            box = Box(ann['translation'], ann['size'], Quaternion(ann['rotation']))
            
            # Transform to lidar coordinates
            box = self._transform_box_to_sensor(box, ego_pose, calib)
            
            # Filter by point cloud range
            if not self._box_in_range(box):
                continue
            
            # Get velocity
            velocity = self.nusc.box_velocity(ann_token)
            if np.any(np.isnan(velocity)):
                velocity = np.zeros(3)
            
            # Get number of lidar and radar points
            num_lidar_pts = ann.get('num_lidar_pts', 0)
            num_radar_pts = ann.get('num_radar_pts', 0)
            
            annotations.append({
                'box': box,
                'name': self._get_class_name(category),
                'velocity': velocity[:2],  # Only x, y velocity
                'num_lidar_pts': num_lidar_pts,
                'num_radar_pts': num_radar_pts
            })
        
        if len(annotations) == 0:
            return {
                'gt_boxes': np.zeros((0, 7)),
                'gt_names': np.array([]),
                'gt_velocity': np.zeros((0, 2)),
                'num_lidar_pts': np.array([]),
                'num_radar_pts': np.array([]),
                'valid_flag': np.array([], dtype=bool)
            }
        
        # Convert to arrays
        gt_boxes = np.array([
            [ann['box'].center[0], ann['box'].center[1], ann['box'].center[2],
             ann['box'].wlh[0], ann['box'].wlh[1], ann['box'].wlh[2],
             ann['box'].orientation.yaw_pitch_roll[0]]
            for ann in annotations
        ])
        
        gt_names = np.array([ann['name'] for ann in annotations])
        gt_velocity = np.array([ann['velocity'] for ann in annotations])
        num_lidar_pts = np.array([ann['num_lidar_pts'] for ann in annotations])
        num_radar_pts = np.array([ann['num_radar_pts'] for ann in annotations])
        valid_flag = np.array([True] * len(annotations), dtype=bool)
        
        return {
            'gt_boxes': gt_boxes,
            'gt_names': gt_names,
            'gt_velocity': gt_velocity,
            'num_lidar_pts': num_lidar_pts,
            'num_radar_pts': num_radar_pts,
            'valid_flag': valid_flag
        }
    
    def _transform_box_to_sensor(self, box: Box, ego_pose: Dict, calib: Dict) -> Box:
        """Transform box from global to sensor coordinates"""
        # Move box to ego vehicle coord system
        box.translate(-np.array(ego_pose['translation']))
        box.rotate(Quaternion(ego_pose['rotation']).inverse)
        
        # Move box to sensor coord system
        box.translate(-np.array(calib['translation']))
        box.rotate(Quaternion(calib['rotation']).inverse)
        
        return box
    
    def _box_in_range(self, box: Box) -> bool:
        """Check if box is within point cloud range"""
        center = box.center
        return (
            self.pc_range[0] <= center[0] <= self.pc_range[3] and
            self.pc_range[1] <= center[1] <= self.pc_range[4] and
            self.pc_range[2] <= center[2] <= self.pc_range[5]
        )
    
    def _filter_category(self, category_name: str) -> bool:
        """Filter categories based on detection classes"""
        for class_name in self.classes:
            if class_name in category_name:
                return True
        return False
    
    def _get_class_name(self, category_name: str) -> str:
        """Map category to class name"""
        for class_name in self.classes:
            if class_name in category_name:
                return class_name
        return 'unknown'
    
    def convert_split(self, split: str) -> List[Dict]:
        """Convert a specific split (train/val/test)"""
        print(f"\nProcessing {split} split...")
        
        # Get scenes for this split
        scene_names = self._get_split_scenes(split)
        
        # Get all samples from these scenes
        infos = []
        for scene in self.nusc.scene:
            if scene['name'] not in scene_names:
                continue
            
            # Get all samples in this scene
            sample_token = scene['first_sample_token']
            while sample_token:
                try:
                    sample_info = self.get_sample_data(sample_token)
                    infos.append(sample_info)
                except Exception as e:
                    print(f"Warning: Failed to process sample {sample_token}: {e}")
                
                sample = self.nusc.get('sample', sample_token)
                sample_token = sample['next']
        
        print(f"Collected {len(infos)} samples for {split} split")
        return infos
    
    def _get_split_scenes(self, split: str) -> List[str]:
        """Get scene names for a split"""
        all_scenes = [s['name'] for s in self.nusc.scene]
        
        # Use split ratios from config
        num_scenes = len(all_scenes)
        train_end = int(num_scenes * self.split_ratios['train'])
        val_end = train_end + int(num_scenes * self.split_ratios['val'])
        
        if split == 'train':
            return all_scenes[:train_end]
        elif split == 'val':
            return all_scenes[train_end:val_end]
        elif split == 'test':
            return all_scenes[val_end:]
        else:
            raise ValueError(f"Unknown split: {split}")
    
    def save_infos(self, infos: List[Dict], split: str):
        """Save infos to pickle file"""
        # Use output paths from config
        dataset_config = self.config['dataset']
        
        if split == 'train':
            output_path = dataset_config['ann_file_train']
        elif split == 'val':
            output_path = dataset_config['ann_file_val']
        elif split == 'test':
            output_path = dataset_config['ann_file_test']
        else:
            output_path = self.out_dir / f'nuscenes_infos_{split}.pkl'
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create metadata
        metadata = {
            'version': self.nusc.version,
            'classes': self.classes,
            'num_classes': len(self.classes),
            'point_cloud_range': self.pc_range,
            'cameras': self.camera_types,
            'radars': self.radar_types,
            'max_points': {
                'lidar': self.max_lidar_points,
                'radar_per_sensor': self.max_radar_points
            }
        }
        
        # Save with metadata
        data = {
            'infos': infos,
            'metadata': metadata
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Saved {len(infos)} samples to {output_path}")
        
        # Print statistics
        self._print_statistics(infos, split)
    
    def _print_statistics(self, infos: List[Dict], split: str):
        """Print dataset statistics"""
        print(f"\n{split.upper()} Split Statistics:")
        print(f"  Total samples: {len(infos)}")
        
        # Count objects per class
        class_counts = {cls: 0 for cls in self.classes}
        total_objects = 0
        
        for info in infos:
            for name in info['gt_names']:
                if name in class_counts:
                    class_counts[name] += 1
                    total_objects += 1
        
        print(f"  Total objects: {total_objects}")
        print(f"  Objects per class:")
        for cls, count in class_counts.items():
            if count > 0:
                print(f"    {cls}: {count}")
    
    def convert_all(self):
        """Convert all splits"""
        print("="*80)
        print("NuScenes to MMDetection3D Conversion (Config-Driven)")
        print("="*80)
        print(f"\nDataset version: {self.nusc.version}")
        print(f"Output directory: {self.out_dir}")
        print(f"Point cloud range: {self.pc_range}")
        print(f"Classes ({len(self.classes)}): {', '.join(self.classes)}")
        
        # Get splits to process
        splits = ['train', 'val', 'test']
        
        for split in splits:
            try:
                infos = self.convert_split(split)
                if len(infos) > 0:
                    self.save_infos(infos, split)
                else:
                    print(f"Warning: No samples found for {split} split")
            except Exception as e:
                print(f"Error processing {split} split: {e}")
                continue
        
        print("\n" + "="*80)
        print("Conversion completed!")
        print("="*80)
        print(f"\nOutput files saved to: {self.out_dir}")
        print("\nTo validate the converted data, run:")
        print(f"  python validate_converted_data.py --data-dir {self.out_dir}")
    
    def print_config_summary(self):
        """Print configuration summary"""
        print("\n" + "="*80)
        print("CONFIGURATION SUMMARY")
        print("="*80)
        
        dataset_config = self.config['dataset']
        
        print("\nDataset:")
        print(f"  Name: {dataset_config['name']}")
        print(f"  Version: {dataset_config['version']}")
        print(f"  Data root: {dataset_config['data_root']}")
        
        print("\nSplits:")
        print(f"  Train: {dataset_config['ann_file_train']}")
        print(f"  Val: {dataset_config['ann_file_val']}")
        print(f"  Test: {dataset_config['ann_file_test']}")
        
        print("\nClasses:")
        for i, cls in enumerate(self.classes, 1):
            print(f"  {i}. {cls}")
        
        print("\nSensors:")
        print(f"  Cameras ({len(self.camera_types)}): {', '.join(self.camera_types)}")
        print(f"  Radars ({len(self.radar_types)}): {', '.join(self.radar_types)}")
        
        print("\nPoint Cloud:")
        print(f"  Range: {self.pc_range}")
        print(f"  Max LiDAR points: {self.max_lidar_points}")
        print(f"  Max radar points per sensor: {self.max_radar_points}")
        
        print("\nSplit Ratios:")
        print(f"  Train: {self.split_ratios['train']*100:.0f}%")
        print(f"  Val: {self.split_ratios['val']*100:.0f}%")
        print(f"  Test: {self.split_ratios['test']*100:.0f}%")
        
        print("\n" + "="*80)


def main():
    """Main conversion function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Convert NuScenes to MMDetection3D format using config.yaml'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/base.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    parser.add_argument(
        '--split',
        type=str,
        default=None,
        choices=['train', 'val', 'test'],
        help='Convert specific split only (default: all splits)'
    )
    parser.add_argument(
        '--show-config',
        action='store_true',
        help='Show configuration summary and exit'
    )
    
    args = parser.parse_args()
    
    # Create converter
    try:
        converter = ConfigDrivenNuScenesConverter(config_path=args.config)
    except FileNotFoundError:
        print(f"Error: Configuration file '{args.config}' not found!")
        print("\nPlease ensure config.yaml exists in the current directory.")
        print("You can create it using:")
        print("  python config_loader.py create --modality all --fusion bev")
        return
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return
    
    # Show config and exit if requested
    if args.show_config:
        converter.print_config_summary()
        return
    
    # Convert dataset
    try:
        if args.split:
            # Convert single split
            print(f"Converting {args.split} split only...")
            infos = converter.convert_split(args.split)
            converter.save_infos(infos, args.split)
        else:
            # Convert all splits
            converter.convert_all()
            
    except Exception as e:
        print(f"\nError during conversion: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n✓ Conversion completed successfully!")


if __name__ == '__main__':
    main()