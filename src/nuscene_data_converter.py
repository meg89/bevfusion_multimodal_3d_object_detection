"""
NuScenes to MMDetection3D Converter
Converts NuScenes mini dataset to MMDetection3D format pickle files
Supports camera, radar, and lidar data
"""

import os
import pickle
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points, transform_matrix
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud, Box


class NuScenesConverter:
    """Convert NuScenes dataset to MMDetection3D format"""
    
    def __init__(self, dataroot: str, version: str = 'v1.0-mini', out_dir: str = './data/nuscenes'):
        """
        Args:
            dataroot: Path to NuScenes dataset root
            version: Dataset version (v1.0-mini, v1.0-trainval, etc.)
            out_dir: Output directory for pickle files
        """
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=True)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        
        # MMDetection3D class names for NuScenes
        self.classes = [
            'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
            'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
        ]
        
        # Camera names
        self.camera_types = [
            'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
        ]
        
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
        
        # Radar data (5 radars in NuScenes)
        radar_types = [
            'RADAR_FRONT', 'RADAR_FRONT_LEFT', 'RADAR_FRONT_RIGHT',
            'RADAR_BACK_LEFT', 'RADAR_BACK_RIGHT'
        ]
        radar_infos = {}
        for radar_type in radar_types:
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
        if split == 'train':
            scene_names = self._get_train_scenes()
        elif split == 'val':
            scene_names = self._get_val_scenes()
        elif split == 'test':
            scene_names = self._get_test_scenes()
        else:
            raise ValueError(f"Unknown split: {split}")
        
        # Get all samples from these scenes
        infos = []
        for scene in self.nusc.scene:
            if scene['name'] not in scene_names:
                continue
            
            # Get all samples in this scene
            sample_token = scene['first_sample_token']
            while sample_token:
                sample_info = self.get_sample_data(sample_token)
                infos.append(sample_info)
                
                sample = self.nusc.get('sample', sample_token)
                sample_token = sample['next']
        
        print(f"Collected {len(infos)} samples for {split} split")
        return infos
    
    def _get_train_scenes(self) -> List[str]:
        """Get training scene names for mini split"""
        # For mini dataset, use first 7 scenes for training
        all_scenes = [s['name'] for s in self.nusc.scene]
        return all_scenes[:7]
    
    def _get_val_scenes(self) -> List[str]:
        """Get validation scene names for mini split"""
        # For mini dataset, use next 2 scenes for validation
        all_scenes = [s['name'] for s in self.nusc.scene]
        return all_scenes[7:9]
    
    def _get_test_scenes(self) -> List[str]:
        """Get test scene names for mini split"""
        # For mini dataset, use last scene for test
        all_scenes = [s['name'] for s in self.nusc.scene]
        return all_scenes[9:10]
    
    def save_infos(self, infos: List[Dict], split: str):
        """Save infos to pickle file"""
        output_path = self.out_dir / f'nuscenes_infos_{split}.pkl'
        
        # Create metadata
        metadata = {
            'version': self.nusc.version,
            'classes': self.classes
        }
        
        # Save with metadata
        data = {
            'infos': infos,
            'metadata': metadata
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Saved {len(infos)} samples to {output_path}")
    
    def convert_all(self):
        """Convert all splits"""
        print("Starting NuScenes to MMDetection3D conversion...")
        print(f"Dataset version: {self.nusc.version}")
        print(f"Output directory: {self.out_dir}")
        
        for split in ['train', 'val', 'test']:
            infos = self.convert_split(split)
            self.save_infos(infos, split)
        
        print("\nConversion completed!")
        print(f"Files saved to: {self.out_dir}")



def main():
    """Main conversion function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert NuScenes to MMDetection3D format')
    parser.add_argument(
        '--dataroot',
        type=str,
        default='./data/nuscenes',
        help='Path to NuScenes dataset root'
    )
    parser.add_argument(
        '--version',
        type=str,
        default='v1.0-mini',
        help='Dataset version (v1.0-mini, v1.0-trainval, etc.)'
    )
    parser.add_argument(
        '--out-dir',
        type=str,
        default='./data/nuscenes',
        help='Output directory for pickle files'
    )
    
    args = parser.parse_args()
    
    # Create converter and run
    converter = NuScenesConverter(
        dataroot=args.dataroot,
        version=args.version,
        out_dir=args.out_dir
    )
    converter.convert_all()


if __name__ == '__main__':
    main()