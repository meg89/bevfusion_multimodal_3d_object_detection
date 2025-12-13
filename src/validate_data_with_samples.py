"""
Validate Converted NuScenes Data (Config-Driven)
Reads configuration from config.yaml and validates converted pickle files
NOW WITH: Sample ground truth boxes printing for inspection
"""

import pickle
import numpy as np
import yaml
from pathlib import Path
from typing import Dict, List
import argparse


class ConfigDrivenDataValidator:
    """Validate converted NuScenes data using config.yaml"""
    
    def __init__(self, config_path: str = 'config.yaml'):
        """
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.dataset_config = self.config['dataset']
        self.classes = self.dataset_config['classes']
        self.num_classes = len(self.classes)
        
        print("Configuration loaded successfully!")
        print(f"Expected classes: {self.classes}")
    
    def load_data(self, split: str) -> Dict:
        """Load data for a split"""
        # Get file path from config
        if split == 'train':
            file_path = self.dataset_config['ann_file_train']
        elif split == 'val':
            file_path = self.dataset_config['ann_file_val']
        elif split == 'test':
            file_path = self.dataset_config['ann_file_test']
        else:
            raise ValueError(f"Unknown split: {split}")
        
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        print(f"\nLoading {split} data from: {file_path}")
        
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        return data
    
    def validate_split(self, split: str, print_samples: int = 5) -> bool:
        """
        Validate a single split
        
        Args:
            split: Split name ('train', 'val', 'test')
            print_samples: Number of sample ground truth boxes to print (default: 5)
        """
        print(f"\n{'='*80}")
        print(f"VALIDATING {split.upper()} SPLIT")
        print('='*80)
        
        try:
            data = self.load_data(split)
        except FileNotFoundError as e:
            print(f"✗ {e}")
            return False
        except Exception as e:
            print(f"✗ Error loading data: {e}")
            return False
        
        # Validate structure
        if not self._validate_structure(data):
            return False
        
        infos = data['infos']
        metadata = data['metadata']
        
        print(f"\n✓ Loaded {len(infos)} samples")
        print(f"✓ Metadata present: {list(metadata.keys())}")
        
        # Validate metadata
        if not self._validate_metadata(metadata):
            return False
        
        # Validate samples
        if not self._validate_samples(infos):
            return False
        
        # ✨ NEW: Print sample ground truth boxes
        self._print_sample_boxes(infos, num_samples=print_samples)
        
        # Print statistics
        self._print_statistics(infos, split)
        
        print(f"\n✓ {split.upper()} split validation passed!")
        return True
    
    def _validate_structure(self, data: Dict) -> bool:
        """Validate data structure"""
        print("\nValidating data structure...")
        
        if 'infos' not in data:
            print("✗ Missing 'infos' key in data")
            return False
        
        if 'metadata' not in data:
            print("✗ Missing 'metadata' key in data")
            return False
        
        print("✓ Data structure valid")
        return True
    
    def _validate_metadata(self, metadata: Dict) -> bool:
        """Validate metadata"""
        print("\nValidating metadata...")
        
        required_keys = ['version', 'classes', 'num_classes']
        for key in required_keys:
            if key not in metadata:
                print(f"✗ Missing metadata key: {key}")
                return False
        
        # Check classes match config
        if metadata['classes'] != self.classes:
            print(f"✗ Classes mismatch!")
            print(f"  Config: {self.classes}")
            print(f"  Metadata: {metadata['classes']}")
            return False
        
        if metadata['num_classes'] != self.num_classes:
            print(f"✗ Number of classes mismatch!")
            print(f"  Config: {self.num_classes}")
            print(f"  Metadata: {metadata['num_classes']}")
            return False
        
        print("✓ Metadata valid")
        return True
    
    def _validate_samples(self, infos: List[Dict]) -> bool:
        """Validate all samples"""
        print("\nValidating samples...")
        
        if len(infos) == 0:
            print("✗ No samples found")
            return False
        
        required_keys = [
            'token', 'timestamp', 'lidar_path', 'cams', 'radars',
            'gt_boxes', 'gt_names', 'gt_velocity'
        ]
        
        errors = []
        
        for i, info in enumerate(infos):
            # Check required keys
            for key in required_keys:
                if key not in info:
                    errors.append(f"Sample {i}: Missing key '{key}'")
            
            # Validate ground truth
            if 'gt_boxes' in info and 'gt_names' in info:
                if len(info['gt_boxes']) != len(info['gt_names']):
                    errors.append(
                        f"Sample {i}: gt_boxes and gt_names length mismatch "
                        f"({len(info['gt_boxes'])} vs {len(info['gt_names'])})"
                    )
                
                # Check box dimensions
                if len(info['gt_boxes']) > 0:
                    if info['gt_boxes'].shape[1] != 7:
                        errors.append(
                            f"Sample {i}: gt_boxes should have 7 values per box, "
                            f"got {info['gt_boxes'].shape[1]}"
                        )
                
                # Check for NaN values
                if np.any(np.isnan(info['gt_boxes'])):
                    errors.append(f"Sample {i}: NaN values in gt_boxes")
                
                if 'gt_velocity' in info and np.any(np.isnan(info['gt_velocity'])):
                    errors.append(f"Sample {i}: NaN values in gt_velocity")
            
            # Validate cameras
            if 'cams' in info:
                expected_cameras = self.dataset_config['cameras']['names']
                for cam in expected_cameras:
                    if cam not in info['cams']:
                        errors.append(f"Sample {i}: Missing camera '{cam}'")
            
            # Validate radars
            if 'radars' in info:
                expected_radars = self.dataset_config['radars']['names']
                for radar in expected_radars:
                    if radar not in info['radars']:
                        errors.append(f"Sample {i}: Missing radar '{radar}'")
            
            # Stop after 10 errors to avoid spam
            if len(errors) >= 10:
                break
        
        if errors:
            print("✗ Validation errors found:")
            for error in errors[:10]:
                print(f"  - {error}")
            if len(errors) > 10:
                print(f"  ... and {len(errors) - 10} more errors")
            return False
        
        print(f"✓ All {len(infos)} samples valid")
        return True
    
    def _print_sample_boxes(self, infos: List[Dict], num_samples: int = 5):
        """
        Print sample ground truth boxes for inspection
        
        Args:
            infos: List of sample infos
            num_samples: Number of samples to print (default: 5)
        """
        print(f"\n{'='*80}")
        print(f"SAMPLE GROUND TRUTH BOXES (First {num_samples} samples)")
        print('='*80)
        
        # Print box format explanation
        print("\nBox format: [x, y, z, width, length, height, yaw]")
        print("  x, y, z: Center position in meters (LiDAR coordinate frame)")
        print("  width, length, height: Box dimensions in meters")
        print("  yaw: Rotation around z-axis in radians")
        print()
        
        for i in range(min(num_samples, len(infos))):
            info = infos[i]
            
            print(f"\n{'-'*80}")
            print(f"Sample {i + 1}/{num_samples}")
            print(f"{'-'*80}")
            print(f"Token: {info['token']}")
            print(f"Timestamp: {info['timestamp']}")
            
            gt_boxes = info['gt_boxes']
            gt_names = info['gt_names']
            gt_velocity = info.get('gt_velocity', None)
            
            num_objects = len(gt_boxes)
            print(f"Number of objects: {num_objects}")
            
            if num_objects == 0:
                print("  (No objects in this sample)")
                continue
            
            print(f"\nGround Truth Boxes:")
            print(f"{'#':<4} {'Class':<25} {'X':>8} {'Y':>8} {'Z':>8} {'W':>7} {'L':>7} {'H':>7} {'Yaw':>8} {'Vx':>7} {'Vy':>7}")
            print("-" * 115)
            
            for j, (box, name) in enumerate(zip(gt_boxes, gt_names)):
                x, y, z, w, l, h, yaw = box
                
                # Get velocity if available
                if gt_velocity is not None and j < len(gt_velocity):
                    vx, vy = gt_velocity[j]
                    vel_str = f"{vx:>7.2f} {vy:>7.2f}"
                else:
                    vel_str = "  N/A     N/A"
                
                print(f"{j:<4} {name:<25} {x:>8.2f} {y:>8.2f} {z:>8.2f} "
                      f"{w:>7.2f} {l:>7.2f} {h:>7.2f} {yaw:>8.4f} {vel_str}")
            
            # Print box statistics for this sample
            if num_objects > 0:
                print(f"\nSample statistics:")
                print(f"  Position range:")
                print(f"    X: [{gt_boxes[:, 0].min():>7.2f}, {gt_boxes[:, 0].max():>7.2f}] m")
                print(f"    Y: [{gt_boxes[:, 1].min():>7.2f}, {gt_boxes[:, 1].max():>7.2f}] m")
                print(f"    Z: [{gt_boxes[:, 2].min():>7.2f}, {gt_boxes[:, 2].max():>7.2f}] m")
                
                print(f"  Size range:")
                print(f"    Width:  [{gt_boxes[:, 3].min():>6.2f}, {gt_boxes[:, 3].max():>6.2f}] m")
                print(f"    Length: [{gt_boxes[:, 4].min():>6.2f}, {gt_boxes[:, 4].max():>6.2f}] m")
                print(f"    Height: [{gt_boxes[:, 5].min():>6.2f}, {gt_boxes[:, 5].max():>6.2f}] m")
                
                # Check if boxes are in point cloud range
                pc_range = self.dataset_config['point_cloud_range']
                in_range = (
                    (gt_boxes[:, 0] >= pc_range[0]) & (gt_boxes[:, 0] <= pc_range[3]) &
                    (gt_boxes[:, 1] >= pc_range[1]) & (gt_boxes[:, 1] <= pc_range[4]) &
                    (gt_boxes[:, 2] >= pc_range[2]) & (gt_boxes[:, 2] <= pc_range[5])
                )
                print(f"  Boxes in PC range: {in_range.sum()}/{num_objects} "
                      f"({in_range.sum()/num_objects*100:.1f}%)")
                
                # Class distribution in this sample
                unique_classes = np.unique(gt_names)
                print(f"  Classes present: {', '.join(unique_classes)}")
        
        print(f"\n{'='*80}")
    
    def _print_statistics(self, infos: List[Dict], split: str):
        """Print dataset statistics"""
        print(f"\n{split.upper()} SPLIT STATISTICS")
        print("-" * 80)
        
        print(f"\nTotal samples: {len(infos)}")
        
        # Count objects per class
        class_counts = {cls: 0 for cls in self.classes}
        total_objects = 0
        
        for info in infos:
            for name in info['gt_names']:
                if name in class_counts:
                    class_counts[name] += 1
                    total_objects += 1
        
        print(f"Total objects: {total_objects}")
        print(f"Average objects per sample: {total_objects / len(infos):.2f}")
        
        print(f"\nObjects per class:")
        for cls in self.classes:
            count = class_counts[cls]
            percentage = (count / total_objects * 100) if total_objects > 0 else 0
            print(f"  {cls:<25} {count:>6} ({percentage:>5.1f}%)")
        
        # Box statistics
        all_boxes = np.concatenate([info['gt_boxes'] for info in infos if len(info['gt_boxes']) > 0])
        
        if len(all_boxes) > 0:
            print(f"\nBounding box statistics:")
            print(f"  Total boxes: {len(all_boxes)}")
            print(f"\n  Dimension statistics (mean ± std):")
            print(f"    Width (x):  {all_boxes[:, 3].mean():.2f} ± {all_boxes[:, 3].std():.2f} m")
            print(f"    Length (y): {all_boxes[:, 4].mean():.2f} ± {all_boxes[:, 4].std():.2f} m")
            print(f"    Height (z): {all_boxes[:, 5].mean():.2f} ± {all_boxes[:, 5].std():.2f} m")
            
            print(f"\n  Position statistics (mean ± std):")
            print(f"    X: {all_boxes[:, 0].mean():.2f} ± {all_boxes[:, 0].std():.2f} m")
            print(f"    Y: {all_boxes[:, 1].mean():.2f} ± {all_boxes[:, 1].std():.2f} m")
            print(f"    Z: {all_boxes[:, 2].mean():.2f} ± {all_boxes[:, 2].std():.2f} m")
            
            print(f"\n  Yaw statistics:")
            print(f"    Range: [{all_boxes[:, 6].min():.4f}, {all_boxes[:, 6].max():.4f}] rad")
            print(f"    Mean: {all_boxes[:, 6].mean():.4f} rad")
            
            # Check point cloud range
            pc_range = self.dataset_config['point_cloud_range']
            in_range = (
                (all_boxes[:, 0] >= pc_range[0]) & (all_boxes[:, 0] <= pc_range[3]) &
                (all_boxes[:, 1] >= pc_range[1]) & (all_boxes[:, 1] <= pc_range[4]) &
                (all_boxes[:, 2] >= pc_range[2]) & (all_boxes[:, 2] <= pc_range[5])
            )
            print(f"\n  Boxes in point cloud range: {in_range.sum()} / {len(all_boxes)} ({in_range.sum()/len(all_boxes)*100:.1f}%)")
            
            if in_range.sum() < len(all_boxes):
                out_of_range = len(all_boxes) - in_range.sum()
                print(f"  ⚠️ {out_of_range} boxes ({out_of_range/len(all_boxes)*100:.1f}%) are outside PC range!")
                print(f"     These objects may not be detected during training/inference.")
        
        # Velocity statistics
        all_velocities = np.concatenate([info['gt_velocity'] for info in infos if len(info['gt_velocity']) > 0])
        
        if len(all_velocities) > 0:
            print(f"\nVelocity statistics:")
            print(f"  Mean velocity (x): {all_velocities[:, 0].mean():.2f} m/s")
            print(f"  Mean velocity (y): {all_velocities[:, 1].mean():.2f} m/s")
            print(f"  Mean speed: {np.linalg.norm(all_velocities, axis=1).mean():.2f} m/s")
            print(f"  Max speed: {np.linalg.norm(all_velocities, axis=1).max():.2f} m/s")
    
    def validate_all(self, print_samples: int = 5) -> bool:
        """
        Validate all splits
        
        Args:
            print_samples: Number of sample ground truth boxes to print per split
        """
        print("\n" + "="*80)
        print("NUSCENES DATA VALIDATION (CONFIG-DRIVEN)")
        print("="*80)
        print(f"\nConfiguration: {self.dataset_config['name']} {self.dataset_config['version']}")
        print(f"Classes: {self.num_classes}")
        
        results = {}
        
        for split in ['train', 'val', 'test']:
            results[split] = self.validate_split(split, print_samples=print_samples)
        
        # Summary
        print("\n" + "="*80)
        print("VALIDATION SUMMARY")
        print("="*80)
        
        for split, passed in results.items():
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"{split.upper():<10} {status}")
        
        all_passed = all(results.values())
        
        if all_passed:
            print("\n✓ All splits validated successfully!")
        else:
            print("\n✗ Some splits failed validation")
        
        return all_passed


def main():
    """Main validation function"""
    parser = argparse.ArgumentParser(
        description='Validate converted NuScenes data using config.yaml'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/base.yaml',
        help='Path to configuration file (default: configs/base.yaml)'
    )
    parser.add_argument(
        '--split',
        type=str,
        default=None,
        choices=['train', 'val', 'test'],
        help='Validate specific split only (default: all splits)'
    )
    parser.add_argument(
        '--samples',
        type=int,
        default=5,
        help='Number of sample ground truth boxes to print (default: 5)'
    )
    
    args = parser.parse_args()
    
    try:
        validator = ConfigDrivenDataValidator(config_path=args.config)
    except FileNotFoundError:
        print(f"Error: Configuration file '{args.config}' not found!")
        return
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return
    
    try:
        if args.split:
            # Validate single split
            success = validator.validate_split(args.split, print_samples=args.samples)
        else:
            # Validate all splits
            success = validator.validate_all(print_samples=args.samples)
        
        exit(0 if success else 1)
        
    except Exception as e:
        print(f"\nError during validation: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == '__main__':
    main()