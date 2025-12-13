"""
Validate Converted NuScenes Data (Config-Driven)
Reads configuration from config.yaml and validates converted pickle files
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
    
    def validate_split(self, split: str) -> bool:
        """Validate a single split"""
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
            
            # Check point cloud range
            pc_range = self.dataset_config['point_cloud_range']
            in_range = (
                (all_boxes[:, 0] >= pc_range[0]) & (all_boxes[:, 0] <= pc_range[3]) &
                (all_boxes[:, 1] >= pc_range[1]) & (all_boxes[:, 1] <= pc_range[4]) &
                (all_boxes[:, 2] >= pc_range[2]) & (all_boxes[:, 2] <= pc_range[5])
            )
            print(f"\n  Boxes in point cloud range: {in_range.sum()} / {len(all_boxes)} ({in_range.sum()/len(all_boxes)*100:.1f}%)")
        
        # Velocity statistics
        all_velocities = np.concatenate([info['gt_velocity'] for info in infos if len(info['gt_velocity']) > 0])
        
        if len(all_velocities) > 0:
            print(f"\nVelocity statistics:")
            print(f"  Mean velocity (x): {all_velocities[:, 0].mean():.2f} m/s")
            print(f"  Mean velocity (y): {all_velocities[:, 1].mean():.2f} m/s")
            print(f"  Mean speed: {np.linalg.norm(all_velocities, axis=1).mean():.2f} m/s")
    
    def validate_all(self) -> bool:
        """Validate all splits"""
        print("\n" + "="*80)
        print("NUSCENES DATA VALIDATION (CONFIG-DRIVEN)")
        print("="*80)
        print(f"\nConfiguration: {self.dataset_config['name']} {self.dataset_config['version']}")
        print(f"Classes: {self.num_classes}")
        
        results = {}
        
        for split in ['train', 'val', 'test']:
            results[split] = self.validate_split(split)
        
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
        help='Path to configuration file (default: config.yaml)'
    )
    parser.add_argument(
        '--split',
        type=str,
        default=None,
        choices=['train', 'val', 'test'],
        help='Validate specific split only (default: all splits)'
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
            success = validator.validate_split(args.split)
        else:
            # Validate all splits
            success = validator.validate_all()
        
        exit(0 if success else 1)
        
    except Exception as e:
        print(f"\nError during validation: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == '__main__':
    main()