#python evaluate.py --model checkpoints/best_model.pth --config config.yaml


"""
Config-Driven Evaluation Script for Multi-Modal 3D Detection
Loads trained model, runs evaluation, computes metrics, and saves results to CSV
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import yaml
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from tqdm import tqdm
import pickle
import argparse
from datetime import datetime
import torch.nn.functional as F

from train_detect import NuScenesDataset, collate_fn
from fusion import create_detector
from utils_v2 import compute_metrics
from fusion_detection import decode_centernet_predictions


class ModelEvaluator:
    """
    Config-driven model evaluator
    Handles model loading, evaluation, metric computation, and result saving
    """
    
    def __init__(
        self,
        model_path: str,
        config_path: str = 'config.yaml',
        device: str = 'cuda',
        output_dir: str = './eval_results'
    ):
        """
        Args:
            model_path: Path to model checkpoint
            config_path: Path to configuration file
            device: Device to use ('cuda' or 'cpu')
            output_dir: Directory to save evaluation results
        """
        self.model_path = model_path
        self.config_path = config_path
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load configuration
        print("="*80)
        print("LOADING CONFIGURATION")
        print("="*80)
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.dataset_config = self.config['dataset']
        self.model_config = self.config['model']
        self.classes = self.dataset_config['classes']
        self.num_classes = len(self.classes)
        
        print(f"Config loaded: {config_path}")
        print(f"Classes: {self.num_classes}")
        print(f"Device: {self.device}")
        
        # Load model
        self.model = self._load_model()
    
    def _load_model(self) -> nn.Module:
        """Load model from checkpoint"""
        
        
        print("\n" + "="*80)
        print("LOADING MODEL")
        print("="*80)
        print(f"Model path: {self.model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Get model configuration (from checkpoint or config file)
        if 'config' in checkpoint:
            model_config = checkpoint['config']
            print("Using model config from checkpoint")
        else:
            model_config = self.model_config
            print("Using model config from config.yaml")
        
        # Extract modality settings
        modality_config = model_config.get('modality_config', 'camera+lidar')
        fusion_type = model_config.get('fusion_type', 'bev')
        detection_head = model_config.get('detection_head', 'centernet')
        
        print(f"Modality: {modality_config}")
        print(f"Fusion: {fusion_type}")
        print(f"Detection head: {detection_head}")
        
        # Create model
        model = create_detector(
            modality_config=modality_config,
            fusion_type=fusion_type,
            detection_head=detection_head,
            num_classes=self.num_classes
        )
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()
        
        epoch = checkpoint.get('epoch', 'unknown')
        best_map = checkpoint.get('best_map', 0.0)
        
        print(f"✓ Model loaded successfully")
        print(f"  Epoch: {epoch}")
        print(f"  Best mAP: {best_map:.4f}")
        
        return model
    
    @torch.no_grad()
    def evaluate(
        self,
        dataloader: DataLoader,
        save_predictions: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate model on dataset
        
        Args:
            dataloader: DataLoader for evaluation
            save_predictions: Whether to save predictions to CSV
        
        Returns:
            Dictionary of metrics
        """
        print("\n" + "="*80)
        print("RUNNING EVALUATION")
        print("="*80)
        
        self.model.eval()
        
        # Get modality flags from model config
        modality_config = self.model_config.get('modality_config', 'all')
        use_camera = 'camera' in modality_config or modality_config == 'all'
        use_lidar = 'lidar' in modality_config or modality_config == 'all'
        use_radar = 'radar' in modality_config or modality_config == 'all'
        
        print(f"Using modalities: camera={use_camera}, lidar={use_lidar}, radar={use_radar}")
        
        all_predictions = []
        all_ground_truths = []
        all_sample_tokens = []
        
        # Evaluation loop
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            # Prepare inputs
            camera_imgs = batch['camera_imgs'].to(self.device) if use_camera and 'camera_imgs' in batch else None
            lidar_points = batch['lidar_points'].to(self.device) if use_lidar and 'lidar_points' in batch else None
            radar_points = [r.to(self.device) for r in batch['radar_points']] if use_radar and 'radar_points' in batch else None
            
            # Forward pass
            predictions = self.model(
                camera_imgs=camera_imgs,
                lidar_points=lidar_points,
                radar_points=radar_points
            )
            
            # Decode predictions (if CenterNet-style)
            if 'heatmap' in predictions:
                decoded = self._decode_centernet_predictions(predictions)
                all_predictions.extend(decoded)
            else:
                decoded = self._decode_mlp_predictions(predictions)
                all_predictions.extend(decoded)
            
            # Store ground truths
            for i in range(len(batch['gt_boxes'])):
                gt_dict = {
                    'boxes': batch['gt_boxes'][i].cpu().numpy(),
                    'labels': batch['gt_labels'][i].cpu().numpy(),
                    'token': batch.get('token', [f'sample_{batch_idx}_{i}'])[i]
                }
                all_ground_truths.append(gt_dict)
                all_sample_tokens.append(gt_dict['token'])
        
        print(f"\n✓ Processed {len(all_predictions)} samples")
        
        # Save predictions and ground truths to CSV
        if save_predictions:
            self._save_to_csv(all_predictions, all_ground_truths, all_sample_tokens)
        
        # Compute metrics
        metrics = self._compute_metrics(all_predictions, all_ground_truths)
        
        # Save metrics
        self._save_metrics(metrics)
        
        return metrics
    
    def _decode_centernet_predictions(self, predictions: Dict) -> List[Dict]:
        """Decode CenterNet-style predictions"""
        
        
        pc_range = self.dataset_config['point_cloud_range']
        
        decoded = decode_centernet_predictions(
            predictions,
            score_thresh=0.0,  # We'll filter later
            K=100,
            pc_range=pc_range
        )
        
        return decoded
    
    def _decode_mlp_predictions(self, predictions: Dict) -> List[Dict]:
        """Decode MLP-style predictions"""
        
        
        batch_size = predictions['cls'].size(0)
        decoded = []
        
        for i in range(batch_size):
            cls_scores = F.softmax(predictions['cls'][i], dim=-1)
            scores, labels = cls_scores.max(dim=-1)
            boxes = predictions['box'][i]
            
            decoded.append({
                'boxes': boxes.cpu().numpy(),
                'scores': scores.cpu().numpy(),
                'labels': labels.cpu().numpy()
            })
        
        return decoded
    
    def _save_to_csv(
        self,
        predictions: List[Dict],
        ground_truths: List[Dict],
        sample_tokens: List[str]
    ):
        """Save predictions and ground truths to CSV files"""
        print("\n" + "="*80)
        print("SAVING PREDICTIONS AND GROUND TRUTHS")
        print("="*80)
        
        # Prepare predictions DataFrame
        pred_rows = []
        for sample_idx, (pred, token) in enumerate(zip(predictions, sample_tokens)):
            boxes = pred['boxes']
            scores = pred['scores']
            labels = pred['labels']
            
            for box_idx, (box, score, label) in enumerate(zip(boxes, scores, labels)):
                class_name = self.classes[label] if label < len(self.classes) else f'class_{label}'
                
                pred_rows.append({
                    'sample_token': token,
                    'sample_idx': sample_idx,
                    'box_idx': box_idx,
                    'class_id': label,
                    'class_name': class_name,
                    'score': float(score),
                    'x': float(box[0]),
                    'y': float(box[1]),
                    'z': float(box[2]),
                    'width': float(box[3]),
                    'length': float(box[4]),
                    'height': float(box[5]),
                    'yaw': float(box[6])
                })
        
        pred_df = pd.DataFrame(pred_rows)
        pred_path = self.output_dir / 'predictions.csv'
        pred_df.to_csv(pred_path, index=False)
        print(f"✓ Saved predictions to {pred_path}")
        print(f"  Total predictions: {len(pred_rows)}")
        
        # Prepare ground truths DataFrame
        gt_rows = []
        for sample_idx, (gt, token) in enumerate(zip(ground_truths, sample_tokens)):
            boxes = gt['boxes']
            labels = gt['labels']
            
            for box_idx, (box, label) in enumerate(zip(boxes, labels)):
                class_name = self.classes[label] if label < len(self.classes) else f'class_{label}'
                
                gt_rows.append({
                    'sample_token': token,
                    'sample_idx': sample_idx,
                    'box_idx': box_idx,
                    'class_id': label,
                    'class_name': class_name,
                    'x': float(box[0]),
                    'y': float(box[1]),
                    'z': float(box[2]),
                    'width': float(box[3]),
                    'length': float(box[4]),
                    'height': float(box[5]),
                    'yaw': float(box[6])
                })
        
        gt_df = pd.DataFrame(gt_rows)
        gt_path = self.output_dir / 'ground_truths.csv'
        gt_df.to_csv(gt_path, index=False)
        print(f"✓ Saved ground truths to {gt_path}")
        print(f"  Total ground truths: {len(gt_rows)}")
        
        # Save summary statistics
        summary_path = self.output_dir / 'data_summary.csv'
        summary_df = pd.DataFrame([{
            'total_samples': len(predictions),
            'total_predictions': len(pred_rows),
            'total_ground_truths': len(gt_rows),
            'avg_predictions_per_sample': len(pred_rows) / len(predictions) if len(predictions) > 0 else 0,
            'avg_ground_truths_per_sample': len(gt_rows) / len(ground_truths) if len(ground_truths) > 0 else 0
        }])
        summary_df.to_csv(summary_path, index=False)
        print(f"✓ Saved data summary to {summary_path}")
    
    def _compute_metrics(
        self,
        predictions: List[Dict],
        ground_truths: List[Dict]
    ) -> Dict[str, float]:
        """Compute evaluation metrics"""
        print("\n" + "="*80)
        print("COMPUTING METRICS")
        print("="*80)
        
        # Initialize metrics
        metrics = {
            'num_samples': len(predictions),
            'num_predictions': sum(len(p['boxes']) for p in predictions),
            'num_ground_truths': sum(len(gt['boxes']) for gt in ground_truths)
        }
        
        # Compute per-sample metrics
        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_iou = 0
        num_matched = 0
        
        # Per-class metrics
        per_class_tp = {cls: 0 for cls in self.classes}
        per_class_fp = {cls: 0 for cls in self.classes}
        per_class_fn = {cls: 0 for cls in self.classes}
        per_class_gt = {cls: 0 for cls in self.classes}
        per_class_pred = {cls: 0 for cls in self.classes}
        
        iou_threshold = 0.5
        
        for pred, gt in zip(predictions, ground_truths):
            pred_boxes = pred['boxes']
            pred_labels = pred['labels']
            pred_scores = pred['scores']
            
            gt_boxes = gt['boxes']
            gt_labels = gt['labels']
            
            # Count per-class ground truths and predictions
            for label in gt_labels:
                if label < len(self.classes):
                    per_class_gt[self.classes[label]] += 1
            
            for label in pred_labels:
                if label < len(self.classes):
                    per_class_pred[self.classes[label]] += 1
            
            if len(pred_boxes) == 0 or len(gt_boxes) == 0:
                total_fn += len(gt_boxes)
                total_fp += len(pred_boxes)
                for label in gt_labels:
                    if label < len(self.classes):
                        per_class_fn[self.classes[label]] += 1
                for label in pred_labels:
                    if label < len(self.classes):
                        per_class_fp[self.classes[label]] += 1
                continue
            
            # Compute IoU matrix
            ious = self._compute_iou_matrix(pred_boxes, gt_boxes)
            
            # Match predictions to ground truths
            matched_gt = set()
            
            # Sort predictions by score (descending)
            sorted_indices = np.argsort(-pred_scores)
            
            for pred_idx in sorted_indices:
                best_iou = 0
                best_gt_idx = -1
                
                # Find best matching GT
                for gt_idx in range(len(gt_boxes)):
                    if gt_idx in matched_gt:
                        continue
                    
                    if pred_labels[pred_idx] == gt_labels[gt_idx] and ious[pred_idx, gt_idx] > best_iou:
                        best_iou = ious[pred_idx, gt_idx]
                        best_gt_idx = gt_idx
                
                # Check if match is valid
                if best_iou >= iou_threshold:
                    total_tp += 1
                    matched_gt.add(best_gt_idx)
                    total_iou += best_iou
                    num_matched += 1
                    
                    if pred_labels[pred_idx] < len(self.classes):
                        per_class_tp[self.classes[pred_labels[pred_idx]]] += 1
                else:
                    total_fp += 1
                    if pred_labels[pred_idx] < len(self.classes):
                        per_class_fp[self.classes[pred_labels[pred_idx]]] += 1
            
            # Count false negatives
            fn = len(gt_boxes) - len(matched_gt)
            total_fn += fn
            
            for gt_idx, label in enumerate(gt_labels):
                if gt_idx not in matched_gt and label < len(self.classes):
                    per_class_fn[self.classes[label]] += 1
        
        # Compute overall metrics
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        mean_iou = total_iou / num_matched if num_matched > 0 else 0
        
        metrics.update({
            'tp': total_tp,
            'fp': total_fp,
            'fn': total_fn,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'mean_iou': mean_iou,
            'iou_threshold': iou_threshold
        })
        
        # Compute per-class metrics
        for cls in self.classes:
            tp = per_class_tp[cls]
            fp = per_class_fp[cls]
            fn = per_class_fn[cls]
            
            cls_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            cls_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            cls_f1 = 2 * cls_precision * cls_recall / (cls_precision + cls_recall) if (cls_precision + cls_recall) > 0 else 0
            
            metrics[f'{cls}_precision'] = cls_precision
            metrics[f'{cls}_recall'] = cls_recall
            metrics[f'{cls}_f1'] = cls_f1
            metrics[f'{cls}_tp'] = tp
            metrics[f'{cls}_fp'] = fp
            metrics[f'{cls}_fn'] = fn
            metrics[f'{cls}_gt_count'] = per_class_gt[cls]
            metrics[f'{cls}_pred_count'] = per_class_pred[cls]
        
        # Approximate mAP (simplified)
        metrics['mAP'] = f1  # Use F1 as proxy for mAP
        
        # Print metrics
        self._print_metrics(metrics)
        
        return metrics
    
    def _compute_iou_matrix(self, boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
        """Compute IoU matrix between two sets of boxes (BEV IoU)"""
        n1 = len(boxes1)
        n2 = len(boxes2)
        
        ious = np.zeros((n1, n2))
        
        for i in range(n1):
            for j in range(n2):
                # BEV IoU (x, y, w, l)
                x1, y1, w1, l1 = boxes1[i, 0], boxes1[i, 1], boxes1[i, 3], boxes1[i, 4]
                x2, y2, w2, l2 = boxes2[j, 0], boxes2[j, 1], boxes2[j, 3], boxes2[j, 4]
                
                # Compute overlap
                x_overlap = max(0, min(x1 + w1/2, x2 + w2/2) - max(x1 - w1/2, x2 - w2/2))
                y_overlap = max(0, min(y1 + l1/2, y2 + l2/2) - max(y1 - l1/2, y2 - l2/2))
                
                intersection = x_overlap * y_overlap
                union = w1 * l1 + w2 * l2 - intersection
                
                ious[i, j] = intersection / union if union > 0 else 0
        
        return ious
    
    def _print_metrics(self, metrics: Dict[str, float]):
        """Print metrics in formatted way"""
        print("\n" + "="*80)
        print("EVALUATION METRICS")
        print("="*80)
        
        print(f"\nDataset Statistics:")
        print(f"  Samples: {metrics['num_samples']}")
        print(f"  Total Predictions: {metrics['num_predictions']}")
        print(f"  Total Ground Truths: {metrics['num_ground_truths']}")
        
        print(f"\nOverall Performance:")
        print(f"  True Positives:  {metrics['tp']}")
        print(f"  False Positives: {metrics['fp']}")
        print(f"  False Negatives: {metrics['fn']}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1 Score:  {metrics['f1_score']:.4f}")
        print(f"  Mean IoU:  {metrics['mean_iou']:.4f}")
        print(f"  mAP:       {metrics['mAP']:.4f}")
        
        print(f"\nPer-Class Performance:")
        print(f"{'Class':<25} {'Precision':>10} {'Recall':>10} {'F1':>10} {'GT':>6} {'Pred':>6}")
        print("-" * 80)
        for cls in self.classes:
            if f'{cls}_precision' in metrics:
                print(f"{cls:<25} "
                      f"{metrics[f'{cls}_precision']:>10.4f} "
                      f"{metrics[f'{cls}_recall']:>10.4f} "
                      f"{metrics[f'{cls}_f1']:>10.4f} "
                      f"{metrics[f'{cls}_gt_count']:>6} "
                      f"{metrics[f'{cls}_pred_count']:>6}")
    
    def _save_metrics(self, metrics: Dict[str, float]):
        """Save metrics to CSV file"""
        print("\n" + "="*80)
        print("SAVING METRICS")
        print("="*80)
        
        # Overall metrics
        overall_metrics = {
            'metric': ['num_samples', 'num_predictions', 'num_ground_truths',
                      'tp', 'fp', 'fn', 'precision', 'recall', 'f1_score', 
                      'mean_iou', 'mAP', 'iou_threshold'],
            'value': [metrics.get(k, 0) for k in ['num_samples', 'num_predictions', 'num_ground_truths',
                     'tp', 'fp', 'fn', 'precision', 'recall', 'f1_score',
                     'mean_iou', 'mAP', 'iou_threshold']]
        }
        
        overall_df = pd.DataFrame(overall_metrics)
        overall_path = self.output_dir / 'metrics_overall.csv'
        overall_df.to_csv(overall_path, index=False)
        print(f"✓ Saved overall metrics to {overall_path}")
        
        # Per-class metrics
        per_class_rows = []
        for cls in self.classes:
            if f'{cls}_precision' in metrics:
                per_class_rows.append({
                    'class': cls,
                    'precision': metrics[f'{cls}_precision'],
                    'recall': metrics[f'{cls}_recall'],
                    'f1_score': metrics[f'{cls}_f1'],
                    'tp': metrics[f'{cls}_tp'],
                    'fp': metrics[f'{cls}_fp'],
                    'fn': metrics[f'{cls}_fn'],
                    'gt_count': metrics[f'{cls}_gt_count'],
                    'pred_count': metrics[f'{cls}_pred_count']
                })
        
        per_class_df = pd.DataFrame(per_class_rows)
        per_class_path = self.output_dir / 'metrics_per_class.csv'
        per_class_df.to_csv(per_class_path, index=False)
        print(f"✓ Saved per-class metrics to {per_class_path}")
        
        # Save metadata
        metadata = {
            'model_path': str(self.model_path),
            'config_path': str(self.config_path),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'device': str(self.device),
            'num_classes': self.num_classes
        }
        
        metadata_df = pd.DataFrame([metadata])
        metadata_path = self.output_dir / 'eval_metadata.csv'
        metadata_df.to_csv(metadata_path, index=False)
        print(f"✓ Saved metadata to {metadata_path}")


def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description='Evaluate 3D detection model')
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='configs/base.yaml', help='Path to config file')
    parser.add_argument('--data-root', type=str, default='./data/nuscenes', help='Data root directory')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test'], 
                       help='Dataset split to evaluate')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--output-dir', type=str, default='./eval_results', help='Output directory')
    parser.add_argument('--no-save-predictions', action='store_true', 
                       help='Don\'t save predictions to CSV')
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = ModelEvaluator(
        model_path=args.model,
        config_path=args.config,
        device=args.device,
        output_dir=args.output_dir
    )
    
    # Create dataset and dataloader
    print("\n" + "="*80)
    print("LOADING DATASET")
    print("="*80)
    
    
    dataset = NuScenesDataset(
        data_root=args.data_root,
        split=args.split
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        #collate_fn=dataset.collate_fn
        collate_fn=collate_fn
    )
    
    print(f"Dataset: {args.split}")
    print(f"Samples: {len(dataset)}")
    print(f"Batch size: {args.batch_size}")
    
    # Run evaluation
    metrics = evaluator.evaluate(
        dataloader=dataloader,
        save_predictions=not args.no_save_predictions
    )
    
    # Final summary
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {args.output_dir}")
    print(f"  predictions.csv - All predictions")
    print(f"  ground_truths.csv - All ground truths")
    print(f"  metrics_overall.csv - Overall metrics")
    print(f"  metrics_per_class.csv - Per-class metrics")
    print(f"  data_summary.csv - Data summary")
    print(f"  eval_metadata.csv - Evaluation metadata")
    
    print(f"\nKey Metrics:")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1_score']:.4f}")
    print(f"  mAP:       {metrics['mAP']:.4f}")
    
    print("\n✓ Evaluation completed successfully!")


if __name__ == '__main__':
    main()