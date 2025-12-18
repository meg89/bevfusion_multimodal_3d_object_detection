"""
Refined Inference Module for Multi-Modal 3D Detection
Supports:
- Config-driven inference
- Visualization of BEV, 3D boxes, camera projections
- Metrics computation (mAP, NDS)
- Show/no-show modes
- Batch inference
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import pickle
import yaml
import argparse

from fusion_detection import decode_centernet_predictions
from fusion import create_detector
from train_detect import NuScenesDataset


# For 3D visualization
try:
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
except ImportError:
    print("Warning: 3D plotting not available")


class InferenceEngine:
    """
    Complete inference engine with visualization
    Supports config-driven inference with flexible display options
    """
    
    def __init__(
        self,
        model_path: str,
        config_path: str = 'config.yaml',
        device: str = 'cuda',
        show: bool = True,
        save_dir: Optional[str] = None
    ):
        """
        Args:
            model_path: Path to trained model checkpoint
            config_path: Path to configuration file
            device: Device to run on ('cuda' or 'cpu')
            show: Whether to display visualizations
            save_dir: Directory to save visualizations (if None, don't save)
        """
        self.model_path = model_path
        self.config_path = config_path
        self.device = device
        self.show = show
        self.save_dir = Path(save_dir) if save_dir else None
        
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Load configuration    
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Get configuration
        self.dataset_config = self.config['dataset']
        self.classes = self.dataset_config['classes']
        self.num_classes = self.dataset_config['num_classes']
        self.pc_range = self.dataset_config['point_cloud_range']
        
        # Visualization config
        viz_config = self.config.get('visualization', {})
        self.box_colors = viz_config.get('style', {}).get('box_colors', {})
        self.score_thresh = viz_config.get('style', {}).get('score_threshold', 0.3)
        
        # Load model
        #self.model = self._load_model()

        print("="*80)
        print("INFERENCE ENGINE INITIALIZED")
        print("="*80)
        print(f"Model: {model_path}")
        print(f"Config: {config_path}")
        print(f"Device: {device}")
        print(f"Show visualizations: {show}")
        print(f"Save directory: {save_dir}")
        print(f"Classes: {self.num_classes}")
        print("="*80)
    
    def _load_model(self):
        """Load model from checkpoint"""
        
        
        print(f"\nLoading model from {self.model_path}...")
        
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Get model configuration
        if 'config' in checkpoint:
            model_config = checkpoint['config']
        else:
            model_config = self.config['model']
        
        # Create model
        model = create_detector(
            modality_config=model_config.get('modality_config', 'all'),
            fusion_type=model_config.get('fusion_type', 'bev'),
            detection_head=model_config.get('detection_head', 'centernet'),
            num_classes=self.num_classes

        )
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'], strict= False)
        model = model.to(self.device)
        model.eval()
        
        epoch = checkpoint.get('epoch', 'unknown')
        print(f"✓ Model loaded successfully (Epoch: {epoch})")
        
        return model
    
    @torch.no_grad()
    def run_inference(
        self,
        camera_imgs: Optional[torch.Tensor] = None,
        lidar_points: Optional[torch.Tensor] = None,
        radar_points: Optional[List[torch.Tensor]] = None,
        gt_boxes: Optional[np.ndarray] = None,
        gt_labels: Optional[np.ndarray] = None,
        sample_token: Optional[str] = None
    ) -> Dict:
        """
        Run inference on a single sample
        
        Args:
            camera_imgs: (1, 6, 3, H, W) or None
            lidar_points: (1, N, 5) or None
            radar_points: List of 5 tensors (1, M, 7) or None
            gt_boxes: (N, 7) ground truth boxes [x,y,z,w,l,h,yaw]
            gt_labels: (N,) ground truth labels
            sample_token: Sample identifier
        
        Returns:
            Dictionary with predictions and metrics
        """
        print("\n" + "="*80)
        print("RUNNING INFERENCE")
        print("="*80)
        
        # Move to device
        if camera_imgs is not None:
            camera_imgs = camera_imgs.to(self.device)
        if lidar_points is not None:
            lidar_points = lidar_points.to(self.device)
        if radar_points is not None:
            radar_points = [r.to(self.device) for r in radar_points]
        
        # Load model
        self.model = self._load_model()

        # Forward pass
        predictions = self.model(
            camera_imgs=camera_imgs,
            lidar_points=lidar_points,
            radar_points=radar_points
        )
        
        print(f"\nRaw predictions:")
        for key, value in predictions.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape}")
        
        # Decode predictions
        if 'heatmap' in predictions:
            detections = self._decode_centernet(predictions)
        else:
            detections = self._decode_mlp(predictions)
        
        # Filter by score threshold
        keep = detections['scores'] > self.score_thresh
        for key in ['boxes', 'scores', 'labels', 'velocities']:
            if key in detections:
                detections[key] = detections[key][keep]
        
        num_detections = len(detections['boxes'])
        print(f"\n✓ Detected {num_detections} objects (score > {self.score_thresh})")
        
        # Print top detections
        self._print_detections(detections)
        
        # Compute metrics if GT available
        metrics = None
        if gt_boxes is not None and gt_labels is not None:
            metrics = self._compute_metrics(detections, gt_boxes, gt_labels)
            self._print_metrics(metrics)
        
        # Prepare results
        results = {
            'predictions': detections,
            'metrics': metrics,
            'raw_predictions': predictions,
            'sample_token': sample_token
        }
        
        # Visualize
        if self.show or self.save_dir:
            self._visualize(
                detections=detections,
                camera_imgs=camera_imgs,
                lidar_points=lidar_points,
                gt_boxes=gt_boxes,
                gt_labels=gt_labels,
                sample_token=sample_token
            )
        
        return results
    
    def _decode_centernet(self, predictions: Dict) -> Dict:
        """Decode CenterNet predictions"""
        
        
        detections = decode_centernet_predictions(
            predictions,
            score_thresh=0.0,  # We'll filter later
            max_detections=100
            #pc_range=self.pc_range
        )[0]  # Get first (and only) batch item
        
        return detections
    
    def _decode_mlp(self, predictions: Dict) -> Dict:
        """Decode MLP predictions"""
        # For MLP head, we have single prediction
        cls_scores = F.softmax(predictions['cls'], dim=-1)
        scores, labels = cls_scores.max(dim=-1)
        boxes = predictions['box']
        
        return {
            'boxes': boxes.cpu().numpy(),
            'scores': scores.cpu().numpy(),
            'labels': labels.cpu().numpy(),
            'velocities': np.zeros((len(boxes), 2))
        }
    
    def _print_detections(self, detections: Dict, max_show: int = 10):
        """Print detection results"""
        boxes = detections['boxes']
        scores = detections['scores']
        labels = detections['labels']
        
        print(f"\nTop {min(len(boxes), max_show)} Detections:")
        print("-" * 80)
        
        for i in range(min(len(boxes), max_show)):
            class_name = self.classes[labels[i]] if labels[i] < len(self.classes) else f"class_{labels[i]}"
            box = boxes[i]
            score = scores[i]
            
            print(f"\n  [{i+1}] {class_name.upper()}")
            print(f"      Score: {score:.3f}")
            print(f"      Position: x={box[0]:>6.2f}m, y={box[1]:>6.2f}m, z={box[2]:>6.2f}m")
            print(f"      Size: w={box[3]:>5.2f}m, l={box[4]:>5.2f}m, h={box[5]:>5.2f}m")
            print(f"      Yaw: {box[6]:>6.2f} rad ({np.rad2deg(box[6]):>6.1f}°)")
            
            if 'velocities' in detections and len(detections['velocities']) > i:
                vel = detections['velocities'][i]
                speed = np.linalg.norm(vel)
                print(f"      Velocity: vx={vel[0]:>5.2f}, vy={vel[1]:>5.2f} m/s (speed: {speed:.2f} m/s)")
    
    def _compute_metrics(
        self,
        detections: Dict,
        gt_boxes: np.ndarray,
        gt_labels: np.ndarray
    ) -> Dict:
        """Compute detection metrics"""
        pred_boxes = detections['boxes']
        pred_labels = detections['labels']
        pred_scores = detections['scores']
        
        # Compute IoU between predictions and ground truth
        ious = self._compute_iou_3d(pred_boxes, gt_boxes)
        
        # Match predictions to ground truth
        matches = self._match_detections(ious, pred_labels, gt_labels, iou_thresh=0.5)
        
        # Compute metrics
        tp = matches['tp']
        fp = matches['fp']
        fn = len(gt_boxes) - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Per-class metrics
        per_class_metrics = {}
        for cls_idx, cls_name in enumerate(self.classes):
            cls_gt = gt_labels == cls_idx
            cls_pred = pred_labels == cls_idx
            
            if cls_gt.sum() > 0 or cls_pred.sum() > 0:
                per_class_metrics[cls_name] = {
                    'gt_count': int(cls_gt.sum()),
                    'pred_count': int(cls_pred.sum())
                }
        
        return {
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'mean_iou': ious.max(axis=1).mean() if len(ious) > 0 else 0,
            'per_class': per_class_metrics
        }
    
    def _compute_iou_3d(self, boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
        """Compute 3D IoU between two sets of boxes"""
        # Simplified 2D BEV IoU for now
        n1 = len(boxes1)
        n2 = len(boxes2)
        
        if n1 == 0 or n2 == 0:
            return np.zeros((n1, n2))
        
        ious = np.zeros((n1, n2))
        
        for i, box1 in enumerate(boxes1):
            for j, box2 in enumerate(boxes2):
                # BEV IoU (x, y, w, l)
                x1, y1, w1, l1 = box1[0], box1[1], box1[3], box1[4]
                x2, y2, w2, l2 = box2[0], box2[1], box2[3], box2[4]
                
                # Compute overlap
                x_overlap = max(0, min(x1 + w1/2, x2 + w2/2) - max(x1 - w1/2, x2 - w2/2))
                y_overlap = max(0, min(y1 + l1/2, y2 + l2/2) - max(y1 - l1/2, y2 - l2/2))
                
                intersection = x_overlap * y_overlap
                union = w1 * l1 + w2 * l2 - intersection
                
                ious[i, j] = intersection / union if union > 0 else 0
        
        return ious
    
    def _match_detections(
        self,
        ious: np.ndarray,
        pred_labels: np.ndarray,
        gt_labels: np.ndarray,
        iou_thresh: float = 0.5
    ) -> Dict:
        """Match predictions to ground truth"""
        n_pred = len(pred_labels)
        n_gt = len(gt_labels)
        
        if n_pred == 0 or n_gt == 0:
            return {'tp': 0, 'fp': n_pred}
        
        tp = 0
        fp = 0
        matched_gt = set()
        
        # Sort predictions by score (assuming they're already sorted)
        for i in range(n_pred):
            # Find best matching GT
            best_iou = 0
            best_j = -1
            
            for j in range(n_gt):
                if j in matched_gt:
                    continue
                
                if pred_labels[i] == gt_labels[j] and ious[i, j] > best_iou:
                    best_iou = ious[i, j]
                    best_j = j
            
            if best_iou >= iou_thresh:
                tp += 1
                matched_gt.add(best_j)
            else:
                fp += 1
        
        return {'tp': tp, 'fp': fp}
    
    def _print_metrics(self, metrics: Dict):
        """Print computed metrics"""
        print("\n" + "="*80)
        print("METRICS")
        print("="*80)
        
        print(f"\nOverall:")
        print(f"  True Positives:  {metrics['tp']}")
        print(f"  False Positives: {metrics['fp']}")
        print(f"  False Negatives: {metrics['fn']}")
        print(f"  Precision: {metrics['precision']:.3f}")
        print(f"  Recall:    {metrics['recall']:.3f}")
        print(f"  F1 Score:  {metrics['f1']:.3f}")
        print(f"  Mean IoU:  {metrics['mean_iou']:.3f}")
        
        if metrics.get('per_class'):
            print(f"\nPer-class counts:")
            for cls_name, cls_metrics in metrics['per_class'].items():
                print(f"  {cls_name:<25} GT: {cls_metrics['gt_count']:>3}, Pred: {cls_metrics['pred_count']:>3}")
    
    def _visualize(
        self,
        detections: Dict,
        camera_imgs: Optional[torch.Tensor] = None,
        lidar_points: Optional[torch.Tensor] = None,
        gt_boxes: Optional[np.ndarray] = None,
        gt_labels: Optional[np.ndarray] = None,
        sample_token: Optional[str] = None
    ):
        """Create visualizations"""
        print("\n" + "="*80)
        print("GENERATING VISUALIZATIONS")
        print("="*80)
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 12))
        
        # 1. BEV view
        ax1 = fig.add_subplot(2, 3, 1)
        self._plot_bev(ax1, detections, gt_boxes, gt_labels)
        
        # 2. 3D view
        ax2 = fig.add_subplot(2, 3, 2, projection='3d')
        self._plot_3d(ax2, detections, lidar_points, gt_boxes, gt_labels)
        
        # 3. Front camera with projected boxes
        if camera_imgs is not None:
            ax3 = fig.add_subplot(2, 3, 3)
            self._plot_camera(ax3, camera_imgs, detections, 0)  # Front camera
        
        # 4. Heatmap (if available)
        if 'heatmap' in detections.get('raw_predictions', {}):
            ax4 = fig.add_subplot(2, 3, 4)
            self._plot_heatmap(ax4, detections['raw_predictions'])
        
        # 5. Detection scores
        ax5 = fig.add_subplot(2, 3, 5)
        self._plot_scores(ax5, detections)
        
        # 6. Class distribution
        ax6 = fig.add_subplot(2, 3, 6)
        self._plot_class_distribution(ax6, detections, gt_labels)
        
        plt.suptitle(
            f"3D Object Detection Results\n"
            f"Sample: {sample_token if sample_token else 'Unknown'} | "
            f"Detections: {len(detections['boxes'])} | "
            f"Score Threshold: {self.score_thresh}",
            fontsize=14,
            fontweight='bold'
        )
        
        plt.tight_layout()
        
        # Save
        if self.save_dir:
            save_path = self.save_dir / f"inference_{sample_token if sample_token else 'sample'}.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved visualization to {save_path}")
        
        # Show
        if self.show:
            plt.show()
        else:
            plt.close()
    
    def _plot_bev(
        self,
        ax: plt.Axes,
        detections: Dict,
        gt_boxes: Optional[np.ndarray] = None,
        gt_labels: Optional[np.ndarray] = None
    ):
        """Plot bird's eye view"""
        ax.set_title("Bird's Eye View", fontweight='bold')
        ax.set_xlabel("X (meters)")
        ax.set_ylabel("Y (meters)")
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # Set limits from point cloud range
        ax.set_xlim(self.pc_range[0], self.pc_range[3])
        ax.set_ylim(self.pc_range[1], self.pc_range[4])
        
        # Plot ground truth boxes (blue)
        if gt_boxes is not None and len(gt_boxes) > 0:
            for box, label in zip(gt_boxes, gt_labels):
                self._draw_box_bev(ax, box, 'blue', alpha=0.3, label='GT' if label == gt_labels[0] else "")
        
        # Plot predicted boxes (red)
        pred_boxes = detections['boxes']
        pred_labels = detections['labels']
        pred_scores = detections['scores']
        
        for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
            class_name = self.classes[label] if label < len(self.classes) else f"cls_{label}"
            color = self._get_box_color(class_name)
            self._draw_box_bev(ax, box, color, alpha=0.7, score=score, class_name=class_name)
        
        ax.legend(loc='upper right')
    
    def _plot_3d(
        self,
        ax: Axes3D,
        detections: Dict,
        lidar_points: Optional[torch.Tensor] = None,
        gt_boxes: Optional[np.ndarray] = None,
        gt_labels: Optional[np.ndarray] = None
    ):
        """Plot 3D view"""
        ax.set_title("3D View", fontweight='bold')
        ax.set_xlabel("X (meters)")
        ax.set_ylabel("Y (meters)")
        ax.set_zlabel("Z (meters)")
        
        # Plot LiDAR points (if available)
        if lidar_points is not None:
            points = lidar_points[0].cpu().numpy()  # (N, 5)
            # Subsample for visualization
            if len(points) > 5000:
                indices = np.random.choice(len(points), 5000, replace=False)
                points = points[indices]
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                      c=points[:, 3], cmap='gray', s=1, alpha=0.3)
        
        # Plot boxes
        pred_boxes = detections['boxes']
        pred_labels = detections['labels']
        
        for box, label in zip(pred_boxes, pred_labels):
            class_name = self.classes[label] if label < len(self.classes) else f"cls_{label}"
            color = self._get_box_color(class_name)
            self._draw_box_3d(ax, box, color, alpha=0.3)
        
        # Set limits
        ax.set_xlim(self.pc_range[0], self.pc_range[3])
        ax.set_ylim(self.pc_range[1], self.pc_range[4])
        ax.set_zlim(self.pc_range[2], self.pc_range[5])
    
    def _plot_camera(
        self,
        ax: plt.Axes,
        camera_imgs: torch.Tensor,
        detections: Dict,
        cam_idx: int = 0
    ):
        """Plot camera image with projected boxes"""
        ax.set_title(f"Camera View (CAM_{cam_idx})", fontweight='bold')
        ax.axis('off')
        
        # Get camera image
        img = camera_imgs[0, cam_idx].cpu().numpy()  # (3, H, W)
        img = np.transpose(img, (1, 2, 0))  # (H, W, 3)
        
        # Denormalize
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = img * std + mean
        img = np.clip(img, 0, 1)
        
        ax.imshow(img)
        
        # Project boxes (simplified - would need camera calibration for proper projection)
        # For now, just show we have the capability
        ax.text(
            0.5, 0.95,
            f"Detections: {len(detections['boxes'])}",
            transform=ax.transAxes,
            ha='center',
            va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            fontsize=10
        )
    
    def _plot_heatmap(self, ax: plt.Axes, predictions: Dict):
        """Plot detection heatmap"""
        ax.set_title("Detection Heatmap", fontweight='bold')
        ax.axis('off')
        
        heatmap = predictions['heatmap'][0].cpu().numpy()  # (num_classes, H, W)
        
        # Take max over classes
        heatmap_max = heatmap.max(axis=0)
        
        im = ax.imshow(heatmap_max, cmap='hot', interpolation='nearest')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    def _plot_scores(self, ax: plt.Axes, detections: Dict):
        """Plot detection scores"""
        ax.set_title("Detection Scores", fontweight='bold')
        ax.set_xlabel("Detection Index")
        ax.set_ylabel("Confidence Score")
        ax.grid(True, alpha=0.3)
        
        scores = detections['scores']
        
        if len(scores) > 0:
            ax.bar(range(len(scores)), scores, color='steelblue', alpha=0.7)
            ax.axhline(y=self.score_thresh, color='r', linestyle='--', 
                      label=f'Threshold ({self.score_thresh})')
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'No detections', ha='center', va='center', 
                   transform=ax.transAxes)
    
    def _plot_class_distribution(
        self,
        ax: plt.Axes,
        detections: Dict,
        gt_labels: Optional[np.ndarray] = None
    ):
        """Plot class distribution"""
        ax.set_title("Class Distribution", fontweight='bold')
        ax.set_xlabel("Class")
        ax.set_ylabel("Count")
        
        pred_labels = detections['labels']
        
        # Count predictions per class
        pred_counts = np.zeros(len(self.classes))
        for label in pred_labels:
            if label < len(self.classes):
                pred_counts[label] += 1
        
        # Count GT per class
        gt_counts = np.zeros(len(self.classes))
        if gt_labels is not None:
            for label in gt_labels:
                if label < len(self.classes):
                    gt_counts[label] += 1
        
        # Plot
        x = np.arange(len(self.classes))
        width = 0.35
        
        ax.bar(x - width/2, gt_counts, width, label='Ground Truth', alpha=0.7)
        ax.bar(x + width/2, pred_counts, width, label='Predictions', alpha=0.7)
        
        ax.set_xticks(x)
        ax.set_xticklabels(self.classes, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    def _draw_box_bev(
        self,
        ax: plt.Axes,
        box: np.ndarray,
        color: str,
        alpha: float = 0.5,
        score: Optional[float] = None,
        class_name: Optional[str] = None,
        label: Optional[str] = None
    ):
        """Draw 3D box in BEV"""
        x, y, z, w, l, h, yaw = box
        
        # Create rectangle
        rect = Rectangle(
            (x - w/2, y - l/2), w, l,
            angle=np.rad2deg(yaw),
            fill=False,
            edgecolor=color,
            linewidth=2,
            alpha=alpha,
            label=label
        )
        ax.add_patch(rect)
        
        # Add direction arrow
        arrow_len = l / 2
        dx = arrow_len * np.cos(yaw)
        dy = arrow_len * np.sin(yaw)
        ax.arrow(x, y, dx, dy, head_width=0.5, head_length=0.3, 
                fc=color, ec=color, alpha=alpha)
        
        # Add label
        if class_name and score is not None:
            ax.text(x, y, f"{class_name}\n{score:.2f}", 
                   ha='center', va='center', fontsize=8,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    def _draw_box_3d(self, ax: Axes3D, box: np.ndarray, color: str, alpha: float = 0.3):
        """Draw 3D box"""
        x, y, z, w, l, h, yaw = box
        
        # Create box corners
        corners = np.array([
            [-w/2, -l/2, -h/2],
            [w/2, -l/2, -h/2],
            [w/2, l/2, -h/2],
            [-w/2, l/2, -h/2],
            [-w/2, -l/2, h/2],
            [w/2, -l/2, h/2],
            [w/2, l/2, h/2],
            [-w/2, l/2, h/2]
        ])
        
        # Rotate
        rot_mat = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])
        corners = corners @ rot_mat.T
        
        # Translate
        corners += np.array([x, y, z])
        
        # Define faces
        faces = [
            [corners[0], corners[1], corners[2], corners[3]],  # bottom
            [corners[4], corners[5], corners[6], corners[7]],  # top
            [corners[0], corners[1], corners[5], corners[4]],  # front
            [corners[2], corners[3], corners[7], corners[6]],  # back
            [corners[0], corners[3], corners[7], corners[4]],  # left
            [corners[1], corners[2], corners[6], corners[5]]   # right
        ]
        
        # Plot
        poly3d = Poly3DCollection(faces, alpha=alpha, facecolor=color, edgecolor='black')
        ax.add_collection3d(poly3d)
    
    def _get_box_color(self, class_name: str) -> str:
        """Get color for class"""
        default_colors = {
            'car': 'green',
            'truck': 'orange',
            'trailer': 'yellow',
            'bus': 'magenta',
            'construction_vehicle': 'olive',
            'bicycle': 'cyan',
            'motorcycle': 'purple',
            'pedestrian': 'red',
            'traffic_cone': 'pink',
            'barrier': 'gray'
        }
        
        return self.box_colors.get(class_name, default_colors.get(class_name, 'blue'))


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def run_inference(
    model_path: str,
    config_path: str = 'config.yaml',
    data_root: str = './data/nuscenes',
    sample_idx: int = 0,
    split: str = 'val',
    device: str = 'cuda',
    show: bool = True,
    save_dir: Optional[str] = './inference_results'
):
    """
    Convenience function to run inference on a single sample
    
    Args:
        model_path: Path to model checkpoint
        config_path: Path to configuration file
        data_root: Root directory of dataset
        sample_idx: Index of sample to use
        split: Dataset split ('train', 'val', 'test')
        device: Device to use
        show: Whether to show visualizations
        save_dir: Directory to save results
    
    Returns:
        Dictionary with results
    """
    # Load dataset
    
    
    print("Loading dataset...")
    dataset = NuScenesDataset(data_root=data_root, split=split)
    sample = dataset[sample_idx]
    
    # Create inference engine
    engine = InferenceEngine(
        model_path=model_path,
        config_path=config_path,
        device=device,
        show=show,
        save_dir=save_dir
    )
    
    # Prepare inputs
    camera_imgs = sample['camera_imgs'].unsqueeze(0) if 'camera_imgs' in sample else None
    lidar_points = sample['lidar_points'].unsqueeze(0) if 'lidar_points' in sample else None
    radar_points = [r.unsqueeze(0) for r in sample['radar_points']] if 'radar_points' in sample else None
    
    gt_boxes = sample.get('gt_boxes')
    gt_labels = sample.get('gt_labels')
    sample_token = sample.get('token', f'sample_{sample_idx}')
    
    # Run inference
    results = engine.run_inference(
        camera_imgs=camera_imgs,
        lidar_points=lidar_points,
        radar_points=radar_points,
        gt_boxes=gt_boxes,
        gt_labels=gt_labels,
        sample_token=sample_token
    )
    
    return results


def batch_inference(
    model_path: str,
    config_path: str = 'config.yaml',
    data_root: str = './data/nuscenes',
    split: str = 'val',
    num_samples: int = 10,
    device: str = 'cuda',
    show: bool = False,
    save_dir: str = './inference_results'
):
    """
    Run inference on multiple samples
    
    Args:
        model_path: Path to model checkpoint
        config_path: Path to configuration file
        data_root: Root directory of dataset
        split: Dataset split
        num_samples: Number of samples to process
        device: Device to use
        show: Whether to show visualizations
        save_dir: Directory to save results
    
    Returns:
        List of results for each sample
    """
    
    
    # Load dataset
    dataset = NuScenesDataset(data_root=data_root, split=split)
    
    # Create inference engine
    engine = InferenceEngine(
        model_path=model_path,
        config_path=config_path,
        device=device,
        show=show,
        save_dir=save_dir
    )
    
    all_results = []
    
    for i in range(min(num_samples, len(dataset))):
        print(f"\n{'='*80}")
        print(f"Processing sample {i+1}/{num_samples}")
        print(f"{'='*80}")
        
        sample = dataset[i]
        
        # Prepare inputs
        camera_imgs = sample['camera_imgs'].unsqueeze(0) if 'camera_imgs' in sample else None
        lidar_points = sample['lidar_points'].unsqueeze(0) if 'lidar_points' in sample else None
        radar_points = [r.unsqueeze(0) for r in sample['radar_points']] if 'radar_points' in sample else None
        
        gt_boxes = sample.get('gt_boxes')
        gt_labels = sample.get('gt_labels')
        sample_token = sample.get('token', f'sample_{i}')
        
        # Run inference
        results = engine.run_inference(
            camera_imgs=camera_imgs,
            lidar_points=lidar_points,
            radar_points=radar_points,
            gt_boxes=gt_boxes,
            gt_labels=gt_labels,
            sample_token=sample_token
        )
        
        all_results.append(results)
    
    # Aggregate metrics
    print("\n" + "="*80)
    print("AGGREGATE METRICS")
    print("="*80)
    
    total_tp = sum(r['metrics']['tp'] for r in all_results if r['metrics'])
    total_fp = sum(r['metrics']['fp'] for r in all_results if r['metrics'])
    total_fn = sum(r['metrics']['fn'] for r in all_results if r['metrics'])
    
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\nOverall ({num_samples} samples):")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall:    {recall:.3f}")
    print(f"  F1 Score:  {f1:.3f}")
    
    return all_results


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='3D Detection Inference')
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='configs/base.yaml', help='Path to config file')
    parser.add_argument('--data-root', type=str, default='./data/nuscenes', help='Data root directory')
    parser.add_argument('--sample-idx', type=int, default=0, help='Sample index')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--device', type=str, default='cpu', choices=['cuda', 'cpu'])
    parser.add_argument('--no-show', action='store_true', help='Don\'t show visualizations')
    parser.add_argument('--save-dir', type=str, default='./inference_results', help='Save directory')
    parser.add_argument('--batch', type=int, default=None, help='Run batch inference on N samples')
    
    args = parser.parse_args()
    
    if args.batch:
        # Batch inference
        batch_inference(
            model_path=args.model,
            config_path=args.config,
            data_root=args.data_root,
            split=args.split,
            num_samples=args.batch,
            device=args.device,
            show=not args.no_show,
            save_dir=args.save_dir
        )
    else:
        # Single inference
        run_inference(
            model_path=args.model,
            config_path=args.config,
            data_root=args.data_root,
            sample_idx=args.sample_idx,
            split=args.split,
            device=args.device,
            show=not args.no_show,
            save_dir=args.save_dir
        )