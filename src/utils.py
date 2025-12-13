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
import sys
import torchvision.transforms as T

def compute_center_distance_matrix(pred_boxes: np.ndarray, gt_boxes: np.ndarray) -> np.ndarray:
    """
    Compute 2D center distance matrix between predictions and ground truth
    
    Args:
        pred_boxes: (N, 7) prediction boxes
        gt_boxes: (M, 7) ground truth boxes
    
    Returns:
        distance_matrix: (N, M) distances between centers
    """
    pred_centers = pred_boxes[:, :2]  # (N, 2) - x, y
    gt_centers = gt_boxes[:, :2]  # (M, 2) - x, y
    
    # Compute pairwise distances
    # distance[i, j] = ||pred_i - gt_j||
    distances = np.sqrt(
        np.sum((pred_centers[:, np.newaxis, :] - gt_centers[np.newaxis, :, :]) ** 2, axis=2)
    )
    
    return distances


def match_predictions_to_gt(
    distance_matrix: np.ndarray,
    pred_scores: np.ndarray,
    threshold: float = 2.0
) -> List[Tuple[int, int]]:
    """
    Match predictions to ground truth using greedy matching based on scores
    
    Args:
        distance_matrix: (N, M) distance matrix
        pred_scores: (N,) confidence scores
        threshold: Maximum distance for valid match
    
    Returns:
        List of (pred_idx, gt_idx) matches
    """
    N, M = distance_matrix.shape
    
    # Sort predictions by score (descending)
    sorted_indices = np.argsort(-pred_scores)
    
    matches = []
    matched_gt = set()
    
    for pred_idx in sorted_indices:
        # Find closest unmatched GT
        distances_to_gt = distance_matrix[pred_idx]
        
        # Mask out already matched GTs
        available_mask = np.ones(M, dtype=bool)
        for gt_idx in matched_gt:
            available_mask[gt_idx] = False
        
        if not available_mask.any():
            break
        
        # Get best match among available GTs
        available_distances = distances_to_gt.copy()
        available_distances[~available_mask] = np.inf
        
        best_gt_idx = np.argmin(available_distances)
        best_distance = available_distances[best_gt_idx]
        
        # Check if within threshold
        if best_distance <= threshold:
            matches.append((pred_idx, best_gt_idx))
            matched_gt.add(best_gt_idx)
    
    return matches


def angle_diff(angle1: float, angle2: float) -> float:
    """
    Compute smallest angle difference between two angles
    
    Args:
        angle1: First angle in radians
        angle2: Second angle in radians
    
    Returns:
        Smallest angle difference in radians
    """
    diff = angle1 - angle2
    diff = np.arctan2(np.sin(diff), np.cos(diff))
    return np.abs(diff)


def calculate_ap_for_class(
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    gt_boxes: np.ndarray,
    distance_matrix: np.ndarray,
    threshold: float = 2.0
) -> float:
    """
    Calculate Average Precision for a single class
    
    Args:
        pred_boxes: (N, 7) prediction boxes
        pred_scores: (N,) confidence scores
        gt_boxes: (M, 7) ground truth boxes
        distance_matrix: (N, M) distance matrix
        threshold: Distance threshold for TP
    
    Returns:
        Average Precision
    """
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return 0.0
    
    # Sort by confidence
    sorted_indices = np.argsort(-pred_scores)
    
    tp = np.zeros(len(pred_boxes))
    fp = np.zeros(len(pred_boxes))
    matched_gt = set()
    
    for i, pred_idx in enumerate(sorted_indices):
        # Find best matching GT
        distances = distance_matrix[pred_idx]
        
        # Get unmatched GTs
        available_mask = np.ones(len(gt_boxes), dtype=bool)
        for gt_idx in matched_gt:
            available_mask[gt_idx] = False
        
        if available_mask.any():
            available_distances = distances.copy()
            available_distances[~available_mask] = np.inf
            
            best_gt_idx = np.argmin(available_distances)
            best_distance = available_distances[best_gt_idx]
            
            if best_distance <= threshold:
                tp[i] = 1
                matched_gt.add(best_gt_idx)
            else:
                fp[i] = 1
        else:
            fp[i] = 1
    
    # Compute precision-recall curve
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    
    recalls = tp_cumsum / len(gt_boxes)
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)
    
    # Compute AP using 11-point interpolation
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        if np.sum(recalls >= t) == 0:
            p = 0
        else:
            p = np.max(precisions[recalls >= t])
        ap += p / 11.0
    
    return ap

def compute_metrics(
    predictions: List[Dict],
    ground_truths: List[Dict],
    num_classes: int = 10,
    iou_thresholds: List[float] = None,
    distance_thresholds: List[float] = None,
    modality_info: Dict[str, bool] = None
) -> Dict[str, float]:
    """
    Compute detection metrics (AP, mAP, NDS) with modality-aware evaluation
    
    Args:
        predictions: List of prediction dicts, each containing:
            - boxes: (N, 7) - [x, y, z, w, l, h, yaw]
            - scores: (N,) - confidence scores
            - labels: (N,) - class labels
            - velocities: (N, 2) - [vx, vy] (optional)
        ground_truths: List of ground truth dicts, each containing:
            - boxes: (M, 7) - [x, y, z, w, l, h, yaw]
            - labels: (M,) - class labels
            - velocities: (M, 2) - [vx, vy] (optional)
        num_classes: Number of detection classes
        iou_thresholds: IoU thresholds for AP calculation
        distance_thresholds: Distance thresholds for matching
        modality_info: Dict with modality usage info (for reporting)
    
    Returns:
        Dictionary of metrics:
            - mAP: mean Average Precision
            - NDS: NuScenes Detection Score
            - mATE: mean Average Translation Error
            - mASE: mean Average Scale Error
            - mAOE: mean Average Orientation Error
            - mAVE: mean Average Velocity Error
            - Per-class APs
            - Modality-specific metrics
    """
    if iou_thresholds is None:
        iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    
    if distance_thresholds is None:
        # NuScenes uses distance-based matching
        distance_thresholds = [0.5, 1.0, 2.0, 4.0]
    
    if modality_info is None:
        modality_info = {'use_camera': True, 'use_lidar': True, 'use_radar': True}
    
    # Class names (NuScenes 10 classes)
    class_names = [
        'car', 'truck', 'bus', 'trailer', 'construction_vehicle',
        'pedestrian', 'motorcycle', 'bicycle', 'traffic_cone', 'barrier'
    ]
    
    # Initialize metrics storage
    all_aps = {cls_id: [] for cls_id in range(num_classes)}
    translation_errors = []
    scale_errors = []
    orientation_errors = []
    velocity_errors = []
    
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    # Process each sample
    for pred_dict, gt_dict in zip(predictions, ground_truths):
        if pred_dict is None or len(pred_dict.get('boxes', [])) == 0:
            # No predictions
            if len(gt_dict.get('boxes', [])) > 0:
                false_negatives += len(gt_dict['boxes'])
            continue
        
        pred_boxes = pred_dict['boxes']  # (N, 7)
        pred_scores = pred_dict['scores']  # (N,)
        pred_labels = pred_dict['labels']  # (N,)
        
        gt_boxes = gt_dict['boxes']  # (M, 7)
        gt_labels = gt_dict['labels']  # (M,)
        
        # Filter out padding (-1 labels)
        if isinstance(gt_labels, torch.Tensor):
            valid_gt_mask = gt_labels >= 0
            gt_boxes = gt_boxes[valid_gt_mask]
            gt_labels = gt_labels[valid_gt_mask]
            if 'velocities' in gt_dict:
                gt_vels = gt_dict['velocities'][valid_gt_mask]
            else:
                gt_vels = None
        else:
            gt_vels = gt_dict.get('velocities', None)
        
        if len(gt_boxes) == 0:
            # No ground truth objects
            false_positives += len(pred_boxes)
            continue
        
        # Convert to numpy if needed
        if isinstance(pred_boxes, torch.Tensor):
            pred_boxes = pred_boxes.cpu().numpy()
            pred_scores = pred_scores.cpu().numpy()
            pred_labels = pred_labels.cpu().numpy()
        if isinstance(gt_boxes, torch.Tensor):
            gt_boxes = gt_boxes.cpu().numpy()
            gt_labels = gt_labels.cpu().numpy()
            if gt_vels is not None:
                gt_vels = gt_vels.cpu().numpy()
        
        pred_vels = pred_dict.get('velocities', None)
        if pred_vels is not None and isinstance(pred_vels, torch.Tensor):
            pred_vels = pred_vels.cpu().numpy()
        
        # Match predictions to ground truth per class
        for cls_id in range(num_classes):
            pred_mask = pred_labels == cls_id
            gt_mask = gt_labels == cls_id
            
            if not pred_mask.any() and not gt_mask.any():
                continue
            
            cls_pred_boxes = pred_boxes[pred_mask]
            cls_pred_scores = pred_scores[pred_mask]
            cls_gt_boxes = gt_boxes[gt_mask]
            
            if len(cls_gt_boxes) == 0:
                false_positives += len(cls_pred_boxes)
                continue
            
            if len(cls_pred_boxes) == 0:
                false_negatives += len(cls_gt_boxes)
                continue
            
            # Calculate IoU matrix or distance matrix
            if len(cls_pred_boxes) > 0 and len(cls_gt_boxes) > 0:
                # Use 2D center distance for matching (faster than 3D IoU)
                distance_matrix = compute_center_distance_matrix(
                    cls_pred_boxes, cls_gt_boxes
                )
                
                # Match predictions to GT using Hungarian algorithm
                matches = match_predictions_to_gt(
                    distance_matrix,
                    cls_pred_scores,
                    threshold=2.0  # 2 meter matching threshold
                )
                
                # Calculate metrics for matched pairs
                for pred_idx, gt_idx in matches:
                    pred_box = cls_pred_boxes[pred_idx]
                    gt_box = cls_gt_boxes[gt_idx]
                    
                    # Translation Error (ATE)
                    trans_error = np.linalg.norm(pred_box[:2] - gt_box[:2])
                    translation_errors.append(trans_error)
                    
                    # Scale Error (ASE)
                    pred_size = pred_box[3:6]  # [w, l, h]
                    gt_size = gt_box[3:6]
                    scale_error = np.mean(np.abs(pred_size - gt_size) / (gt_size + 1e-6))
                    scale_errors.append(scale_error)
                    
                    # Orientation Error (AOE)
                    pred_yaw = pred_box[6]
                    gt_yaw = gt_box[6]
                    orient_error = angle_diff(pred_yaw, gt_yaw)
                    orientation_errors.append(orient_error)
                    
                    # Velocity Error (AVE)
                    if pred_vels is not None and gt_vels is not None:
                        pred_vel = pred_vels[pred_mask][pred_idx]
                        gt_vel = gt_vels[gt_mask][gt_idx]
                        vel_error = np.linalg.norm(pred_vel - gt_vel)
                        velocity_errors.append(vel_error)
                    
                    true_positives += 1
                
                # Unmatched predictions are false positives
                matched_pred_indices = set([m[0] for m in matches])
                false_positives += len(cls_pred_boxes) - len(matched_pred_indices)
                
                # Unmatched GTs are false negatives
                matched_gt_indices = set([m[1] for m in matches])
                false_negatives += len(cls_gt_boxes) - len(matched_gt_indices)
                
                # Calculate AP for this class at different thresholds
                cls_ap = calculate_ap_for_class(
                    cls_pred_boxes, cls_pred_scores, cls_gt_boxes, distance_matrix
                )
                all_aps[cls_id].append(cls_ap)
    
    # Aggregate metrics
    metrics = {}
    
    # Calculate mAP (mean over all classes and samples)
    class_aps = {}
    valid_class_count = 0
    total_ap = 0.0
    
    for cls_id in range(num_classes):
        if len(all_aps[cls_id]) > 0:
            class_ap = np.mean(all_aps[cls_id])
            class_aps[f'AP_{class_names[cls_id]}'] = float(class_ap)
            total_ap += class_ap
            valid_class_count += 1
        else:
            class_aps[f'AP_{class_names[cls_id]}'] = 0.0
    
    mAP = total_ap / valid_class_count if valid_class_count > 0 else 0.0
    metrics['mAP'] = float(mAP)
    metrics.update(class_aps)
    
    # Calculate error metrics
    metrics['mATE'] = float(np.mean(translation_errors)) if translation_errors else 0.0
    metrics['mASE'] = float(np.mean(scale_errors)) if scale_errors else 0.0
    metrics['mAOE'] = float(np.mean(orientation_errors)) if orientation_errors else 0.0
    metrics['mAVE'] = float(np.mean(velocity_errors)) if velocity_errors else 0.0
    
    # Calculate NDS (NuScenes Detection Score)
    # NDS = 1/10 * (5*mAP + mATE + mASE + mAOE + mAVE + mAAE)
    # Normalized version
    nds_components = [
        5 * mAP,  # mAP weighted 5x
        1.0 - min(metrics['mATE'] / 4.0, 1.0),  # Lower is better, normalize
        1.0 - min(metrics['mASE'] / 1.0, 1.0),  # Lower is better
        1.0 - min(metrics['mAOE'] / np.pi, 1.0),  # Lower is better
        1.0 - min(metrics['mAVE'] / 4.0, 1.0),  # Lower is better
    ]
    metrics['NDS'] = float(np.mean(nds_components))
    
    # Precision, Recall, F1
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    metrics['precision'] = float(precision)
    metrics['recall'] = float(recall)
    metrics['f1'] = float(f1)
    
    # Add modality information to metrics
    if modality_info:
        modality_str = '+'.join([
            k.replace('use_', '') for k, v in modality_info.items() if v
        ])
        metrics['modality'] = modality_str
        metrics['num_modalities'] = sum(modality_info.values())
    
    # Detection counts
    metrics['true_positives'] = int(true_positives)
    metrics['false_positives'] = int(false_positives)
    metrics['false_negatives'] = int(false_negatives)
    
    return metrics

