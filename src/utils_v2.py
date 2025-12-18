import numpy as np
from typing import List, Dict, Tuple

# -------------------------------
# 1. Distance-based matching
# -------------------------------
def compute_center_distance_matrix(pred_boxes: np.ndarray, gt_boxes: np.ndarray) -> np.ndarray:
    pred_centers = pred_boxes[:, :2]
    gt_centers   = gt_boxes[:, :2]
    return np.sqrt(((pred_centers[:, None, :] - gt_centers[None, :, :]) ** 2).sum(axis=2))


def match_predictions_to_gt(distance_matrix: np.ndarray,
                            pred_scores: np.ndarray,
                            threshold: float = 2.0) -> List[Tuple[int, int]]:
    N, M = distance_matrix.shape
    sorted_indices = np.argsort(-pred_scores)  # high → low
    matches, matched_gt = [], set()

    for pred_idx in sorted_indices:
        distances = distance_matrix[pred_idx]

        # available GTs
        mask = np.ones(M, dtype=bool)
        mask[list(matched_gt)] = False
        if not mask.any():
            break

        candidate_dist = distances.copy()
        candidate_dist[~mask] = np.inf
        gt_idx = np.argmin(candidate_dist)
        if candidate_dist[gt_idx] <= threshold:
            matches.append((pred_idx, gt_idx))
            matched_gt.add(gt_idx)

    return matches


# -------------------------------
# 2. AP computation (11-point interpolated)
# -------------------------------
def calculate_ap(pred_boxes: np.ndarray,
                 pred_scores: np.ndarray,
                 gt_boxes: np.ndarray,
                 distance_matrix: np.ndarray,
                 threshold: float = 2.0) -> float:

    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return 0.0

    sorted_idx = np.argsort(-pred_scores)
    tp = np.zeros(len(pred_boxes))
    fp = np.zeros(len(pred_boxes))
    matched_gt = set()

    for i, pred_idx in enumerate(sorted_idx):
        distances = distance_matrix[pred_idx]

        # available GTs
        mask = np.ones(len(gt_boxes), dtype=bool)
        mask[list(matched_gt)] = False

        if mask.any():
            dist = distances.copy()
            dist[~mask] = np.inf
            best_gt = np.argmin(dist)
            if dist[best_gt] <= threshold:
                tp[i] = 1
                matched_gt.add(best_gt)
            else:
                fp[i] = 1
        else:
            fp[i] = 1

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)

    recalls = tp_cum / len(gt_boxes)
    precisions = tp_cum / (tp_cum + fp_cum + 1e-10)

    # 11-point AP
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        valid = precisions[recalls >= t]
        p = valid.max() if len(valid) > 0 else 0
        ap += p / 11.0

    return ap


# -------------------------------
# 3. Final Metric Computation: mAP + NDS
# -------------------------------
def compute_metrics(predictions: List[Dict],
                    ground_truths: List[Dict]
                    ) -> Dict[str, float]:

    class_names = [
        'car', 'truck', 'bus', 'trailer', 'construction_vehicle',
        'pedestrian', 'motorcycle', 'bicycle', 'traffic_cone', 'barrier'
    ]
    num_classes =10
    distance_threshold= 2.0
    aps_per_class = {c: [] for c in range(num_classes)}
    mATEs, mASEs, mAOEs = [], [], []

    for pred, gt in zip(predictions, ground_truths):

        pred_boxes = pred["boxes"]
        pred_scores = pred["scores"]
        pred_labels = pred["labels"]

        gt_boxes = gt["boxes"]
        gt_labels = gt["labels"]

        # Filter invalid GT
        if isinstance(gt_labels, np.ndarray):
            mask = gt_labels >= 0
            gt_boxes = gt_boxes[mask]
            gt_labels = gt_labels[mask]

        if len(gt_boxes) == 0 and len(pred_boxes) == 0:
            continue

        # Convert to numpy if tensors
        if not isinstance(pred_boxes, np.ndarray):
            pred_boxes = pred_boxes.cpu().numpy()
            pred_scores = pred_scores.cpu().numpy()
            pred_labels = pred_labels.cpu().numpy()
        if not isinstance(gt_boxes, np.ndarray):
            gt_boxes = gt_boxes.cpu().numpy()
            gt_labels = gt_labels.cpu().numpy()

        # Class-wise evaluation
        for cls in range(num_classes):
            p_mask = pred_labels == cls
            g_mask = gt_labels == cls

            cls_preds = pred_boxes[p_mask]
            cls_scores = pred_scores[p_mask]
            cls_gts = gt_boxes[g_mask]

            if len(cls_gts) == 0 and len(cls_preds) == 0:
                continue

            if len(cls_gts) == 0:
                aps_per_class[cls].append(0.0)
                continue

            if len(cls_preds) == 0:
                aps_per_class[cls].append(0.0)
                continue

            dist_mat = compute_center_distance_matrix(cls_preds, cls_gts)
            ap = calculate_ap(cls_preds, cls_scores, cls_gts, dist_mat, threshold=distance_threshold)
            aps_per_class[cls].append(ap)

            # For NDS components
            matches = match_predictions_to_gt(dist_mat, cls_scores, threshold=distance_threshold)
            for pi, gi in matches:
                pb, gb = cls_preds[pi], cls_gts[gi]

                # ATE (center distance)
                mATEs.append(np.linalg.norm(pb[:2] - gb[:2]))

                # ASE (scale difference)
                mASEs.append(np.mean(np.abs(pb[3:6] - gb[3:6]) / (gb[3:6] + 1e-6)))

                # AOE (rotation difference)
                ang = pb[6] - gb[6]
                ang = np.arctan2(np.sin(ang), np.cos(ang))
                mAOEs.append(abs(ang))

    # ---------------------------
    # Compute mAP
    # ---------------------------
    class_aps = []
    for cls in range(num_classes):
        if len(aps_per_class[cls]) > 0:
            class_aps.append(np.mean(aps_per_class[cls]))
        else:
            class_aps.append(0.0)

    mAP = float(np.mean(class_aps))

    # ---------------------------
    # Compute NDS
    # ---------------------------
    mATE = float(np.mean(mATEs)) if mATEs else 1.0
    mASE = float(np.mean(mASEs)) if mASEs else 1.0
    mAOE = float(np.mean(mAOEs)) if mAOEs else 1.0

    # Official NDS formula (simplified)
    NDS = np.mean([
        5 * mAP,
        1 - min(mATE / 4.0, 1.0),
        1 - min(mASE / 1.0, 1.0),
        1 - min(mAOE / np.pi, 1.0),
    ])

    return {
        "mAP": float(mAP),
        "NDS": float(NDS),
        "AP_per_class": {class_names[i]: float(class_aps[i]) for i in range(num_classes)},
    }


def save_and_print_metrics(metrics: dict, save_path: str = "metrics_output.txt"):
    """
    Pretty-print mAP, NDS, and per-class APs, and save them into a .txt file.
    """
    # --------------------------
    # Pretty-print to console
    # --------------------------
    print("\n===== Evaluation Metrics =====")
    print(f"mAP : {metrics['mAP']:.4f}")
    print(f"NDS : {metrics['NDS']:.4f}")
    print("\n--- AP Per Class ---")
    for cls_name, ap_val in metrics["AP_per_class"].items():
        print(f"{cls_name:20s}: {ap_val:.4f}")

    # --------------------------
    # Save to file
    # --------------------------
    with open(save_path, "w") as f:
        f.write("===== Evaluation Metrics =====\n")
        f.write(f"mAP : {metrics['mAP']:.4f}\n")
        f.write(f"NDS : {metrics['NDS']:.4f}\n")
        f.write("\n--- AP Per Class ---\n")
        for cls_name, ap_val in metrics["AP_per_class"].items():
            f.write(f"{cls_name:20s}: {ap_val:.4f}\n")

    print(f"\nMetrics saved to {save_path}")









