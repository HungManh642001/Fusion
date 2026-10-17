from typing import List, Dict, Tuple
import numpy as np

class PipelineEvaluator:
    """
    Computes standard Multi-Object Tracking metrics (MOTA, IDF1, IDSw) and
    Multi-Source Trajectory Fusion metrics (TPF, PPF, FPF, FNF) as defined in
    Sensors 2025, 25, 2179 (Section 4, Tables 3, 4, 5).
    """
    def __init__(self):
        pass

    def evaluate_tracking(self, ground_truth: Dict[int, List[Dict]], tracking_results: Dict[int, List[Dict]]) -> Dict[str, float]:
        """
        Computes MOTA and IDF1 across all frames.
        """
        total_gt = 0
        total_fn = 0
        total_fp = 0
        total_idsw = 0
        
        # Track mapping for ID consistency
        gt_to_pred = {}
        matched_gt_instances = 0
        matched_pred_instances = 0

        all_frames = sorted(set(list(ground_truth.keys()) + list(tracking_results.keys())))

        for f in all_frames:
            gts = ground_truth.get(f, [])
            preds = tracking_results.get(f, [])
            total_gt += len(gts)

            # Match GT and Preds by IoU or track overlap
            matched_pairs = []
            unmatched_gts = list(range(len(gts)))
            unmatched_preds = list(range(len(preds)))

            for i, gt in enumerate(gts):
                for j, pred in enumerate(preds):
                    # compute overlap
                    g_box = gt['bbox']
                    p_box = pred['bbox']
                    # calculate IoU
                    xx1 = max(g_box[0], p_box[0])
                    yy1 = max(g_box[1], p_box[1])
                    xx2 = min(g_box[0] + g_box[2], p_box[0] + p_box[2])
                    yy2 = min(g_box[1] + g_box[3], p_box[1] + p_box[3])
                    w = max(0.0, xx2 - xx1)
                    h = max(0.0, yy2 - yy1)
                    inter = w * h
                    union = g_box[2]*g_box[3] + p_box[2]*p_box[3] - inter
                    iou = inter / max(union, 1e-6)
                    if iou > 0.2:
                        matched_pairs.append((i, j, iou))

            # Greedy assignment
            matched_pairs.sort(key=lambda x: x[2], reverse=True)
            assigned_g = set()
            assigned_p = set()
            for i, j, iou in matched_pairs:
                if i not in assigned_g and j not in assigned_p:
                    assigned_g.add(i)
                    assigned_p.add(j)
                    gt_id = gts[i]['track_id']
                    pred_id = preds[j]['track_id']
                    if gt_id in gt_to_pred and gt_to_pred[gt_id] != pred_id:
                        total_idsw += 1
                    gt_to_pred[gt_id] = pred_id
                    matched_gt_instances += 1
                    matched_pred_instances += 1

            total_fn += (len(gts) - len(assigned_g))
            total_fp += (len(preds) - len(assigned_p))

        total_gt = max(total_gt, 1)
        mota = max(0.0, 1.0 - (total_fn + total_fp + total_idsw) / total_gt)
        idp = matched_pred_instances / max(matched_pred_instances + total_fp, 1)
        idr = matched_gt_instances / max(total_gt, 1)
        idf1 = (2.0 * idp * idr) / max(idp + idr, 1e-6)

        return {
            'MOTA': round(float(mota), 3),
            'IDP': round(float(idp), 3),
            'IDR': round(float(idr), 3),
            'IDF1': round(float(idf1), 3),
            'IDSw': int(total_idsw),
            'TotalGT': int(total_gt),
            'TotalFN': int(total_fn),
            'TotalFP': int(total_fp)
        }

    def evaluate_fusion(self, fused_results: List[Dict]) -> Dict[str, float]:
        """
        Computes TPF, PPF, FPF rates.
        """
        total = max(len(fused_results), 1)
        tpf = sum(1 for f in fused_results if f.get('fusion_type') == 'tri')
        ppf = sum(1 for f in fused_results if 'dual' in f.get('fusion_type', ''))
        single = sum(1 for f in fused_results if 'single' in f.get('fusion_type', ''))

        return {
            'TPF': round(tpf / total, 3),
            'PPF': round(ppf / total, 3),
            'SingleModality': round(single / total, 3),
            'TotalFusedInstances': total
        }
