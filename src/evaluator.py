"""Mô-đun đánh giá định lượng hiệu năng bám bắt đa mục tiêu và hợp nhất quỹ đạo.

Tính toán các chỉ số tiêu chuẩn theo Sensors 2025, 25, 2179:
- Bám bắt: MOTA, IDF1, IDP, IDR, IDSw (Bảng 4).
- Hợp nhất: TPF (True Positive Fusion), PPF (Part Positive Fusion), Single (Bảng 5).
"""

import logging
from collections.abc import Sequence

logger = logging.getLogger(__name__)


class PipelineEvaluator:
    """Bộ đánh giá định lượng các chỉ số chất lượng bám bắt và hợp nhất quỹ đạo."""

    def evaluate_tracking(
        self,
        ground_truth: dict[int, list[dict[str, object]]],
        tracking_results: dict[int, list[dict[str, object]]],
    ) -> dict[str, float]:
        """Tính toán MOTA, IDF1, IDP, IDR và ID Switches trên toàn bộ khung hình.

        Args:
            ground_truth: Tập nhãn thực tế theo khung hình {frame_idx: [{...}]}.
            tracking_results: Kết quả thuật toán bám bắt theo khung hình.

        Returns:
            Từ điển chứa các chỉ số đo lường định lượng MOTA, IDF1, IDP, IDR, IDSw.
        """
        total_gt = 0
        total_fn = 0
        total_fp = 0
        total_idsw = 0
        matched_gt_instances = 0
        matched_pred_instances = 0
        gt_to_pred: dict[int, int] = {}

        all_frames = sorted(set(ground_truth.keys()) | set(tracking_results.keys()))

        for f in all_frames:
            gts = ground_truth.get(f, [])
            preds = tracking_results.get(f, [])
            total_gt += len(gts)

            fn_cnt, fp_cnt, idsw_cnt, m_gt, m_pred = self._evaluate_frame(
                gts, preds, gt_to_pred
            )
            total_fn += fn_cnt
            total_fp += fp_cnt
            total_idsw += idsw_cnt
            matched_gt_instances += m_gt
            matched_pred_instances += m_pred

        total_gt_safe = max(total_gt, 1)
        mota = max(0.0, 1.0 - (total_fn + total_fp + total_idsw) / total_gt_safe)
        idp = matched_pred_instances / max(matched_pred_instances + total_fp, 1)
        idr = matched_gt_instances / total_gt_safe
        idf1 = (2.0 * idp * idr) / max(idp + idr, 1e-6)

        return {
            "MOTA": round(float(mota), 3),
            "IDP": round(float(idp), 3),
            "IDR": round(float(idr), 3),
            "IDF1": round(float(idf1), 3),
            "IDSw": float(total_idsw),
            "TotalGT": float(total_gt),
            "TotalFN": float(total_fn),
            "TotalFP": float(total_fp),
        }

    def _evaluate_frame(
        self,
        gts: list[dict[str, object]],
        preds: list[dict[str, object]],
        gt_to_pred: dict[int, int],
    ) -> tuple[int, int, int, int, int]:
        """Đánh giá chi tiết cho từng khung hình độc lập.

        Args:
            gts: Danh sách đối tượng nhãn chuẩn trong khung hình.
            preds: Danh sách đối tượng dự đoán trong khung hình.
            gt_to_pred: Bảng tra cứu đối chiếu nhãn chuẩn sang nhãn dự đoán.

        Returns:
            Bộ 5 giá trị: (số FN, số FP, số IDSw, số GT khớp, số Pred khớp).
        """
        matched_pairs = self._find_iou_pairs(gts, preds)
        matched_pairs.sort(key=lambda x: x[2], reverse=True)

        assigned_g: set[int] = set()
        assigned_p: set[int] = set()
        idsw_count = 0
        matched_count = 0

        for i, j, _ in matched_pairs:
            if i in assigned_g or j in assigned_p:
                continue
            assigned_g.add(i)
            assigned_p.add(j)
            gt_id = int(gts[i]["track_id"])  # type: ignore[arg-type]
            pred_id = int(preds[j]["track_id"])  # type: ignore[arg-type]

            if gt_id in gt_to_pred and gt_to_pred[gt_id] != pred_id:
                idsw_count += 1
            gt_to_pred[gt_id] = pred_id
            matched_count += 1

        fn_count = len(gts) - len(assigned_g)
        fp_count = len(preds) - len(assigned_p)
        return fn_count, fp_count, idsw_count, matched_count, matched_count

    @staticmethod
    def _find_iou_pairs(
        gts: list[dict[str, object]],
        preds: list[dict[str, object]],
        *,
        iou_thresh: float = 0.2,
    ) -> list[tuple[int, int, float]]:
        """Tìm các cặp nhãn chuẩn và dự đoán có chỉ số IoU vượt ngưỡng.

        Args:
            gts: Danh sách nhãn thực tế.
            preds: Danh sách dự đoán.
            iou_thresh: Ngưỡng IoU tối thiểu.

        Returns:
            Danh sách tuple (chỉ số gt, chỉ số pred, giá trị iou).
        """
        pairs: list[tuple[int, int, float]] = []
        for i, gt in enumerate(gts):
            g_box: Sequence[float] = gt["bbox"]  # type: ignore[assignment]
            for j, pred in enumerate(preds):
                p_box: Sequence[float] = pred["bbox"]  # type: ignore[assignment]
                xx1 = max(float(g_box[0]), float(p_box[0]))
                yy1 = max(float(g_box[1]), float(p_box[1]))
                xx2 = min(
                    float(g_box[0]) + float(g_box[2]),
                    float(p_box[0]) + float(p_box[2]),
                )
                yy2 = min(
                    float(g_box[1]) + float(p_box[3]),
                    float(p_box[1]) + float(p_box[3]),
                )
                w = max(0.0, xx2 - xx1)
                h = max(0.0, yy2 - yy1)
                inter = w * h
                union = (
                    float(g_box[2]) * float(g_box[3])
                    + float(p_box[2]) * float(p_box[3])
                    - inter
                )
                iou = inter / max(union, 1e-6)
                if iou > iou_thresh:
                    pairs.append((i, j, iou))
        return pairs

    @staticmethod
    def evaluate_fusion(
        fused_results: list[dict[str, object]],
    ) -> dict[str, float]:
        """Tính toán tỷ lệ hợp nhất 3 nguồn TPF và 2 nguồn PPF.

        Args:
            fused_results: Danh sách các thể hiện quỹ đạo hợp nhất thu thập được.

        Returns:
            Từ điển kết quả tỷ lệ hợp nhất đa nguồn.
        """
        total = max(len(fused_results), 1)
        tpf = sum(1 for f in fused_results if f.get("fusion_type") == "tri")
        ppf = sum(1 for f in fused_results if "dual" in str(f.get("fusion_type", "")))
        single = sum(
            1 for f in fused_results if "single" in str(f.get("fusion_type", ""))
        )

        return {
            "TPF": round(tpf / float(total), 3),
            "PPF": round(ppf / float(total), 3),
            "SingleModality": round(single / float(total), 3),
            "TotalFusedInstances": float(total),
        }
