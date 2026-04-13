from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_auc_score


class Evaluator:
    """
    模块化分割评估器。

    支持：
    - 全局 overall 指标
    - 按类别 per_class 指标
    - 各类别宏平均 macro_avg 指标

    同时保留顶层全局指标键，兼容现有训练/验证代码。
    """

    def __init__(
        self,
        threshold: Optional[float] = 0.5,
        eps: float = 1e-7,
        default_group_name: str = "__default__",
    ) -> None:
        self.threshold = threshold if threshold is None else float(threshold)
        self.eps = float(eps)
        self.default_group_name = default_group_name
        self.global_group_name = "__all__"
        self.reset()

    def reset(self) -> None:
        self.img_gts_dict: Dict[str, List[int]] = {}
        self.img_preds_dict: Dict[str, List[float]] = {}
        self.px_gts_dict: Dict[str, List[np.ndarray]] = {}
        self.px_preds_dict: Dict[str, List[np.ndarray]] = {}
        self.seg_stats_dict: Dict[str, Dict[str, float]] = {}

    def _init_group(self, cls_name: str) -> None:
        if cls_name not in self.img_gts_dict:
            self.img_gts_dict[cls_name] = []
            self.img_preds_dict[cls_name] = []
            self.px_gts_dict[cls_name] = []
            self.px_preds_dict[cls_name] = []
            self.seg_stats_dict[cls_name] = {
                "num_samples": 0,
                "tp": 0.0,
                "fp": 0.0,
                "fn": 0.0,
            }

    @staticmethod
    def _to_numpy(x: Any) -> np.ndarray:
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    @staticmethod
    def _binary_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
        y_true = y_true.astype(np.int64).reshape(-1)
        y_score = y_score.astype(np.float64).reshape(-1)
        try:
            return float(roc_auc_score(y_true, y_score))
        except ValueError:
            return float("nan")

    def _binary_ap(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        y_true = y_true.astype(np.int64).reshape(-1)
        y_score = y_score.astype(np.float64).reshape(-1)
        try:
            return float(average_precision_score(y_true, y_score))
        except ValueError:
            return float("nan")
    
    def _binary_aupr(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        y_true = y_true.astype(np.int64).reshape(-1)
        y_score = y_score.astype(np.float64).reshape(-1)
        try:
            precision, recall, _ = precision_recall_curve(y_true, y_score)
            return float(auc(recall, precision))
        except ValueError:
            return float("nan")

    def _best_f1_threshold(self, y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
        y_true = y_true.astype(np.int64).reshape(-1)
        y_score = y_score.astype(np.float64).reshape(-1)

        pos = int((y_true == 1).sum())
        if pos == 0:
            return float("nan"), float("nan")

        order = np.argsort(-y_score)
        y_true_sorted = y_true[order]
        y_score_sorted = y_score[order]

        tp_cum = np.cumsum(y_true_sorted == 1)
        fp_cum = np.cumsum(y_true_sorted == 0)

        change_idx = np.where(np.diff(y_score_sorted) != 0)[0]
        idxs = np.concatenate([change_idx, np.array([len(y_score_sorted) - 1])])

        tp = tp_cum[idxs].astype(np.float64)
        fp = fp_cum[idxs].astype(np.float64)
        fn = float(pos) - tp

        f1 = 2.0 * tp / np.maximum(2.0 * tp + fp + fn, self.eps)
        best = int(np.argmax(f1))
        return float(f1[best]), float(y_score_sorted[idxs[best]])

    def _iou_at_threshold(self, y_true: np.ndarray, y_score: np.ndarray, thr: float) -> float:
        y_true = y_true.astype(np.int64).reshape(-1)
        y_pred = (y_score.reshape(-1) >= float(thr)).astype(np.int64)

        tp = float(((y_pred == 1) & (y_true == 1)).sum())
        fp = float(((y_pred == 1) & (y_true == 0)).sum())
        fn = float(((y_pred == 0) & (y_true == 1)).sum())
        return float((tp + self.eps) / (tp + fp + fn + self.eps))

    @staticmethod
    def _flatten_pixel_lists(gts: List[np.ndarray], preds: List[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        if len(gts) == 0 or len(preds) == 0:
            return np.array([], dtype=np.uint8), np.array([], dtype=np.float32)
        y_true = np.concatenate([x.reshape(-1) for x in gts], axis=0).astype(np.uint8)
        y_score = np.concatenate([x.reshape(-1) for x in preds], axis=0).astype(np.float32)
        return y_true, y_score

    @staticmethod
    def _sanitize_group_name(name: Optional[str], default_name: str) -> str:
        if name is None:
            return default_name
        text = str(name).strip()
        return text if text else default_name

    @staticmethod
    def _nanmean_dict(metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
        if not metrics_list:
            return {}

        keys = set()
        for metrics in metrics_list:
            keys.update(metrics.keys())

        out: Dict[str, float] = {}
        for key in sorted(keys):
            values = []
            for metrics in metrics_list:
                value = metrics.get(key, float("nan"))
                if isinstance(value, (int, float)):
                    values.append(float(value))
            if not values:
                out[key] = float("nan")
                continue
            arr = np.asarray(values, dtype=np.float64)
            if key == "num_samples":
                out[key] = float(np.sum(arr))
            else:
                out[key] = float(np.nanmean(arr)) if not np.all(np.isnan(arr)) else float("nan")
        return out

    def update(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        image_preds: Optional[Sequence[float] | np.ndarray | torch.Tensor] = None,
        image_gts: Optional[Sequence[int] | np.ndarray | torch.Tensor] = None,
        img_clsname: Optional[List[Optional[str]]] = None,
    ) -> None:
        preds_np = self._to_numpy(preds).astype(np.float32)
        targets_np = self._to_numpy(targets).astype(np.float32)

        if preds_np.ndim == 3:
            preds_np = preds_np[:, None, :, :]
        if targets_np.ndim == 3:
            targets_np = targets_np[:, None, :, :]

        if preds_np.shape[0] != targets_np.shape[0]:
            raise ValueError(f"Batch size mismatch: preds={preds_np.shape}, targets={targets_np.shape}")

        if preds_np.shape[1] != 1:
            raise ValueError(f"Expected preds with shape [B,1,H,W], got {preds_np.shape}")

        bsz = preds_np.shape[0]
        anomaly_probs = preds_np[:, 0]
        mask_np = (targets_np > 0.5).astype(np.uint8)

        if image_gts is not None:
            img_gt_batch = self._to_numpy(image_gts).reshape(-1)
            if len(img_gt_batch) != bsz:
                raise ValueError(f"image_gts length {len(img_gt_batch)} != batch size {bsz}")
            img_gt_batch = (img_gt_batch > 0).astype(np.int64)
        else:
            img_gt_batch = (mask_np.reshape(bsz, -1).max(axis=1) > 0).astype(np.int64)

        if image_preds is not None:
            img_pred_batch = self._to_numpy(image_preds).reshape(-1).astype(np.float32)
            if len(img_pred_batch) != bsz:
                raise ValueError(f"image_preds length {len(img_pred_batch)} != batch size {bsz}")
        else:
            img_pred_batch = anomaly_probs.reshape(bsz, -1).max(axis=1).astype(np.float32)

        if img_clsname is None or len(img_clsname) != bsz:
            group_names = [self.default_group_name] * bsz
        else:
            group_names = [self._sanitize_group_name(x, self.default_group_name) for x in img_clsname]

        for i in range(bsz):
            cls_name = group_names[i]
            self._init_group(cls_name)

            self.img_gts_dict[cls_name].append(int(img_gt_batch[i]))
            self.img_preds_dict[cls_name].append(float(img_pred_batch[i]))

            px_gt = mask_np[i, 0].astype(np.uint8)
            px_pred = anomaly_probs[i].astype(np.float32)
            self.px_gts_dict[cls_name].append(px_gt)
            self.px_preds_dict[cls_name].append(px_pred)

            seg_stats = self.seg_stats_dict[cls_name]
            seg_stats["num_samples"] += 1

            if self.threshold is not None:
                px_bin = (px_pred >= float(self.threshold)).astype(np.uint8)
                tp = float(((px_bin == 1) & (px_gt == 1)).sum())
                fp = float(((px_bin == 1) & (px_gt == 0)).sum())
                fn = float(((px_bin == 0) & (px_gt == 1)).sum())
                seg_stats["tp"] += tp
                seg_stats["fp"] += fp
                seg_stats["fn"] += fn

    def get_groups(self) -> List[str]:
        return list(self.img_gts_dict.keys())

    def get_group_data(self, group_name: str) -> Dict[str, Any]:
        if group_name not in self.img_gts_dict:
            raise ValueError(f"Group '{group_name}' not found. Available groups: {self.get_groups()}")
        return {
            "img_gts": self.img_gts_dict[group_name],
            "img_preds": self.img_preds_dict[group_name],
            "px_gts": self.px_gts_dict[group_name],
            "px_preds": self.px_preds_dict[group_name],
            "seg_stats": self.seg_stats_dict[group_name],
        }

    def _finalize_group_metrics(self, group_name: str) -> Dict[str, float]:
        data = self.get_group_data(group_name)
        seg_stats = data["seg_stats"]

        n = max(int(seg_stats["num_samples"]), 1)
        out: Dict[str, float] = {
            "num_samples": float(n),
        }

        if self.threshold is not None:
            tp = float(seg_stats["tp"])
            fp = float(seg_stats["fp"])
            fn = float(seg_stats["fn"])
            out[f"f1@{self.threshold}"] = float((2.0 * tp + self.eps) / (2.0 * tp + fp + fn + self.eps))
            out[f"iou@{self.threshold}"] = float((tp + self.eps) / (tp + fp + fn + self.eps))
        else:
            out["f1"] = float("nan")
            out["iou"] = float("nan")

        img_gts = np.asarray(data["img_gts"], dtype=np.int64)
        img_preds = np.asarray(data["img_preds"], dtype=np.float32)
        if img_gts.size > 0 and img_preds.size > 0:
            out["image_auroc"] = self._binary_auroc(img_gts, img_preds)
            out["image_aupr"] = self._binary_aupr(img_gts, img_preds)
            out["image_ap"] = self._binary_ap(img_gts, img_preds)
            image_best_f1, image_best_thr = self._best_f1_threshold(img_gts, img_preds)
            out["image_f1"] = image_best_f1
            out["image_f1_threshold"] = image_best_thr
        else:
            out["image_auroc"] = float("nan")
            out["image_aupr"] = float("nan")
            out["image_ap"] = float("nan")
            out["image_f1"] = float("nan")
            out["image_f1_threshold"] = float("nan")

        px_true, px_score = self._flatten_pixel_lists(data["px_gts"], data["px_preds"])
        if px_true.size > 0:
            out["pixel_auroc"] = self._binary_auroc(px_true, px_score)
            out["pixel_aupr"] = self._binary_aupr(px_true, px_score)
            out["pixel_ap"] = self._binary_ap(px_true, px_score)
            best_f1, best_thr = self._best_f1_threshold(px_true, px_score)
            out["pixel_f1"] = best_f1
            out["pixel_f1_threshold"] = best_thr

            if np.isfinite(best_thr):
                out["iou"] = self._iou_at_threshold(px_true, px_score, best_thr)
        else:
            out["pixel_auroc"] = float("nan")
            out["pixel_aupr"] = float("nan")
            out["pixel_ap"] = float("nan")
            out["pixel_f1"] = float("nan")
            out["pixel_f1_threshold"] = float("nan")

        return out

    def compute_group(self, group_name: str) -> Dict[str, float]:
        return self._finalize_group_metrics(group_name)

    def compute_all_groups(self) -> Dict[str, Dict[str, float]]:
        return {g: self._finalize_group_metrics(g) for g in self.get_groups() if g != self.global_group_name}

    def _build_global_group(self) -> None:
        self._init_group(self.global_group_name)
        self.img_gts_dict[self.global_group_name] = []
        self.img_preds_dict[self.global_group_name] = []
        self.px_gts_dict[self.global_group_name] = []
        self.px_preds_dict[self.global_group_name] = []
        self.seg_stats_dict[self.global_group_name] = {
            "num_samples": 0,
            "tp": 0.0,
            "fp": 0.0,
            "fn": 0.0,
        }

        for g in self.get_groups():
            if g == self.global_group_name:
                continue
            self.img_gts_dict[self.global_group_name].extend(self.img_gts_dict[g])
            self.img_preds_dict[self.global_group_name].extend(self.img_preds_dict[g])
            self.px_gts_dict[self.global_group_name].extend(self.px_gts_dict[g])
            self.px_preds_dict[self.global_group_name].extend(self.px_preds_dict[g])

            s = self.seg_stats_dict[g]
            gs = self.seg_stats_dict[self.global_group_name]
            gs["num_samples"] += int(s["num_samples"])
            gs["tp"] += float(s["tp"])
            gs["fp"] += float(s["fp"])
            gs["fn"] += float(s["fn"])

    def compute(self) -> Dict[str, Any]:
        print(f"[Evaluator] Computing grouped metrics...")
        groups = self.compute_all_groups()
        num_classes = len(groups)

        if num_classes == 0:
            print("[Evaluator] No valid class groups found; returning empty metrics.")
            overall: Dict[str, float] = {}
            per_class = groups
        elif num_classes == 1:
            class_name = next(iter(groups.keys()))
            print(f"[Evaluator] Single-class evaluation detected: {class_name}. Reusing class metrics for overall and macro_avg.")
            overall = dict(next(iter(groups.values())))
            per_class = groups
        else:
            print("[Evaluator] Computing macro average across classes...")
            overall = self._nanmean_dict(list(groups.values()))
            per_class = groups

        out: Dict[str, Any] = dict(overall)
        out["average"] = overall
        out["per_class"] = per_class
        out["num_classes"] = float(num_classes)
        return out
