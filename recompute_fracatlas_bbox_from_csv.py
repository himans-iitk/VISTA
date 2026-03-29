#!/usr/bin/env python3
"""Recompute FracAtlas bbox metrics from detailed_results.csv using normalized-corner parsing.

Uses image size from each row's image file and bbox_text; updates pred_boxes (pixel xyxy),
pred_boxes_norm, best_iou, best_gt_overlap, bbox_correct_50, then summary_metrics.csv.
"""
from __future__ import annotations

import ast
import json
import os
import re
import sys
from typing import List, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score


def _vals_look_normalized(vals: Tuple[float, float, float, float], eps: float = 1e-3) -> bool:
    return all(-eps <= v <= 1.0 + eps for v in vals)


def parse_bboxes_from_text(
    text: str, image_wh: Tuple[int, int]
) -> Tuple[List[List[float]], List[List[float]]]:
    W, H = image_wh
    if not text or W <= 0 or H <= 0:
        return [], []
    stripped = str(text).strip()
    if re.match(r"^none\s*\.?\s*$", stripped, re.I):
        return [], []

    work = str(text)
    for sep in ("---fallback_no_aug---", "---retry_strict---"):
        if sep in work:
            work = work.split(sep)[-1]

    nums = re.findall(r"-?\d+(?:\.\d+)?(?:e[-+]?\d+)?", work, re.I)
    vals = [float(x) for x in nums]

    def _quad_to_norm(a: float, b: float, c: float, d: float):
        if _vals_look_normalized((a, b, c, d)):
            x1n, y1n, x2n, y2n = a, b, c, d
            if x2n <= x1n or y2n <= y1n:
                wn, hn = c, d
                x2n, y2n = a + wn, b + hn
            x1n = max(0.0, min(1.0, x1n))
            y1n = max(0.0, min(1.0, y1n))
            x2n = max(0.0, min(1.0, x2n))
            y2n = max(0.0, min(1.0, y2n))
            if x2n > x1n and y2n > y1n:
                return [x1n, y1n, x2n, y2n]
            return None
        x1p, y1p, x2p, y2p = a, b, c, d
        if x2p <= x1p or y2p <= y1p:
            return None
        return [x1p / W, y1p / H, x2p / W, y2p / H]

    last_norm = None
    for i in range(0, len(vals) - 3):
        qn = _quad_to_norm(vals[i], vals[i + 1], vals[i + 2], vals[i + 3])
        if qn is not None:
            last_norm = qn

    norm_boxes: List[List[float]] = [last_norm] if last_norm is not None else []
    pixel_boxes = [[bn[0] * W, bn[1] * H, bn[2] * W, bn[3] * H] for bn in norm_boxes]
    return norm_boxes, pixel_boxes


def box_area(b: List[float]) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def intersection_area(a: List[float], b: List[float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return (x2 - x1) * (y2 - y1)


def iou(a: List[float], b: List[float]) -> float:
    inter = intersection_area(a, b)
    union = box_area(a) + box_area(b) - inter
    return inter / union if union > 0 else 0.0


def gt_overlap_ratio(pred: List[float], gt: List[float]) -> float:
    inter = intersection_area(pred, gt)
    g = box_area(gt)
    return inter / g if g > 0 else 0.0


def parse_list_cell(x) -> list:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    if isinstance(x, list):
        return x
    s = str(x).strip()
    if not s or s == "[]":
        return []
    return ast.literal_eval(s)


def recompute_row(row: pd.Series) -> dict:
    path = row["image_path"]
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(path)
    with Image.open(path) as im:
        W, H = im.size

    gt_boxes = parse_list_cell(row.get("gt_boxes"))
    bbox_text = row.get("bbox_text")
    if pd.isna(bbox_text):
        bbox_text = ""

    pred_norm, pred_pixel = parse_bboxes_from_text(str(bbox_text), (W, H))

    best_iou = 0.0
    best_gt_overlap = 0.0
    if len(gt_boxes) > 0 and len(pred_norm) > 0:
        gt_norm = [[b[0] / W, b[1] / H, b[2] / W, b[3] / H] for b in gt_boxes]
        for pb in pred_norm:
            for gb in gt_norm:
                best_iou = max(best_iou, iou(pb, gb))
                best_gt_overlap = max(best_gt_overlap, gt_overlap_ratio(pb, gb))

    bbox_correct_50 = int(best_gt_overlap >= 0.5)
    return {
        "pred_boxes": pred_pixel,
        "pred_boxes_norm": pred_norm,
        "best_iou": float(best_iou),
        "best_gt_overlap": float(best_gt_overlap),
        "bbox_correct_50": bbox_correct_50,
    }


def build_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    summary_rows = []
    for variant_name, g in results_df.groupby("variant"):
        y_true = g["gt_label"].astype(int).values
        y_pred = g["pred_label"].astype(int).values
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        gf = g[g["gt_label"] == 1].copy()
        loc_hit_rate_50 = gf["bbox_correct_50"].mean() if len(gf) else np.nan
        gf_pred_pos = gf[gf["pred_label"] == 1]
        loc_hit_rate_50_pred_pos = gf_pred_pos["bbox_correct_50"].mean() if len(gf_pred_pos) else np.nan
        gf_pred_neg = gf[gf["pred_label"] == 0]
        loc_hit_rate_50_pred_neg = gf_pred_neg["bbox_correct_50"].mean() if len(gf_pred_neg) else np.nan
        summary_rows.append(
            {
                "variant": variant_name,
                "n_samples": len(g),
                "n_gt_fractured": int((g["gt_label"] == 1).sum()),
                "accuracy": acc,
                "f1": f1,
                "loc_hit_rate_50_on_gt_fractured": loc_hit_rate_50,
                "loc_hit_rate_50_when_pred_fracture": loc_hit_rate_50_pred_pos,
                "loc_hit_rate_50_when_pred_non_fracture": loc_hit_rate_50_pred_neg,
            }
        )
    return pd.DataFrame(summary_rows).sort_values("variant").reset_index(drop=True)


def main() -> int:
    default = os.path.join(os.path.dirname(__file__), "exp_results", "fracatlas_vista_benchmark", "detailed_results.csv")
    csv_path = sys.argv[1] if len(sys.argv) > 1 else default
    out_dir = os.path.dirname(os.path.abspath(csv_path))

    df = pd.read_csv(csv_path)
    updates = []
    for _, row in df.iterrows():
        updates.append(recompute_row(row))
    up = pd.DataFrame(updates)
    for c in up.columns:
        df[c] = up[c]

    summary_df = build_summary(df)

    summary_path = os.path.join(out_dir, "summary_metrics.csv")
    json_path = os.path.join(out_dir, "detailed_results.jsonl")

    summary_df.to_csv(summary_path, index=False)
    df.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            rec = row.to_dict()
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")

    print("Updated:", csv_path)
    print("Updated:", summary_path)
    print("Updated:", json_path)
    print(summary_df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
