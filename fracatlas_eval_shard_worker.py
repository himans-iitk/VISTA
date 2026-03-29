#!/usr/bin/env python3
"""Run one shard of FracAtlas eval on a single physical GPU (subprocess)."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


def _bump_shared_image_progress(path: str) -> int:
    """Increment global completed-image count (file lock; safe across GPU subprocesses)."""
    import fcntl

    with open(path, "r+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            raw = (f.read() or "0").strip()
            n = int(raw) + 1
        except ValueError:
            n = 1
        f.seek(0)
        f.truncate()
        f.write(str(n) + "\n")
        f.flush()
        fcntl.flock(f, fcntl.LOCK_UN)
    return n


def _parse_body_region(text: str) -> str:
    allowed = (
        "shoulder",
        "ankle",
        "wrist",
        "elbow",
        "spine",
        "pelvis",
        "chest",
        "skull",
        "knee",
        "hip",
        "hand",
        "foot",
        "leg",
        "arm",
        "other",
    )
    t = (text or "").lower()
    for w in sorted(set(allowed), key=len, reverse=True):
        if re.search(rf"\b{re.escape(w)}\b", t):
            return w
    return "other"


def _vals_look_normalized(vals: Tuple[float, float, float, float], eps: float = 1e-3) -> bool:
    return all(-eps <= v <= 1.0 + eps for v in vals)


def _parse_pred_label(text: str) -> int:
    t = (text or "").lower()
    if re.search(r"\b(no fracture|non[- ]?fracture|not fractured|normal)\b", t):
        return 0
    if re.search(r"\b(fracture|fractured)\b", t):
        return 1
    return 0


def _parse_bboxes_from_text(
    text: str, image_wh: Tuple[int, int]
) -> Tuple[List[List[float]], List[List[float]]]:
    W, H = image_wh
    if not text or W <= 0 or H <= 0:
        return [], []
    stripped = text.strip()
    if re.match(r"^none\s*\.?\s*$", stripped, re.I):
        return [], []

    work = text
    for sep in ("---fallback_no_aug---", "---retry_strict---"):
        if sep in work:
            work = work.split(sep)[-1]

    nums = re.findall(r"-?\d+(?:\.\d+)?(?:e[-+]?\d+)?", work, re.I)
    vals = [float(x) for x in nums]

    def _quad_to_norm(a: float, b: float, c: float, d: float) -> Optional[List[float]]:
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

    last_norm: Optional[List[float]] = None
    for i in range(0, len(vals) - 3):
        qn = _quad_to_norm(vals[i], vals[i + 1], vals[i + 2], vals[i + 3])
        if qn is not None:
            last_norm = qn

    norm_boxes: List[List[float]] = [last_norm] if last_norm is not None else []
    pixel_boxes = [[bn[0] * W, bn[1] * H, bn[2] * W, bn[3] * H] for bn in norm_boxes]
    return norm_boxes, pixel_boxes


def _box_area(b: List[float]) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _intersection_area(a: List[float], b: List[float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return (x2 - x1) * (y2 - y1)


def _iou(a: List[float], b: List[float]) -> float:
    inter = _intersection_area(a, b)
    union = _box_area(a) + _box_area(b) - inter
    return inter / union if union > 0 else 0.0


def _gt_overlap_ratio(pred: List[float], gt: List[float]) -> float:
    inter = _intersection_area(pred, gt)
    g = _box_area(gt)
    return inter / g if g > 0 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--project-root", required=True)
    ap.add_argument("--config-json", required=True)
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

    from pathlib import Path

    for _dp in Path(os.environ.get("VISTA_TORCH_PREFIX", "/workspace/torch-cuda")).rglob("dist-packages"):
        if _dp.is_dir() and (_dp / "torch").is_dir():
            _p = str(_dp)
            if _p not in sys.path:
                sys.path.insert(0, _p)
            break

    sys.path.insert(0, args.project_root)

    with open(args.config_json) as f:
        cfg = json.load(f)

    import xml.etree.ElementTree as ET
    import argparse as ap_mod
    import torch
    from llava.utils import disable_torch_init
    import myutils
    from model_loader import ModelLoader
    from steering_vector import obtain_vsv, add_logits_flag, remove_logits_flag
    from llm_layers import add_vsv_layers, remove_vsv_layers
    from PIL import Image
    from tqdm import tqdm

    voc_dir = cfg["voc_dir"]
    img_dir_f = cfg["img_dir_f"]
    img_dir_nf = cfg["img_dir_nf"]
    MODEL_NAME = cfg.get("model_name", "llava-1.5")

    DESC_PROMPT = cfg["desc_prompt"]
    CLS_PROMPT = cfg["cls_prompt"]
    BBOX_PROMPT = cfg["bbox_prompt"]
    BBOX_RETRY_PROMPT = cfg["bbox_retry_prompt"]
    BODY_REGION_PROMPT = cfg["body_region_prompt"]

    mt = cfg["max_tokens"]
    MAX_NEW_TOKENS_DESC = mt["desc"]
    MAX_NEW_TOKENS_CLS = mt["cls"]
    MAX_NEW_TOKENS_BBOX = mt["bbox"]
    MAX_NEW_TOKENS_BBOX_RETRY = mt["bbox_retry"]
    MAX_NEW_TOKENS_BODY = mt.get("body_region", 32)

    @dataclass
    class VariantCfg:
        name: str
        vsv: bool = False
        logits_aug: bool = False
        vsv_lambda: float = 0.0
        layers: Optional[str] = None
        logits_layers: str = "25,30"
        logits_alpha: float = 0.3
        vsv_neg_mode: str = "null"

    VARIANTS = [VariantCfg(**v) for v in cfg["variants"]]
    nf_by_region: Dict[str, List[str]] = {k: list(v) for k, v in cfg["nf_by_region"].items()}
    all_nf_ids = list(cfg["all_nf_ids"])
    rows = cfg["rows"]

    def resolve_image_path(image_id: str, fractured_label: int) -> str:
        if fractured_label == 1:
            path = os.path.join(img_dir_f, image_id)
        else:
            path = os.path.join(img_dir_nf, image_id)
        if not os.path.exists(path):
            alt_dir = img_dir_nf if fractured_label == 1 else img_dir_f
            alt = os.path.join(alt_dir, image_id)
            if os.path.exists(alt):
                return alt
            raise FileNotFoundError(f"Image not found: {image_id}")
        return path

    def parse_voc_bboxes(image_id: str) -> List[List[float]]:
        xml_path = os.path.join(voc_dir, image_id.replace(".jpg", ".xml"))
        if not os.path.exists(xml_path):
            return []
        root = ET.parse(xml_path).getroot()
        boxes = []
        for obj in root.findall("object"):
            name = (obj.findtext("name") or "").strip().lower()
            if name != "fractured":
                continue
            bb = obj.find("bndbox")
            if bb is None:
                continue
            xmin = float(bb.findtext("xmin", "0"))
            ymin = float(bb.findtext("ymin", "0"))
            xmax = float(bb.findtext("xmax", "0"))
            ymax = float(bb.findtext("ymax", "0"))
            if xmax > xmin and ymax > ymin:
                boxes.append([xmin, ymin, xmax, ymax])
        return boxes

    disable_torch_init()
    ml = ModelLoader(MODEL_NAME)
    dev = torch.device("cuda:0")
    ml.vlm_model = ml.vlm_model.to(dev)
    ml.llm_model = ml.llm_model.to(dev)
    ml.vlm_model.eval()
    ml.llm_model.eval()

    def make_args(variant: VariantCfg):
        return ap_mod.Namespace(
            model=MODEL_NAME,
            vsv=variant.vsv,
            vsv_lambda=variant.vsv_lambda,
            layers=variant.layers,
            logits_aug=variant.logits_aug,
            logits_layers=variant.logits_layers,
            logits_alpha=variant.logits_alpha,
        )

    def run_generation(
        image_pil: Image.Image,
        prompt: str,
        variant: VariantCfg,
        max_new_tokens: int,
        neg_image_pil: Optional[Image.Image] = None,
    ) -> str:
        a = make_args(variant)
        template = myutils.prepare_template(a)
        image = ml.image_processor(image_pil)
        query = [prompt]

        with torch.inference_mode():
            with myutils.maybe_autocast(MODEL_NAME, ml.vlm_model.device):
                questions, kwargs = ml.prepare_inputs_for_model(template, query, image)

                if variant.vsv:
                    neg_mode = getattr(variant, "vsv_neg_mode", "null")
                    if neg_mode == "matched_nf" and neg_image_pil is not None:
                        neg_proc = ml.image_processor(neg_image_pil)
                        neg_kwargs = ml.prepare_llava_kwargs_from_processed(
                            template, query, neg_proc
                        )
                    else:
                        neg_kwargs = ml.prepare_neg_prompt(a, questions, template=template)
                    pos_kwargs = ml.prepare_pos_prompt(a, kwargs)
                    visual_vector, _ = obtain_vsv(a, ml.llm_model, [[neg_kwargs, pos_kwargs]], rank=1)
                    llm_device = next(ml.llm_model.parameters()).device
                    vsv_tensor = torch.stack([visual_vector], dim=1).to(llm_device)
                    add_vsv_layers(
                        ml.llm_model,
                        vsv_tensor,
                        [variant.vsv_lambda],
                        variant.layers,
                    )

                for _k in ["logits_aug", "logits_layers", "logits_alpha"]:
                    if hasattr(ml.llm_model, _k):
                        delattr(ml.llm_model, _k)
                add_logits_flag(ml.llm_model, a)

                outputs = ml.llm_model.generate(
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                    output_attentions=False,
                    output_hidden_states=True if variant.logits_aug else False,
                    return_dict=True,
                    **kwargs,
                )

                remove_logits_flag(ml.llm_model)
                if variant.vsv:
                    remove_vsv_layers(ml.llm_model)

        return ml.decode(outputs)[0].strip()

    records: List[Dict[str, Any]] = []
    body_region_cache: Dict[str, str] = {}

    prog_path = cfg.get("_progress_path")
    prog_total = cfg.get("_progress_total_images")

    for row in tqdm(rows, desc=f"eval shard GPU {args.gpu}"):
        image_id = row["image_id"]
        gt_label = int(row["fractured"])
        image_path = resolve_image_path(image_id, gt_label)
        gt_boxes = parse_voc_bboxes(image_id)

        image = Image.open(image_path).convert("RGB")
        W, H = image.size

        if image_id not in body_region_cache:
            body_region_cache[image_id] = _parse_body_region(
                run_generation(image, BODY_REGION_PROMPT, VARIANTS[0], MAX_NEW_TOKENS_BODY)
            )
        region = body_region_cache[image_id]

        pool = [i for i in nf_by_region.get(region, []) if i != image_id]
        if not pool:
            pool = [i for i in all_nf_ids if i != image_id]
        if not pool:
            neg_image_id = image_id
        else:
            rid = int(hashlib.md5(image_id.encode("utf-8")).hexdigest()[:8], 16)
            neg_image_id = pool[rid % len(pool)]
        neg_path = os.path.join(img_dir_nf, neg_image_id)
        if os.path.isfile(neg_path):
            neg_pil = Image.open(neg_path).convert("RGB")
        else:
            neg_pil = image

        for variant in VARIANTS:
            neg_arg = None
            if variant.vsv and getattr(variant, "vsv_neg_mode", "null") == "matched_nf":
                neg_arg = neg_pil

            desc = run_generation(image, DESC_PROMPT, variant, MAX_NEW_TOKENS_DESC, neg_image_pil=neg_arg)
            cls_text = run_generation(image, CLS_PROMPT, variant, MAX_NEW_TOKENS_CLS, neg_image_pil=neg_arg)
            pred_label = _parse_pred_label(cls_text)

            bbox_text = run_generation(image, BBOX_PROMPT, variant, MAX_NEW_TOKENS_BBOX, neg_image_pil=neg_arg)
            pred_boxes_norm, pred_boxes = _parse_bboxes_from_text(bbox_text, (W, H))

            if variant.name in ("vista_model", "vista_focused") and len(pred_boxes_norm) == 0:
                retry_txt = run_generation(
                    image, BBOX_RETRY_PROMPT, variant, MAX_NEW_TOKENS_BBOX_RETRY, neg_image_pil=neg_arg
                )
                rn, rp = _parse_bboxes_from_text(retry_txt, (W, H))
                if len(rn) > 0:
                    bbox_text = bbox_text + "\n---retry_strict---\n" + retry_txt
                    pred_boxes_norm, pred_boxes = rn, rp
                else:
                    fb = VariantCfg(
                        name=variant.name + "_fb",
                        vsv=False,
                        logits_aug=False,
                        vsv_lambda=0.0,
                        layers=None,
                        logits_layers=variant.logits_layers,
                        logits_alpha=variant.logits_alpha,
                        vsv_neg_mode="null",
                    )
                    fb_txt = run_generation(image, BBOX_RETRY_PROMPT, fb, MAX_NEW_TOKENS_BBOX_RETRY)
                    fn, fp = _parse_bboxes_from_text(fb_txt, (W, H))
                    if len(fn) > 0:
                        bbox_text = (
                            bbox_text
                            + "\n---retry_strict---\n"
                            + retry_txt
                            + "\n---fallback_no_aug---\n"
                            + fb_txt
                        )
                        pred_boxes_norm, pred_boxes = fn, fp

            best_iou = 0.0
            best_gt_overlap = 0.0
            if len(gt_boxes) > 0 and len(pred_boxes_norm) > 0:
                gt_boxes_norm = [[b[0] / W, b[1] / H, b[2] / W, b[3] / H] for b in gt_boxes]
                for pb in pred_boxes_norm:
                    for gb in gt_boxes_norm:
                        best_iou = max(best_iou, _iou(pb, gb))
                        best_gt_overlap = max(best_gt_overlap, _gt_overlap_ratio(pb, gb))

            bbox_correct_50 = int(best_gt_overlap >= 0.5)

            records.append(
                {
                    "variant": variant.name,
                    "image_id": image_id,
                    "image_path": image_path,
                    "gt_label": gt_label,
                    "pred_label": int(pred_label),
                    "description": desc,
                    "classification_text": cls_text,
                    "bbox_text": bbox_text,
                    "gt_boxes": gt_boxes,
                    "pred_boxes": pred_boxes,
                    "pred_boxes_norm": pred_boxes_norm,
                    "best_iou": float(best_iou),
                    "best_gt_overlap": float(best_gt_overlap),
                    "bbox_correct_50": bbox_correct_50,
                    "body_region": region,
                    "neg_image_id_matched_nf": neg_image_id if neg_arg is not None else "",
                }
            )

        if prog_path and prog_total:
            n_done = _bump_shared_image_progress(prog_path)
            print(
                f"[eval progress] images completed {n_done}/{prog_total} "
                f"({100.0 * n_done / prog_total:.1f}%)  [shard GPU {args.gpu}]",
                flush=True,
            )

    with open(args.out_json, "w") as f:
        json.dump(records, f)


if __name__ == "__main__":
    main()
