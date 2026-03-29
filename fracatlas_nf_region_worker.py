#!/usr/bin/env python3
"""Label non-fracture images by body region on a single GPU (subprocess). CUDA_VISIBLE_DEVICES set before torch."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, required=True, help="Physical CUDA device index for this process")
    ap.add_argument("--project-root", required=True)
    ap.add_argument("--img-dir-nf", required=True)
    ap.add_argument("--ids-json", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--body-region-prompt", required=True)
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

    import torch
    from llava.utils import disable_torch_init
    import myutils
    from model_loader import ModelLoader
    from steering_vector import add_logits_flag, remove_logits_flag
    from PIL import Image
    from tqdm import tqdm
    import argparse as ap_mod

    disable_torch_init()
    ml = ModelLoader("llava-1.5")
    dev = torch.device("cuda:0")
    ml.vlm_model = ml.vlm_model.to(dev)
    ml.llm_model = ml.llm_model.to(dev)
    ml.vlm_model.eval()
    ml.llm_model.eval()

    MODEL_NAME = "llava-1.5"

    def make_args():
        return ap_mod.Namespace(
            model=MODEL_NAME,
            vsv=False,
            vsv_lambda=0.0,
            layers=None,
            logits_aug=False,
            logits_layers="25,30",
            logits_alpha=0.3,
        )

    def run_baseline_generation(image_pil, prompt: str, max_new_tokens: int) -> str:
        a = make_args()
        template = myutils.prepare_template(a)
        image = ml.image_processor(image_pil)
        query = [prompt]
        with torch.inference_mode():
            with myutils.maybe_autocast(MODEL_NAME, ml.vlm_model.device):
                questions, kwargs = ml.prepare_inputs_for_model(template, query, image)
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
                    output_hidden_states=False,
                    return_dict=True,
                    **kwargs,
                )
                remove_logits_flag(ml.llm_model)
        return ml.decode(outputs)[0].strip()

    with open(args.ids_json) as f:
        ids = json.load(f)

    results = []
    for iid in tqdm(ids, desc=f"NF regions cuda:{args.gpu}"):
        p = os.path.join(args.img_dir_nf, iid)
        if not os.path.isfile(p):
            continue
        im = Image.open(p).convert("RGB")
        txt = run_baseline_generation(im, args.body_region_prompt, 24)
        results.append({"image_id": iid, "region": _parse_body_region(txt)})

    with open(args.out_json, "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    main()
