"""Split FracAtlas benchmark work across multiple GPUs via subprocesses (one CUDA device per process)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Sequence


def _split_evenly(items: Sequence[Any], n: int) -> List[List[Any]]:
    if n <= 1:
        return [list(items)]
    k = len(items)
    base, rem = divmod(k, n)
    out, j = [], 0
    for i in range(n):
        sz = base + (1 if i < rem else 0)
        out.append(list(items[j : j + sz]))
        j += sz
    return out


def run_nf_region_index_parallel(
    *,
    project_root: str,
    img_dir_nf: str,
    image_ids: List[str],
    body_region_prompt: str,
    physical_gpus: Sequence[int],
) -> Dict[str, List[str]]:
    """Run baseline body-region labeling on ``image_ids`` split across ``physical_gpus``."""
    gpus = list(physical_gpus)
    if len(gpus) <= 1:
        raise ValueError("need at least 2 physical GPU indices")

    chunks = _split_evenly(image_ids, len(gpus))
    script = Path(__file__).resolve().parent / "fracatlas_nf_region_worker.py"
    if not script.is_file():
        raise FileNotFoundError(script)

    tmpdir = tempfile.mkdtemp(prefix="vista_nf_")
    cmds = []
    for gi, (gpu, chunk) in enumerate(zip(gpus, chunks)):
        if not chunk:
            continue
        ids_path = os.path.join(tmpdir, f"ids_{gi}.json")
        out_path = os.path.join(tmpdir, f"out_{gi}.json")
        with open(ids_path, "w") as f:
            json.dump(chunk, f)
        cmds.append(
            (
                gpu,
                [
                    sys.executable,
                    str(script),
                    "--gpu",
                    str(gpu),
                    "--project-root",
                    project_root,
                    "--img-dir-nf",
                    img_dir_nf,
                    "--ids-json",
                    ids_path,
                    "--out-json",
                    out_path,
                    "--body-region-prompt",
                    body_region_prompt,
                ],
                out_path,
            )
        )

    def _run(cmd_tuple):
        gpu, cmd, _ = cmd_tuple
        env = os.environ.copy()
        # Child sets CUDA_VISIBLE_DEVICES itself; keep parent env unchanged.
        r = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(
                f"NF worker GPU {gpu} failed ({r.returncode}): {r.stderr[-4000:]}"
            )
        return cmd_tuple

    with ThreadPoolExecutor(max_workers=len(cmds)) as ex:
        list(ex.map(_run, cmds))

    nf_by_region: Dict[str, List[str]] = defaultdict(list)
    for _, _, out_path in cmds:
        with open(out_path) as f:
            rows = json.load(f)
        for item in rows:
            nf_by_region[item["region"]].append(item["image_id"])

    return dict(nf_by_region)


def run_eval_shard_subprocess(
    *,
    project_root: str,
    config_path: str,
    out_path: str,
    physical_gpu: int,
) -> None:
    script = Path(__file__).resolve().parent / "fracatlas_eval_shard_worker.py"
    if not script.is_file():
        raise FileNotFoundError(script)
    cmd = [
        sys.executable,
        str(script),
        "--gpu",
        str(physical_gpu),
        "--project-root",
        project_root,
        "--config-json",
        config_path,
        "--out-json",
        out_path,
    ]
    # Inherit stdout/stderr so worker progress prints show live in Jupyter.
    r = subprocess.run(cmd, text=True)
    if r.returncode != 0:
        raise RuntimeError(
            f"Eval shard GPU {physical_gpu} failed ({r.returncode}); see traceback/output above."
        )


def run_eval_parallel(
    *,
    project_root: str,
    config_dict: Dict[str, Any],
    physical_gpus: Sequence[int],
) -> List[Dict[str, Any]]:
    """Split config_dict[\"rows\"] across GPUs; each worker returns a list of record dicts."""
    gpus = list(physical_gpus)
    rows = config_dict.get("rows") or []
    if len(gpus) <= 1 or len(rows) <= 1:
        raise ValueError("parallel eval needs 2+ GPUs and 2+ rows")

    chunks = _split_evenly(rows, len(gpus))
    n_var = len(config_dict.get("variants") or [])
    shard_parts = [
        f"GPU {gpu}: {len(ch)} images" for gpu, ch in zip(gpus, chunks) if ch
    ]
    print(
        "[run_eval_parallel] "
        f"{len(rows)} eval images across {len(gpus)} workers | "
        + " | ".join(shard_parts)
    )
    print(
        "[run_eval_parallel] "
        f"Each image × {n_var} variants → {len(rows) * n_var} output rows; "
        f"≥{len(rows) * (1 + n_var * 3)} generate() calls minimum per shard sum "
        f"(+ bbox retries)."
    )
    print(
        "[run_eval_parallel] Live progress: workers print [eval progress] "
        "after each image (global % across all GPUs)."
    )
    tmpdir = tempfile.mkdtemp(prefix="vista_eval_")
    progress_path = os.path.join(tmpdir, "images_done.count")
    with open(progress_path, "w") as _pf:
        _pf.write("0\n")

    futures = []
    out_paths = []
    with ThreadPoolExecutor(max_workers=len(gpus)) as ex:
        for gi, (gpu, chunk) in enumerate(zip(gpus, chunks)):
            if not chunk:
                out_paths.append(None)
                continue
            sub = dict(config_dict)
            sub["rows"] = chunk
            sub["_progress_path"] = progress_path
            sub["_progress_total_images"] = len(rows)
            cfg_path = os.path.join(tmpdir, f"cfg_{gi}.json")
            out_path = os.path.join(tmpdir, f"out_{gi}.json")
            with open(cfg_path, "w") as f:
                json.dump(sub, f)
            out_paths.append(out_path)
            futures.append(
                ex.submit(
                    run_eval_shard_subprocess,
                    project_root=project_root,
                    config_path=cfg_path,
                    out_path=out_path,
                    physical_gpu=gpu,
                )
            )
        for f in as_completed(futures):
            f.result()

    records: List[Dict[str, Any]] = []
    for p in out_paths:
        if p is None:
            continue
        with open(p) as f:
            records.extend(json.load(f))
    return records
