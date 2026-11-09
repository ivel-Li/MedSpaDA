#!/usr/bin/env python3
"""Evaluate a jointly trained MedicalSAM3D Parallel+C0 checkpoint."""

from __future__ import annotations

import argparse
from argparse import Namespace
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, os.environ.get("MEDSPADA_SAM3_REPO", ""))
from medspada.data import DATASETS, LEGACY, open_dataset, protocol_sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--benchmark", choices=tuple(DATASETS), default=LEGACY)
    parser.add_argument("--dataset", choices=DATASETS[LEGACY], default=None)
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--max-cases-per-dataset", type=int)
    args = parser.parse_args()
    if args.split == "test" and args.max_cases_per_dataset is not None:
        parser.error("test evaluation must include every locked-test case")

    # Recovery workers provide MEDSODA_GPU; honor it when no explicit CUDA map exists.
    if "CUDA_VISIBLE_DEVICES" not in os.environ and "MEDSODA_GPU" in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["MEDSODA_GPU"]

    import torch
    import torch.nn.functional as F
    from torchvision.transforms import v2

    from medspada.refinement import GeneralistSpatialResidualC0
    from medspada.runner import Sam3PatientVolumeRunner
    from medspada.evaluation import build_model
    from medspada.refinement_evaluation import evaluate

    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if saved.get("format") != "medicalsam3d-generalist-joint-c0-v2":
        raise ValueError("expected jointly trained Parallel+C0 checkpoint")
    if saved.get("model_name") not in ("medicalsam3d", "medicalsam3d_serial", "sam3_ft_hr1"):
        raise ValueError("unsupported fixed-HR1 joint VSR architecture")
    saved_config = saved.get("config", {})
    if saved_config.get("benchmark", LEGACY) != args.benchmark:
        raise ValueError("checkpoint benchmark mismatch")
    if saved_config.get("benchmark_protocol_sha256") != protocol_sha256(args.benchmark):
        raise ValueError("checkpoint benchmark protocol mismatch")
    if saved_config.get("dataset") != args.dataset:
        raise ValueError("checkpoint task-specific dataset mismatch")
    model, fusion, _ = build_model(saved["model_name"], torch)
    model.load_state_dict(saved["model"], strict=True)
    model.eval()
    model.backbone.act_ckpt_whole_vision_backbone = False
    model.backbone.act_ckpt_whole_language_backbone = False
    model.use_act_checkpoint_seg_head = False
    runner = Sam3PatientVolumeRunner(
        model, fusion, slice_microbatch=12, head_slice_microbatch=1,
        encoder_halo=0, singleton_halo=1,
    )
    c0 = GeneralistSpatialResidualC0().cuda()
    c0.load_state_dict(saved["c0"], strict=True)
    c0.eval()
    settings = Namespace(
        core_depth=8, halo=2, max_concepts=5, score_threshold=0.60,
        max_dev_cases_per_dataset=args.max_cases_per_dataset,
        eval_split=args.split,
        datasets=(args.dataset,) if args.dataset else DATASETS[args.benchmark],
    )
    dataset = open_dataset(args.benchmark, args.split)
    result = evaluate(
        model, runner, c0, dataset, settings, torch, F, v2
    )
    summary = {
        "status": "passed", "split": args.split, "epoch": int(saved["epoch"]),
        "benchmark": args.benchmark,
        "dataset": args.dataset,
        "benchmark_protocol_sha256": protocol_sha256(args.benchmark),
        "checkpoint": str(args.checkpoint.resolve()),
        "score_threshold": 0.60,
        "patient_macro_dice": result["c0_macro"],
        "by_dataset": {
            name: {
                "cases": row["cases"],
                "patient_macro_dice": row["c0_patient_macro_dice"],
                "base_patient_macro_dice": row["base_patient_macro_dice"],
            }
            for name, row in result["by_dataset"].items()
        },
        "cases": result["cases"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "cases"}))


if __name__ == "__main__":
    main()
