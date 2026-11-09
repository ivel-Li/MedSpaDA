#!/usr/bin/env python3
"""Evaluate a frozen joint SAM3 checkpoint on generalist dev or test cases."""

from __future__ import annotations

import argparse
import os
from argparse import Namespace
from collections import defaultdict
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, os.environ.get("MEDSPADA_SAM3_REPO", ""))
from medspada.data import DATASETS, LEGACY, open_dataset, protocol_sha256


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--benchmark", choices=tuple(DATASETS), default=LEGACY)
    parser.add_argument("--dataset", choices=DATASETS[LEGACY], default=None)
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--max-cases-per-dataset", type=int, default=None)
    parser.add_argument("--core-depth", type=int, default=8)
    parser.add_argument("--halo", type=int, default=2)
    parser.add_argument("--max-concepts", type=int, default=5)
    parser.add_argument("--score-threshold", type=float, default=0.60)
    parser.add_argument("--mask-threshold", type=float, default=0.0)
    args = parser.parse_args()
    if args.max_cases_per_dataset is not None and args.max_cases_per_dataset < 1:
        parser.error("max cases must be positive")
    if args.split == "test" and args.max_cases_per_dataset is not None:
        parser.error("test evaluation must include every locked-test case")
    return args


def concept_chunks(count: int, maximum: int):
    return [list(range(start, min(start + maximum, count)))
            for start in range(0, count, maximum)]


def window_context(sample, start: int, core_depth: int, halo: int):
    if sample.depth == 1:
        return sample.images, sample.z_mm, 0, 1
    valid = min(core_depth, sample.depth - start)
    indices = np.arange(start - halo, start + core_depth + halo, dtype=np.int64)
    indices = np.clip(indices, 0, sample.depth - 1)
    return (
        [sample.images[int(index)] for index in indices],
        sample.z_mm[indices],
        halo,
        valid,
    )


def dice(prediction: np.ndarray, target: np.ndarray) -> float:
    denominator = float(prediction.sum() + target.sum())
    if denominator == 0:
        return 1.0
    return float(2.0 * np.logical_and(prediction, target).sum() / denominator)


def build_model(model_name, torch):
    from medspada.adapters import install_encoder_adaptation
    from medspada.resolution import install_high_resolution_decoder
    from medspada.query_identity import install_patient_query_fusion
    from medspada.native import load_model

    model, _, _ = load_model(
        Namespace(
            sam3_repo=Path(os.environ["MEDSPADA_SAM3_REPO"]),
            stage_a_checkpoint=Path(os.environ["MEDSPADA_SAM3_CHECKPOINT"]),
        ), torch,
    )
    fusion = install_patient_query_fusion(model, "F0")
    medical = model_name.startswith("medicalsam3d")
    if medical:
        encoder, _ = install_encoder_adaptation(
            model,
            encoder_arm="M3",
            bottleneck=384,
            spatial_kind=(
                "mona_serial" if model_name == "medicalsam3d_serial"
                else "mona_parallel"
            ),
            peft_kind="fact_qv",
            rank=32,
            scale=1.0,
            dropout=0.1,
        )
    else:
        encoder, _ = install_encoder_adaptation(
            model, encoder_arm="E0", bottleneck=64,
            spatial_kind="legacy", peft_kind="auto", rank=32,
            scale=1.0, dropout=0.1,
        )
    hr1 = model_name in {"medicalsam3d", "medicalsam3d_serial", "sam3_ft_hr1"}
    install_high_resolution_decoder(
        model, "HR1" if hr1 else "HR0", half_size=504, full_size=1008
    )
    return model, fusion, encoder


def main():
    args = parse_args()
    import torch
    import torch.nn.functional as F
    from torchvision.transforms import v2

    from medspada.runner import Sam3PatientVolumeRunner
    from medspada.native import transform_native_window

    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if saved.get("format") != "medicalsam3d-generalist-joint-v1":
        raise ValueError("unexpected checkpoint format")
    saved_config = saved.get("config", {})
    if saved_config.get("benchmark", LEGACY) != args.benchmark:
        raise ValueError("checkpoint benchmark mismatch")
    if saved_config.get("benchmark_protocol_sha256") != protocol_sha256(args.benchmark):
        raise ValueError("checkpoint benchmark protocol mismatch")
    if saved_config.get("dataset") != args.dataset:
        raise ValueError("checkpoint task-specific dataset mismatch")
    model_name = saved["model_name"]
    model, fusion, _ = build_model(model_name, torch)
    model.load_state_dict(saved["model"], strict=True)
    model.eval()
    model.backbone.act_ckpt_whole_vision_backbone = False
    model.backbone.act_ckpt_whole_language_backbone = False
    model.use_act_checkpoint_seg_head = False
    runner = Sam3PatientVolumeRunner(
        model, fusion,
        slice_microbatch=args.core_depth + 2 * args.halo,
        head_slice_microbatch=1,
        encoder_halo=0,
        singleton_halo=1,
    )
    dataset = open_dataset(args.benchmark, args.split)
    if args.dataset is not None and args.dataset not in DATASETS[args.benchmark]:
        raise ValueError(f"{args.dataset} is not in {args.benchmark}")
    dataset_names = (args.dataset,) if args.dataset else DATASETS[args.benchmark]
    case_rows = []
    started = time.time()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for dataset_name in dataset_names:
            cases = dataset.cases(dataset=dataset_name, split=args.split)
            if args.max_cases_per_dataset is not None:
                cases = cases[:args.max_cases_per_dataset]
            for case in cases:
                sample = dataset.load(case)
                predictions = np.zeros_like(sample.target, dtype=np.bool_)
                for concept_indices in concept_chunks(
                    len(sample.concepts), args.max_concepts
                ):
                    selected = sample.select_concepts(concept_indices)
                    categories = len(concept_indices)
                    for volume_start in range(0, sample.depth, args.core_depth):
                        images, z_np, core_offset, valid_depth = window_context(
                            selected, volume_start, args.core_depth, args.halo
                        )
                        # Validation uses the full 1008 canvas deterministically.
                        dummy = selected.target[
                            :, volume_start:volume_start + valid_depth
                        ].reshape(
                            categories * valid_depth,
                            *selected.target.shape[-2:],
                        )
                        image_batch, _, _ = transform_native_window(
                            images, dummy,
                            resolution=1008,
                            min_resolution=1008,
                            rng=random.Random(0),
                            torch=torch,
                            v2=v2,
                        )
                        z_mm = torch.from_numpy(np.asarray(z_np)).to(
                            "cuda", dtype=torch.float32
                        )
                        encoding = runner.encode_patient(
                            image_batch, selected.prompts
                        )
                        fused, _ = runner.fuse_patient(encoding, z_mm)
                        core_end = core_offset + valid_depth
                        for start, end, output in runner.iter_final_outputs(
                            encoding, fused
                        ):
                            if end <= core_offset:
                                continue
                            if start >= core_end:
                                break
                            logits = output["pred_logits"].squeeze(-1)
                            presence = output["presence_logit_dec"]
                            if presence.ndim == 1:
                                presence = presence[:, None]
                            scores = logits.sigmoid() * presence.sigmoid()
                            best = scores.argmax(dim=1)
                            rows = torch.arange(categories, device="cuda")
                            selected_masks = output["pred_masks"][rows, best]
                            accepted = scores[rows, best] >= args.score_threshold
                            selected_masks = F.interpolate(
                                selected_masks[:, None].float(),
                                size=selected.target.shape[-2:],
                                mode="bilinear",
                                align_corners=False,
                            )[:, 0]
                            binary = (
                                selected_masks > args.mask_threshold
                            ) & accepted[:, None, None]
                            relative = start - core_offset
                            z_index = volume_start + relative
                            for local_category, global_category in enumerate(
                                concept_indices
                            ):
                                predictions[global_category, z_index] = (
                                    binary[local_category].cpu().numpy()
                                )
                            del output, selected_masks, binary
                        del encoding, fused
                class_dice = {
                    concept.name: dice(predictions[index], sample.target[index] > 0.5)
                    for index, concept in enumerate(sample.concepts)
                }
                positive_slice_dice = []
                for category in range(len(sample.concepts)):
                    for z_index in range(sample.depth):
                        target = sample.target[category, z_index] > 0.5
                        if target.any():
                            positive_slice_dice.append(
                                dice(predictions[category, z_index], target)
                            )
                row = {
                    "dataset": dataset_name,
                    "case_id": case.case_id,
                    "depth": sample.depth,
                    "class_dice": class_dice,
                    "patient_macro_dice": float(np.mean(list(class_dice.values()))),
                    "positive_slice_mean_dice": (
                        float(np.mean(positive_slice_dice))
                        if positive_slice_dice else math.nan
                    ),
                }
                case_rows.append(row)
                print(json.dumps(row), flush=True)
    by_dataset = {}
    for dataset_name in dataset_names:
        rows = [row for row in case_rows if row["dataset"] == dataset_name]
        by_dataset[dataset_name] = {
            "cases": len(rows),
            "patient_macro_dice": float(np.mean(
                [row["patient_macro_dice"] for row in rows]
            )),
            "positive_slice_mean_dice": float(np.nanmean(
                [row["positive_slice_mean_dice"] for row in rows]
            )),
        }
    summary = {
        "status": "passed",
        "model": model_name,
        "benchmark": args.benchmark,
        "dataset": args.dataset,
        "benchmark_protocol_sha256": protocol_sha256(args.benchmark),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": int(saved["epoch"]),
        "split": args.split,
        "score_threshold": args.score_threshold,
        "mask_threshold": args.mask_threshold,
        "by_dataset": by_dataset,
        "cases": case_rows,
        "elapsed_seconds": time.time() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "cases"}, indent=2))


if __name__ == "__main__":
    main()
