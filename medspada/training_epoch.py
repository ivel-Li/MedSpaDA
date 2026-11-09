#!/usr/bin/env python3
"""Train or calibrate the joint generalist SAM3 models on real data.

Without ``--balanced-updates-per-epoch`` this is a short throughput/gradient
calibration.  Formal mode makes every optimizer update accumulate one patient
window from each of the five datasets and writes a resumable epoch checkpoint.
"""

from __future__ import annotations

import argparse
import os
from argparse import Namespace
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


from medspada.data import DATASETS, LEGACY, protocol_sha256, open_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=(
            "sam3_ft",
            "sam3_ft_hr1",
                        "medicalsam3d_serial",
            "medicalsam3d",
        ),
        required=True,
    )
    parser.add_argument("--benchmark", choices=tuple(DATASETS), default=LEGACY)
    parser.add_argument("--dataset", choices=DATASETS[LEGACY], default=None)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument(
        "--planned-epochs",
        type=int,
        default=None,
        help=(
            "fixed full-run horizon used by the LR schedule; this remains "
            "constant when --epochs is increased one epoch at a time"
        ),
    )
    parser.add_argument(
        "--balanced-updates-per-epoch",
        type=int,
        default=None,
        help="formal mode: optimizer updates per epoch, each using every benchmark dataset",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--core-depth", type=int, default=8)
    parser.add_argument("--halo", type=int, default=2)
    parser.add_argument("--max-concepts", type=int, default=5)
    parser.add_argument("--min-train-resolution", type=int, default=480)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--with-c0", action="store_true")
    parser.add_argument("--init-base-checkpoint", type=Path)
    parser.add_argument("--init-c0-checkpoint", type=Path)
    parser.add_argument("--base-lr-factor", type=float, default=1.0)
    parser.add_argument("--c0-learning-rate", type=float, default=1e-4)
    parser.add_argument("--c0-loss-weight", type=float, default=1.0)
    parser.add_argument("--amos-sampler", choices=("uniform", "fg_bg_1to1"), default="uniform")
    parser.add_argument("--amos-all-concepts", action="store_true")
    parser.add_argument("--lung-positive-window-prob", type=float, default=0.0)
    parser.add_argument("--amos-positive-window-prob", type=float, default=0.0)
    parser.add_argument("--safe-gradient-norm", action="store_true")
    parser.add_argument("--lr-warmup-steps", type=int, default=20)
    parser.add_argument("--lr-cooldown-steps", type=int, default=20)
    parser.add_argument("--lr-timescale", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not all(0 <= p <= 1 for p in (args.lung_positive_window_prob, args.amos_positive_window_prob)):
        parser.error("positive-window probabilities must be within [0, 1]")
    if args.amos_positive_window_prob and args.amos_sampler != "uniform":
        parser.error("choose only one AMOS sampler")
    if min(args.steps, args.epochs, args.core_depth, args.max_concepts) < 1 or args.halo < 0:
        parser.error("invalid positive calibration dimensions")
    if not 32 <= args.min_train_resolution <= 1008:
        parser.error("min-train-resolution must be between 32 and 1008")
    if args.balanced_updates_per_epoch is not None and args.balanced_updates_per_epoch < 1:
        parser.error("balanced updates per epoch must be positive")
    if args.planned_epochs is None:
        args.planned_epochs = args.epochs
    if args.planned_epochs < args.epochs:
        parser.error("planned epochs cannot be smaller than target epochs")
    if min(args.lr_warmup_steps, args.lr_cooldown_steps) < 0:
        parser.error("LR warmup/cooldown steps cannot be negative")
    if args.lr_timescale < 1:
        parser.error("LR timescale must be positive")
    if args.base_lr_factor <= 0 or args.c0_learning_rate <= 0 or args.c0_loss_weight < 0:
        parser.error("invalid joint learning rate or C0 loss weight")
    if args.init_c0_checkpoint and not args.with_c0:
        parser.error("C0 initialization requires --with-c0")
    if args.init_c0_checkpoint and not args.init_base_checkpoint:
        parser.error("C0 initialization requires its paired base checkpoint")
    if args.c0_loss_weight == 0 and not (
        args.with_c0 and args.init_c0_checkpoint
    ):
        parser.error("zero C0 loss is reserved for a paired frozen-C0 control")
    if (args.amos_sampler != "uniform" or args.amos_all_concepts) and not args.with_c0:
        parser.error("new AMOS ablations require --with-c0")
    if (
        args.balanced_updates_per_epoch is not None
        and args.planned_epochs * args.balanced_updates_per_epoch
        <= args.lr_warmup_steps
    ):
        parser.error(
            "planned formal updates must exceed LR warmup steps"
        )
    return args


def atomic_torch_save(payload, destination: Path, torch) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        torch.save(payload, temporary)
        temporary.replace(destination)
    finally:
        # Failed quota-limited saves must not strand another full model on disk.
        temporary.unlink(missing_ok=True)


def choose_concepts(sample, maximum: int, rng: random.Random):
    count = len(sample.concepts)
    if count <= maximum:
        return sample
    chosen = sorted(rng.sample(range(count), maximum))
    return sample.select_concepts(chosen)


def choose_core_with_context(sample, core_depth: int, halo: int, rng: random.Random,
                             start_override: int | None = None):
    """Return explicit encoder context and the supervised core metadata."""

    if sample.depth == 1:
        return sample.images, sample.target, sample.z_mm, 0, 1, [0]
    valid_depth = min(core_depth, sample.depth)
    start = (rng.randint(0, sample.depth - valid_depth)
             if start_override is None else start_override)
    if not 0 <= start <= sample.depth - valid_depth:
        raise ValueError("invalid core start")
    context_indices = np.arange(
        start - halo, start + core_depth + halo, dtype=np.int64
    )
    context_indices = np.clip(context_indices, 0, sample.depth - 1)
    core_indices = np.arange(start, start + valid_depth, dtype=np.int64)
    return (
        [sample.images[int(index)] for index in context_indices],
        sample.target[:, core_indices],
        sample.z_mm[context_indices],
        halo,
        valid_depth,
        context_indices.tolist(),
    )


def amos_fg_bg_start(sample, target_index: int, core_depth: int, positive: bool,
                     rng: random.Random) -> int | None:
    """Sample a core positive/negative for one *prompted* AMOS class."""
    valid_depth = min(core_depth, sample.depth)
    presence = sample.target[target_index].reshape(sample.depth, -1).any(axis=1)
    prefix = np.concatenate(([0], np.cumsum(presence, dtype=np.int64)))
    hits = prefix[valid_depth:] - prefix[:-valid_depth]
    candidates = np.flatnonzero(hits > 0 if positive else hits == 0)
    return int(rng.choice(candidates.tolist())) if len(candidates) else None


def positive_window_override(sample, core_depth, probability, rng):
    """Force a positive core for one of the supervised concepts, else uniform.

    Select the case before this function: retries must not reweight patients.
    Absent concepts are excluded from the target draw and remain supervised.
    """
    if probability <= 0 or rng.random() >= probability:
        return None, None
    present = np.flatnonzero(sample.target.reshape(len(sample.concepts), -1).any(axis=1))
    if not len(present):
        return None, None
    index = int(rng.choice(present.tolist()))
    return amos_fg_bg_start(sample, index, core_depth, True, rng), sample.concepts[index].prompt


def make_optimizer(model, fusion, encoder_3d, vision_peft, high_resolution, torch):
    from medspada.optim import _optimizer

    values = Namespace(
        hr_learning_rate=1e-4,
        encoder_3d_learning_rate=1e-4,
        fact_learning_rate=1e-4,
        fusion_learning_rate=1e-4,
        vision_learning_rate=2.5e-5,
        vision_layer_decay=0.9,
        language_learning_rate=5e-6,
        decoder_learning_rate=1e-5,
        model_learning_rate=8e-5,
        weight_decay=0.1,
    )
    return _optimizer(
        model, fusion, encoder_3d, vision_peft, high_resolution, values, torch
    )


def main() -> None:
    args = parse_args()
    # Resolve the official SAM3 package before importing any ``sam3.*``
    # module.  Some environments also contain Medical-SAM3 under the same
    # top-level package name.
    sys.path.insert(0, os.environ.get("MEDSPADA_SAM3_REPO", ""))
    import torch
    from sam3.train.optim.schedulers import InverseSquareRootParamScheduler
    from torchvision.transforms import v2

    from medspada.adapters import install_encoder_adaptation
    from medspada.refinement import GeneralistSpatialResidualC0
    from medspada.resolution import install_high_resolution_decoder
    from medspada.query_identity import install_patient_query_fusion
    from medspada.runner import Sam3PatientVolumeRunner
    from medspada.native import (
        build_native_find_target,
        build_native_stage_a_loss,
        load_model,
        native_stage_a_loss,
        transform_native_window,
    )
    from medspada.refinement_loss import c0_window_loss, selected_q0

    rng = random.Random(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "metrics.jsonl"
    if args.dataset is not None and args.dataset not in DATASETS[args.benchmark]:
        raise ValueError(f"{args.dataset} is not in {args.benchmark}")
    dataset_names = (args.dataset,) if args.dataset else DATASETS[args.benchmark]
    benchmark_hash = protocol_sha256(args.benchmark)

    build_started = time.time()
    model, _, _ = load_model(
        Namespace(
            sam3_repo=Path(os.environ["MEDSPADA_SAM3_REPO"]),
            stage_a_checkpoint=Path(os.environ["MEDSPADA_SAM3_CHECKPOINT"]),
        ),
        torch,
    )
    fusion = install_patient_query_fusion(model, "F0")
    use_medical_encoder = args.model.startswith("medicalsam3d")
    use_hr1 = args.model in {
        "sam3_ft_hr1", "medicalsam3d_serial", "medicalsam3d"
    }
    if use_medical_encoder:
        encoder_3d, vision_peft = install_encoder_adaptation(
            model,
            encoder_arm="M3",
            bottleneck=384,
            spatial_kind=(
                "mona_serial"
                if args.model == "medicalsam3d_serial"
                else "mona_parallel"
            ),
            peft_kind="fact_qv",
            rank=32,
            scale=1.0,
            dropout=0.1,
        )
    else:
        encoder_3d, vision_peft = install_encoder_adaptation(
            model,
            encoder_arm="E0",
            bottleneck=64,
            spatial_kind="legacy",
            peft_kind="auto",
            rank=32,
            scale=1.0,
            dropout=0.1,
        )
    high_resolution = install_high_resolution_decoder(
        model, "HR1" if use_hr1 else "HR0", half_size=504, full_size=1008
    )
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    if use_medical_encoder:
        for parameter in model.backbone.vision_backbone.parameters():
            parameter.requires_grad_(False)
        for parameter in encoder_3d.parameters():
            parameter.requires_grad_(True)
        for parameter in vision_peft.parameters():
            parameter.requires_grad_(True)
    model.backbone.act_ckpt_whole_vision_backbone = True
    model.backbone.act_ckpt_whole_language_backbone = True
    model.use_act_checkpoint_seg_head = True
    model.train()
    optimizer = make_optimizer(
        model, fusion, encoder_3d, vision_peft, high_resolution, torch
    )
    for group in optimizer.param_groups:
        group["base_lr"] *= args.base_lr_factor
        group["lr"] *= args.base_lr_factor
    c0 = GeneralistSpatialResidualC0().cuda() if args.with_c0 else None
    if args.init_base_checkpoint:
        initial = torch.load(args.init_base_checkpoint, map_location="cpu", weights_only=False)
        if initial.get("format") != "medicalsam3d-generalist-joint-v1" or initial.get("model_name") != args.model:
            raise ValueError("initialization requires a matching base checkpoint")
        initial_config = initial.get("config", {})
        if initial_config.get("benchmark", LEGACY) != args.benchmark:
            raise ValueError("base checkpoint benchmark mismatch")
        if initial_config.get("benchmark_protocol_sha256") != benchmark_hash:
            raise ValueError("base checkpoint benchmark protocol mismatch")
        if initial_config.get("dataset") != args.dataset:
            raise ValueError("base checkpoint task-specific dataset mismatch")
        base_epoch = int(initial["epoch"])
        model.load_state_dict(initial["model"], strict=True)
        del initial
    if args.init_c0_checkpoint:
        paired = torch.load(args.init_c0_checkpoint, map_location="cpu", weights_only=False)
        if paired.get("format") != "medicalsam3d-generalist-c0-v1":
            raise ValueError("unexpected C0 initialization format")
        if Path(paired["base_checkpoint"]).resolve() != args.init_base_checkpoint.resolve():
            raise ValueError("C0 checkpoint was not trained on the selected base")
        if paired["base_model"] != args.model:
            raise ValueError("C0/base model mismatch")
        if int(paired["base_epoch"]) != base_epoch:
            raise ValueError("C0/base selected epoch mismatch")
        raise ValueError("C0 initialization for new benchmark requires a separately audited paired checkpoint")
        c0.load_state_dict(paired["model"], strict=True)
        del paired
    if c0 is not None and args.c0_loss_weight > 0:
        optimizer.add_param_group({
            "params": list(c0.parameters()), "lr": args.c0_learning_rate,
            "base_lr": args.c0_learning_rate, "weight_decay": 1e-2,
            "group_name": "c0",
        })
    elif c0 is not None:
        for parameter in c0.parameters():
            parameter.requires_grad_(False)
        c0.eval()
    schedulers = [
        InverseSquareRootParamScheduler(
            base_lr=float(group["base_lr"]),
            warmup_steps=args.lr_warmup_steps,
            cooldown_steps=args.lr_cooldown_steps,
            timescale=args.lr_timescale,
        )
        for group in optimizer.param_groups
    ]
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    criterion = build_native_stage_a_loss(torch, o2m_topk=2)
    model.matcher = criterion.matcher
    # Context is explicitly materialized below, so this runner must not add a
    # second halo.  D=1 still invokes its singleton halo=1 policy.
    runner = Sam3PatientVolumeRunner(
        model,
        fusion,
        slice_microbatch=args.core_depth + 2 * args.halo,
        head_slice_microbatch=1,
        encoder_halo=0,
        singleton_halo=1,
    )
    adaptation_ids = {
        id(parameter)
        for parameter in (
            *tuple(fusion.parameters()),
            *tuple(encoder_3d.parameters()),
            *tuple(vision_peft.parameters()),
            *tuple(high_resolution.parameters()),
        )
    }
    base_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in adaptation_ids
    ]
    adaptation_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) in adaptation_ids
    ]
    if c0 is not None and args.c0_loss_weight > 0:
        adaptation_parameters.extend(c0.parameters())
    dataset = open_dataset(args.benchmark, "train_dev")
    train_cases = {
        name: dataset.cases(dataset=name, split="train") for name in dataset_names
    }
    formal = args.balanced_updates_per_epoch is not None
    accumulation_steps = len(dataset_names) if formal else 1
    total_steps = (
        args.epochs * args.balanced_updates_per_epoch * len(dataset_names)
        if formal
        else args.steps
    )
    checkpoint_path = args.output_dir / "checkpoint_last.pt"
    start_step = 0
    optimizer_updates = 0
    if args.resume:
        if not checkpoint_path.exists():
            raise FileNotFoundError(checkpoint_path)
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        expected_format = ("medicalsam3d-generalist-joint-c0-v2" if args.with_c0
                           else "medicalsam3d-generalist-joint-v1")
        if saved.get("format") != expected_format:
            raise ValueError("unexpected generalist checkpoint format")
        if saved["model_name"] != args.model:
            raise ValueError("resume model mismatch")
        saved_config = saved.get("config", {})
        for key, expected in (
            ("planned_epochs", args.planned_epochs),
            ("balanced_updates_per_epoch", args.balanced_updates_per_epoch),
            ("lr_warmup_steps", args.lr_warmup_steps),
            ("lr_cooldown_steps", args.lr_cooldown_steps),
            ("lr_timescale", args.lr_timescale),
            ("amos_sampler", args.amos_sampler),
            ("amos_all_concepts", args.amos_all_concepts),
            ("lung_positive_window_prob", args.lung_positive_window_prob),
            ("amos_positive_window_prob", args.amos_positive_window_prob),
            ("safe_gradient_norm", args.safe_gradient_norm),
            ("min_train_resolution", args.min_train_resolution),
            ("base_lr_factor", args.base_lr_factor),
            ("c0_loss_weight", args.c0_loss_weight),
            ("benchmark", args.benchmark),
            ("dataset", args.dataset),
            ("benchmark_protocol_sha256", benchmark_hash),
        ):
            actual = saved_config.get(key, 480 if key == "min_train_resolution" else False if key == "safe_gradient_norm" else 0.0 if key.endswith("positive_window_prob") else LEGACY if key == "benchmark" else None)
            # A diagnostic replay may enable precise norm aggregation on an old
            # checkpoint. This cannot change weights, targets, loss or LR.
            if key == "safe_gradient_norm" and key not in saved_config:
                continue
            if actual != expected:
                raise ValueError(
                    f"resume {key} mismatch: {actual} != {expected}"
                )
        model.load_state_dict(saved["model"], strict=True)
        if c0 is not None:
            c0.load_state_dict(saved["c0"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        start_step = int(saved["next_step"])
        optimizer_updates = int(saved["optimizer_updates"])
        rng.setstate(saved["python_rng_state"])
        np.random.set_state(saved["numpy_rng_state"])
        torch.set_rng_state(saved["torch_rng_state"])
        torch.cuda.set_rng_state(saved["cuda_rng_state"])
    config = {
        "purpose": (
            "generalist_formal_training" if formal
            else "short_optimizer_calibration_not_reportable_result"
        ),
        "model": args.model,
        "benchmark": args.benchmark,
        "dataset": args.dataset,
        "benchmark_protocol_sha256": benchmark_hash,
        "medical_encoder": use_medical_encoder,
        "base_checkpoint": os.environ["MEDSPADA_SAM3_CHECKPOINT"],
        "init_base_checkpoint": str(args.init_base_checkpoint) if args.init_base_checkpoint else None,
        "init_c0_checkpoint": str(args.init_c0_checkpoint) if args.init_c0_checkpoint else None,
        "with_c0": args.with_c0,
        "c0_loss_weight": args.c0_loss_weight,
        "c0_frozen_control": c0 is not None and args.c0_loss_weight == 0,
        "base_lr_factor": args.base_lr_factor,
        "amos_sampler": args.amos_sampler,
        "amos_all_concepts": args.amos_all_concepts,
        "lung_positive_window_prob": args.lung_positive_window_prob,
        "amos_positive_window_prob": args.amos_positive_window_prob,
        "safe_gradient_norm": args.safe_gradient_norm,
        "query_fusion_arm": "F0",
        "mona_topology": (
            "serial" if args.model == "medicalsam3d_serial" else
            "parallel" if use_medical_encoder else "none"
        ),
        "hr_arm": "HR1" if use_hr1 else "HR0",
        "datasets": list(dataset_names),
        "dataset_sampling": (
            f"{len(dataset_names)}_dataset_equal_update_with_amos_target_fg_bg_1to1"
            if args.amos_sampler == "fg_bg_1to1" else
            f"{len(dataset_names)}_dataset_equal_update_target_positive_mixture"
            if args.lung_positive_window_prob or args.amos_positive_window_prob else
            f"{len(dataset_names)}_dataset_equal_update_uniform_window"
        ),
        "steps": total_steps,
        "formal": formal,
        "epochs": args.epochs,
        "planned_epochs": args.planned_epochs,
        "balanced_updates_per_epoch": args.balanced_updates_per_epoch,
        "gradient_accumulation_datasets": accumulation_steps,
        "core_depth": args.core_depth,
        "halo": args.halo,
        "head_slice_microbatch": 1,
        "max_concepts": args.max_concepts,
        "resolution": 1008,
        "min_train_resolution": args.min_train_resolution,
        "o2m_topk": 2,
        "native_sam3_loss_unchanged": True,
        "window_loss_normalized_by_valid_depth": True,
        "backward_policy": (
            "one backward per AMOS 5-concept chunk, average chunks; "
            f"one optimizer update after {len(dataset_names)} dataset-balanced windows"
            if args.amos_all_concepts else
            "one backward per dataset window; one optimizer update after "
            f"{len(dataset_names)} dataset-balanced windows"
        ),
        "dataset_loss_weighting": f"equal_dataset_1_over_{len(dataset_names)}",
        "singleton_policy": {
            "slice_microbatch": 1,
            "halo": 1,
            "depth_residual_enabled": False,
        },
        "learning_rates": {
            group["group_name"]: group["base_lr"] for group in optimizer.param_groups
        },
        "lr_warmup_steps": args.lr_warmup_steps,
        "lr_cooldown_steps": args.lr_cooldown_steps,
        "lr_timescale": args.lr_timescale,
        "lr_schedule": "sam3_inverse_square_root",
        "gradient_clipping_policy": "separate_base_0.1_adaptation_1.0",
        "image_encoder_base_frozen": use_medical_encoder,
        "image_encoder_adapters_trainable": use_medical_encoder,
        "image_encoder_fact_trainable": use_medical_encoder,
        "outside_image_encoder_trainable": True,
        "trainable_parameter_count": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ) + (sum(parameter.numel() for parameter in c0.parameters() if parameter.requires_grad) if c0 is not None else 0),
        "frozen_parameter_count": sum(
            parameter.numel()
            for parameter in model.parameters()
            if not parameter.requires_grad
        ),
        "activation_checkpointing": {
            "whole_vision_backbone": True,
            "whole_language_backbone": True,
            "segmentation_head": True,
        },
        "locked_test_access": False,
        "build_seconds": time.time() - build_started,
    }
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    records = []
    optimizer.zero_grad(set_to_none=True)
    for step in range(start_step, total_steps):
        if step % accumulation_steps == 0:
            if formal:
                planned_updates = (
                    args.planned_epochs * args.balanced_updates_per_epoch
                )
            else:
                planned_updates = max(1, total_steps)
            where = optimizer_updates / max(1, planned_updates)
            for group, scheduler in zip(optimizer.param_groups, schedulers):
                group["lr"] = scheduler(optimizer_updates, where)
        dataset_name = dataset_names[step % len(dataset_names)]
        load_started = time.time()
        balanced_amos = dataset_name == "AMOS22" and args.amos_sampler == "fg_bg_1to1"
        target_prompt = None
        target_positive = None
        selected_five = None
        for attempt in range(32 if balanced_amos else 1):
            case = rng.choice(train_cases[dataset_name])
            full_sample = dataset.load(case)
            scaffold = choose_concepts(full_sample, args.max_concepts, rng)
            if not balanced_amos:
                sample = full_sample if dataset_name == "AMOS22" and args.amos_all_concepts else scaffold
                probability = (args.amos_positive_window_prob if dataset_name == "AMOS22"
                               else args.lung_positive_window_prob if dataset_name == "MSD_Task06_Lung" else 0.0)
                start_override, target_prompt = positive_window_override(sample, args.core_depth, probability, rng)
                target_positive = True if start_override is not None else None
                break
            # Both 1:1 arms consume the same RNG calls and use the same 5
            # sampled prompts as the sampler scaffold. B1 alone expands the
            # supervised prompts to all 15 for this identical patient/core.
            target_local = rng.randrange(len(scaffold.concepts))
            target_concept = scaffold.concepts[target_local]
            target_index = full_sample.concepts.index(target_concept)
            target_positive = optimizer_updates % 2 == 0
            start_override = amos_fg_bg_start(
                full_sample, target_index, args.core_depth, target_positive, rng
            )
            if start_override is not None:
                sample = full_sample if args.amos_all_concepts else scaffold
                target_prompt = target_concept.prompt
                selected_five = scaffold.prompts
                break
        else:
            raise RuntimeError("could not sample requested AMOS foreground/background core")
        images, target_np, z_np, core_offset, valid_depth, context_indices = (
            choose_core_with_context(
                sample, args.core_depth, args.halo, rng, start_override
            )
        )
        load_seconds = time.time() - load_started
        categories = len(sample.concepts)
        first_count = min(categories, args.max_concepts)
        flat_target = target_np[:first_count].reshape(
            first_count * valid_depth, *target_np.shape[-2:]
        )
        with torch.no_grad():
            image_batch, target, transform = transform_native_window(
                images,
                flat_target,
                resolution=1008,
                min_resolution=args.min_train_resolution,
                rng=rng,
                torch=torch,
                v2=v2,
            )
        z_mm = torch.from_numpy(np.asarray(z_np)).to("cuda", dtype=torch.float32)
        torch.cuda.reset_peak_memory_stats()
        train_started = time.time()
        loss_total = 0.0
        supervised_slices = 0
        native_loss_total = 0.0
        c0_loss_total = 0.0
        core_end = core_offset + valid_depth
        if dataset_name == "AMOS22" and args.amos_all_concepts:
            chunks = [list(range(start, min(start + args.max_concepts, categories)))
                      for start in range(0, categories, args.max_concepts)]
        else:
            chunks = [list(range(categories))]
        transformed_first = target.reshape(first_count, valid_depth, *target.shape[-2:])
        for concept_indices in chunks:
            selected = sample.select_concepts(concept_indices)
            if concept_indices[-1] < first_count:
                transformed_chunk = transformed_first[concept_indices]
            else:
                with torch.no_grad():
                    raw = torch.from_numpy(target_np[concept_indices].reshape(
                        len(concept_indices) * valid_depth, *target_np.shape[-2:]
                    )).to("cuda", dtype=torch.float32)
                    scale = transform["train_resize"]
                    left, top = transform["pad_left"], transform["pad_top"]
                    raw = torch.nn.functional.interpolate(
                        raw[:, None], size=(scale, scale), mode="nearest"
                    ).squeeze(1)
                    raw = torch.nn.functional.pad(
                        raw, (left, 1008 - scale - left, top, 1008 - scale - top)
                    )
                    transformed_chunk = raw.reshape(
                        len(concept_indices), valid_depth, *raw.shape[-2:]
                    )
            local_window_loss = None
            q0_rows, residual_rows, score_rows, target_rows = [], [], [], []
            with torch.autocast("cuda", dtype=torch.bfloat16):
                encoding = runner.encode_patient(image_batch, selected.prompts)
                fused, _ = runner.fuse_patient(encoding, z_mm)
                output_iterator = runner.iter_final_outputs(encoding, fused)
                for start, end, output in output_iterator:
                    if end <= core_offset:
                        continue
                    if start >= core_end:
                        break
                    if end - start != 1:
                        raise RuntimeError("calibration requires one-slice head chunks")
                    relative = start - core_offset
                    local_target = transformed_chunk[:, relative]
                    find_target = build_native_find_target(local_target, torch)
                    model._compute_matching(output, model.back_convert(find_target))
                    slice_loss, _ = native_stage_a_loss(
                        output, find_target, criterion, model, torch
                    )
                    scaled_loss = slice_loss / float(valid_depth)
                    local_window_loss = (
                        scaled_loss if local_window_loss is None
                        else local_window_loss + scaled_loss
                    )
                    native_loss_total += float(scaled_loss.detach()) / len(chunks)
                    supervised_slices += 1
                    if c0 is not None and args.c0_loss_weight > 0:
                        q0, score, _ = selected_q0(output, runner.num_o2o, torch)
                        refined, residual = c0(q0.float(), score.float())
                        q0_rows.append(refined.float())
                        score_rows.append(score.float())
                        target_rows.append(local_target.float())
                        # C0's volume loss uses all core slices below.
                        residual_rows.append(residual.float())
                    del output, find_target, slice_loss, scaled_loss
                if c0 is not None and args.c0_loss_weight > 0:
                    refined = torch.stack(q0_rows, dim=1)
                    residual = torch.stack(residual_rows, dim=1)
                    scores = torch.stack(score_rows, dim=1)
                    c0_target = torch.stack(target_rows, dim=1)
                    c0_loss, _ = c0_window_loss(
                        refined, residual, scores, c0_target, 0.60, torch
                    )
                    c0_loss_total += float(c0_loss.detach()) / len(chunks)
                    local_window_loss = local_window_loss + args.c0_loss_weight * c0_loss
                    del refined, residual, scores, c0_target, c0_loss
            if local_window_loss is None:
                raise RuntimeError("window chunk produced no loss")
            scaler.scale(local_window_loss / (accumulation_steps * len(chunks))).backward()
            del local_window_loss, encoding, fused, transformed_chunk
        loss_total = native_loss_total + args.c0_loss_weight * c0_loss_total
        if supervised_slices != valid_depth * len(chunks):
            raise RuntimeError(
                f"supervised {supervised_slices} slices, expected {valid_depth * len(chunks)}"
            )
        update_after_step = (
            (step + 1) % accumulation_steps == 0 or step + 1 == total_steps
        )
        base_gradient_norm = math.nan
        adaptation_gradient_norm = math.nan
        if update_after_step:
            scaler.unscale_(optimizer)
            if args.safe_gradient_norm:
                from medspada.finite_gradient_clip import finite_clip
                named = dict(model.named_parameters())
                if c0 is not None:
                    named.update({"c0." + n: p for n, p in c0.named_parameters()})
                base_ids = {id(p) for p in base_parameters}
                adaptation_clip_ids = {id(p) for p in adaptation_parameters}
                base_gradient_norm, base_extreme = finite_clip(
                    [(n, p) for n, p in named.items() if id(p) in base_ids], .1, torch)
                adaptation_gradient_norm, adaptation_extreme = finite_clip(
                    [(n, p) for n, p in named.items() if id(p) in adaptation_clip_ids], 1., torch)
                if base_extreme or adaptation_extreme:
                    print(json.dumps({"stage": "extreme_finite_gradients", "step": step + 1,
                                      "base": base_extreme, "adaptation": adaptation_extreme}), flush=True)
            else:
                base_gradient_norm = float(torch.nn.utils.clip_grad_norm_(base_parameters, .1))
                adaptation_gradient_norm = float(torch.nn.utils.clip_grad_norm_(adaptation_parameters, 1.)
                                                 if adaptation_parameters else 0.)
            if not all(
                math.isfinite(value)
                for value in (base_gradient_norm, adaptation_gradient_norm)
            ):
                raise RuntimeError("non-finite separately clipped gradient")
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            optimizer_updates += 1
        record = {
            "step": step + 1,
            "dataset": dataset_name,
            "case_id": case.case_id,
            "concepts": sample.prompts,
            "categories": categories,
            "sampler": ("target_positive_mixture" if
                        (dataset_name == "AMOS22" and args.amos_positive_window_prob) or
                        (dataset_name == "MSD_Task06_Lung" and args.lung_positive_window_prob)
                        else args.amos_sampler if dataset_name == "AMOS22" else "uniform"),
            "window_has_foreground": bool(target_np.any()),
            "sampler_target_prompt": target_prompt,
            "sampler_target_positive": target_positive,
            "sampler_scaffold_prompts": selected_five,
            "concept_chunks": len(chunks),
            "native_loss": native_loss_total,
            "c0_loss": c0_loss_total,
            "patient_depth": sample.depth,
            "valid_core_depth": valid_depth,
            "context_depth": len(images),
            "context_indices": context_indices,
            "supervised_slices": supervised_slices,
            "loss": loss_total,
            "base_gradient_norm_before_clip": base_gradient_norm,
            "adaptation_gradient_norm_before_clip": adaptation_gradient_norm,
            "optimizer_update": optimizer_updates if update_after_step else None,
            "learning_rates": {
                group["group_name"]: float(group["lr"])
                for group in optimizer.param_groups
            },
            "load_seconds": load_seconds,
            "train_seconds": time.time() - train_started,
            "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30,
            "execution_policy": {
                "slice_microbatch": runner.last_execution_policy.slice_microbatch,
                "encoder_halo": runner.last_execution_policy.encoder_halo,
                "depth_residual_enabled": (
                    runner.last_execution_policy.depth_residual_enabled
                ),
            },
            "transform": transform,
        }
        records.append(record)
        with log_path.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)

        if formal and update_after_step:
            completed_epoch = (
                optimizer_updates % args.balanced_updates_per_epoch == 0
            )
            if completed_epoch:
                epoch = optimizer_updates // args.balanced_updates_per_epoch
                payload = {
                    "format": ("medicalsam3d-generalist-joint-c0-v2" if c0 is not None
                               else "medicalsam3d-generalist-joint-v1"),
                    "model_name": args.model,
                    "epoch": epoch,
                    "next_step": step + 1,
                    "optimizer_updates": optimizer_updates,
                    "model": model.state_dict(),
                    "c0": c0.state_dict() if c0 is not None else None,
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "python_rng_state": rng.getstate(),
                    "numpy_rng_state": np.random.get_state(),
                    "torch_rng_state": torch.get_rng_state(),
                    "cuda_rng_state": torch.cuda.get_rng_state(),
                    "config": config,
                }
                atomic_torch_save(payload, checkpoint_path, torch)
                print(
                    json.dumps(
                        {"stage": "epoch_checkpoint", "epoch": epoch,
                         "path": str(checkpoint_path),
                         "retention": "last_only_until_dev_promotes_best"}
                    ),
                    flush=True,
                )

    summary = {
        **config,
        "status": "passed",
        "median_train_seconds": (
            float(np.median([r["train_seconds"] for r in records]))
            if records else math.nan
        ),
        "max_peak_vram_gib": (
            max(r["peak_vram_gib"] for r in records) if records else math.nan
        ),
        "optimizer_updates": optimizer_updates,
        "records": records,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    main()
