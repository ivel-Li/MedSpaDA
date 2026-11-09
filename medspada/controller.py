#!/usr/bin/env python3
"""Run one generalist SAM3 arm with train->dev validation every epoch.

The controller is restart-safe.  It retains only ``checkpoint_last.pt`` and a
hard-linked ``checkpoint_best.pt`` and never reads the locked test split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PYTHON = Path(sys.executable)
from medspada.data import DATASETS, LEGACY, protocol_sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=("medicalsam3d", "medicalsam3d_serial", "sam3_ft", "sam3_ft_hr1"),
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--training-entry", type=Path, default=Path("scripts/train_epoch.py"))
    parser.add_argument("--benchmark", choices=tuple(DATASETS), default=LEGACY)
    parser.add_argument("--dataset", choices=DATASETS[LEGACY],
                        default=None, help="Train and validate one task only")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument(
        "--stop-after-epoch",
        type=int,
        default=None,
        help="Stop after this epoch while preserving the original --epochs LR horizon",
    )
    parser.add_argument("--balanced-updates-per-epoch", type=int, default=64)
    parser.add_argument("--min-train-resolution", type=int, default=480)
    parser.add_argument("--joint-c0", action="store_true")
    parser.add_argument("--init-base-checkpoint", type=Path)
    parser.add_argument("--init-c0-checkpoint", type=Path)
    parser.add_argument("--base-lr-factor", type=float, default=1.0)
    parser.add_argument("--c0-loss-weight", type=float, default=1.0)
    parser.add_argument("--amos-sampler", choices=("uniform", "fg_bg_1to1"), default="uniform")
    parser.add_argument("--amos-all-concepts", action="store_true")
    parser.add_argument("--lung-positive-window-prob", type=float, default=0.0)
    parser.add_argument("--amos-positive-window-prob", type=float, default=0.0)
    parser.add_argument("--safe-gradient-norm", action="store_true")
    parser.add_argument(
        "--max-dev-cases-per-dataset",
        type=int,
        default=None,
        help="smoke-only cap; omit for every formal run",
    )
    return parser.parse_args()


def run_logged(command: list[str], log_path: Path, env: dict[str, str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as handle:
        handle.write("\n$ " + " ".join(command) + "\n")
        handle.flush()
        subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=True,
        )


def validation_score(path: Path, dataset_names: tuple[str, ...]) -> tuple[float, dict[str, float]]:
    from medspada.data import verify_report
    verify_report(path, "dev", dataset_names)
    summary = json.loads(path.read_text())
    if summary.get("status") != "passed" or summary.get("split") != "dev":
        raise RuntimeError(f"invalid dev summary: {path}")
    values = {
        name: float(summary["by_dataset"][name]["patient_macro_dice"])
        for name in dataset_names
    }
    return sum(values.values()) / len(values), values


def promote_best(last: Path, best: Path) -> None:
    temporary = best.with_suffix(best.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    os.link(last, temporary)
    temporary.replace(best)


def write_json_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if min(args.epochs, args.balanced_updates_per_epoch) < 1:
        raise ValueError("epochs and balanced updates must be positive")
    if args.stop_after_epoch is not None and not (1 <= args.stop_after_epoch <= args.epochs):
        raise ValueError("stop-after-epoch must be within the planned training horizon")
    if args.joint_c0 and args.model not in ("medicalsam3d", "medicalsam3d_serial", "sam3_ft_hr1"):
        raise ValueError("joint VSR requires an explicitly supported fixed-HR1 architecture")
    if args.dataset is not None and args.dataset not in DATASETS[args.benchmark]:
        raise ValueError(f"{args.dataset} is not in {args.benchmark}")
    dataset_names = (args.dataset,) if args.dataset else DATASETS[args.benchmark]
    benchmark_hash = protocol_sha256(args.benchmark)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_last = args.output_dir / "checkpoint_last.pt"
    checkpoint_best = args.output_dir / "checkpoint_best.pt"
    state_path = args.output_dir / "controller_state.json"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    state = {
        "status": "running",
        "model": args.model,
        "benchmark": args.benchmark,
        "dataset": args.dataset,
        "benchmark_protocol_sha256": benchmark_hash,
        "gpu": args.gpu,
        "planned_epochs": args.epochs,
        "balanced_updates_per_epoch": args.balanced_updates_per_epoch,
        "min_train_resolution": args.min_train_resolution,
        "windows_per_epoch": args.balanced_updates_per_epoch * len(dataset_names),
        "best_epoch": None,
        "best_dev_equal_dataset_patient_macro_dice": None,
        "epochs": [],
        "locked_test_access": False,
        "joint_c0": args.joint_c0,
        "init_base_checkpoint": str(args.init_base_checkpoint) if args.init_base_checkpoint else None,
        "init_c0_checkpoint": str(args.init_c0_checkpoint) if args.init_c0_checkpoint else None,
        "base_lr_factor": args.base_lr_factor,
        "c0_loss_weight": args.c0_loss_weight,
        "amos_sampler": args.amos_sampler,
        "amos_all_concepts": args.amos_all_concepts,
        "lung_positive_window_prob": args.lung_positive_window_prob,
        "amos_positive_window_prob": args.amos_positive_window_prob,
        "safe_gradient_norm": args.safe_gradient_norm,
        "training_entry": str(args.training_entry),
        "expert_optimization_sha256": None,
        "started_unix": time.time(),
    }
    if state_path.exists():
        previous = json.loads(state_path.read_text())
        for key in (
            "model", "planned_epochs", "balanced_updates_per_epoch", "joint_c0",
            "benchmark", "benchmark_protocol_sha256",
            "dataset",
            "init_base_checkpoint", "init_c0_checkpoint", "base_lr_factor",
            "c0_loss_weight", "amos_sampler", "amos_all_concepts",
            "lung_positive_window_prob", "amos_positive_window_prob",
            "safe_gradient_norm",
            "training_entry", "expert_optimization_sha256",
            "min_train_resolution",
        ):
            prior = previous.get(key, 480 if key == "min_train_resolution" else "scripts/train_epoch.py" if key == "training_entry" else False if key == "safe_gradient_norm" else 0.0 if key.endswith("positive_window_prob") else LEGACY if key == "benchmark" else None)
            if prior != state[key]:
                raise ValueError(f"controller resume mismatch for {key}")
        state = previous
        state["status"] = "running"

    target_epoch = args.stop_after_epoch or args.epochs
    if any(int(row["epoch"]) > target_epoch for row in state.get("epochs", [])):
        raise ValueError("existing run has already passed the requested stop epoch")
    state["stop_after_epoch"] = target_epoch
    completed = {int(row["epoch"]) for row in state.get("epochs", [])}
    best_score = state.get("best_dev_equal_dataset_patient_macro_dice")
    best_score = float(best_score) if best_score is not None else float("-inf")

    for epoch in range(1, target_epoch + 1):
        if epoch in completed:
            continue
        train_command = [
            str(PYTHON),
            str(args.training_entry),
            "--model", args.model,
            "--benchmark", args.benchmark,
            *(["--dataset", args.dataset] if args.dataset else []),
            "--epochs", str(epoch),
            "--planned-epochs", str(args.epochs),
            "--balanced-updates-per-epoch",
            str(args.balanced_updates_per_epoch),
            "--output-dir", str(args.output_dir),
            "--min-train-resolution", str(args.min_train_resolution),
        ]
        train_command.extend(["--base-lr-factor", str(args.base_lr_factor)])
        if args.safe_gradient_norm:
            train_command.append("--safe-gradient-norm")
        train_command.extend(["--lung-positive-window-prob", str(args.lung_positive_window_prob),
                              "--amos-positive-window-prob", str(args.amos_positive_window_prob)])
        if args.init_base_checkpoint:
            train_command.extend(["--init-base-checkpoint", str(args.init_base_checkpoint)])
        if args.joint_c0:
            train_command.append("--with-c0")
            train_command.extend(["--c0-loss-weight", str(args.c0_loss_weight)])
            train_command.extend(["--amos-sampler", args.amos_sampler])
            if args.amos_all_concepts:
                train_command.append("--amos-all-concepts")
            if args.init_c0_checkpoint:
                train_command.extend(["--init-c0-checkpoint", str(args.init_c0_checkpoint)])
        if checkpoint_last.exists():
            train_command.append("--resume")
        epoch_started = time.time()
        run_logged(
            train_command,
            args.output_dir / f"train_epoch_{epoch:03d}.log",
            env,
        )
        dev_path = args.output_dir / f"dev_epoch_{epoch:03d}.json"
        eval_command = [
            str(PYTHON),
            ("scripts/evaluate_vsr.py" if args.joint_c0
             else "scripts/evaluate_base.py"),
            "--checkpoint", str(checkpoint_last),
            "--output", str(dev_path),
            "--benchmark", args.benchmark,
            *(["--dataset", args.dataset] if args.dataset else []),
        ]
        if args.max_dev_cases_per_dataset is not None:
            eval_command.extend([
                "--max-cases-per-dataset",
                str(args.max_dev_cases_per_dataset),
            ])
        run_logged(
            eval_command,
            args.output_dir / f"dev_epoch_{epoch:03d}.log",
            env,
        )
        score, per_dataset = validation_score(dev_path, dataset_names)
        improved = score > best_score
        if improved:
            promote_best(checkpoint_last, checkpoint_best)
            best_score = score
            state["best_epoch"] = epoch
            state["best_dev_equal_dataset_patient_macro_dice"] = score
            write_json_atomic(
                args.output_dir / "best_checkpoint.json",
                {
                    "epoch": epoch,
                    "selection_metric": "dev_equal_dataset_patient_macro_dice",
                    "benchmark": args.benchmark,
                    "benchmark_protocol_sha256": benchmark_hash,
                    "value": score,
                    "by_dataset": per_dataset,
                    "checkpoint": str(checkpoint_best.resolve()),
                },
            )
        row = {
            "epoch": epoch,
            "dev_equal_dataset_patient_macro_dice": score,
            "by_dataset": per_dataset,
            "promoted_best": improved,
            "elapsed_seconds": time.time() - epoch_started,
        }
        state.setdefault("epochs", []).append(row)
        state["last_completed_epoch"] = epoch
        state["updated_unix"] = time.time()
        write_json_atomic(state_path, state)
        print(json.dumps(row), flush=True)

    state["status"] = "complete"
    state["completed_unix"] = time.time()
    write_json_atomic(state_path, state)
    print(json.dumps({
        "status": "complete",
        "model": args.model,
        "best_epoch": state["best_epoch"],
        "best_score": state["best_dev_equal_dataset_patient_macro_dice"],
    }), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        # The per-command logs contain the traceback from a failed child.  The
        # controller itself exits non-zero so a parent queue will not continue.
        print(json.dumps({
            "status": "failed",
            "error": f"{type(error).__name__}: {error}",
        }), file=sys.stderr, flush=True)
        raise
