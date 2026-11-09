"""Patient-balanced base training followed by joint base/VSR refinement."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from medspada.cli import ROOT, add_common, configure
from medspada.data import DATASETS, LEGACY, ManifestDataset, verify_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common(parser)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true", help="ten real training windows; no test access or weight promotion")
    args = parser.parse_args()
    config = configure(args)
    data = ManifestDataset()
    for task in DATASETS[LEGACY]:
        for split in ("train", "dev", "test"):
            if not data.cases(dataset=task, split=split):
                raise ValueError(f"missing {task}/{split} cohort")
    models = {"medspada_parallel": "medicalsam3d", "medspada_serial": "medicalsam3d_serial",
              "sam3_ft": "sam3_ft", "without_A": "sam3_ft_hr1", "without_VSR": "medicalsam3d"}
    model = models[config["model"]]
    args.output_dir = args.output_dir.resolve()
    common = ["--benchmark", LEGACY, "--model", model,
              "--lung-positive-window-prob", str(config["lung_positive_window_prob"]),
              "--amos-positive-window-prob", str(config["amos_positive_window_prob"]),
              "--min-train-resolution", str(config["min_train_resolution"])]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    def run(command):
        subprocess.run([sys.executable, *map(str, command)], cwd=ROOT, env=env, check=True)

    if args.smoke:
        run(["scripts/train_epoch.py", *common, "--steps", "10", "--output-dir", args.output_dir])
        return
    baseline = config["model"] == "sam3_ft"
    with_vsr = config["model"] not in ("sam3_ft", "without_VSR")
    phases = [("train30", 30, False)] if baseline else [
        ("base20", int(config["base_epochs"]), False),
        ("joint10", int(config["joint_epochs"]), with_vsr)]
    for name, epochs, joint in phases:
        output = args.output_dir / name
        command = ["scripts/train_stage.py", *common, "--gpu", args.gpu,
                   "--balanced-updates-per-epoch", config["windows_per_dataset_per_epoch"],
                   "--epochs", epochs, "--output-dir", output]
        if name == "joint10":
            command += ["--init-base-checkpoint", args.output_dir / "base20/checkpoint_best.pt",
                        "--base-lr-factor", config["joint_base_lr_factor"]]
        if joint:
            command.append("--joint-c0")
        run(command)
    last_phase, _, joint = phases[-1]
    checkpoint = args.output_dir / last_phase / "checkpoint_best.pt"
    report = args.output_dir / "test_best.json"
    evaluator = "scripts/evaluate_vsr.py" if joint else "scripts/evaluate_base.py"
    if report.exists():
        saved = verify_report(report, "test")
        if Path(saved["checkpoint"]).resolve() != checkpoint:
            raise ValueError("existing test report belongs to another checkpoint")
    else:
        run([evaluator, "--benchmark", LEGACY, "--checkpoint", checkpoint,
             "--split", "test", "--output", report])
        verify_report(report, "test")
    for name, _, _ in phases:
        (args.output_dir / name / "checkpoint_last.pt").unlink(missing_ok=True)
    print(json.dumps({"status": "tested", "checkpoint": str(checkpoint), "test_report": str(report)}))


if __name__ == "__main__":
    main()
