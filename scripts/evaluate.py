"""Evaluate a dev-selected model using case-volume Dice and dataset macro means."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

from medspada.cli import ROOT, add_common, configure
from medspada.data import verify_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common(parser)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("dev", "test"), default="test")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    configure(args)
    import torch
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True)
    vsr = saved.get("c0") is not None
    del saved
    env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    subprocess.run([sys.executable, str(ROOT / "scripts" / ("evaluate_vsr.py" if vsr else "evaluate_base.py")),
                    "--checkpoint", str(args.checkpoint.resolve()), "--split", args.split,
                    "--output", str(args.output.resolve())], cwd=ROOT, env=env, check=True)
    verify_report(args.output, args.split)


if __name__ == "__main__":
    main()
