"""Shared public CLI configuration; paths are supplied by the caller."""
import os
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]


def configure(args):
    from medspada.data import configure_manifest
    config = yaml.safe_load(args.config.read_text())
    manifest = configure_manifest(args.manifest)
    checkpoint = args.sam3_checkpoint.expanduser().resolve()
    repo = args.sam3_repo.expanduser().resolve()
    if not checkpoint.is_file() or not (repo / "sam3/assets/bpe_simple_vocab_16e6.txt.gz").is_file():
        raise FileNotFoundError("provide a local official SAM3 checkpoint and the pinned SAM3 source checkout")
    os.environ.update(MEDSPADA_SAM3_REPO=str(repo), MEDSPADA_SAM3_CHECKPOINT=str(checkpoint),
                      MEDSPADA_MANIFEST=str(manifest), PYTHONUNBUFFERED="1",
                      PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    os.environ["PYTHONPATH"] = os.pathsep.join((str(ROOT), str(repo), os.environ.get("PYTHONPATH", "")))
    sys.path.insert(0, str(repo))
    return config


def add_common(parser):
    parser.add_argument("--config", type=Path, default=ROOT / "configs/medspada.yaml")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--sam3-repo", type=Path, required=True)
    parser.add_argument("--sam3-checkpoint", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
