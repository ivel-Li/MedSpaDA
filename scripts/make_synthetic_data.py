"""Generate tiny synthetic arrays for loader checks; no medical images."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    cases = []
    for split in ("train", "dev", "test"):
        label = np.zeros((12, 32, 32), dtype=np.uint8)
        label[3:9, 8:24, 8:24] = 1
        images = rng.integers(0, 100, (*label.shape, 3), dtype=np.uint8)
        images[label > 0] = 220
        np.savez_compressed(args.output_dir / f"{split}.npz", images=images,
                            label=label, z_mm=np.arange(12, dtype=np.float32)*3)
        cases.append(dict(dataset="ExampleMRI", case_id=f"synthetic-{split}",
                          patient_id=f"synthetic-patient-{split}", split=split, path=f"{split}.npz"))
    manifest = dict(datasets={"ExampleMRI": {"concepts": [
        {"name": "example structure", "label_id": 1, "prompt": "example structure"}]}}, cases=cases)
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")


if __name__ == "__main__":
    main()
