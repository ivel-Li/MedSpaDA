"""Validate identities, patient splits, labels, RGB arrays and slice coordinates."""
import argparse
import json
from pathlib import Path

from medspada.data import DATASETS, LEGACY, ManifestDataset, configure_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--check-arrays", action="store_true")
    args = parser.parse_args()
    configure_manifest(args.manifest)
    data = ManifestDataset()
    counts = {}
    for task in DATASETS[LEGACY]:
        counts[task] = {}
        for split in ("train", "dev", "test"):
            cases = data.cases(dataset=task, split=split)
            if not cases: raise ValueError(f"empty {task}/{split}")
            counts[task][split] = len(cases)
            for case in cases:
                if not case.path.is_file(): raise FileNotFoundError(case.path)
                if args.check_arrays: data.load(case)
    print(json.dumps({"status": "passed", "by_dataset": counts}, indent=2))


if __name__ == "__main__":
    main()
