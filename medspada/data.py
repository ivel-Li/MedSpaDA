"""Portable NPZ volume manifests with patient-disjoint train/dev/test splits."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
from PIL import Image

LEGACY = "manifest"
DATASETS = {LEGACY: ("ExampleMRI",)}
_manifest = None


def configure_manifest(path):
    global _manifest
    _manifest = Path(path).expanduser().resolve()
    spec = json.loads(_manifest.read_text())
    if not spec.get("datasets") or not spec.get("cases"):
        raise ValueError("manifest needs datasets and cases")
    DATASETS[LEGACY] = tuple(spec["datasets"])
    os.environ["MEDSPADA_MANIFEST"] = str(_manifest)
    return _manifest


def protocol_sha256(benchmark=LEGACY):
    if benchmark != LEGACY or _manifest is None:
        raise ValueError("configure a manifest before training/evaluation")
    return hashlib.sha256(_manifest.read_bytes()).hexdigest()


@dataclass(frozen=True)
class Concept:
    name: str
    label_id: int
    prompt: str


@dataclass(frozen=True)
class Case:
    dataset: str
    case_id: str
    patient_id: str
    split: str
    path: Path


@dataclass
class Volume:
    images: list
    target: np.ndarray
    z_mm: np.ndarray
    concepts: tuple

    @property
    def depth(self):
        return len(self.images)

    @property
    def prompts(self):
        return [c.prompt for c in self.concepts]

    def select_concepts(self, indices):
        return Volume(self.images, self.target[indices], self.z_mm,
                      tuple(self.concepts[i] for i in indices))


class ManifestDataset:
    def __init__(self):
        if _manifest is None:
            raise ValueError("no data manifest configured")
        self.spec = json.loads(_manifest.read_text())
        self.concepts = {}
        for name, info in self.spec["datasets"].items():
            concepts = tuple(Concept(c["name"], int(c["label_id"]), c["prompt"])
                             for c in info["concepts"])
            ids = [c.label_id for c in concepts]
            if not ids or min(ids) < 1 or len(ids) != len(set(ids)):
                raise ValueError(f"foreground labels must be unique and positive: {name}")
            self.concepts[name] = concepts
        self.records = []
        patients, keys = {}, set()
        for row in self.spec["cases"]:
            name, cid, pid, split = (row[k] for k in ("dataset", "case_id", "patient_id", "split"))
            if name not in self.concepts or split not in ("train", "dev", "test") or not cid or not pid:
                raise ValueError("unknown task, missing identity, or invalid split")
            if (name, cid) in keys:
                raise ValueError(f"duplicate case: {name}/{cid}")
            keys.add((name, cid))
            if patients.setdefault((name, pid), split) != split:
                raise ValueError(f"patient occurs across splits: {name}/{pid}")
            self.records.append(Case(name, cid, pid, split, (_manifest.parent / row["path"]).resolve()))

    def cases(self, *, dataset=None, split=None):
        return [c for c in self.records if (dataset is None or c.dataset == dataset)
                and (split is None or c.split == split)]

    def load(self, case):
        with np.load(case.path, allow_pickle=False) as saved:
            images = saved["images"]
            label = saved["label"]
            z_mm = saved["z_mm"].astype(np.float32)
        if images.dtype != np.uint8 or images.ndim != 4 or images.shape[-1] != 3:
            raise ValueError("images must be uint8 [D,H,W,3], already normalized to RGB")
        if label.shape != images.shape[:3] or not np.issubdtype(label.dtype, np.integer):
            raise ValueError("label must be integer [D,H,W], on the same grid as images")
        if z_mm.shape != (len(images),) or not np.isfinite(z_mm).all() or (np.diff(z_mm) <= 0).any():
            raise ValueError("z_mm must be finite and strictly increasing")
        concepts = self.concepts[case.dataset]
        allowed = {0, *(c.label_id for c in concepts)}
        if not set(np.unique(label)).issubset(allowed):
            raise ValueError("volume has labels absent from its task ontology")
        target = np.stack([label == c.label_id for c in concepts]).astype(np.float32)
        return Volume([Image.fromarray(image) for image in images], target, z_mm, concepts)


def open_dataset(benchmark, split):
    if benchmark != LEGACY or split not in ("train_dev", "dev", "test"):
        raise ValueError((benchmark, split))
    return ManifestDataset()


def verify_report(path, split, dataset_names=None):
    report = json.loads(Path(path).read_text())
    tasks = tuple(dataset_names or DATASETS[LEGACY])
    expected = {(c.dataset, c.case_id) for c in ManifestDataset().cases(split=split) if c.dataset in tasks}
    actual = [(r["dataset"], r["case_id"]) for r in report["cases"]]
    if (report.get("status") != "passed" or report.get("split") != split
            or report.get("benchmark_protocol_sha256") != protocol_sha256()
            or len(actual) != len(set(actual)) or set(actual) != expected
            or set(report["by_dataset"]) != set(tasks)):
        raise ValueError("incomplete cohort, changed manifest, or wrong evaluation split")
    for task in tasks:
        scores = [r.get("c0_patient_macro_dice", r.get("patient_macro_dice"))
                  for r in report["cases"] if r["dataset"] == task]
        if not scores or any(v is None or not math.isfinite(v) or not 0 <= v <= 1 for v in scores):
            raise ValueError("invalid case Dice")
        aggregate = report["by_dataset"][task]
        if aggregate["cases"] != len(scores) or abs(sum(scores)/len(scores)-aggregate["patient_macro_dice"]) > 1e-8:
            raise ValueError("case count or Dice aggregation mismatch")
    return report


if os.environ.get("MEDSPADA_MANIFEST"):
    configure_manifest(os.environ["MEDSPADA_MANIFEST"])
