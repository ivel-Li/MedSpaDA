import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from medspada.data import ManifestDataset, configure_manifest, protocol_sha256, verify_report


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.spec = {"datasets": {"ExampleMRI": {"concepts": [
            {"name": "organ", "label_id": 1, "prompt": "organ"}]}}, "cases": [
            {"dataset": "ExampleMRI", "case_id": "case1", "patient_id": "subject1", "split": "dev", "path": "case.npz"}]}
        self.path = self.root / "manifest.json"
        self.path.write_text(json.dumps(self.spec))
        np.savez(self.root/"case.npz", images=np.zeros((1, 4, 4, 3), np.uint8),
                 label=np.zeros((1, 4, 4), np.uint8), z_mm=np.array([0.], np.float32))
        configure_manifest(self.path)

    def tearDown(self):
        os.environ.pop("MEDSPADA_MANIFEST", None)
        self.temp.cleanup()

    def test_singleton_and_negative_case_are_valid(self):
        data = ManifestDataset(); volume = data.load(data.cases()[0])
        self.assertEqual(volume.depth, 1)
        self.assertEqual(volume.target.shape, (1, 1, 4, 4))
        self.assertEqual(volume.target.sum(), 0)

    def test_patient_leakage_is_rejected(self):
        self.spec["cases"].append({**self.spec["cases"][0], "case_id": "case2", "split": "train"})
        self.path.write_text(json.dumps(self.spec))
        with self.assertRaisesRegex(ValueError, "across splits"):
            ManifestDataset()

    def test_duplicate_case_is_rejected(self):
        self.spec["cases"].append(self.spec["cases"][0])
        self.path.write_text(json.dumps(self.spec))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            ManifestDataset()

    def test_report_requires_exact_case_cohort(self):
        report = {"status": "passed", "split": "dev", "benchmark_protocol_sha256": protocol_sha256(),
                  "cases": [{"dataset": "ExampleMRI", "case_id": "case1", "patient_macro_dice": 0.5}],
                  "by_dataset": {"ExampleMRI": {"cases": 1, "patient_macro_dice": 0.5}}}
        p = self.root / "report.json"; p.write_text(json.dumps(report)); verify_report(p, "dev")
        report["cases"][0]["case_id"] = "different-case"
        p.write_text(json.dumps(report))
        with self.assertRaises(ValueError): verify_report(p, "dev")


if __name__ == "__main__":
    unittest.main()
