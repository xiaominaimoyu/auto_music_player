import unittest
from unittest.mock import patch
import tempfile
from pathlib import Path
import json
import hashlib

import pytest
pytest.importorskip("torch")

from training.run_sequence_pilot import paired_work_comparison, timing_summary, seal_pilot


class SequenceComparisonTests(unittest.TestCase):
    def test_pairing_macro_and_selection_lock(self):
        base = {"sample_id_file_sha256": "same", "work_values": {"model:x": {"w1": 2, "w2": 10}}}
        candidate = {"sample_id_file_sha256": "same", "work_values": {"model:x": {"w1": 1, "w2": 8}}}
        result = paired_work_comparison([base], [candidate])["targets"]["model:x"]
        self.assertEqual(result["delta_candidate_minus_baseline"], -1.5)
        candidate["sample_id_file_sha256"] = "different"
        with self.assertRaises(ValueError):
            paired_work_comparison([base], [candidate])

    def test_no_timing_windows_is_not_zero_error(self):
        self.assertIsNone(timing_summary({})["model:fixed_score"]["bar_drift"]["16"]["sample_macro_mae_ms"])

    def test_parent_marker_and_child_tamper_gate(self):
        def digest(path):
            return hashlib.sha256(path.read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "reference"
            reference.mkdir()
            (reference / "summary.json").write_text("{}")
            for kind in ("runs", "audits"):
                child = root / kind / "a"
                child.mkdir(parents=True)
                artifact = child / "report.json"
                artifact.write_text(json.dumps({"timing_metrics": {"present": True}}))
                inventory = child / "integrity.json"
                inventory.write_text(json.dumps({"files": [{"path": "report.json", "size_bytes": artifact.stat().st_size,
                                                             "sha256": digest(artifact)}]}))
                (child / "_SUCCESS").write_text(digest(inventory))
            (root / "pilot_summary.json").write_text(json.dumps({"runs": [{"label": "a"}]}))
            with patch("training.tempo_reference.load_timing_reference", return_value={}):
                result = seal_pilot(root, reference)
                self.assertTrue(result["parent_integrity_verified"])
                self.assertEqual((root / "_SUCCESS").read_text().strip(), digest(root / "pilot_summary.json"))
                (root / "runs" / "a" / "report.json").write_text("tampered")
                with self.assertRaises(ValueError):
                    seal_pilot(root, reference)
