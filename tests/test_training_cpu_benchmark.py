import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
torch = pytest.importorskip("torch")

from training.benchmark_sequence_inference import benchmark
from training.train_expression import _sha256
from training.train_sequence_expression import write_integrity


def test_benchmark_passes_zero_labels_and_no_oracle_geometry(tmp_path: Path):
    audit, run, output = tmp_path / "audit", tmp_path / "run", tmp_path / "out"
    audit.mkdir()
    run.mkdir()
    checkpoint = run / "checkpoint_best.pt"
    checkpoint.write_bytes(b"local-test-artifact")
    arrays = {"note_features": np.zeros((3, 17), np.float32), "chord_features": np.zeros((3, 13), np.float32),
              "note_to_chord": np.arange(3)}
    np.savez_compressed(audit / "score.npz", **arrays)
    (audit / "report.json").write_text(json.dumps({"checkpoint_sha256": _sha256(checkpoint),
                                                  "prediction_files": {"sample": "score.npz"}}))
    write_integrity(audit)
    write_integrity(run)
    def prediction(model, sample, **kwargs):
        assert kwargs["device"] == "cpu"
        assert not sample.note_targets.any() and not sample.chord_targets.any()
        assert not sample.note_masks.any() and not sample.chord_masks.any()
        assert sample.score_ioi_seconds is None
        return np.zeros((3, 6)), np.zeros((3, 2))
    threads = torch.get_num_threads()
    try:
        with patch("training.benchmark_sequence_inference.load_model", return_value=(object(), {"architecture": "gru"})), \
             patch("training.benchmark_sequence_inference.predict_sample", side_effect=prediction):
            result = benchmark(audit=audit, checkpoint=checkpoint, output=output, cpu_threads=2)
    finally:
        torch.set_num_threads(threads)
    assert result["bitwise_repeat_passed"]
    assert not result["actual_game_input_sent"] and not result["fallback_interrupt_timeout_verified"]
