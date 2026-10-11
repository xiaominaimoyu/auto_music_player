import json

import numpy as np

from training.quality_mask import apply_quality_mask, load_quality_mask
from training.review_quality_mask import TARGETS, _default_mask, _make_mask


def _row() -> dict[str, str]:
    return {"id": "sample-1", "performance_dataset": "TEST", "split": "validation"}


def _review(**overrides: object) -> dict[str, object]:
    data: dict[str, object] = {
        "refined_recall": 1.0,
        "interpolation_rate": 0.0,
        "pedal_tail_after_last_note_s": 0.0,
        "target_stats": {"log_key_duration_ratio": {"median": 0.0}},
        "diagnostics": {
            "micro_offset_abs_p95_ms": 0.0,
            "onset_delta_abs_p95_ms": 0.0,
        },
    }
    data.update(overrides)
    return data


def test_unreviewed_samples_keep_all_targets() -> None:
    decision = _default_mask(_row())
    assert decision["reviewed"] is False
    assert decision["context_keep"] is True
    assert decision["status"] == "keep"
    assert all(decision["target_mask"][target] for target in TARGETS)
    assert all(decision["target_weight"][target] == 1.0 for target in TARGETS)


def test_review_mask_disables_only_targets_with_concrete_evidence() -> None:
    decision = _make_mask(
        _row(),
        _review(
            refined_recall=0.88,
            interpolation_rate=0.12,
            pedal_tail_after_last_note_s=3.0,
            target_stats={"log_key_duration_ratio": {"median": -1.2}},
            diagnostics={
                "micro_offset_abs_p95_ms": 55.0,
                "onset_delta_abs_p95_ms": 2_100.0,
            },
        ),
    )
    assert decision["status"] == "mask_targets"
    assert decision["context_keep"] is True
    assert decision["target_mask"] == {
        "micro_offset_ms": False,
        "log_key_duration_ratio": False,
        "velocity_0_1": True,
        "velocity_robust_z": True,
        "pedal_at_onset": False,
        "pedal_coverage": False,
        "onset_delta_ms": False,
        "log_ioi_ratio": True,
    }
    assert all(value == 0.5 for value in decision["target_weight"].values())
    assert len(decision["findings"]) == 5


def test_quality_sidecar_applies_after_native_masks(tmp_path) -> None:
    decision = _make_mask(
        _row(),
        _review(refined_recall=0.88, interpolation_rate=0.12),
        policy_version="quality-mask-v2-calibrated",
    )
    path = tmp_path / "quality_mask.jsonl"
    path.write_text(json.dumps(decision) + "\n", encoding="utf-8")
    loaded = load_quality_mask(path)
    arrays = {
        "sample_ids": np.asarray(["sample-1", "sample-2"]),
        "sample_note_offsets": np.asarray([0, 2, 3], dtype=np.int64),
        "note_masks": np.asarray([[True] * 6, [False] + [True] * 5, [True] * 6]),
    }
    masks, weights = apply_quality_mask(
        arrays, kind="note", decisions=loaded.decisions
    )
    assert masks[0].tolist() == [True] * 6
    assert masks[1].tolist() == [False] + [True] * 5
    assert weights[0].tolist() == [0.5] * 6
    assert weights[2].tolist() == [1.0] * 6


def test_v3_low_alignment_only_weights_timing_targets() -> None:
    decision = _make_mask(
        _row(),
        _review(refined_recall=0.88, interpolation_rate=0.12),
        policy_version="quality-mask-v3-target-specific",
    )
    assert decision["status"] == "downweight"
    assert decision["target_mask"] == {target: True for target in TARGETS}
    assert decision["target_weight"] == {
        "micro_offset_ms": 0.5,
        "log_key_duration_ratio": 0.5,
        "velocity_0_1": 1.0,
        "velocity_robust_z": 1.0,
        "pedal_at_onset": 1.0,
        "pedal_coverage": 1.0,
        "onset_delta_ms": 0.5,
        "log_ioi_ratio": 0.5,
    }


def test_v3_pedal_tail_remains_independent() -> None:
    decision = _make_mask(
        _row(),
        _review(
            refined_recall=0.88,
            interpolation_rate=0.12,
            pedal_tail_after_last_note_s=3.0,
        ),
        policy_version="quality-mask-v3-target-specific",
    )
    assert decision["target_weight"]["pedal_at_onset"] == 0.5
    assert decision["target_weight"]["pedal_coverage"] == 0.5
    assert decision["target_weight"]["velocity_0_1"] == 1.0
