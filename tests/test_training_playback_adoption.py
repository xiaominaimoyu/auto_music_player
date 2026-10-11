import json
from pathlib import Path

import pytest

from training.playback_adoption import POLICY, adoption_decision, compare_work_metrics, verify_inventory
from training.compare_quality_multiseed import _sha256


def reports(value):
    return [{"seed": seed, "selection_sha256": "locked", "source_by_work": {f"w{i}": "s" for i in range(6)},
             "work_values": {name: {f"w{i}": value for i in range(6)}
                             for name in POLICY["primary_metrics"] + POLICY["noninferiority_metrics"]}}
            for seed in (1, 2, 3)]


def test_numeric_pass_does_not_enable_playback_or_production():
    comparison = compare_work_metrics(reports(10), reports(8))
    result = adoption_decision({name: comparison for name in POLICY["required_baselines"]}, safety_passed=True, expected_sources=["s"])
    assert result["numerical_gate_passed"]
    assert not result["experimental_entry_authorized"]
    assert not result["production_approved"]


def test_missing_baseline_and_small_source_cannot_pass():
    assert not adoption_decision({}, safety_passed=True, expected_sources=[])["numerical_gate_passed"]
    comparison = compare_work_metrics(reports(10), reports(8))
    assert not adoption_decision({"identity": comparison, "rules": comparison}, safety_passed=True,
                                expected_sources=["unmeasured"])["numerical_gate_passed"]


def test_missing_windows_are_not_zero_and_seed_selection_locked():
    base, candidate = reports(10), reports(8)
    for row in base + candidate:
        row["work_values"][POLICY["noninferiority_metrics"][-1]] = {}
    comparison = compare_work_metrics(base, candidate)
    assert comparison["metrics"][POLICY["noninferiority_metrics"][-1]]["candidate_mean"] is None
    assert not adoption_decision({"identity": comparison, "rules": comparison}, safety_passed=True,
                                expected_sources=["s"])["numerical_gate_passed"]
    candidate[0]["seed"] = 5
    with pytest.raises(ValueError, match="pairing"):
        compare_work_metrics(base, candidate)


def test_no_silent_work_intersection():
    base, candidate = reports(10), reports(8)
    candidate[0]["work_values"][POLICY["primary_metrics"][0]].pop("w0")
    with pytest.raises(ValueError, match="coverage"):
        compare_work_metrics(base, candidate)


def test_duplicate_seeds_and_omitted_sources_cannot_pass():
    base, candidate = reports(10), reports(8)
    with pytest.raises(ValueError, match="distinct"):
        compare_work_metrics([base[0]] * 3, [candidate[0]] * 3)
    comparison = compare_work_metrics(base, candidate)
    result = adoption_decision({name: comparison for name in POLICY["required_baselines"]},
                               safety_passed=True, expected_sources=[])
    assert not result["numerical_gate_passed"]


def test_inventory_rejects_tamper_and_escape(tmp_path: Path):
    artifact = tmp_path / "artifact.json"
    artifact.write_text("{}")
    inventory = tmp_path / "integrity.json"
    inventory.write_text(json.dumps({"files": [{"path": "artifact.json", "size_bytes": artifact.stat().st_size,
                                               "sha256": _sha256(artifact)}]}))
    (tmp_path / "_SUCCESS").write_text(_sha256(inventory))
    verify_inventory(tmp_path)
    artifact.write_text("changed")
    with pytest.raises(ValueError, match="modified"):
        verify_inventory(tmp_path)
    inventory.write_text(json.dumps({"files": [{"path": "../escape", "size_bytes": 0, "sha256": "x"}]}))
    (tmp_path / "_SUCCESS").write_text(_sha256(inventory))
    with pytest.raises(ValueError, match="unsafe"):
        verify_inventory(tmp_path)


def test_empty_or_omitted_inventory_fails_closed(tmp_path):
    inventory = tmp_path / "integrity.json"
    inventory.write_text(json.dumps({"files": []}))
    (tmp_path / "_SUCCESS").write_text(_sha256(inventory))
    with pytest.raises(ValueError, match="empty"):
        verify_inventory(tmp_path)
