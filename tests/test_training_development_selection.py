import hashlib
import json
from pathlib import Path

import pytest

from training.build_development_selection import (
    SelectionCapacityError, SelectionError, build_selection, plan_selection,
)
from tests.test_training_validate_pianocore_dataset import _dataset, _sha256


def _digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def _row(sample_id, source, piece, split="train", **changes):
    row = {"id": sample_id, "performance_dataset": source,
           "piece_group": _digest("piece:" + piece),
           "score_semantic_sha256": _digest("semantic:" + piece),
           "score_midi_sha256": _digest("raw:" + piece),
           "official_split": "test" if split == "test" else "train", "split": split}
    row.update(changes)
    return row


def _balanced_rows():
    return [_row(f"{split}-{source}-{work}-{take}", source, f"{split}-{source}-{work}", split)
            for split in ("train", "validation") for source in ("A", "B")
            for work in range(3) for take in range(2)]


def test_reproducible_independent_of_manifest_order_and_one_take_per_work():
    rows = _balanced_rows()
    first, report = plan_selection(rows, [], train_per_source=2, validation_per_source=2, sources=("A", "B"))
    second, _ = plan_selection(list(reversed(rows)), [], train_per_source=2, validation_per_source=2, sources=("A", "B"))
    assert first == second
    assert len(first) == len({row["piece_group"] for row in first}) == 8
    assert report["selected_source_counts"] == {"train": {"A": 2, "B": 2}, "validation": {"A": 2, "B": 2}}
    assert report["development_only"] is True
    assert report["untouched_blind_holdout"] is False


@pytest.mark.parametrize("identity", ["piece_group", "score_semantic_sha256", "score_midi_sha256"])
def test_prior_identity_excluded_globally_across_sources(identity):
    prior = _row("prior", "A", "previous")
    alias = _row("alias", "B", "alias", **{identity: prior[identity]})
    rows = [prior, alias, *_balanced_rows()]
    selected, report = plan_selection(rows, ["prior"], train_per_source=2, validation_per_source=2, sources=("A", "B"))
    assert {row["id"] for row in selected}.isdisjoint({"prior", "alias"})
    assert report["excluded_manifest_record_count"] == 2
    assert report["excluded_global_work_count"] == 1


def test_connected_exclusion_propagates_transitively():
    prior = _row("prior", "A", "previous")
    bridge = _row("bridge", "B", "bridge", score_semantic_sha256=prior["score_semantic_sha256"])
    indirect = _row("indirect", "B", "indirect", score_midi_sha256=bridge["score_midi_sha256"])
    selected, report = plan_selection([prior, bridge, indirect, *_balanced_rows()], ["prior"],
        train_per_source=2, validation_per_source=2, sources=("A", "B"))
    assert report["excluded_manifest_record_count"] == 3
    assert {row["id"] for row in selected}.isdisjoint({"prior", "bridge", "indirect"})


def test_shared_work_counted_once_and_matching_respects_source_capacity():
    rows = []
    for split in ("train", "validation"):
        rows += [_row(f"{split}-A-shared", "A", split + "shared", split),
                 _row(f"{split}-B-shared", "B", split + "shared", split),
                 _row(f"{split}-A-only", "A", split + "only", split)]
    selected, _ = plan_selection(rows, [], train_per_source=1, validation_per_source=1, sources=("A", "B"))
    assert all(row["id"].endswith("only") if row["performance_dataset"] == "A" else row["id"].endswith("shared") for row in selected)


def test_capacity_fails_instead_of_reusing_work_for_two_sources():
    rows = [_row(f"{split}-{source}", source, split + "shared", split)
            for split in ("train", "validation") for source in ("A", "B")]
    with pytest.raises(SelectionCapacityError) as captured:
        plan_selection(rows, [], train_per_source=1, validation_per_source=1, sources=("A", "B"))
    assert captured.value.report["status"] == "insufficient_capacity"
    assert sum(captured.value.report["selected_source_counts"]["train"].values()) == 1


def test_official_test_is_not_available_to_selection():
    rows = [*_balanced_rows(), _row("heldout", "A", "heldout", "test")]
    selected, report = plan_selection(rows, [], train_per_source=2, validation_per_source=2, sources=("A", "B"))
    assert "heldout" not in {row["id"] for row in selected}
    assert report["official_test_metadata_record_count"] == 1
    assert report["official_test_midi_or_label_files_opened"] == 0


def test_fail_closed_on_global_identity_crossing_original_splits():
    rows = [_row("train", "A", "same"), _row("val", "A", "same", "validation")]
    with pytest.raises(SelectionError, match="crosses original"):
        plan_selection(rows, [], train_per_source=1, validation_per_source=1, sources=("A",))


def test_fail_closed_on_official_test_moved_to_train():
    with pytest.raises(SelectionError, match="official test"):
        plan_selection([_row("test", "A", "test", official_split="test")], [], sources=("A",))


def test_unknown_prior_id_is_not_silently_ignored():
    with pytest.raises(SelectionError, match="absent from manifest"):
        plan_selection(_balanced_rows(), ["missing"], sources=("A", "B"))


def _bound_fixture(tmp_path):
    dataset = _dataset(tmp_path / "dataset")
    cache = tmp_path / "cache.json"
    cache.write_text(json.dumps({"dataset_success_sha256": _sha256(dataset / "_SUCCESS"),
        "validation_report": {"status": "valid", "dataset_root": str(dataset.resolve()), "accepted_count": 2}}))
    excluded = tmp_path / "excluded.txt"
    excluded.write_text("")
    return dataset, cache, excluded


def test_build_seals_ids_original_manifest_and_selected_shard_checks(tmp_path):
    dataset, cache, excluded = _bound_fixture(tmp_path)
    original_normalizer = _sha256(dataset / "normalizer.json")
    output = tmp_path / "selection"
    report = build_selection(dataset, output, validation_cache=cache, excluded_id_files=[excluded],
        train_per_source=1, validation_per_source=1, sources=("Demo",))
    assert report["status"] == "verified"
    assert report["selected_note_count"] == 3
    assert report["selected_count"] == report["selected_global_work_count"] == 2
    assert (output / "train_sample_ids.txt").read_text() == "sample-1\n"
    assert (output / "validation_sample_ids.txt").read_text() == "sample-2\n"
    assert (output / "_SUCCESS").read_text().strip() == _sha256(output / "integrity.json")
    assert not (output / "_INCOMPLETE").exists()
    assert _sha256(dataset / "normalizer.json") == original_normalizer
    for entry in json.loads((output / "integrity.json").read_text())["files"]:
        assert _sha256(output / entry["path"]) == entry["sha256"]


def test_inspection_writes_nothing_and_never_opens_any_shard(tmp_path, monkeypatch):
    dataset, cache, excluded = _bound_fixture(tmp_path)
    output = tmp_path / "selection"
    def forbidden(*args, **kwargs):
        raise AssertionError("inspection opened a shard")
    monkeypatch.setattr("training.build_development_selection._validate_shard", forbidden)
    report = build_selection(dataset, output, validation_cache=cache, excluded_id_files=[excluded],
        train_per_source=1, validation_per_source=1, sources=("Demo",), inspect_only=True)
    assert report["status"] == "planned"
    assert not output.exists()


def test_refuses_existing_output_and_source_tree_output(tmp_path):
    dataset, cache, excluded = _bound_fixture(tmp_path)
    for output in (tmp_path, dataset / "new-selection"):
        with pytest.raises(SelectionError, match="output"):
            build_selection(dataset, output, validation_cache=cache, excluded_id_files=[excluded])


def test_corrupt_selected_shard_never_gets_success_seal(tmp_path):
    dataset, cache, excluded = _bound_fixture(tmp_path)
    shard = next((dataset / "shards").rglob("*.npz"))
    shard.write_bytes(shard.read_bytes() + b"tamper")
    output = tmp_path / "selection"
    with pytest.raises(SelectionError, match="shard binding mismatch"):
        build_selection(dataset, output, validation_cache=cache, excluded_id_files=[excluded],
            train_per_source=1, validation_per_source=1, sources=("Demo",))
    assert not (output / "_SUCCESS").exists()
    assert (output / "_INCOMPLETE").is_file()


def test_stale_cache_and_modified_metadata_rejected(tmp_path):
    dataset, cache, excluded = _bound_fixture(tmp_path)
    cache_data = json.loads(cache.read_text())
    cache_data["dataset_success_sha256"] = "0" * 64
    cache.write_text(json.dumps(cache_data))
    with pytest.raises(SelectionError, match="validation cache"):
        build_selection(dataset, None, validation_cache=cache, excluded_id_files=[excluded], inspect_only=True)
    cache_data["dataset_success_sha256"] = _sha256(dataset / "_SUCCESS")
    cache.write_text(json.dumps(cache_data))
    (dataset / "manifest.jsonl").write_text((dataset / "manifest.jsonl").read_text() + "\n")
    with pytest.raises(SelectionError, match="metadata binding mismatch"):
        build_selection(dataset, None, validation_cache=cache, excluded_id_files=[excluded], inspect_only=True)


@pytest.mark.parametrize("value", [0, -1, True, 1.2])
def test_invalid_quota_rejected(value):
    with pytest.raises(SelectionError, match="positive integers"):
        plan_selection(_balanced_rows(), [], train_per_source=value, sources=("A", "B"))
