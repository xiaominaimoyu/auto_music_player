from __future__ import annotations

import ast
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from training import audit_game_expressibility as game
from training import audit_source_game_mapping as audit
from training.pianocore_dataset import ZIP_PREFIX
from training.tempo_reference import build_timing_reference, load_timing_reference
from tests.test_training_tempo_reference import _dataset_fixture


def _inputs(tmp_path, *, split="train", styles=("piano", "preserve"), score_member=None):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    dataset, archive, selection = _dataset_fixture(inputs, split=split)
    if score_member is not None:
        manifest = dataset / "manifest.jsonl"
        row = json.loads(manifest.read_text(encoding="utf-8"))
        with zipfile.ZipFile(archive) as handle:
            members = {name: handle.read(name) for name in handle.namelist()}
        score = members.pop(f"{ZIP_PREFIX}/{row['score_midi_path']}")
        members[f"{ZIP_PREFIX}/{score_member}"] = score
        with zipfile.ZipFile(archive, "w") as handle:
            for name, data in members.items():
                handle.writestr(name, data)
        row["score_midi_path"] = score_member
        manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    reference, game_root = tmp_path / "reference", tmp_path / "game"
    build_timing_reference(dataset_root=dataset, archive_path=archive,
                           sample_id_files=[selection], output_dir=reference)
    game.audit_reference_root(timing_reference=reference, output_dir=game_root, styles=styles)
    return reference, game_root, dataset, archive


def _run(reference, game_root, output, **kwargs):
    return audit.audit_source_game_mapping(timing_reference=reference, game_audit=game_root,
                                           output_dir=output, **kwargs)


def _summary(root):
    return json.loads((root / "summary.json").read_text(encoding="utf-8"))


def _seal(root, summary):
    (root / "summary.json").write_bytes(audit._canonical(summary) + b"\n")
    (root / "_SUCCESS").write_text(audit._sha256(root / "summary.json") + "\n", encoding="ascii")


def _change_case(root, summary, change, position=0):
    entry = summary["samples"][position]
    path = root / entry["path"]
    row = json.loads(path.read_text(encoding="utf-8"))
    change(row)
    path.write_bytes(audit._canonical(row) + b"\n")
    entry["sha256"] = audit._sha256(path)
    return row


@pytest.mark.parametrize("split", ["train", "validation"])
def test_real_reference_game_mapping_chain_seals_both_styles_and_preserves_native_arrays(tmp_path, split):
    reference, game_root, dataset, archive = _inputs(tmp_path, split=split)
    before = {path: audit._sha256(path) for parent in (reference, game_root, dataset)
              for path in parent.rglob("*") if path.is_file()}
    before[archive] = audit._sha256(archive)
    output = tmp_path / "mapping"
    result = _run(reference / "summary.json", game_root / "summary.json", output)
    assert result["sample_count"] == 1
    assert result["sample_style_count"] == 2
    assert result["status_counts"] == {"complete": 2}
    assert result["styles"] == ["piano", "preserve"]
    assert all(not value for value in result["acceptance"].values())
    assert not (output / "_INCOMPLETE").exists()
    assert (output / "_SUCCESS").read_text().strip() == audit._sha256(output / "summary.json")
    native = load_timing_reference(reference)["sample-one"]
    for entry in result["samples"]:
        assert audit._sha256(output / entry["path"]) == entry["sha256"]
        row = json.loads((output / entry["path"]).read_text(encoding="utf-8"))
        assert audit._sha256(output / row["arrays_path"]) == row["arrays_sha256"]
        assert row["mapping"]["diagnostic_only"] is True
        with np.load(output / row["arrays_path"], allow_pickle=False) as arrays:
            for original, copied in (("note_native_masks", "native_note_masks"),
                                     ("chord_native_masks", "native_chord_masks"),
                                     ("note_to_chord", "native_note_to_chord")):
                np.testing.assert_array_equal(arrays[copied], native[original])
            assert not any("performance" in key or "target" in key for key in arrays.files)
        assert set(row["mapping"]["profiles"]) == {"default", "delta_force_harmonica"}
    groups = result["aggregate"]["groups"]
    assert len(groups) == 2
    assert sum(group["sample_style_count"] for group in groups) == 2
    assert all(group["split"] == split and group["stages"]["source"]["voice_count"] == 6 for group in groups)
    assert {path: audit._sha256(path) for path in before} == before
    bindings = result["implementation_and_config_sha256"]
    assert str(Path(audit.__file__).resolve()) in bindings
    assert str(audit._ROOT / "training" / "source_game_mapping.py") in bindings


def test_cli_main_and_deterministic_arrays(tmp_path, capsys):
    reference, game_root, _, _ = _inputs(tmp_path)
    first, second = tmp_path / "first", tmp_path / "second"
    _run(reference, game_root, first)
    assert audit.main(["--timing-reference", str(reference), "--game-audit", str(game_root),
                       "--output-dir", str(second)]) == 0
    assert "sample_count" in capsys.readouterr().out
    first_files = {path.relative_to(first): audit._sha256(path) for path in first.rglob("*") if path.is_file()}
    second_files = {path.relative_to(second): audit._sha256(path) for path in second.rglob("*") if path.is_file()}
    assert first_files == second_files


@pytest.mark.parametrize("status", ["import_rejected", "transport_rejected"])
def test_bound_rejections_remain_in_denominators_and_are_never_mapped(tmp_path, monkeypatch, status):
    reference, game_root, _, _ = _inputs(tmp_path)
    summary = _summary(game_root)
    def reject(row):
        row.pop("audit")
        row.update(status=status, error="bound production gate refusal")
        if status == "transport_rejected":
            row["transport_errors"] = ["bound storage error"]
    _change_case(game_root, summary, reject)
    summary["samples"][0]["status"] = status
    summary["status_counts"] = {status: 1, "complete": 1}
    _seal(game_root, summary)
    actual = audit.build_sample_game_mapping
    calls = []
    def counted(**kwargs):
        calls.append(kwargs["options"].style)
        return actual(**kwargs)
    monkeypatch.setattr(audit, "build_sample_game_mapping", counted)
    output = tmp_path / "mapping"
    result = _run(reference, game_root, output)
    assert result["sample_count"] == 1 and result["sample_style_count"] == 2
    assert result["status_counts"] == {status: 1, "complete": 1}
    assert len(calls) == 1
    rejected = next(row for row in result["samples"] if row["status"] == status)
    row = json.loads((output / rejected["path"]).read_text(encoding="utf-8"))
    assert row["error"] == "bound production gate refusal"
    assert "arrays_path" not in row and "mapping" not in row
    group = next(group for group in result["aggregate"]["groups"] if group["style"] == rejected["style"])
    assert group["sample_style_count"] == 1 and group["completed_sample_style_count"] == 0
    assert group["stages"] == group["profiles"] == {}


@pytest.mark.parametrize("change", ["summary_hash", "incomplete", "case_hash", "extra_file", "count", "missing_case",
                                    "duplicate_case", "unknown_status", "unknown_split", "case_path", "zip_path",
                                    "metadata", "options", "profile", "settings", "implementation", "rejection_error"])
def test_invalid_game_audit_fails_before_creating_output(tmp_path, change):
    reference, game_root, _, _ = _inputs(tmp_path)
    summary = _summary(game_root)
    if change == "summary_hash":
        (game_root / "summary.json").write_bytes((game_root / "summary.json").read_bytes() + b" ")
    elif change == "incomplete":
        (game_root / "_INCOMPLETE").write_text("unfinished")
    elif change == "case_hash":
        (game_root / summary["samples"][0]["path"]).write_text("{}")
    elif change == "extra_file":
        (game_root / "unbound.txt").write_text("unbound")
    else:
        if change == "count":
            summary["sample_style_count"] += 1
        elif change == "missing_case":
            summary["samples"].pop()
        elif change == "duplicate_case":
            summary["samples"].append(summary["samples"][0])
        elif change == "unknown_status":
            summary["samples"][0]["status"] = "unexpected"
        elif change == "unknown_split":
            summary["samples"][0]["split"] = "mystery"
        elif change == "case_path":
            summary["samples"][0]["path"] = "../outside.json"
        elif change == "zip_path":
            summary["samples"][0]["score_midi_path"] = "../outside.mid"
        elif change == "metadata":
            _change_case(game_root, summary, lambda row: row.update(piece_group="different"))
        elif change == "options":
            _change_case(game_root, summary, lambda row: row["audit"]["options"].update(transpose=12))
        elif change == "profile":
            _change_case(game_root, summary, lambda row: row["audit"].update(projection=[]))
        elif change == "settings":
            summary["playback_settings"]["gap_ms"] += 1
        elif change == "implementation":
            summary["implementation_and_config_sha256"]["unknown"] = "0" * 64
        elif change == "rejection_error":
            def reject(row):
                row.pop("audit")
                row["status"] = "import_rejected"
            _change_case(game_root, summary, reject)
            summary["samples"][0]["status"] = "import_rejected"
        _seal(game_root, summary)
    output = tmp_path / "mapping"
    expected = {
        "summary_hash": "summary hash mismatch", "incomplete": "incomplete", "case_hash": "case duplicate path or hash mismatch",
        "extra_file": "unbound files", "count": "denominator mismatch", "missing_case": "denominator mismatch",
        "duplicate_case": "selection/status", "unknown_status": "selection/status", "unknown_split": "selected source binding",
        "case_path": "unsafe sidecar/source path", "zip_path": "unsafe source ZIP member", "metadata": "case metadata",
        "options": "importer options", "profile": "required free-play", "settings": "playback settings",
        "implementation": "implementation/config binding", "rejection_error": "rejected case lacks an error",
    }
    with pytest.raises(ValueError, match=expected[change]):
        _run(reference, game_root, output)
    assert not output.exists()


@pytest.mark.parametrize("path", ["", ".", "..", "../score.mid", "/score.mid", "C:/score.mid", "C:score.mid", "a\\b.mid", "a/../b.mid", "score\x00.mid"])
def test_unsafe_zip_member_rejected(path):
    with pytest.raises(ValueError, match="unsafe"):
        audit._member_path(path)


def test_archive_work_title_colon_is_not_a_windows_destination(tmp_path):
    member = "Bach,_Johann_Sebastian/No.1_in_C_major,_BWV_846:_Fugue/score_ASAP_refined.mid"
    assert audit._member_path(member) == member
    reference, game_root, _, _ = _inputs(tmp_path, score_member=member)
    result = _run(reference, game_root, tmp_path / "mapping")
    assert result["status_counts"] == {"complete": 2}
    assert {row["score_midi_path"] for row in result["samples"]} == {member}


@pytest.mark.parametrize("target", ["reference", "game", "dataset", "archive", "parent"])
def test_output_must_not_overlap_bound_inputs(tmp_path, target):
    reference, game_root, dataset, archive = _inputs(tmp_path)
    output = {"reference": reference / "out", "game": game_root / "out", "dataset": dataset / "out",
              "archive": archive.parent / "out", "parent": tmp_path}[target]
    with pytest.raises(ValueError, match="output must be isolated"):
        _run(reference, game_root, output)
    assert not (output / "_INCOMPLETE").exists()


def test_nonempty_output_is_never_overwritten(tmp_path):
    reference, game_root, _, _ = _inputs(tmp_path)
    output = tmp_path / "mapping"
    output.mkdir()
    sentinel = output / "original.txt"
    sentinel.write_bytes(b"preserve me")
    with pytest.raises(ValueError, match="new or empty"):
        _run(reference, game_root, output)
    assert sentinel.read_bytes() == b"preserve me" and len(list(output.iterdir())) == 1


def test_archive_member_corruption_retains_incomplete_marker(tmp_path):
    reference, game_root, _, archive = _inputs(tmp_path)
    with zipfile.ZipFile(archive) as handle:
        data = {name: handle.read(name) for name in handle.namelist()}
    data[f"{ZIP_PREFIX}/data/score_midi.bin"] += b"corrupt"
    with zipfile.ZipFile(archive, "w") as handle:
        for name, value in data.items():
            handle.writestr(name, value)
    output = tmp_path / "mapping"
    with pytest.raises(ValueError, match="score ZIP member hash"):
        _run(reference, game_root, output)
    assert (output / "_INCOMPLETE").exists() and not (output / "_SUCCESS").exists()


@pytest.mark.parametrize("change", ["internal", "source_shard", "reference", "game_case", "implementation", "output_arrays", "output_case", "voice_count", "archive", "missing_profile"])
def test_errors_or_mutation_during_mapping_cannot_publish_success(tmp_path, monkeypatch, change):
    reference, game_root, dataset, archive = _inputs(tmp_path, styles=("preserve",) if change == "archive" else ("piano", "preserve"))
    output = tmp_path / "mapping"
    original, original_bindings = audit.build_sample_game_mapping, audit._implementation_bindings
    changed = False
    def mutate(**kwargs):
        nonlocal changed
        if change == "internal":
            raise RuntimeError("mapping implementation failed")
        value = original(**kwargs)
        if not changed:
            if change == "source_shard":
                path = dataset / "part.npz"
                path.write_bytes(path.read_bytes() + b"changed")
            elif change == "reference":
                path = reference / "summary.json"
                path.write_bytes(path.read_bytes() + b" ")
            elif change == "game_case":
                path = game_root / _summary(game_root)["samples"][0]["path"]
                path.write_bytes(path.read_bytes() + b" ")
            elif change == "voice_count":
                value["summary"]["stages"]["source"]["voice_count"] += 1
            elif change == "missing_profile":
                value["summary"]["profiles"].pop("default")
            elif change == "archive":
                with zipfile.ZipFile(archive) as handle:
                    data = {name: handle.read(name) for name in handle.namelist()}
                data[f"{ZIP_PREFIX}/data/score_midi.bin"] += b"changed"
                with zipfile.ZipFile(archive, "w") as handle:
                    for name, content in data.items():
                        handle.writestr(name, content)
            changed = True
        elif change in {"output_arrays", "output_case"}:
            suffix = "*.npz" if change == "output_arrays" else "*.json"
            path = next((output / "samples").glob(suffix))
            path.write_bytes(path.read_bytes() + b"changed")
        return value
    def bindings(*args, **kwargs):
        values = original_bindings(*args, **kwargs)
        if changed and change == "implementation":
            values["changed"] = "0" * 64
        return values
    monkeypatch.setattr(audit, "build_sample_game_mapping", mutate)
    monkeypatch.setattr(audit, "_implementation_bindings", bindings)
    expected = {"internal": "mapping implementation failed", "source_shard": "source shard hash mismatch",
                "reference": "summary hash mismatch", "game_case": "case duplicate path or hash mismatch",
                "implementation": "implementation/config changed", "output_arrays": "mapping arrays changed",
                "output_case": "mapping case changed", "voice_count": "voice counts differ",
                "archive": "ZIP member changed during audit", "missing_profile": "required game profiles"}
    with pytest.raises((ValueError, RuntimeError), match=expected[change]):
        _run(reference, game_root, output)
    assert (output / "_INCOMPLETE").exists() and not (output / "_SUCCESS").exists()


def test_invalid_timing_reference_rejected_before_output(tmp_path):
    reference, game_root, _, _ = _inputs(tmp_path)
    entry = _summary(reference)["samples"][0]
    path = reference / entry["path"]
    path.write_bytes(path.read_bytes() + b"changed")
    output = tmp_path / "mapping"
    with pytest.raises(ValueError, match="sample hash"):
        _run(reference, game_root, output)
    assert not output.exists()


def test_module_does_not_import_gui_driver_or_trainer():
    tree = ast.parse(Path(audit.__file__).read_text(encoding="utf-8"))
    imports = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert not imports & {"core.player", "core.event_player", "core.keyboard_driver", "gui.player_tab", "torch"}
