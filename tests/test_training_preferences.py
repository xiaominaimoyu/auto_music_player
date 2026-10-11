"""Offline preference pipeline tests; all answers here are disposable fixtures.

No fixture is a real participant response and no test output is production data.
"""

from __future__ import annotations

import copy
import json
import struct
import uuid
import wave
import zipfile
from pathlib import Path

import pytest

from training.audit_preferences import audit_responses, preference_statistics
from training.preference_bundle import (
    CONSENT_PURPOSE, RESPONSE_SCHEMA, PreferenceError, build_bundle,
    file_hash, read_json, write_json,
)
from training.preference_dataset import load_response, validate_bundle


def _wav(path: Path, level: int, rate: int = 16000) -> None:
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(rate)
        audio.writeframes(struct.pack("<h", level) * (rate // 10))


@pytest.fixture
def bundle_factory(tmp_path):
    _wav(tmp_path / "baseline.wav", 100)
    _wav(tmp_path / "candidate.wav", 200)

    def create(*, synthetic=False, split="local_dev", name="bundle"):
        spec = {
            "study_id": "unit-only-test", "study_version": "1", "consent_version": "test-v1",
            "render_config": {"renderer": "fixture-only", "instrument": "same-test-instrument"},
            "trials": [
                {"piece_group": "PRIVATE_WORK_" + str(index), "source_split": split, "excerpt_id": "private_excerpt_" + str(index),
                 "baseline": {"candidate_id": "SECRET_BASELINE", "wav_path": "baseline.wav"},
                 "candidate": {"candidate_id": "SECRET_MODEL", "wav_path": "candidate.wav"}}
                for index in range(2)
            ],
        }
        spec_path = tmp_path / (name + "_spec.json")
        write_json(spec_path, spec)
        destination = tmp_path / name
        build_bundle(spec_path=spec_path, output=destination, synthetic=synthetic)
        return destination
    return create


def _fixture_answer(bundle_root: Path, *, synthetic: bool = True) -> dict:
    manifest = read_json(bundle_root / "public" / "manifest.json")
    answers = []
    for index, trial in enumerate(reversed(manifest["trials"]), 1):
        a, b = reversed(trial["assets"])
        answers.append({"trial_id": trial["trial_id"], "trial_sha256": trial["trial_sha256"], "position": index,
                        "a_asset_id": a["asset_id"], "a_sha256": a["sha256"], "b_asset_id": b["asset_id"], "b_sha256": b["sha256"],
                        "choice": "a", "a_listened": True, "b_listened": True})
    return {"schema": RESPONSE_SCHEMA, "schema_version": 1,
            **{field: manifest[field] for field in ("study_id", "study_version", "bundle_id", "manifest_sha256", "consent_version")},
            "participant_id": str(uuid.uuid4()), "session_id": str(uuid.uuid4()), "response_id": str(uuid.uuid4()),
            "synthetic": synthetic, "consent": {"accepted": True, "version": manifest["consent_version"], "purpose": CONSENT_PURPOSE},
            "responses": answers}


def test_public_zip_is_blind_and_research_artifact_verifies(bundle_factory):
    root = bundle_factory()
    bundle = validate_bundle(root)
    assert len(bundle["trials"]) == 2
    with zipfile.ZipFile(root / "listening_bundle.zip") as archive:
        assert set(archive.namelist()) == {item["path"] for item in bundle["private"]["public_files"]}
        assert "private_manifest.json" not in archive.namelist()
        public_text = archive.read("index.html") + archive.read("manifest.json")
        for secret in (b"SECRET_MODEL", b"SECRET_BASELINE", b"PRIVATE_WORK", b"source_split", b"wav_path"):
            assert secret not in public_text
        assert b"__BUNDLE_MANIFEST__" not in public_text
        assert b"connect-src 'none'" in public_text
        assert b"fetch(" not in public_text


def test_locked_test_source_is_rejected_before_output(bundle_factory):
    with pytest.raises(PreferenceError, match="never locked test"):
        bundle_factory(split="test")


def test_wav_metadata_cannot_unblind_listener(bundle_factory, tmp_path):
    root = bundle_factory()
    source = tmp_path / "candidate.wav"
    wav = source.read_bytes()
    secret = b"SECRET_MODEL_METADATA\x00"
    payload = b"INFO" + b"INAM" + struct.pack("<I", len(secret)) + secret + (b"\x00" if len(secret) % 2 else b"")
    chunk = b"LIST" + struct.pack("<I", len(payload)) + payload
    updated = wav[:4] + struct.pack("<I", len(wav) + len(chunk) - 8) + wav[8:] + chunk
    source.write_bytes(updated)
    other = tmp_path / "metadata_bundle"
    build_bundle(spec_path=tmp_path / "bundle_spec.json", output=other)
    validate_bundle(other)
    for audio in (other / "public" / "assets").glob("*.wav"):
        assert b"SECRET_MODEL_METADATA" not in audio.read_bytes()


def test_nonempty_output_is_not_overwritten(bundle_factory):
    root = bundle_factory()
    previous = file_hash(root / "summary.json")
    with pytest.raises(PreferenceError, match="absent or empty"):
        bundle_factory()
    assert file_hash(root / "summary.json") == previous


def test_wav_tamper_invalidates_bundle(bundle_factory):
    root = bundle_factory()
    wav = next((root / "public" / "assets").glob("*.wav"))
    wav.write_bytes(wav.read_bytes()[:-1] + b"\x08")
    with pytest.raises(PreferenceError, match="hash/size mismatch"):
        validate_bundle(root)


def test_response_resolves_actual_shuffled_assets_without_modifying_source(bundle_factory, tmp_path):
    root = bundle_factory()
    answer = _fixture_answer(root)
    path = tmp_path / "fixture_synthetic_response.json"
    write_json(path, answer)
    previous = file_hash(path)
    loaded = load_response(path, bundle_root=root)
    assert loaded["synthetic"] is True
    assert loaded["rows"][0]["outcome"] in {"baseline", "candidate"}
    assert loaded["rows"][0]["piece_group"].startswith("PRIVATE_WORK_")
    assert file_hash(path) == previous


@pytest.mark.parametrize("mutation,match", [
    (lambda answer: answer["consent"].update(accepted=False), "consent"),
    (lambda answer: answer["consent"].update(version="wrong"), "consent"),
    (lambda answer: answer.update(email="not-collected@example.invalid"), "unexpected fields"),
    (lambda answer: answer["responses"][0].update(trial_id="unknown"), "unknown or duplicate trial"),
    (lambda answer: answer["responses"][0].update(trial_sha256="0" * 64), "trial hash"),
    (lambda answer: answer["responses"][0].update(a_sha256="0" * 64), "modified audio"),
    (lambda answer: answer["responses"][0].update(choice="generated-winner"), "invalid choice"),
    (lambda answer: answer["responses"][0].update(a_listened=False), "both playback"),
    (lambda answer: answer["responses"][0].update(a_listened=1), "must be boolean"),
    (lambda answer: answer["responses"][0].update(position=True), "presentation order"),
    (lambda answer: answer.update(participant_id="participant@example.invalid"), "UUID"),
    (lambda answer: answer["responses"].pop(), "every trial"),
])
def test_strict_loader_rejects_invalid_answers(bundle_factory, tmp_path, mutation, match):
    root = bundle_factory()
    answer = _fixture_answer(root)
    mutation(answer)
    path = tmp_path / "invalid_synthetic_fixture.json"
    write_json(path, answer)
    with pytest.raises(PreferenceError, match=match):
        load_response(path, bundle_root=root)


def test_duplicate_json_keys_are_rejected(bundle_factory, tmp_path):
    root = bundle_factory()
    path = tmp_path / "duplicate_keys_fixture.json"
    path.write_text('{"schema":1,"schema":2}', encoding="utf-8")
    with pytest.raises(PreferenceError, match="duplicate JSON"):
        load_response(path, bundle_root=root)


def test_synthetic_bundle_cannot_be_downgraded_by_response(bundle_factory, tmp_path):
    root = bundle_factory(synthetic=True)
    answer = _fixture_answer(root, synthetic=False)
    path = root / "responses" / "fixture_claims_non_synthetic.json"
    write_json(path, answer)
    assert load_response(path, bundle_root=root)["synthetic"] is True
    summary = audit_responses(bundle_root=root, responses_root=root / "responses", output=tmp_path / "audit")
    assert summary["human_preference_count"] == 0
    assert summary["excluded_counts"]["synthetic_response_never_human"] == 1
    assert summary["training_ready"] is False


def test_empty_and_synthetic_answers_never_count_as_human(bundle_factory, tmp_path):
    root = bundle_factory()
    empty = audit_responses(bundle_root=root, responses_root=root / "responses", output=tmp_path / "empty_audit")
    assert empty["scanned_response_files"] == empty["human_preference_count"] == 0
    assert empty["statistics"]["crossed_cluster_ci95"] is None
    assert empty["training_ready"] is False
    write_json(root / "responses" / "synthetic_fixture.json", _fixture_answer(root))
    result = audit_responses(bundle_root=root, responses_root=root / "responses", output=tmp_path / "synthetic_audit")
    assert result["has_human_preferences"] is False
    assert result["human_participant_count"] == 0


def test_tie_skip_and_unanswered_remain_distinct(bundle_factory, tmp_path):
    root = bundle_factory()
    answer = _fixture_answer(root)
    answer["responses"][0]["choice"] = "tie"
    answer["responses"][1].update(choice=None, a_listened=False, b_listened=False)
    path = tmp_path / "choice_fixture.json"
    write_json(path, answer)
    result = load_response(path, bundle_root=root)
    stats = preference_statistics(result["rows"])
    assert stats["counts"]["tie"] == 1
    assert stats["counts"]["no_selection"] == 1
    assert stats["candidate_score_with_half_ties"] == 0.5
    answer["responses"][1]["choice"] = "skip"
    write_json(path, answer)
    stats = preference_statistics(load_response(path, bundle_root=root)["rows"])
    assert stats["counts"]["skip"] == 1
    assert stats["counts"]["no_selection"] == 0


def test_duplicate_and_conflicting_hypothetical_responses_are_not_extra_votes(bundle_factory, tmp_path):
    # Disposable unit fixtures exercise the non-synthetic branch.  They are not
    # real returned files and must never be copied into a collection directory.
    root = bundle_factory()
    answer = _fixture_answer(root, synthetic=False)
    first = root / "responses" / "unit_fixture_a.json"
    second = root / "responses" / "unit_fixture_duplicate.json"
    write_json(first, answer)
    second.write_bytes(first.read_bytes())
    initial_hash = file_hash(first)
    result = audit_responses(bundle_root=root, responses_root=root / "responses", output=tmp_path / "duplicate_audit")
    assert result["human_trial_records"] == 2
    assert result["excluded_counts"]["duplicate_response"] == 1
    assert result["training_ready"] is False
    conflict = copy.deepcopy(answer)
    conflict["responses"][0]["choice"] = "b"
    write_json(second, conflict)
    result = audit_responses(bundle_root=root, responses_root=root / "responses", output=tmp_path / "conflict_audit")
    assert result["human_trial_records"] == 0
    assert result["conflicting_response_id_count"] == 1
    assert result["excluded_counts"]["conflicting_response_or_session_identity"] == 2
    assert file_hash(first) == initial_hash


def test_conflicting_same_person_new_session_is_explicitly_rejected(bundle_factory, tmp_path):
    root = bundle_factory()
    answer = _fixture_answer(root, synthetic=False)
    write_json(root / "responses" / "unit_fixture_a.json", answer)
    alternate = copy.deepcopy(answer)
    alternate.update(session_id=str(uuid.uuid4()), response_id=str(uuid.uuid4()))
    alternate["responses"][0]["choice"] = "b"
    write_json(root / "responses" / "unit_fixture_b.json", alternate)
    summary = audit_responses(bundle_root=root, responses_root=root / "responses", output=tmp_path / "audit")
    assert summary["conflicting_participant_trial_count"] == 1
    assert summary["human_trial_records"] == 1
    assert summary["excluded_counts"]["conflicting_repeat_answer"] == 2
    assert summary["excluded_counts"]["duplicate_participant_trial_answer"] == 1


def test_audit_does_not_write_inside_returned_answers(bundle_factory):
    root = bundle_factory()
    with pytest.raises(PreferenceError, match="non-nested"):
        audit_responses(bundle_root=root, responses_root=root / "responses", output=root / "responses" / "audit")


def test_small_cluster_count_withholds_confidence_interval(bundle_factory, tmp_path):
    root = bundle_factory()
    answer = _fixture_answer(root)
    path = tmp_path / "synthetic_stats_fixture.json"
    write_json(path, answer)
    rows = load_response(path, bundle_root=root)["rows"]
    stats = preference_statistics(rows, bootstrap=200)
    assert stats["crossed_cluster_ci95"] is None
    assert "insufficient independent clusters" in stats["interval_interpretation"]
    # Pure in-memory synthetic data checks the crossed-cluster implementation.
    large_fixture = [{**rows[0], "participant_id": str(person), "piece_group": str(piece), "outcome": "tie", "candidate_score": 0.5} for person in range(5) for piece in range(5)]
    stats = preference_statistics(large_fixture, bootstrap=200)
    assert stats["crossed_cluster_ci95"] == [0.5, 0.5]
    assert stats["participant_equal_candidate_score"] == 0.5
    assert stats["piece_equal_candidate_score"] == 0.5
