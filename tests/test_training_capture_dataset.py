import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from core.performance_capture import SCHEMA_NAME, canonical_notes_hash
from training.capture_dataset import (
    CaptureValidationError,
    iter_completed_captures,
    load_capture,
)
from training.local_capture_manifest import build_capture_manifest


MAP_HASH = "b" * 64
SESSION_ID = "00000000-0000-4000-8000-000000000001"


def _valid_records(*, eligible=True, synthetic_close=False):
    canonical_notes = [
        {"notes": ["mid_1", "mid_3"], "dur": 1},
        {"notes": ["high_2"], "dur": 1, "semitone": 1},
    ]
    records = [
        {
            "record": "start",
            "schema": SCHEMA_NAME,
            "schema_version": 1,
            "session_id": SESSION_ID,
            "status": "active",
            "mapping_hash": MAP_HASH,
            "bpm": 120,
            "quantize": 0.25,
        },
        {
            "record": "event",
            "session_id": SESSION_ID,
            "seq": 1,
            "elapsed_ns": 10,
            "event": {
                "type": "note_down",
                "press_id": 1,
                "relative_ns": 100,
                "note_id": "mid_1",
                "semitone": 0,
                "modifier": "natural",
                "synthetic_close": False,
            },
        },
        {
            "record": "event",
            "session_id": SESSION_ID,
            "seq": 2,
            "elapsed_ns": 20,
            "event": {
                "type": "note_down",
                "press_id": 2,
                "relative_ns": 100,
                "note_id": "mid_3",
                "semitone": 0,
                "modifier": "natural",
                "synthetic_close": False,
            },
        },
        {
            "record": "event",
            "session_id": SESSION_ID,
            "seq": 3,
            "elapsed_ns": 30,
            "event": {
                "type": "note_up",
                "press_id": 1,
                "relative_ns": 300,
                "note_id": "mid_1",
                "semitone": 0,
                "modifier": "natural",
                "synthetic_close": False,
            },
        },
        {
            "record": "event",
            "session_id": SESSION_ID,
            "seq": 4,
            "elapsed_ns": 40,
            "event": {
                "type": "note_up",
                "press_id": 2,
                "relative_ns": 400,
                "note_id": "mid_3",
                "semitone": 0,
                "modifier": "natural",
                "synthetic_close": False,
            },
        },
        {
            "record": "event",
            "session_id": SESSION_ID,
            "seq": 5,
            "elapsed_ns": 50,
            "event": {
                "type": "note_down",
                "press_id": 3,
                "relative_ns": 500,
                "note_id": "high_2",
                "semitone": 1,
                "modifier": "semitone",
                "synthetic_close": False,
            },
        },
        {
            "record": "event",
            "session_id": SESSION_ID,
            "seq": 6,
            "elapsed_ns": 60,
            "event": {
                "type": "note_up",
                "press_id": 3,
                "relative_ns": 700,
                "note_id": "high_2",
                "semitone": 1,
                "modifier": "semitone",
                "synthetic_close": synthetic_close,
            },
        },
    ]
    ignored_count = 0
    if not eligible:
        ignored_count = 1
        records.append(
            {
                "record": "event",
                "session_id": SESSION_ID,
                "seq": 7,
                "elapsed_ns": 70,
                "event": {
                    "type": "ignored",
                    "reason": "unmapped_or_conflicting_input",
                },
            }
        )
    raw = "".join(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n" for record in records)
    flags = []
    if ignored_count:
        flags.append("high_ignored_ratio")
    if synthetic_close:
        flags.append("held_note_closed_on_stop")
    records.append(
        {
            "record": "end",
            "session_id": SESSION_ID,
            "status": "completed",
            "event_count": 6 + ignored_count,
            "last_seq": 6 + ignored_count,
            "content_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
            "quality": {
                "input_count": 3 + ignored_count,
                "mapped_event_count": 6,
                "mapped_press_count": 3,
                "paired_press_count": 3,
                "unclosed_press_count": 0,
                "ignored_count": ignored_count,
                "ignored_ratio": ignored_count / (3 + ignored_count),
                "synthetic_close": synthetic_close,
                "synthetic_close_count": 1 if synthetic_close else 0,
                "flags": flags,
                "training_eligible": not flags,
            },
            "captured_count": 3,
            "canonical_notes": canonical_notes,
            "canonical_notes_sha256": canonical_notes_hash(canonical_notes),
            "canonical_notes_count": 2,
        }
    )
    return records


def _write_capture(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    # The production writer hashes binary UTF-8 lines, so fixtures must avoid
    # Windows text-mode newline conversion as well.
    path.write_bytes(
        "".join(
            json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            for record in records
        ).encode("utf-8")
    )


class TestCaptureDataset(unittest.TestCase):
    def test_valid_fixture_derives_notes_and_piece_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "completed" / "valid.jsonl"
            _write_capture(path, _valid_records())
            sample = load_capture(path)

            self.assertEqual(sample.session_id, SESSION_ID)
            self.assertEqual(sample.piece_group, canonical_notes_hash(sample.canonical_notes))
            self.assertEqual([note.midi_pitch for note in sample.notes], [60, 64, 75])
            self.assertEqual([note.onset_s for note in sample.notes], [0.0, 0.0, 0.0000004])
            self.assertEqual([note.duration_s for note in sample.notes], [0.0000002, 0.0000003, 0.0000002])

    def test_rejects_content_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "completed" / "bad-hash.jsonl"
            records = _valid_records()
            records[-1]["content_sha256"] = "0" * 64
            _write_capture(path, records)
            with self.assertRaisesRegex(CaptureValidationError, "content_sha256"):
                load_capture(path)

    def test_rejects_sequence_and_time_regression(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "completed" / "bad-seq.jsonl"
            records = _valid_records()
            records[2]["seq"] = 9
            _rehash(records)
            _write_capture(path, records)
            with self.assertRaisesRegex(CaptureValidationError, "seq"):
                load_capture(path)

            records = _valid_records()
            records[3]["event"]["relative_ns"] = 50
            _rehash(records)
            _write_capture(path, records)
            with self.assertRaisesRegex(CaptureValidationError, "relative_ns"):
                load_capture(path)

    def test_rejects_unpaired_or_mismatched_note(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "completed" / "bad-pair.jsonl"
            records = _valid_records()
            records[3]["event"]["note_id"] = "mid_2"
            _rehash(records)
            _write_capture(path, records)
            with self.assertRaisesRegex(CaptureValidationError, "签名"):
                load_capture(path)

            records = _valid_records()
            records.pop(4)
            for seq, record in enumerate(records[1:-1], 1):
                record["seq"] = seq
            records[-1]["event_count"] = 5
            records[-1]["last_seq"] = 5
            _rehash(records)
            _write_capture(path, records)
            with self.assertRaisesRegex(CaptureValidationError, "没有 note_up"):
                load_capture(path)

    def test_derived_notes_are_ordered_by_onset_not_release(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "completed" / "release-order.jsonl"
            records = _valid_records()
            first_up = records[3]["event"]
            second_up = records[4]["event"]
            first_up["press_id"], second_up["press_id"] = 2, 1
            first_up["note_id"], second_up["note_id"] = "mid_3", "mid_1"
            _rehash(records)
            _write_capture(path, records)

            sample = load_capture(path)
            self.assertEqual([note.press_id for note in sample.notes], [1, 2, 3])

    def test_ineligible_rejection_and_explicit_allow(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "completed" / "ineligible.jsonl"
            _write_capture(path, _valid_records(eligible=False))
            with self.assertRaisesRegex(CaptureValidationError, "training_eligible"):
                load_capture(path)
            self.assertFalse(load_capture(path, require_training_eligible=False).training_eligible)

    def test_rejects_canonical_pitch_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "completed" / "canonical-mismatch.jsonl"
            records = _valid_records()
            records[-1]["canonical_notes"][1]["notes"] = ["high_3"]
            records[-1]["canonical_notes_sha256"] = canonical_notes_hash(records[-1]["canonical_notes"])
            _rehash(records)
            _write_capture(path, records)
            with self.assertRaisesRegex(CaptureValidationError, "音高多重集合"):
                load_capture(path)

    def test_rejects_empty_bool_counts_invalid_score_and_private_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "completed" / "invalid.jsonl"

            records = _valid_records()
            records = [records[0], records[-1]]
            records[-1].update(
                {
                    "event_count": 0,
                    "last_seq": 0,
                    "captured_count": 0,
                    "canonical_notes": [{"notes": [], "dur": 1}],
                }
            )
            records[-1]["canonical_notes_sha256"] = canonical_notes_hash(
                records[-1]["canonical_notes"]
            )
            records[-1]["canonical_notes_count"] = 1
            _rehash(records)
            _write_capture(path, records)
            with self.assertRaisesRegex(CaptureValidationError, "captured_count"):
                load_capture(path)

            records = _valid_records()
            records[-1]["captured_count"] = True
            _rehash(records)
            _write_capture(path, records)
            with self.assertRaisesRegex(CaptureValidationError, "captured_count"):
                load_capture(path)

            records = _valid_records()
            records[-1]["canonical_notes"][0]["dur"] = 0
            records[-1]["canonical_notes_sha256"] = canonical_notes_hash(
                records[-1]["canonical_notes"]
            )
            _rehash(records)
            _write_capture(path, records)
            with self.assertRaisesRegex(CaptureValidationError, "乐谱校验"):
                load_capture(path)

            records = _valid_records()
            records[1]["event"]["nested"] = {"scan_code": 30}
            _rehash(records)
            _write_capture(path, records)
            with self.assertRaisesRegex(CaptureValidationError, "隐私校验"):
                load_capture(path)

            records = _valid_records()
            records[-1]["quality"]["ignored_ratio"] = 1.0
            _write_capture(path, records)
            with self.assertRaisesRegex(CaptureValidationError, "quality"):
                load_capture(path)

    def test_enumerates_completed_captures_deterministically(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("z.jsonl", "a.jsonl"):
                _write_capture(root / "completed" / name, _valid_records())
            _write_capture(root / "pending" / "ignored.jsonl", _valid_records())
            self.assertEqual([path.name for path in iter_completed_captures(root)], ["a.jsonl", "z.jsonl"])

    def test_manifest_builder_separates_rejects_and_is_repeatable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "captures"
            valid = root / "completed" / "valid.jsonl"
            invalid = root / "completed" / "invalid.jsonl"
            _write_capture(valid, _valid_records())
            invalid_records = _valid_records()
            invalid_records[-1]["content_sha256"] = "0" * 64
            _write_capture(invalid, invalid_records)
            source_hashes_before = {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in (valid, invalid)
            }

            first = Path(tmp) / "out-first"
            second = Path(tmp) / "out-second"
            first_summary = build_capture_manifest(capture_root=root, output_path=first)
            second_summary = build_capture_manifest(capture_root=root, output_path=second)

            self.assertEqual(first_summary, second_summary)
            self.assertEqual(first_summary["scanned_count"], 2)
            self.assertEqual(first_summary["accepted_count"], 1)
            self.assertEqual(first_summary["rejected_count"], 1)
            self.assertTrue((first / "_SUCCESS").is_file())
            for name in (
                "build_config.json",
                "manifest.jsonl",
                "rejects.jsonl",
                "summary.json",
                "integrity.json",
                "_SUCCESS",
            ):
                self.assertEqual((first / name).read_bytes(), (second / name).read_bytes())
            source_hashes_after = {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in (valid, invalid)
            }
            self.assertEqual(source_hashes_before, source_hashes_after)

    def test_empty_manifest_audit_is_explicitly_not_training_ready(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "captures"
            (root / "completed").mkdir(parents=True)
            output = Path(tmp) / "audit"
            summary = build_capture_manifest(capture_root=root, output_path=output)

            self.assertEqual(summary["accepted_count"], 0)
            self.assertFalse(summary["training_ready"])
            self.assertTrue((output / "_NO_ACCEPTED_SAMPLES").is_file())
            self.assertTrue((output / "_SUCCESS").is_file())


def _rehash(records):
    raw = "".join(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n" for record in records[:-1])
    records[-1]["content_sha256"] = hashlib.sha256(raw.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    unittest.main()
