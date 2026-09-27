import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from core.performance_capture import (
    PerformanceCapture,
    canonical_notes_hash,
)


MAP_HASH = "a" * 64


class FakeClock:
    def __init__(self):
        self.mono = 1_000_000
        self.wall = 1_700_000_000_000_000_000

    def monotonic_ns(self):
        return self.mono

    def wall_time_ns(self):
        return self.wall

    def advance(self, mono=1_000, wall=1_000):
        self.mono += mono
        self.wall += wall


def make_capture(root, *, clock=None, session_id=None, notes=None):
    return PerformanceCapture.start(
        root,
        app_version="1.5.1",
        profile_id="default",
        mapping_mode="mapped_performance",
        mapping_hash=MAP_HASH,
        bpm=120,
        quantize=0.25,
        notes=notes if notes is not None else [{"notes": ["mid_1"], "dur": 1}],
        session_id=session_id,
        clock=clock,
    )


def finalize_capture(capture, **kwargs):
    return capture.finalize(
        canonical_notes=[{"notes": ["mid_1"], "dur": 1}],
        captured_count=1,
        duration_s=0.5,
        **kwargs,
    )


class TestPerformanceCapture(unittest.TestCase):
    def test_finalize_writes_jsonl_and_moves_atomically(self):
        with tempfile.TemporaryDirectory() as tmp:
            clock = FakeClock()
            capture = make_capture(tmp, clock=clock)
            clock.advance()
            capture.append_event({"type": "note_on", "note_id": "mid_1", "semitone": 0})
            clock.advance()
            capture.append_event({"type": "note_off", "note_id": "mid_1"})
            result = finalize_capture(capture, reason="completed")

            self.assertEqual(result.status, "completed")
            self.assertTrue(result.path.is_file())
            self.assertIn("completed", result.path.parts)
            self.assertFalse((Path(tmp) / "pending" / f"{capture.session_id}.jsonl").exists())
            lines = [json.loads(line) for line in result.path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([line["record"] for line in lines], ["start", "event", "event", "end"])
            self.assertEqual(lines[-1]["event_count"], 2)
            self.assertEqual(lines[-1]["last_seq"], 2)
            self.assertEqual(lines[-1]["quality"]["ignored_ratio"], 0.0)
            self.assertTrue(lines[-1]["quality"]["training_eligible"])
            self.assertNotIn("wall_time_ns", lines[0])
            self.assertNotIn("monotonic_ns", lines[0])
            self.assertNotIn("monotonic_ns", lines[1])
            self.assertEqual(
                lines[-1]["canonical_notes"],
                [{"notes": ["mid_1"], "dur": 1}],
            )
            content = "\n".join(result.path.read_text(encoding="utf-8").splitlines()[:-1]) + "\n"
            self.assertEqual(hashlib.sha256(content.encode("utf-8")).hexdigest(), lines[-1]["content_sha256"])

    def test_reject_is_idempotent_and_is_not_completed(self):
        with tempfile.TemporaryDirectory() as tmp:
            capture = make_capture(tmp)
            first = capture.reject("user_cancelled")
            second = capture.reject("different_reason")
            self.assertEqual(first, second)
            self.assertEqual(first.status, "rejected")
            self.assertIn("rejected", first.path.parts)
            self.assertFalse(list((Path(tmp) / "completed").glob("*.jsonl")))
            end = json.loads(first.path.read_text(encoding="utf-8").splitlines()[-1])
            self.assertEqual(end["status"], "rejected")
            self.assertEqual(end["reason"], "user_cancelled")

    def test_finalize_is_idempotent_and_empty_capture_cannot_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            capture = make_capture(tmp)
            with self.assertRaises(ValueError):
                finalize_capture(capture)
            self.assertFalse(capture.closed)
            capture.append_event({"type": "note_on", "note_id": "mid_1"})
            first = finalize_capture(capture)
            second = finalize_capture(capture, reason="ignored_second_call")
            self.assertEqual(first, second)

    def test_session_id_uniqueness_across_all_status_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            session_id = "00000000-0000-4000-8000-000000000001"
            first = make_capture(tmp, session_id=session_id)
            first.reject()
            with self.assertRaises(FileExistsError):
                make_capture(tmp, session_id=session_id)

    def test_notes_hash_is_canonical_and_persisted(self):
        notes_a = [{"dur": 1, "notes": ["mid_1"]}]
        notes_b = [{"notes": ["mid_1"], "dur": 1}]
        self.assertEqual(canonical_notes_hash(notes_a), canonical_notes_hash(notes_b))
        with tempfile.TemporaryDirectory() as tmp:
            capture = make_capture(tmp, notes=notes_a)
            capture.append_event({"type": "note_on", "note_id": "mid_1"})
            result = finalize_capture(capture)
            start = json.loads(result.path.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(start["notes_sha256"], canonical_notes_hash(notes_b))

    def test_privacy_forbidden_fields_are_rejected_recursively(self):
        forbidden = [
            {"type": "note_on", "key": "A"},
            {"type": "note_on", "nested": {"scan_code": 30}},
            {"type": "note_on", "mouse_button": "left"},
            {"type": "note_on", "window_title": "Game"},
            {"type": "note_on", "x": 10, "y": 20},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            capture = make_capture(tmp)
            for event in forbidden:
                with self.assertRaises(ValueError):
                    capture.append_event(event)
            capture.append_event({"type": "note_on", "note_id": "mid_1", "semitone": 0})
            finalize_capture(capture)

    def test_ignored_ratio_and_synthetic_close_quality_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            capture = make_capture(tmp)
            capture.append_event({"type": "note_on", "note_id": "mid_1"})
            capture.append_ignored("duplicate_active_note")
            finalize_capture(capture, synthetic_close=True)
            end = json.loads(capture.result.path.read_text(encoding="utf-8").splitlines()[-1])
            quality = end["quality"]
            self.assertEqual(quality["input_count"], 2)
            self.assertEqual(quality["ignored_count"], 1)
            self.assertAlmostEqual(quality["ignored_ratio"], 0.5)
            self.assertTrue(quality["synthetic_close"])
            self.assertEqual(quality["synthetic_close_count"], 1)

    def test_write_after_close_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            capture = make_capture(tmp)
            capture.append_event({"type": "note_on", "note_id": "mid_1"})
            finalize_capture(capture)
            with self.assertRaises(RuntimeError):
                capture.append_event({"type": "note_off", "note_id": "mid_1"})
            with self.assertRaises(RuntimeError):
                capture.append_ignored("late")

    def test_invalid_duration_or_count_cannot_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            capture = make_capture(tmp)
            capture.append_event({"type": "note_on", "note_id": "mid_1"})
            with self.assertRaises(ValueError):
                capture.finalize(
                    canonical_notes=[{"notes": ["mid_1"], "dur": 1}],
                    captured_count=True,
                    duration_s=0.5,
                )
            with self.assertRaises(ValueError):
                capture.finalize(
                    canonical_notes=[{"notes": ["mid_1"], "dur": 1}],
                    captured_count=1,
                    duration_s=float("nan"),
                )
            capture.reject("invalid_metadata")

    def test_mapped_press_ids_must_pair_before_completion(self):
        event = {
            "type": "note_down",
            "press_id": 1,
            "relative_ns": 10,
            "note_id": "mid_1",
            "semitone": 0,
            "modifier": "legacy",
            "synthetic_close": False,
        }
        with tempfile.TemporaryDirectory() as tmp:
            capture = make_capture(tmp)
            capture.append_event(event)
            with self.assertRaises(ValueError):
                finalize_capture(capture)
            with self.assertRaises(ValueError):
                capture.append_event({**event, "type": "note_down"})
            capture.append_event(
                {
                    **event,
                    "type": "note_up",
                    "relative_ns": 20,
                }
            )
            result = finalize_capture(capture)
            quality = result.summary["quality"]
            self.assertEqual(quality["paired_press_count"], 1)
            self.assertEqual(quality["unclosed_press_count"], 0)

    def test_release_without_press_or_with_different_note_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            capture = make_capture(tmp)
            with self.assertRaises(ValueError):
                capture.append_event(
                    {
                        "type": "note_up",
                        "press_id": 1,
                        "relative_ns": 20,
                        "note_id": "mid_1",
                        "semitone": 0,
                        "modifier": "legacy",
                    }
                )
            capture.append_event(
                {
                    "type": "note_down",
                    "press_id": 1,
                    "relative_ns": 30,
                    "note_id": "mid_1",
                    "semitone": 0,
                    "modifier": "legacy",
                }
            )
            with self.assertRaises(ValueError):
                capture.append_event(
                    {
                        "type": "note_up",
                        "press_id": 1,
                        "relative_ns": 40,
                        "note_id": "mid_2",
                        "semitone": 0,
                        "modifier": "legacy",
                    }
                )
            capture.reject("invalid_pair")


if __name__ == "__main__":
    unittest.main()
