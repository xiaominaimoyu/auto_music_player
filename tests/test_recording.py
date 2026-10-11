import unittest

from core.keymap import KeyMap
from core.profile import GameProfile
from core.recording import PerformanceRecorder, PhysicalNoteResolver


MAPPING = {
    "high": list("QWERTYU"),
    "mid": list("ASDFGHJ"),
    "low": list("ZXCVBNM"),
}


class FakeClock:
    def __init__(self):
        self.value = 100.0

    def __call__(self):
        return self.value


class TestPerformanceRecorder(unittest.TestCase):
    def make_legacy(self):
        clock = FakeClock()
        recorder = PerformanceRecorder(
            PhysicalNoteResolver(keymap=KeyMap(MAPPING)), clock=clock
        )
        recorder.start()
        return recorder, clock

    def test_timing_quantizes_and_keeps_gap_as_rest(self):
        recorder, clock = self.make_legacy()
        clock.value = 100.1
        recorder.press("a")
        clock.value = 100.6
        recorder.release("A")
        clock.value = 101.1
        recorder.press("s")
        clock.value = 101.6
        recorder.release("s")
        result = recorder.stop(bpm=120, quantize_beats=0.25)
        self.assertEqual(
            result.notes,
            [
                {"notes": ["mid_1"], "dur": 1.0},
                {"notes": [], "dur": 1.0},
                {"notes": ["mid_2"], "dur": 1.0},
            ],
        )
        self.assertEqual(result.captured_count, 2)

    def test_near_simultaneous_keys_become_chord_after_quantize(self):
        recorder, clock = self.make_legacy()
        clock.value = 100.10
        recorder.press("a")
        clock.value = 100.12
        recorder.press("s")
        clock.value = 100.60
        recorder.release("a")
        clock.value = 100.61
        recorder.release("s")
        result = recorder.stop(bpm=120, quantize_beats=0.25)
        self.assertEqual(
            result.notes, [{"notes": ["mid_1", "mid_2"], "dur": 1.0}]
        )

    def test_stop_closes_held_note_and_ignores_repeat_down(self):
        recorder, clock = self.make_legacy()
        clock.value = 100.2
        self.assertTrue(recorder.press("a"))
        self.assertFalse(recorder.press("a"))
        clock.value = 100.7
        result = recorder.stop(bpm=120, quantize_beats=0.25)
        self.assertEqual(result.captured_count, 1)
        self.assertEqual(result.notes, [{"notes": ["mid_1"], "dur": 1.0}])
        self.assertEqual([event.action for event in result.raw_events], ["down", "up"])
        self.assertFalse(result.raw_events[0].closed_by_stop)
        self.assertTrue(result.raw_events[1].closed_by_stop)

    def test_raw_events_keep_unquantized_relative_nanoseconds(self):
        clock = FakeClock()
        streamed = []
        resolver = PhysicalNoteResolver(keymap=KeyMap(MAPPING))
        recorder = PerformanceRecorder(
            resolver,
            clock=clock,
            event_sink=streamed.append,
        )
        recorder.start()
        clock.value = 100.123
        self.assertTrue(recorder.press("a"))
        clock.value = 100.456
        self.assertTrue(recorder.release("a"))

        result = recorder.stop(bpm=120, quantize_beats=0.25)

        self.assertEqual(result.notes, [{"notes": ["mid_1"], "dur": 0.75}])
        self.assertEqual(len(result.raw_events), 2)
        self.assertEqual(result.raw_events, streamed)
        self.assertEqual(result.raw_events[0].relative_ns, 123_000_000)
        self.assertEqual(result.raw_events[1].relative_ns, 456_000_000)
        self.assertEqual(result.raw_events[0].press_id, 1)
        self.assertEqual(result.raw_events[1].press_id, 1)
        self.assertEqual(result.raw_events[0].note_id, "mid_1")
        self.assertEqual(result.raw_events[0].modifier, "legacy")
        self.assertEqual(
            result.raw_events[0].to_performance_event()["modifier"], "legacy"
        )
        self.assertFalse(hasattr(result.raw_events[0], "key"))

    def test_unmapped_inputs_are_counted_but_never_enter_raw_events(self):
        recorder, clock = self.make_legacy()
        clock.value = 100.1
        self.assertFalse(recorder.press("escape"))
        self.assertEqual(recorder.ignored_count, 1)
        self.assertEqual(recorder.raw_events, ())
        clock.value = 100.2
        recorder.press("a")
        clock.value = 100.4
        recorder.release("a")

        result = recorder.stop(bpm=120)

        self.assertEqual(result.ignored_count, 1)
        self.assertEqual({event.note_id for event in result.raw_events}, {"mid_1"})

    def test_capture_sink_failure_does_not_break_score_recording(self):
        def broken_sink(_event):
            raise OSError("disk full")

        clock = FakeClock()
        recorder = PerformanceRecorder(
            PhysicalNoteResolver(keymap=KeyMap(MAPPING)),
            clock=clock,
            event_sink=broken_sink,
        )
        recorder.start()
        clock.value = 100.1
        recorder.press("a")
        clock.value = 100.6
        recorder.release("a")

        result = recorder.stop(bpm=120)

        self.assertEqual(result.captured_count, 1)
        self.assertEqual(len(result.capture_errors), 1)
        self.assertIn("disk full", result.capture_errors[0])

    def test_event_profile_mouse_modifier_maps_octave_and_semitone(self):
        profile = GameProfile(
            id="delta",
            name="Delta",
            pitch_keys=("Z", "X", "C", "V", "B", "N", "M"),
            modifier_buttons={"lower": "left", "semitone": "middle", "higher": "right"},
            legacy_keymap=None,
        )
        resolver = PhysicalNoteResolver(profile=profile)
        self.assertEqual(resolver.resolve("z", ["right"]).note_id, "high_1")
        semitone = resolver.resolve("x", ["middle"])
        self.assertEqual((semitone.note_id, semitone.semitone), ("mid_2", 1))
        self.assertEqual(semitone.modifier, "semitone")
        self.assertIsNone(resolver.resolve("z", ["left", "right"]))
        self.assertEqual(resolver.mapping_mode, "event_profile")
        self.assertEqual(len(resolver.mapping_hash), 64)

    def test_empty_capture_is_rejected(self):
        recorder, _clock = self.make_legacy()
        with self.assertRaises(ValueError):
            recorder.stop(bpm=120)


if __name__ == "__main__":
    unittest.main()
