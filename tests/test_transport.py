import unittest

from core.transport import MIN_GAME_NOTE_MS, normalize_transport_preferences, prepare_score


class TestPrepareScore(unittest.TestCase):
    def test_slice_is_end_exclusive_and_preserves_storage(self):
        notes = [
            {"notes": ["mid_1"], "dur": 1.0},
            {"notes": [], "dur": 0.5},
            {"notes": ["mid_2"], "dur": 2.0, "semitone": 1},
        ]
        result = prepare_score(notes, start_index=1, end_index=3)
        self.assertEqual(result.source_start, 1)
        self.assertEqual(result.source_end, 3)
        self.assertEqual(result.notes, notes[1:])

    def test_chromatic_transpose_is_preserved(self):
        result = prepare_score(
            [{"notes": ["mid_1"], "dur": 1.0}], transpose=1
        )
        self.assertEqual(
            result.notes, [{"notes": ["mid_1"], "dur": 1.0, "semitone": 1}]
        )

    def test_octave_fold_keeps_time(self):
        result = prepare_score(
            [{"notes": ["high_7"], "dur": 2.0}], transpose=12
        )
        self.assertEqual(result.notes[0]["notes"], ["high_7"])
        self.assertEqual(result.notes[0]["dur"], 2.0)
        self.assertTrue(any("八度折叠" in str(item) for item in result.degradations))

    def test_no_fold_turns_out_of_range_into_rest(self):
        result = prepare_score(
            [{"notes": ["high_7"], "dur": 2.0}],
            transpose=12,
            fold_octaves=False,
        )
        self.assertEqual(result.notes, [{"notes": [], "dur": 2.0}])
        self.assertEqual(len(result.degradations), 1)

    def test_mixed_accidental_chord_degrades_deterministically(self):
        result = prepare_score(
            [{"notes": ["mid_3", "mid_4"], "dur": 1.0}], transpose=1
        )
        self.assertEqual(result.notes, [{"notes": ["mid_4"], "dur": 1.0, "semitone": 1}])
        self.assertTrue(any("混合升半音和弦" in str(item) for item in result.degradations))

    def test_rejects_bad_range_and_transpose(self):
        notes = [{"notes": ["mid_1"], "dur": 1.0}]
        for kwargs in (
            {"start_index": -1},
            {"end_index": 2},
            {"start_index": 1, "end_index": 0},
            {"transpose": 25},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                prepare_score(notes, **kwargs)

    def test_duplicate_chord_notes_are_removed_before_playback(self):
        prepared = prepare_score(
            [{"notes": ["mid_3", "mid_6", "mid_6"], "dur": 1.0}]
        )
        self.assertEqual(
            prepared.notes,
            [{"notes": ["mid_3", "mid_6"], "dur": 1.0}],
        )
        self.assertEqual(len(prepared.degradations), 1)
        self.assertIn("重复按键", prepared.degradations[0].reason)

    def test_midi_timing_fragments_are_absorbed_without_changing_duration(self):
        notes = [
            {"notes": [], "dur": 0.01},
            {"notes": ["mid_1"], "dur": 0.5},
            {"notes": ["mid_2"], "dur": 0.02},
            {"notes": ["mid_3"], "dur": 0.5},
        ]
        prepared = prepare_score(
            notes,
            bpm=120,
            min_playable_ms=MIN_GAME_NOTE_MS,
        )
        self.assertEqual(len(prepared.notes), 2)
        self.assertAlmostEqual(
            sum(item["dur"] for item in prepared.notes),
            sum(item["dur"] for item in notes),
        )
        self.assertEqual(prepared.notes[0]["notes"], ["mid_1"])
        self.assertEqual(prepared.notes[1]["notes"], ["mid_3"])
        self.assertEqual(len(prepared.degradations), 2)

    def test_all_short_fragments_still_respect_storage_duration_limit(self):
        notes = [{"notes": ["mid_1"], "dur": 0.2} for _ in range(100)]
        prepared = prepare_score(
            notes,
            bpm=300,
            min_playable_ms=MIN_GAME_NOTE_MS,
        )
        self.assertAlmostEqual(sum(item["dur"] for item in prepared.notes), 20.0)
        self.assertTrue(all(item["dur"] <= 16.0 for item in prepared.notes))


class TestPreferences(unittest.TestCase):
    def test_valid_values_survive(self):
        self.assertEqual(
            normalize_transport_preferences(
                {"bpm": 128, "transpose": -3, "segment": [2, 8]}, 10
            ),
            {"version": 1, "bpm": 128, "transpose": -3, "segment": [2, 8]},
        )

    def test_invalid_values_fall_back_without_escaping_bounds(self):
        self.assertEqual(
            normalize_transport_preferences(
                {"bpm": "bad", "transpose": 99, "segment": [-1, 99]}, 4
            ),
            {"version": 1, "bpm": 100, "transpose": 0, "segment": [0, 4]},
        )


if __name__ == "__main__":
    unittest.main()
