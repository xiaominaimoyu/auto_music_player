import copy
import math
import unittest

from core.score_model import MAX_DUR_BEATS, ScoreValidationError, require_valid
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

    def test_absorption_caps_leading_and_trailing_overflow_without_short_tail(self):
        for fragments in (
            [{"notes": [], "dur": .04}, {"notes": ["mid_1"], "dur": 16}],
            [{"notes": ["mid_1"], "dur": 16}, {"notes": ["mid_2"], "dur": .04}],
        ):
            with self.subTest(fragments=fragments):
                prepared = prepare_score(fragments, bpm=120, min_playable_ms=60)
                self.assertEqual([item["notes"] for item in prepared.notes], [["mid_1"], ["mid_1"]])
                self.assertAlmostEqual(prepared.notes[0]["dur"], 15.92)
                self.assertAlmostEqual(prepared.notes[1]["dur"], .12)
                self.assertAlmostEqual(math.fsum(item["dur"] for item in prepared.notes), 16.04)
                self.assertEqual(len(prepared.degradations), 2)
                self.assertIn("补足末段", prepared.degradations[-1].reason)
                self.assertIn("重新触发", prepared.degradations[-1].reason)
                require_valid(prepared.notes)

    def test_split_preserves_chord_semitone_source_indices_and_input(self):
        notes = [
            {"notes": ["low_1"], "dur": 1},
            {"notes": [], "dur": .04},
            {"notes": ["mid_1", "mid_3"], "dur": 16, "semitone": 1},
            {"notes": ["mid_2"], "dur": .04},
            {"notes": ["high_1"], "dur": 1},
        ]
        original = copy.deepcopy(notes)
        prepared = prepare_score(notes, start_index=1, end_index=4, bpm=120, min_playable_ms=60)
        self.assertEqual((prepared.source_start, prepared.source_end), (1, 4))
        self.assertEqual([d.index for d in prepared.degradations], [1, 3, 2])
        self.assertEqual([item["notes"] for item in prepared.notes], [["mid_1", "mid_3"]] * 2)
        self.assertEqual([item["semitone"] for item in prepared.notes], [1, 1])
        self.assertAlmostEqual(math.fsum(item["dur"] for item in prepared.notes), 16.08)
        self.assertEqual(notes, original)
        prepared.notes[0]["notes"].append("mid_5")
        self.assertEqual(prepared.notes[1]["notes"], ["mid_1", "mid_3"])
        self.assertEqual(notes, original)

    def test_long_rest_preserved_and_following_boundary_unchanged(self):
        notes = [{"notes": [], "dur": 16}, {"notes": ["mid_1"], "dur": .04},
                 {"notes": ["mid_2"], "dur": 1}]
        prepared = prepare_score(notes, bpm=120, min_playable_ms=60)
        self.assertEqual([item["notes"] for item in prepared.notes], [[], [], ["mid_2"]])
        self.assertAlmostEqual(math.fsum(item["dur"] for item in prepared.notes[:-1]), 16.04)
        self.assertEqual(prepared.notes[-1], notes[-1])
        self.assertIn("休止切分不触发按键", prepared.degradations[-1].reason)

    def test_all_short_split_keeps_existing_cap_boundaries_except_tiny_tail(self):
        for count, tail, expected in (
            (160, 0, [16]),
            (160, .02, [15.9, .12]),
            (164, 0, [16, .4]),
            (320, 0, [16, 16]),
            (320, .02, [16, 15.9, .12]),
        ):
            notes = [{"notes": ["mid_1"], "dur": .1} for _ in range(count)]
            if tail:
                notes.append({"notes": ["mid_2"], "dur": tail})
            with self.subTest(count=count, tail=tail):
                prepared = prepare_score(notes, bpm=120, min_playable_ms=60)
                self.assertEqual(len(prepared.notes), len(expected))
                for item, duration in zip(prepared.notes, expected):
                    self.assertAlmostEqual(item["dur"], duration)
                    self.assertEqual(item["notes"], ["mid_1"])
                    self.assertGreaterEqual(item["dur"] * 500 + 1e-9, 60)
                self.assertAlmostEqual(math.fsum(item["dur"] for item in prepared.notes),
                                       math.fsum(item["dur"] for item in notes))
                require_valid(prepared.notes)

    def test_all_short_split_degradation_retains_absolute_anchor_index(self):
        notes = [{"notes": ["low_1"], "dur": 1},
                 {"notes": ["mid_2"], "dur": .01},
                 {"notes": ["mid_3"], "dur": .119, "semitone": 1}]
        notes.extend({"notes": [], "dur": .1} for _ in range(160))
        prepared = prepare_score(notes, start_index=1, bpm=120, min_playable_ms=60)
        self.assertEqual(prepared.degradations[-1].index, 2)
        self.assertTrue(all(item["notes"] == ["mid_3"] and item["semitone"] == 1 for item in prepared.notes))
        self.assertAlmostEqual(math.fsum(item["dur"] for item in prepared.notes), 16.129)

    def test_split_preserves_just_over_cap_precision_and_highest_minimum(self):
        for bpm, minimum_ms, fragment in ((120, 60, math.ulp(16.0)), (300, 1000, .01)):
            with self.subTest(bpm=bpm, minimum_ms=minimum_ms):
                notes = [{"notes": ["mid_1"], "dur": 16}, {"notes": [], "dur": fragment}]
                prepared = prepare_score(notes, bpm=bpm, min_playable_ms=minimum_ms)
                self.assertEqual(len(prepared.notes), 2)
                self.assertEqual(math.fsum(item["dur"] for item in prepared.notes), 16 + fragment)
                self.assertTrue(all(0 < item["dur"] <= MAX_DUR_BEATS for item in prepared.notes))
                self.assertTrue(all(item["dur"] * 60000 / bpm + 1e-9 >= minimum_ms for item in prepared.notes))

    def test_total_shorter_than_minimum_preserved_even_below_old_epsilon(self):
        for duration in (.01, 1e-12):
            notes = [{"notes": ["mid_1"], "dur": duration}, {"notes": [], "dur": duration}]
            prepared = prepare_score(notes, bpm=120, min_playable_ms=60)
            self.assertEqual(prepared.notes, [{"notes": ["mid_1"], "dur": duration * 2}])
        self.assertEqual(prepare_score([{"notes": [], "dur": .01}], bpm=120, min_playable_ms=60).notes,
                         [{"notes": [], "dur": .01}])

    def test_invalid_source_cannot_be_hidden_by_absorption_or_transposition(self):
        for invalid in (
            {"notes": ["mid_1"], "dur": duration}
            for duration in (16.01, float("nan"), float("inf"), float("-inf"), -1, 0, "1", True)
        ):
            for transpose in (0, 1):
                with self.subTest(invalid=invalid, transpose=transpose), self.assertRaises(ScoreValidationError):
                    prepare_score([invalid, {"notes": ["mid_2"], "dur": .01}],
                                  bpm=120, min_playable_ms=60, transpose=transpose)
        for invalid in ({"notes": ["bad"], "dur": .01}, {"notes": None, "dur": .01}, None):
            with self.subTest(invalid=invalid), self.assertRaises(ScoreValidationError):
                prepare_score([invalid, {"notes": ["mid_2"], "dur": 1}], bpm=120, min_playable_ms=60)

    def test_only_selected_slice_is_validated(self):
        notes = [{"notes": ["bad"], "dur": float("nan")}, {"notes": ["mid_1"], "dur": 1}]
        self.assertEqual(prepare_score(notes, start_index=1).notes, notes[1:])


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
