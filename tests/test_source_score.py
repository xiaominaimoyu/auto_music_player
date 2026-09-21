"""来源曲目适配器测试。运行: python -m unittest tests.test_source_score -v"""

import math
import unittest

from core.score_model import require_valid
from core.source_score import (
    AdaptOptions,
    SourceNote,
    SourceSong,
    adapt_source_song,
    bridge_short_gaps,
    monophonic_highest_active,
    piano_melody,
    recommend_track,
)


def _song(notes, *, tracks=1, duration_s=None, bpm_hint=60, tempo_change_count=1):
    if duration_s is None:
        duration_s = max((note.end_s for note in notes), default=0.0)
    return SourceSong("来源测试", notes, tracks, duration_s, bpm_hint, tempo_change_count)


def _codes(score):
    return {item.code: item.count for item in score.degradations}


class TestTrackAndCleanup(unittest.TestCase):
    def test_recommend_track_prefers_dense_melody_track(self):
        song = _song(
            [
                SourceNote(0, 2, 48, 0),
                SourceNote(0, 1, 60, 1),
                SourceNote(1, 2, 62, 1),
                SourceNote(2, 3, 64, 1),
            ],
            tracks=2,
            duration_s=3,
        )
        self.assertEqual(recommend_track(song), 1)
        adapted = adapt_source_song(song)
        self.assertEqual(adapted.selected_track, 1)
        self.assertEqual([item["notes"] for item in adapted.notes], [["mid_1"], ["mid_2"], ["mid_3"]])

    def test_recommend_track_rejects_high_dense_accompaniment(self):
        accompaniment = [
            SourceNote(index * 0.25, index * 0.25 + 1.0, 76 + index % 4, 0)
            for index in range(12)
        ]
        melody = [
            SourceNote(index, index + 0.8, 60 + index * 2, 1)
            for index in range(4)
        ]
        song = _song(
            accompaniment + melody,
            tracks={0: "Piano accompaniment", 1: "Vocal melody"},
            duration_s=4,
        )
        self.assertEqual(recommend_track(song), 1)

    def test_global_monophonic_keeps_repeated_note_as_new_event(self):
        notes = [
            SourceNote(0, 3, 60),
            SourceNote(1, 2, 64),
            SourceNote(2, 3, 60),
        ]
        melody = monophonic_highest_active(notes)
        self.assertEqual([(note.start_s, note.end_s, note.pitch) for note in melody], [(0, 1, 60), (1, 2, 64), (2, 3, 60)])
        score = adapt_source_song(_song(notes, duration_s=3), AdaptOptions(style="original"))
        self.assertEqual([item["notes"] for item in score.notes], [["mid_1"], ["mid_3"], ["mid_1"]])
        self.assertGreater(_codes(score)["polyphony_cleanup"], 0)

    def test_piano_bridges_short_gap_without_rest(self):
        notes = [SourceNote(0, 1, 60), SourceNote(1.04, 2, 62)]
        bridged = bridge_short_gaps(piano_melody(notes))
        self.assertAlmostEqual(bridged[0].end_s, 1.04)
        score = adapt_source_song(_song(notes, duration_s=2))
        self.assertEqual([item["notes"] for item in score.notes], [["mid_1"], ["mid_2"]])
        self.assertAlmostEqual(score.notes[0]["dur"], 1.04)
        self.assertAlmostEqual(score.notes[1]["dur"], 0.96)
        self.assertEqual(_codes(score)["short_gap_bridged"], 1)

    def test_dense_overlap_uses_sweep_without_changing_highest_rule(self):
        notes = [SourceNote(index / 1000, 6, 48 + index % 36) for index in range(5000)]
        melody = monophonic_highest_active(notes)
        self.assertEqual(melody[-1].pitch, 83)
        self.assertLessEqual(len(melody), len(notes))


class TestMappingAndTiming(unittest.TestCase):
    def test_all_chromatic_mapping_uses_natural_base_and_semitone(self):
        notes = [SourceNote(index, index + 1, 48 + index) for index in range(12)]
        score = adapt_source_song(_song(notes, duration_s=12), AdaptOptions(style="original"))
        self.assertEqual(
            [(item["notes"], item.get("semitone", 0)) for item in score.notes],
            [
                (["low_1"], 0), (["low_1"], 1), (["low_2"], 0), (["low_2"], 1),
                (["low_3"], 0), (["low_4"], 0), (["low_4"], 1), (["low_5"], 0),
                (["low_5"], 1), (["low_6"], 0), (["low_6"], 1), (["low_7"], 0),
            ],
        )

    def test_octave_fold_and_disabled_fold_rest_preserve_duration(self):
        song = _song([SourceNote(0, 1, 36)], duration_s=1)
        folded = adapt_source_song(song, AdaptOptions(style="original", fold_octaves=True))
        self.assertEqual(folded.notes, [{"notes": ["low_1"], "dur": 1.0}])
        self.assertEqual(_codes(folded)["octave_folded"], 1)
        no_fold = adapt_source_song(song, AdaptOptions(style="original", fold_octaves=False))
        self.assertEqual(no_fold.notes, [{"notes": [], "dur": 1.0}])
        self.assertEqual(_codes(no_fold)["unsupported_note"], 1)

    def test_leading_inter_note_and_trailing_rests_keep_absolute_timing(self):
        song = _song([SourceNote(1, 2, 60), SourceNote(3, 4, 62)], duration_s=5)
        score = adapt_source_song(song, AdaptOptions(style="original"))
        self.assertEqual(
            score.notes,
            [
                {"notes": [], "dur": 1.0},
                {"notes": ["mid_1"], "dur": 1.0},
                {"notes": [], "dur": 1.0},
                {"notes": ["mid_2"], "dur": 1.0},
                {"notes": [], "dur": 1.0},
            ],
        )

    def test_variable_tempo_is_explicitly_flattened(self):
        score = adapt_source_song(
            _song([SourceNote(0, 1, 60)], duration_s=1, bpm_hint=120, tempo_change_count=3),
            AdaptOptions(style="original"),
        )
        self.assertEqual(score.bpm, 120)
        self.assertEqual(_codes(score)["tempo_flattened"], 2)
        self.assertTrue(any("固定换算" in warning for warning in score.warnings))

    def test_duration_over_sixteen_beats_is_split_without_time_loss(self):
        score = adapt_source_song(
            _song([SourceNote(0, 17, 60)], duration_s=17), AdaptOptions(style="original")
        )
        self.assertEqual(score.notes, [{"notes": ["mid_1"], "dur": 16.0}, {"notes": ["mid_1"], "dur": 1.0}])
        self.assertEqual(sum(item["dur"] for item in score.notes), 17.0)
        require_valid(score.notes, bpm=score.bpm)


class TestPreserveAndValidation(unittest.TestCase):
    def test_preserve_keeps_compatible_chord_and_reduces_mixed_semitones(self):
        compatible = adapt_source_song(
            _song([SourceNote(0, 1, 60), SourceNote(0, 1, 64)], duration_s=1),
            AdaptOptions(style="preserve"),
        )
        self.assertEqual(compatible.notes, [{"notes": ["mid_1", "mid_3"], "dur": 1.0}])
        mixed = adapt_source_song(
            _song([SourceNote(0, 1, 60), SourceNote(0, 1, 61)], duration_s=1),
            AdaptOptions(style="preserve"),
        )
        self.assertEqual(mixed.notes, [{"notes": ["mid_1"], "dur": 1.0, "semitone": 1}])
        self.assertEqual(_codes(mixed)["mixed_semitone_chord"], 1)

    def test_invalid_non_finite_and_out_of_range_input_is_rejected(self):
        invalid_songs = [
            _song([SourceNote(math.nan, 1, 60)], duration_s=1),
            _song([SourceNote(0, 1, 128)], duration_s=1),
            _song([SourceNote(1, 1, 60)], duration_s=1),
            _song([SourceNote(0, 2, 60)], duration_s=1),
        ]
        for song in invalid_songs:
            with self.subTest(song=song):
                with self.assertRaises(ValueError):
                    adapt_source_song(song)
        with self.assertRaises(ValueError):
            adapt_source_song(_song([], duration_s=0), AdaptOptions(style="wrong"))
        with self.assertRaisesRegex(ValueError, "音轨编号"):
            adapt_source_song(
                SourceSong(
                    "越界轨道",
                    [SourceNote(0, 1, 60, 256)],
                    {256: "invalid"},
                    1,
                    60,
                )
            )


if __name__ == "__main__":
    unittest.main()
