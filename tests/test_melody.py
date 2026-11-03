"""旋律跟踪回归：伴奏不填空、长音不重触发、移调与乐句折回。"""

import unittest

from core.melody import (
    cluster_onsets,
    extract_melody,
    phrase_fold,
    quantize_notes,
    recommend_transpose,
)
from core.source_score import AdaptOptions, SourceNote, SourceSong, adapt_source_song


def _pitches(notes):
    return [(round(note.start_s, 3), round(note.end_s, 3), note.pitch) for note in notes]


class TestOnsetMelody(unittest.TestCase):
    def test_sustained_accompaniment_does_not_fill_the_melody_rest(self):
        notes = [
            SourceNote(0.0, 0.4, 72),
            SourceNote(0.0, 2.0, 48),
            SourceNote(1.0, 1.4, 74),
        ]
        melody = extract_melody(notes)
        self.assertEqual(_pitches(melody), [(0.0, 0.4, 72), (1.0, 1.4, 74)])

    def test_long_note_is_not_reattacked_when_a_higher_voice_ends(self):
        notes = [
            SourceNote(0.0, 3.0, 60),
            SourceNote(1.0, 2.0, 64),
        ]
        melody = extract_melody(notes)
        self.assertEqual(_pitches(melody), [(0.0, 1.0, 60), (1.0, 2.0, 64)])

    def test_genuine_new_onset_of_the_same_pitch_is_kept(self):
        notes = [
            SourceNote(0.0, 3.0, 60),
            SourceNote(1.0, 2.0, 64),
            SourceNote(2.0, 3.0, 60),
        ]
        melody = extract_melody(notes)
        self.assertEqual(
            _pitches(melody),
            [(0.0, 1.0, 60), (1.0, 2.0, 64), (2.0, 3.0, 60)],
        )

    def test_rolled_chord_within_30ms_collapses_to_one_attack(self):
        notes = [
            SourceNote(0.0, 0.5, 60),
            SourceNote(0.02, 0.5, 64),
            SourceNote(0.028, 0.5, 67),
        ]
        clusters = cluster_onsets(notes, 0.032)
        self.assertEqual(len(clusters), 1)
        melody = extract_melody(notes, window_s=0.032)
        self.assertEqual([note.pitch for note in melody], [67])

    def test_notes_50ms_apart_stay_separate(self):
        notes = [SourceNote(0.0, 0.2, 60), SourceNote(0.05, 0.25, 62)]
        self.assertEqual(len(cluster_onsets(notes, 0.032)), 2)
        self.assertEqual([note.pitch for note in extract_melody(notes)], [60, 62])

    def test_cost_mode_prefers_a_small_leap_over_a_high_stab(self):
        notes = [
            SourceNote(0.0, 0.4, 60),
            SourceNote(0.5, 0.9, 62),
            SourceNote(0.5, 0.9, 84),
        ]
        skyline = extract_melody(notes, cost=False)
        smooth = extract_melody(notes, cost=True)
        self.assertEqual(skyline[-1].pitch, 84)
        self.assertEqual(smooth[-1].pitch, 62)


class TestTransposeAndPhrase(unittest.TestCase):
    def test_21key_recommendation_moves_black_keys_onto_naturals(self):
        notes = [SourceNote(index, index + 0.5, pitch) for index, pitch in enumerate((61, 63, 66))]
        advice = recommend_transpose(notes, profile="21key")
        self.assertEqual(advice.semitones, -1)
        self.assertEqual(advice.black_key_count, 0)

    def test_delta_profile_does_not_move_a_playable_black_key_line(self):
        notes = [SourceNote(index, index + 0.5, pitch) for index, pitch in enumerate((61, 63, 66))]
        advice = recommend_transpose(notes, profile="delta")
        self.assertEqual(advice.semitones, 0)

    def test_high_melody_is_brought_down_an_octave(self):
        notes = [
            SourceNote(index, index + 0.4, pitch)
            for index, pitch in enumerate((96, 98, 100, 101))
        ]
        advice = recommend_transpose(notes, profile="21key")
        self.assertEqual(advice.semitones, -24)
        self.assertEqual(advice.fold_count, 0)

    def test_phrase_fold_keeps_the_interval_shape(self):
        # 80 和 82 在音域内，84/86 越界。整句下移 12 后音程仍是 +2。
        notes = [
            SourceNote(0.0, 0.4, 80),
            SourceNote(0.4, 0.8, 82),
            SourceNote(0.8, 1.2, 84),
            SourceNote(1.2, 1.6, 86),
        ]
        folded, moved = phrase_fold(notes)
        self.assertEqual(moved, 1)
        self.assertEqual([note.pitch for note in folded], [68, 70, 72, 74])

    def test_two_phrases_can_fold_independently(self):
        low = [SourceNote(0.0, 0.4, 40), SourceNote(0.4, 0.8, 42)]
        high = [SourceNote(2.0, 2.4, 90), SourceNote(2.4, 2.8, 91)]
        folded, moved = phrase_fold(low + high, gap_s=0.45)
        self.assertEqual(moved, 2)
        self.assertEqual([note.pitch for note in folded], [52, 54, 78, 79])

    def test_quantize_strength_zero_is_identity_and_one_snaps(self):
        notes = [SourceNote(0.08, 0.4, 60)]
        same = quantize_notes(notes, bpm=120, grid_beats=0.5, strength=0.0)
        self.assertEqual(same[0].start_s, 0.08)
        snapped = quantize_notes(notes, bpm=120, grid_beats=0.5, strength=1.0)
        # 120 BPM 的半拍 = 0.25s，0.08 吸附到 0。
        self.assertAlmostEqual(snapped[0].start_s, 0.0)
        self.assertAlmostEqual(snapped[0].end_s - snapped[0].start_s, 0.32)


class TestMelodyAdapt(unittest.TestCase):
    def test_merged_tracks_follow_new_onsets_only(self):
        song = SourceSong(
            "双轨",
            [
                SourceNote(0.0, 2.0, 48, 0),
                SourceNote(0.0, 0.5, 60, 1),
                SourceNote(1.0, 1.5, 64, 1),
            ],
            {0: "伴奏", 1: "旋律"},
            2.0,
            60,
        )
        score = adapt_source_song(song, AdaptOptions(style="melody", track=None, bpm=60))
        self.assertEqual(
            [item["notes"] for item in score.notes if item["notes"]],
            [["mid_1"], ["mid_3"]],
        )
        sounded = [item for item in score.notes if item["notes"]]
        self.assertAlmostEqual(sounded[0]["dur"] + sounded[1]["dur"], 1.0, places=2)

    def test_existing_piano_style_still_bridges_short_gaps(self):
        notes = [SourceNote(0, 1, 60), SourceNote(1.04, 2, 62)]
        score = adapt_source_song(
            SourceSong("旧", notes, 1, 2, 60),
            AdaptOptions(style="piano"),
        )
        self.assertEqual([item["notes"] for item in score.notes], [["mid_1"], ["mid_2"]])
        self.assertAlmostEqual(score.notes[0]["dur"], 1.04)


if __name__ == "__main__":
    unittest.main()
