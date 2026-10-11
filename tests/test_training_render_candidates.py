import io
import tempfile
from pathlib import Path
import unittest
import wave

import numpy as np

from training.midi_features import parse_midi_bytes
from training.render_expression_candidates import (
    ListeningNote, expression_plan, limit_retrigger_overlap,
    original_study_midi, plan_midi_bytes, score_inference_arrays, write_audio_pair,
)


class CandidateRenderingTests(unittest.TestCase):
    def test_original_studies_identity_roundtrip(self):
        for index in range(3):
            score = parse_midi_bytes(original_study_midi(index))
            arrays = score_inference_arrays(score)
            self.assertFalse(arrays["note_masks"].any())
            self.assertFalse(arrays["chord_masks"].any())
            self.assertEqual(float(arrays["note_targets"].sum()), 0)
            plan, meta = expression_plan(score, arrays["note_to_chord"])
            midi = parse_midi_bytes(plan_midi_bytes(plan))
            self.assertEqual(len(midi.notes), len(score.notes))
            self.assertEqual(sorted(n.pitch for n in midi.notes), sorted(n.pitch for n in score.notes))
            self.assertFalse(meta["uses_oracle_performance"])
            for actual, original in zip(midi.notes, score.notes):
                self.assertAlmostEqual(actual.start_s, original.start_s, delta=0.0011)

    def test_reconstruction_is_integrated_and_bounded(self):
        score = parse_midi_bytes(original_study_midi(0))
        arrays = score_inference_arrays(score)
        ny = np.zeros_like(arrays["note_targets"])
        cy = np.zeros_like(arrays["chord_targets"])
        ny[:, 0], ny[:, 1], cy[:, 0], cy[:, 1] = -10000, 10000, 1e6, 10000
        plan, meta = expression_plan(score, arrays["note_to_chord"], ny, cy)
        self.assertGreater(meta["raw_ioi_clipped"], 0)
        self.assertTrue(all(n.start_s >= 0 and n.end_s > n.start_s for n in plan))
        self.assertTrue(all(n.velocity == 76 for n in plan))
        expected = score.notes[-1].start_s * np.exp(0.25 * 0.5) - 0.015
        self.assertAlmostEqual(plan[-1].start_s, expected, places=5)

    def test_retrigger_guard_and_invalid_values(self):
        notes = [ListeningNote(60, 0, 2), ListeningNote(60, 1, 3)]
        self.assertEqual(limit_retrigger_overlap(notes)[0].end_s, 1)
        with self.assertRaises(ValueError):
            limit_retrigger_overlap([ListeningNote(60, 0, 1), ListeningNote(60, 0, 2)])
        with self.assertRaises(ValueError):
            plan_midi_bytes([ListeningNote(60, float("nan"), 1)])

    def test_pair_common_gain_and_pcm(self):
        with tempfile.TemporaryDirectory() as directory:
            left, right = Path(directory) / "left.wav", Path(directory) / "right.wav"
            a, b = np.full((100, 2), 2.0), np.full((100, 2), 1.0)
            metadata = write_audio_pair(a, b, left, right)
            self.assertAlmostEqual(metadata["shared_gain"], 0.4)
            with wave.open(str(left)) as handle:
                pcm_a = np.frombuffer(handle.readframes(100), dtype="<i2")
            with wave.open(str(right)) as handle:
                pcm_b = np.frombuffer(handle.readframes(100), dtype="<i2")
            self.assertAlmostEqual(float(pcm_a[0]) / float(pcm_b[0]), 2, delta=0.001)
            with self.assertRaises(ValueError):
                write_audio_pair(a, b, left, right)


if __name__ == "__main__":
    unittest.main()
