"""曲谱导入导出测试:JSON 往返、MIDI 导出/解析、边界与降级路径。

运行: python -m unittest tests.test_score_io -v
"""

import json
import os
import tempfile
import unittest

from core.score_io import (
    TPQ,
    MidiParseError,
    _chunk,
    _vlq,
    export_json,
    export_midi,
    import_any,
    import_json,
    import_json_many,
    import_midi,
    inspect_midi,
    midi_to_note_id,
    note_id_to_midi,
)
from core.score_model import ScoreValidationError

SCORE = {
    "name": "小星星",
    "bpm_default": 96,
    "notes": [
        {"notes": ["mid_1"], "dur": 1.0},
        {"notes": ["mid_1"], "dur": 1.0},
        {"notes": ["high_5"], "dur": 2.0},
        {"notes": [], "dur": 1.0},
        {"notes": ["mid_1", "mid_3", "mid_5"], "dur": 1.5},
    ],
}


class TestNoteMapping(unittest.TestCase):
    def test_all_21_roundtrip(self):
        for octave in ("low", "mid", "high"):
            for num in range(1, 8):
                nid = f"{octave}_{num}"
                self.assertEqual(midi_to_note_id(note_id_to_midi(nid)), nid)

    def test_boundary_notes(self):
        self.assertEqual(note_id_to_midi("low_1"), 48)
        self.assertEqual(note_id_to_midi("mid_1"), 60)
        self.assertEqual(note_id_to_midi("high_7"), 83)

    def test_black_keys_and_range(self):
        for m in (49, 51, 54, 56, 58):   # C3 八度的黑键
            self.assertIsNone(midi_to_note_id(m))
        for m in (47, 84, 100):          # 范围外
            self.assertIsNone(midi_to_note_id(m))

    def test_invalid_note_id(self):
        with self.assertRaises(ValueError):
            note_id_to_midi("high_8")
        with self.assertRaises(ValueError):
            note_id_to_midi("bogus")


class TestJson(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "score.json")

    def tearDown(self):
        for f in os.listdir(self.dir):
            os.remove(os.path.join(self.dir, f))
        os.rmdir(self.dir)

    def test_roundtrip(self):
        export_json(self.path, SCORE)
        result = import_json(self.path)
        self.assertEqual(result.name, "小星星")
        self.assertEqual(result.bpm, 96)
        self.assertEqual(result.notes, SCORE["notes"])
        self.assertEqual(result.kind, "json")

    def test_import_rejects_invalid_notes(self):
        bad = {"name": "x", "bpm": 100, "notes": [{"notes": ["bad_1"], "dur": 1.0}]}
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(bad, f)
        with self.assertRaises(ScoreValidationError):
            import_json(self.path)

    def test_import_rejects_bad_bpm(self):
        bad = {"name": "x", "bpm": 999, "notes": SCORE["notes"]}
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(bad, f)
        with self.assertRaises(ScoreValidationError):
            import_json(self.path)

    def test_import_rejects_bad_structure(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump({"foo": 1}, f)
        with self.assertRaises(ValueError):
            import_json(self.path)

    def test_import_raw_list(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(SCORE["notes"], f)
        result = import_json(self.path)
        self.assertEqual(result.notes, SCORE["notes"])
        self.assertEqual(result.bpm, 100)


class TestMidiRoundtrip(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "score.mid")

    def tearDown(self):
        for f in os.listdir(self.dir):
            os.remove(os.path.join(self.dir, f))
        os.rmdir(self.dir)

    def _roundtrip(self, score):
        export_midi(self.path, score)
        return import_midi(self.path)

    def test_monophonic(self):
        score = {"name": "旋律", "bpm_default": 96,
                 "notes": [
                     {"notes": ["mid_1"], "dur": 1.0},
                     {"notes": ["mid_2"], "dur": 0.5},
                     {"notes": ["high_5"], "dur": 2.0},
                 ]}
        result = self._roundtrip(score)
        self.assertEqual(result.bpm, 96)
        self.assertEqual(len(result.notes), 3)
        self.assertEqual([e["notes"][0] for e in result.notes], ["mid_1", "mid_2", "high_5"])
        for got, want in zip(result.notes, score["notes"]):
            self.assertAlmostEqual(got["dur"], want["dur"], places=6)

    def test_chord(self):
        score = {"name": "和弦", "bpm_default": 100,
                 "notes": [{"notes": ["mid_1", "mid_3", "mid_5"], "dur": 1.0}]}
        result = self._roundtrip(score)
        self.assertEqual(len(result.notes), 1)
        self.assertEqual(result.notes[0]["notes"], ["mid_1", "mid_3", "mid_5"])
        self.assertAlmostEqual(result.notes[0]["dur"], 1.0, places=6)

    def test_rest_is_preserved_as_explicit_timeline_element(self):
        """绝对时间适配保留真正休止，后续编辑/跳转不再猜测间隙。"""
        score = {"name": "r", "bpm_default": 100,
                 "notes": [
                     {"notes": ["mid_1"], "dur": 1.0},
                     {"notes": [], "dur": 1.0},
                     {"notes": ["mid_3"], "dur": 1.0},
                 ]}
        result = self._roundtrip(score)
        self.assertEqual([e["notes"] for e in result.notes], [["mid_1"], [], ["mid_3"]])
        self.assertEqual(sum(e["dur"] for e in result.notes), 3.0)

    def test_trailing_rest_survives_midi_roundtrip(self):
        score = {
            "name": "尾部休止",
            "bpm_default": 120,
            "notes": [
                {"notes": ["mid_1"], "dur": 1.0},
                {"notes": [], "dur": 2.0},
            ],
        }
        result = self._roundtrip(score)
        self.assertEqual([item["notes"] for item in result.notes], [["mid_1"], []])
        self.assertAlmostEqual(sum(item["dur"] for item in result.notes), 3.0)

    def test_semitone_roundtrip(self):
        score = {
            "name": "半音",
            "bpm_default": 120,
            "notes": [{"notes": ["mid_1"], "dur": 1.0, "semitone": 1}],
        }
        result = self._roundtrip(score)
        self.assertEqual(result.notes, score["notes"])

    def test_dotted_and_sixteenth(self):
        score = {"name": "d", "bpm_default": 100,
                 "notes": [
                     {"notes": ["mid_5"], "dur": 0.75},
                     {"notes": ["mid_6"], "dur": 0.25},
                 ]}
        result = self._roundtrip(score)
        self.assertAlmostEqual(result.notes[0]["dur"], 0.75, places=6)
        self.assertAlmostEqual(result.notes[1]["dur"], 0.25, places=6)


class TestMidiImportEdges(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "t.mid")

    def tearDown(self):
        for f in os.listdir(self.dir):
            os.remove(os.path.join(self.dir, f))
        os.rmdir(self.dir)

    def _write(self, data: bytes):
        with open(self.path, "wb") as f:
            f.write(data)
        return self.path

    @staticmethod
    def _midi(fmt=0, ntrks=1, division=TPQ, track=b""):
        header = (
            b"MThd" + (6).to_bytes(4, "big")
            + fmt.to_bytes(2, "big") + ntrks.to_bytes(2, "big") + division.to_bytes(2, "big")
        )
        return header + _chunk(b"MTrk", track)

    def test_running_status(self):
        track = (
            _vlq(0) + bytes((0x90, 60, 100))     # mid_1 on
            + _vlq(TPQ) + bytes((60, 100))       # running status:同状态再触发
            + _vlq(0) + bytes((0x80, 60, 0))
            + _vlq(TPQ) + bytes((0x80, 60, 0))
            + _vlq(0) + b"\xff\x2f\x00"
        )
        result = import_midi(self._write(self._midi(track=track)))
        self.assertEqual(len(result.notes), 2)
        self.assertEqual([e["notes"][0] for e in result.notes], ["mid_1", "mid_1"])

    def test_black_key_becomes_semitone(self):
        track = (
            _vlq(0) + bytes((0x90, 61, 100))     # C#4 黑键
            + _vlq(TPQ) + bytes((0x80, 61, 0))
            + _vlq(0) + b"\xff\x2f\x00"
        )
        result = import_midi(self._write(self._midi(track=track)))
        self.assertEqual(len(result.notes), 1)
        self.assertEqual(
            result.notes[0], {"notes": ["mid_1"], "dur": 1.0, "semitone": 1}
        )

    def test_out_of_range_folds_by_octave_with_report(self):
        track = (
            _vlq(0) + bytes((0x90, 100, 100))    # 超出 C3~B5
            + _vlq(TPQ) + bytes((0x80, 100, 0))
            + _vlq(0) + b"\xff\x2f\x00"
        )
        result = import_midi(self._write(self._midi(track=track)))
        self.assertEqual(result.notes[0]["notes"], ["high_3"])
        self.assertTrue(any("八度折回" in w for w in result.warnings))
        no_fold = import_midi(self.path, fold_octaves=False)
        self.assertEqual(no_fold.notes[0]["notes"], [])
        self.assertTrue(any("等时值休止" in w for w in no_fold.warnings))

    def test_percussion_channel_skipped(self):
        track = (
            _vlq(0) + bytes((0x99, 60, 100))     # 第 10 通道(ch=9)
            + _vlq(TPQ) + bytes((0x89, 60, 0))
            + _vlq(0) + b"\xff\x2f\x00"
        )
        result = import_midi(self._write(self._midi(track=track)))
        self.assertEqual(result.notes, [])
        self.assertTrue(any("打击乐" in w for w in result.warnings))

    def test_bpm_from_tempo_and_clamp(self):
        track = _vlq(0) + b"\xff\x51\x03" + (120000).to_bytes(3, "big") + _vlq(0) + b"\xff\x2f\x00"
        result = import_midi(self._write(self._midi(track=track)))
        self.assertEqual(result.bpm, 300)   # 500 → 钳制到 300
        self.assertTrue(any("调整为" in w for w in result.warnings))
        track = _vlq(0) + b"\xff\x51\x03" + (625000).to_bytes(3, "big") + _vlq(0) + b"\xff\x2f\x00"
        result = import_midi(self._write(self._midi(track=track)))
        self.assertEqual(result.bpm, 96)    # 60000000/625000 = 96

    def test_inspection_clamps_extreme_tempo_before_track_dialog(self):
        track = (
            _vlq(0) + b"\xff\x51\x03" + (120000).to_bytes(3, "big")
            + _vlq(0) + bytes((0x90, 60, 100))
            + _vlq(TPQ) + bytes((0x80, 60, 0))
            + _vlq(0) + b"\xff\x2f\x00"
        )
        parsed = inspect_midi(self._write(self._midi(track=track)))
        self.assertEqual(parsed.song.bpm_hint, 300)
        self.assertTrue(any("调整为 300" in warning for warning in parsed.warnings))

    def test_smpte_rejected(self):
        with self.assertRaises(MidiParseError):
            import_midi(self._write(self._midi(division=0xE728)))

    def test_type2_rejected(self):
        with self.assertRaises(MidiParseError):
            import_midi(self._write(self._midi(fmt=2, ntrks=2)))

    def test_bad_magic_rejected(self):
        with self.assertRaises(MidiParseError):
            import_midi(self._write(b"not a midi file at all....."))

    def test_global_tempo_map_and_utf8_track_name(self):
        import mido

        midi = mido.MidiFile(type=1, ticks_per_beat=TPQ, charset="utf8")
        tempo = mido.MidiTrack()
        tempo.append(mido.MetaMessage("set_tempo", tempo=500_000, time=0))
        tempo.append(mido.MetaMessage("set_tempo", tempo=1_000_000, time=TPQ))
        midi.tracks.append(tempo)
        melody = mido.MidiTrack()
        melody.append(mido.MetaMessage("track_name", name="主旋律", time=0))
        melody.append(mido.Message("note_on", note=60, velocity=100, time=0))
        melody.append(mido.Message("note_off", note=60, velocity=0, time=TPQ))
        melody.append(mido.Message("note_on", note=62, velocity=100, time=0))
        melody.append(mido.Message("note_off", note=62, velocity=0, time=TPQ))
        midi.tracks.append(melody)
        midi.save(self.path)

        parsed = inspect_midi(self.path)
        self.assertEqual(parsed.song.tracks[1], "主旋律")
        self.assertAlmostEqual(parsed.song.notes[0].end_s, 0.5)
        self.assertAlmostEqual(parsed.song.notes[1].end_s, 1.5)
        result = import_midi(self.path, track=1, style="original")
        self.assertAlmostEqual(sum(item["dur"] for item in result.notes), 3.0)
        self.assertTrue(any("速度变化" in warning for warning in result.warnings))


class TestExternalJsonRecognition(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.temp.name, "external.json")

    def tearDown(self):
        self.temp.cleanup()

    def _write(self, data):
        with open(self.path, "w", encoding="utf-8") as stream:
            json.dump(data, stream, ensure_ascii=False)

    def test_single_simulator_events_are_recognized(self):
        self._write(
            {
                "title": "录制片段",
                "events": [
                    {"t": 0, "d": 0.4, "key": "z", "note": "1", "mouse": []},
                    {"t": 0.5, "d": 0.4, "key": "x", "note": "2", "mouse": ["middle"]},
                ],
            }
        )
        result = import_json(self.path)
        self.assertEqual(result.name, "录制片段")
        self.assertEqual(result.source_format, "koufengqin-events")
        self.assertEqual(result.notes[0]["notes"], ["mid_1"])
        self.assertTrue(any(item.get("semitone") == 1 for item in result.notes))
        self.assertTrue(any("120 BPM" in warning for warning in result.warnings))

    def test_library_envelope_imports_every_song(self):
        event = {"t": 0, "d": 0.5, "key": "z", "note": "1", "mouse": []}
        self._write(
            {
                "app": "口风琴模拟",
                "version": 1,
                "library": [
                    {"title": "一", "events": [event]},
                    {"title": "二", "events": [event]},
                ],
            }
        )
        results = import_json_many(self.path)
        self.assertEqual([result.name for result in results], ["一", "二"])
        with self.assertRaisesRegex(ValueError, "包含 2 首"):
            import_json(self.path)


class TestDispatch(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def tearDown(self):
        for f in os.listdir(self.dir):
            os.remove(os.path.join(self.dir, f))
        os.rmdir(self.dir)

    def test_import_any_dispatch(self):
        p = os.path.join(self.dir, "a.json")
        export_json(p, SCORE)
        self.assertEqual(import_any(p).kind, "json")
        p = os.path.join(self.dir, "b.mid")
        export_midi(p, SCORE)
        self.assertEqual(import_any(p).kind, "midi")
        p = os.path.join(self.dir, "c.mid")
        export_midi(p, SCORE)
        self.assertEqual(import_any(p).name, "c")

    def test_import_any_unknown_ext(self):
        p = os.path.join(self.dir, "d.txt")
        with open(p, "w", encoding="utf-8") as f:
            f.write("1 2 3")
        with self.assertRaises(ValueError):
            import_any(p)


if __name__ == "__main__":
    unittest.main()
