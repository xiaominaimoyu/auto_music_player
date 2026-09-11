"""core.ir 单元测试。"""

import unittest

from core.ir import (
    IRChord,
    IRNote,
    IRRest,
    from_storage,
    to_storage,
)


class TestIRNote(unittest.TestCase):
    def test_note_id_excludes_semitone(self):
        """D1:semitone 绝不出现在持久层 note_id 里。"""
        self.assertEqual(IRNote(3, 0, 0).note_id, "mid_3")
        self.assertEqual(IRNote(3, 0, 1).note_id, "mid_3")
        self.assertEqual(IRNote(1, 1, 0).note_id, "high_1")
        self.assertEqual(IRNote(5, -1, 0).note_id, "low_5")

    def test_valid_range_accepted(self):
        for pitch in range(1, 8):
            for octave in (-1, 0, 1):
                for semi in (0, 1):
                    self.assertIsInstance(IRNote(pitch, octave, semi), IRNote)

    def test_invalid_pitch_raises(self):
        with self.assertRaises(ValueError):
            IRNote(0, 0, 0)
        with self.assertRaises(ValueError):
            IRNote(8, 0, 0)

    def test_invalid_octave_raises(self):
        with self.assertRaises(ValueError):
            IRNote(1, 2, 0)

    def test_invalid_semitone_raises(self):
        with self.assertRaises(ValueError):
            IRNote(1, 0, 2)

    def test_invalid_dur_raises(self):
        with self.assertRaises(ValueError):
            IRNote(1, 0, 0, 0)
        with self.assertRaises(ValueError):
            IRNote(1, 0, 0, 17)


class TestStorageRoundtrip(unittest.TestCase):
    STORAGE = [
        {"notes": ["mid_1"], "dur": 1.0},
        {"notes": [], "dur": 0.5},
        {"notes": ["high_3", "high_5"], "dur": 2.0},
        {"notes": ["low_7"], "dur": 0.75},
    ]

    def test_roundtrip_stable(self):
        ir = from_storage(self.STORAGE)
        self.assertEqual(to_storage(ir), self.STORAGE)

    def test_element_types(self):
        ir = from_storage(self.STORAGE)
        self.assertIsInstance(ir[0], IRNote)
        self.assertIsInstance(ir[1], IRRest)
        self.assertIsInstance(ir[2], IRChord)
        self.assertIsInstance(ir[3], IRNote)

    def test_octave_decoded(self):
        ir = from_storage(self.STORAGE)
        self.assertEqual(ir[0].octave, 0)
        self.assertEqual(ir[2].notes[0].octave, 1)
        self.assertEqual(ir[2].notes[0].pitch, 3)
        self.assertEqual(ir[3].octave, -1)

    def test_semitone_injected_but_not_persisted(self):
        """D8 注入通道:IR 带 semitone,落盘时丢弃。"""
        ir = from_storage([{"notes": ["mid_3"], "dur": 1.0}], semitones=[1])
        self.assertEqual(ir[0].semitone, 1)
        self.assertEqual(to_storage(ir), [{"notes": ["mid_3"], "dur": 1.0}])

    def test_semitone_from_dict_takes_priority(self):
        """dict 内 semitone 字段优先于外部 semitones 参数。"""
        ir = from_storage(
            [{"notes": ["mid_3"], "dur": 1.0, "semitone": 1}],
            semitones=[0],
        )
        self.assertEqual(ir[0].semitone, 1)

    def test_semitone_external_fallback(self):
        """dict 内无 semitone 字段时,使用外部 semitones 参数。"""
        ir = from_storage(
            [{"notes": ["mid_3"], "dur": 1.0}],
            semitones=[1],
        )
        self.assertEqual(ir[0].semitone, 1)

    def test_semitone_dict_field_not_persisted_by_to_storage(self):
        """to_storage 不输出 semitone 字段(向后兼容)。"""
        ir = from_storage([{"notes": ["mid_3"], "dur": 1.0, "semitone": 1}])
        self.assertEqual(to_storage(ir), [{"notes": ["mid_3"], "dur": 1.0}])

    def test_semitones_length_mismatch_raises(self):
        """D9:长度不一致必须报错,绝不静默错位回填。"""
        with self.assertRaises(ValueError):
            from_storage([{"notes": ["mid_1"], "dur": 1.0}], semitones=[1, 0])

    def test_unknown_element_type(self):
        with self.assertRaises(TypeError):
            to_storage([object()])

    def test_invalid_note_id(self):
        for bad in ["mid_8", "top_1", "mid_x", "mid"]:
            with self.assertRaises(ValueError):
                from_storage([{"notes": [bad], "dur": 1.0}])

    def test_empty_chord_raises(self):
        with self.assertRaises(ValueError):
            IRChord((), 1.0)


if __name__ == "__main__":
    unittest.main()
