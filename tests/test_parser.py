"""解析器单元测试。运行: python -m unittest tests.test_parser -v"""

import unittest

from core.parser import parse_jianpu


class TestParser(unittest.TestCase):
    def test_basic_mid(self):
        r = parse_jianpu("1 2 3 4 5 6 7")
        self.assertEqual(
            r,
            [
                {"notes": ["mid_1"], "dur": 1.0},
                {"notes": ["mid_2"], "dur": 1.0},
                {"notes": ["mid_3"], "dur": 1.0},
                {"notes": ["mid_4"], "dur": 1.0},
                {"notes": ["mid_5"], "dur": 1.0},
                {"notes": ["mid_6"], "dur": 1.0},
                {"notes": ["mid_7"], "dur": 1.0},
            ],
        )

    def test_octaves(self):
        r = parse_jianpu("1' 2' 3, 4,")
        self.assertEqual(
            [n["notes"][0] for n in r],
            ["high_1", "high_2", "low_3", "low_4"],
        )

    def test_durations(self):
        r = parse_jianpu("5- 3_ 1__ 7--")
        self.assertEqual([n["dur"] for n in r], [2.0, 0.5, 0.25, 4.0])

    def test_dash_duration_boundary(self):
        """固化减号规则:一个 - = 二分(2拍),-- = 全音符(4拍),--- = 8拍(超协议但行为确定)。"""
        r = parse_jianpu("5- 5-- 5---")
        self.assertEqual([n["dur"] for n in r], [2.0, 4.0, 8.0])

    def test_dotted(self):
        r = parse_jianpu("5_· 1·")
        self.assertEqual([(n["notes"], n["dur"]) for n in r],
                         [(["mid_5"], 0.75), (["mid_1"], 1.5)])

    def test_chord(self):
        r = parse_jianpu("[1' 3' 5']-")
        self.assertEqual(r, [{"notes": ["high_1", "high_3", "high_5"], "dur": 2.0}])

    def test_rest(self):
        r = parse_jianpu("0 0_ 0--")
        self.assertEqual([n["dur"] for n in r], [1.0, 0.5, 4.0])
        self.assertTrue(all(n["notes"] == [] for n in r))

    def test_barline_and_tune_line(self):
        r = parse_jianpu("1=C 4/4\n1 | 2 ‖ 3")
        self.assertEqual([n["notes"][0] for n in r], ["mid_1", "mid_2", "mid_3"])

    def test_lyric_line_skipped(self):
        r = parse_jianpu("一闪一闪亮晶晶\n1 1 5 5 6 6 5-")
        self.assertEqual(len(r), 7)

    def test_dot_alone_is_low_not_dotted(self):
        """`.` 单独出现是下点低音标记(三角洲约定),不是附点。"""
        r = parse_jianpu("5. 6.")
        self.assertEqual([n["notes"][0] for n in r], ["low_5", "low_6"])
        self.assertEqual([n["dur"] for n in r], [1.0, 1.0])

    def test_sample(self):
        """外部 AI 返回的典型规范化简谱(含高低音/附点/和弦/休止)全量解析。"""
        sample = "1 1 5, 5, 6 6 5'- 4 4 3 3 2 2 1- 0 0 [1' 3' 5']- 1 2 3_ 3_ 5_· 5_"

        r = parse_jianpu(sample)
        self.assertEqual(len(r), 23)
        self.assertIn({"notes": ["high_5"], "dur": 2.0}, r)
        self.assertIn({"notes": ["high_1", "high_3", "high_5"], "dur": 2.0}, r)
        self.assertIn({"notes": ["low_5"], "dur": 1.0}, r)


class TestCollectMode(unittest.TestCase):
    def test_collect_clean(self):
        notes, errors = parse_jianpu("1 2 3", collect=True)
        self.assertEqual(errors, [])
        self.assertEqual([n["notes"][0] for n in notes], ["mid_1", "mid_2", "mid_3"])

    def test_collect_garbage(self):
        notes, errors = parse_jianpu("1 @ 2", collect=True)
        self.assertEqual(len(notes), 2)
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0].token, "@")
        self.assertEqual(errors[0].line, 1)
        self.assertIn("无法识别", errors[0].reason)

    def test_collect_line_numbers(self):
        _, errors = parse_jianpu("1 2\n#\n3", collect=True)
        self.assertEqual([e.line for e in errors], [2])

    def test_collect_chord_invalid_inner(self):
        notes, errors = parse_jianpu("[1 8 5]", collect=True)
        self.assertEqual(notes, [{"notes": ["mid_1", "mid_5"], "dur": 1.0}])
        self.assertEqual([e.token for e in errors], ["8"])

    def test_collect_rest_with_octave_mark(self):
        notes, errors = parse_jianpu("0'", collect=True)
        self.assertEqual(notes, [{"notes": [], "dur": 1.0}])
        self.assertIn("八度", errors[0].reason)

    def test_lenient_matches_collect_notes(self):
        text = "1 @ 2 [1 8] 0' 3_"
        a = parse_jianpu(text)
        b, _ = parse_jianpu(text, collect=True)
        self.assertEqual(a, b)

    def test_default_returns_list(self):
        self.assertIsInstance(parse_jianpu("1 2"), list)


class TestTokenOrder(unittest.TestCase):
    def test_mixed_chord_keeps_position(self):
        """和弦与单音混排时按真实顺序输出(旧实现把和弦统一挪到行尾)。"""
        r = parse_jianpu("1 [5] 3")
        self.assertEqual(
            r,
            [
                {"notes": ["mid_1"], "dur": 1.0},
                {"notes": ["mid_5"], "dur": 1.0},
                {"notes": ["mid_3"], "dur": 1.0},
            ],
        )


class TestSemitoneParsing(unittest.TestCase):
    def test_sharp_mid(self):
        r = parse_jianpu("1#")
        self.assertEqual(r, [{"notes": ["mid_1"], "dur": 1.0, "semitone": 1}])

    def test_sharp_high(self):
        r = parse_jianpu("1'#")
        self.assertEqual(r, [{"notes": ["high_1"], "dur": 1.0, "semitone": 1}])

    def test_sharp_low(self):
        r = parse_jianpu("1,#")
        self.assertEqual(r, [{"notes": ["low_1"], "dur": 1.0, "semitone": 1}])

    def test_no_sharp_no_semitone_field(self):
        r = parse_jianpu("1")
        self.assertEqual(r, [{"notes": ["mid_1"], "dur": 1.0}])
        self.assertNotIn("semitone", r[0])

    def test_sharp_in_collect_mode_no_error(self):
        notes, errors = parse_jianpu("1# 2#", collect=True)
        self.assertEqual(len(notes), 2)
        self.assertEqual(errors, [])

    def test_sharp_in_chord(self):
        r = parse_jianpu("[1# 3 5]")
        self.assertEqual(r[0]["notes"], ["mid_1", "mid_3", "mid_5"])
        self.assertEqual(r[0]["semitone"], 1)

    def test_prefix_sharp_matches_suffix_sharp(self):
        self.assertEqual(parse_jianpu("#1 #2' #3,"), parse_jianpu("1# 2'# 3,#"))

    def test_prefix_sharp_in_chord(self):
        r = parse_jianpu("[#1 3 5]")
        self.assertEqual(r[0]["notes"], ["mid_1", "mid_3", "mid_5"])
        self.assertEqual(r[0]["semitone"], 1)


class TestDeltaCommunityNotation(unittest.TestCase):
    def test_chinese_regions_and_single_parenthesis(self):
        r = parse_jianpu("【1 2】 （3 4） (5) #6")
        self.assertEqual(
            r,
            [
                {"notes": ["high_1"], "dur": 1.0},
                {"notes": ["high_2"], "dur": 1.0},
                {"notes": ["low_3"], "dur": 1.0},
                {"notes": ["low_4"], "dur": 1.0},
                {"notes": ["low_5"], "dur": 1.0},
                {"notes": ["mid_6"], "dur": 1.0, "semitone": 1},
            ],
        )

    def test_multi_note_ascii_parentheses_remain_chord(self):
        r = parse_jianpu("(1 3)")
        self.assertEqual(r, [{"notes": ["mid_1", "mid_3"], "dur": 1.0}])


class TestDottedNote(unittest.TestCase):
    def test_dot_dotted_quarter(self):
        r = parse_jianpu("5.")
        self.assertEqual(r, [{"notes": ["low_5"], "dur": 1.0}])

    def test_comma_still_low(self):
        r = parse_jianpu("5,")
        self.assertEqual(r, [{"notes": ["low_5"], "dur": 1.0}])

    def test_dot_eighth(self):
        r = parse_jianpu("5_.")
        self.assertEqual(r, [{"notes": ["mid_5"], "dur": 0.75}])

    def test_dot_in_chord_same_as_single(self):
        r = parse_jianpu("[1 3].")
        self.assertEqual(r, [{"notes": ["mid_1", "mid_3"], "dur": 1.5}])

    def test_chinese_dot_unchanged(self):
        r = parse_jianpu("5·")
        self.assertEqual(r, [{"notes": ["mid_5"], "dur": 1.5}])


class TestChinesePunctuation(unittest.TestCase):
    def test_chinese_quote_error(self):
        _, errors = parse_jianpu("1\u2019 2", collect=True)
        self.assertTrue(any("中文标点" in e.reason for e in errors))

    def test_chinese_comma_error(self):
        _, errors = parse_jianpu("1\uff0c 2", collect=True)
        self.assertTrue(any("中文标点" in e.reason for e in errors))


class TestDotAloneLow(unittest.TestCase):
    """`.` 单独出现为下点低音(三角洲约定),带时值符号为附点。"""

    def test_dot_alone_is_low(self):
        r = parse_jianpu("5.")
        self.assertEqual(r, [{"notes": ["low_5"], "dur": 1.0}])

    def test_dot_alone_low_multiple(self):
        r = parse_jianpu("1. 2. 3.")
        self.assertEqual(
            r,
            [
                {"notes": ["low_1"], "dur": 1.0},
                {"notes": ["low_2"], "dur": 1.0},
                {"notes": ["low_3"], "dur": 1.0},
            ],
        )

    def test_dot_with_duration_is_dotted(self):
        r = parse_jianpu("5._")
        self.assertEqual(r, [{"notes": ["mid_5"], "dur": 0.75}])

    def test_dot_with_dash_is_dotted(self):
        r = parse_jianpu("5.-")
        self.assertEqual(r, [{"notes": ["mid_5"], "dur": 3.0}])

    def test_dot_with_sharp_is_low_semitone(self):
        r = parse_jianpu("5.#")
        self.assertEqual(r, [{"notes": ["low_5"], "dur": 1.0, "semitone": 1}])

    def test_dot_with_octave_and_sharp(self):
        r = parse_jianpu("1'.#")
        self.assertEqual(r, [{"notes": ["high_1"], "dur": 1.0, "semitone": 1}])


class TestStandardDialect(unittest.TestCase):
    def test_community_dash_and_dot_stay_unchanged(self):
        self.assertEqual([n["dur"] for n in parse_jianpu("5- 5--")], [2.0, 4.0])
        self.assertEqual(parse_jianpu("5.")[0]["notes"], ["low_5"])

    def test_standard_dash_adds_one_beat_and_dot_is_dotted(self):
        notes, errors = parse_jianpu("5- 5-- 5.", dialect="standard", collect=True)
        self.assertEqual([n["dur"] for n in notes], [2.0, 3.0, 1.5])
        self.assertEqual(notes[2]["notes"], ["mid_5"])
        self.assertEqual(errors, [])

    def test_extension_line_and_tie_and_triplet(self):
        text = "5\n--\n1~ 1\n{1 2 3}"
        notes = parse_jianpu(text, dialect="standard")
        self.assertAlmostEqual(notes[0]["dur"], 3.0)
        self.assertEqual(notes[1]["notes"], ["mid_1"])
        self.assertAlmostEqual(notes[1]["dur"], 2.0)
        self.assertEqual(len(notes), 5)
        self.assertTrue(all(abs(n["dur"] - 1.0 / 3.0) < 1e-9 for n in notes[2:]))

    def test_flat_becomes_enharmonic_and_warns(self):
        notes, errors = parse_jianpu("b3 ♭1", dialect="standard", collect=True)
        self.assertEqual(notes[0], {"notes": ["mid_2"], "dur": 1.0, "semitone": 1})
        self.assertEqual(notes[1]["notes"], ["low_7"])
        self.assertTrue(errors)

    def test_extra_octave_marks_warn_without_dropping_the_note(self):
        notes, errors = parse_jianpu("1''", dialect="standard", collect=True)
        self.assertEqual(notes[0]["notes"], ["high_1"])
        self.assertTrue(any("八度" in error.reason for error in errors))
        community, community_errors = parse_jianpu("1''", collect=True)
        self.assertEqual(community[0]["notes"], ["high_1"])
        self.assertTrue(community_errors)


if __name__ == "__main__":
    unittest.main()
