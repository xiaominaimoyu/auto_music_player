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

    def test_dot_as_low_fallback(self):
        r = parse_jianpu("5. 6.")
        self.assertEqual([n["notes"][0] for n in r], ["low_5", "low_6"])

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


if __name__ == "__main__":
    unittest.main()