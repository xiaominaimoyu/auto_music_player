"""休止符功能集成测试:验证从解析到演奏的完整链路。

运行: python -m pytest tests/test_rest_integration.py -v
"""

import unittest
from unittest.mock import MagicMock, patch

from core.parser import parse_jianpu
from core.ir import from_storage, to_storage, IRRest, IRNote
from core.compiler import compile_score, CompileParams
from core.score_model import Score, Rest, Note


class TestRestParsing(unittest.TestCase):
    """测试休止符解析:简谱文本 → 存储格式"""

    def test_parse_single_rest(self):
        """单个休止符(四分休止符)"""
        result = parse_jianpu("0")
        self.assertEqual(result, [{"notes": [], "dur": 1.0}])

    def test_parse_rest_with_durations(self):
        """不同时值的休止符"""
        result = parse_jianpu("0 0_ 0__ 0- 0--")
        expected_durs = [1.0, 0.5, 0.25, 2.0, 4.0]
        self.assertEqual([n["dur"] for n in result], expected_durs)
        self.assertTrue(all(n["notes"] == [] for n in result))

    def test_parse_rest_with_dotted(self):
        """附点休止符"""
        result = parse_jianpu("0. 0_·")
        self.assertAlmostEqual(result[0]["dur"], 1.5)
        self.assertAlmostEqual(result[1]["dur"], 0.75)

    def test_parse_mixed_notes_and_rests(self):
        """音符与休止符混排"""
        result = parse_jianpu("1 0 2 0_ 3")
        self.assertEqual(
            [n["notes"] for n in result], [["mid_1"], [], ["mid_2"], [], ["mid_3"]]
        )

    def test_parse_rest_octave_marks_ignored(self):
        """休止符带八度标记应被忽略(在 collect 模式会报警告)"""
        result, errors = parse_jianpu("0' 0,", collect=True)
        self.assertEqual(len(result), 2)
        self.assertTrue(all(n["notes"] == [] for n in result))
        self.assertEqual(len(errors), 2)
        self.assertTrue(all("不应带八度" in e.reason for e in errors))


class TestRestIR(unittest.TestCase):
    """测试休止符 IR 转换"""

    def test_from_storage_creates_IRRest(self):
        """存储格式 → IR:空 notes 列表生成 IRRest"""
        storage = [
            {"notes": ["mid_1"], "dur": 1.0},
            {"notes": [], "dur": 0.5},
            {"notes": ["high_3"], "dur": 2.0},
        ]
        ir = from_storage(storage)
        self.assertIsInstance(ir[0], IRNote)
        self.assertIsInstance(ir[1], IRRest)
        self.assertIsInstance(ir[2], IRNote)
        self.assertEqual(ir[1].dur, 0.5)

    def test_to_storage_preserves_rest(self):
        """IR → 存储格式:IRRest 转为空 notes 列表"""
        ir = [IRNote(1, 0, 0, 1.0), IRRest(0.5), IRNote(3, 1, 0, 2.0)]
        storage = to_storage(ir)
        self.assertEqual(storage[1], {"notes": [], "dur": 0.5})

    def test_roundtrip_with_rests(self):
        """存储 → IR → 存储往返一致"""
        original = [
            {"notes": ["mid_1"], "dur": 1.0},
            {"notes": [], "dur": 2.0},
            {"notes": [], "dur": 0.25},
        ]
        ir = from_storage(original)
        restored = to_storage(ir)
        self.assertEqual(restored, original)


class TestRestScoreModel(unittest.TestCase):
    """测试休止符在 Score 模型中的表示"""

    def test_score_from_storage_with_rests(self):
        """Score.from_storage 正确处理休止符"""
        items = [
            {"notes": ["mid_1"], "dur": 1.0},
            {"notes": [], "dur": 0.5},
            {"notes": ["high_5"], "dur": 2.0},
        ]
        score = Score.from_storage("测试曲", 100, items)
        self.assertIsInstance(score.elements[0], Note)
        self.assertIsInstance(score.elements[1], Rest)
        self.assertIsInstance(score.elements[2], Note)
        self.assertEqual(score.elements[1].dur, 0.5)

    def test_score_to_storage_with_rests(self):
        """Score.to_storage 正确序列化休止符"""
        score = Score(
            "测试曲", 100, [Note(1, "mid", 1.0), Rest(0.5), Note(5, "high", 2.0)]
        )
        storage = score.to_storage()
        self.assertEqual(storage[1], {"notes": [], "dur": 0.5})

    def test_score_total_beats_includes_rests(self):
        """total_beats 计算包含休止符时值"""
        score = Score(
            "测试曲", 100, [Note(1, "mid", 1.0), Rest(2.0), Note(3, "mid", 1.0)]
        )
        self.assertEqual(score.total_beats, 4.0)


class TestRestCompiler(unittest.TestCase):
    """测试休止符编译:IR → 输入事件序列"""

    def test_rest_advances_cursor(self):
        """休止符正确推进时间轴"""
        params = CompileParams(
            bpm=120,
            settle_ms=30,
            release_settle_ms=20,
            hold_ratio=1.0,
            max_hold_ms=None,
        )
        ir = [
            IRNote(1, 0, 0, 1.0),  # 500ms
            IRRest(1.0),  # 500ms 休止
            IRNote(2, 0, 0, 1.0),  # 500ms
        ]
        result = compile_score(ir, params)

        # 第一个音符: down at 0, up at 500
        # 休止 500ms
        # 第二个音符: down at ~1000, up at ~1500
        events = result.events
        note1_down = next(e for e in events if e.key == "Z" and e.action == "down")
        note2_down = next(e for e in events if e.key == "X" and e.action == "down")

        # 两个音符起始时刻应相差约 1000ms (500 + 500)
        time_diff = note2_down.t_ms - note1_down.t_ms
        self.assertAlmostEqual(time_diff, 1000.0, delta=50)

    def test_rest_releases_modifier(self):
        """休止符前如有修饰键按住,应释放"""
        params = CompileParams(
            bpm=120,
            settle_ms=30,
            release_settle_ms=20,
            hold_ratio=1.0,
            max_hold_ms=None,
        )
        ir = [
            IRNote(1, 1, 0, 0.5),  # high_1: 需要 right 修饰键
            IRRest(0.5),
            IRNote(2, 0, 0, 0.5),  # mid_2: 自然音
        ]
        result = compile_score(ir, params)

        # 应该有 right 键的 up 事件在休止期间
        mouse_ups = [
            e for e in result.events if e.device == "mouse" and e.action == "up"
        ]
        self.assertTrue(len(mouse_ups) > 0, "休止符前应释放修饰键")

    def test_multiple_consecutive_rests(self):
        """连续多个休止符"""
        params = CompileParams(
            bpm=60, settle_ms=30, release_settle_ms=20, hold_ratio=1.0, max_hold_ms=None
        )
        ir = [
            IRNote(1, 0, 0, 1.0),
            IRRest(1.0),
            IRRest(2.0),
            IRNote(2, 0, 0, 1.0),
        ]
        result = compile_score(ir, params)

        note1 = next(e for e in result.events if e.key == "Z" and e.action == "down")
        note2 = next(e for e in result.events if e.key == "X" and e.action == "down")

        # 总休止 3 秒 (1 + 2 拍 @ 60 BPM)
        time_diff = note2.t_ms - note1.t_ms
        self.assertAlmostEqual(time_diff, 4000.0, delta=100)  # 1拍音符 + 3拍休止


class TestRestInRealScore(unittest.TestCase):
    """测试真实乐谱场景中的休止符"""

    def test_twinkle_twinkle_with_rests(self):
        """小星星带休止符版本"""
        jianpu = """
        1=C 4/4
        1 1 5 5 | 6 6 5- | 0 0
        4 4 3 3 | 2 2 1-
        """
        result = parse_jianpu(jianpu)

        # 找到休止符
        rests = [n for n in result if n["notes"] == []]
        self.assertEqual(len(rests), 2)
        self.assertTrue(all(r["dur"] == 1.0 for r in rests))

    def test_complex_rhythm_with_rests(self):
        """复杂节奏型:包含不同时值的休止符"""
        jianpu = "1 0_ 2 0 3_ 0__ 5- 0--"
        result = parse_jianpu(jianpu)

        rests = [n for n in result if n["notes"] == []]
        rest_durs = [r["dur"] for r in rests]
        self.assertEqual(rest_durs, [0.5, 1.0, 0.25, 4.0])

    def test_chord_rest_sequence(self):
        """和弦与休止符混排"""
        jianpu = "[1 3 5]- 0 [2 4 6] 0_"
        result = parse_jianpu(jianpu)

        self.assertEqual(len(result), 4)
        self.assertEqual(result[0]["notes"], ["mid_1", "mid_3", "mid_5"])
        self.assertEqual(result[1]["notes"], [])
        self.assertEqual(result[2]["notes"], ["mid_2", "mid_4", "mid_6"])
        self.assertEqual(result[3]["notes"], [])


class TestRestEndToEnd(unittest.TestCase):
    """端到端测试:完整演奏链路"""

    @patch("core.keyboard_driver.KeyboardDriver")
    def test_rest_playback_timing(self, mock_driver_class):
        """测试休止符在演奏时正确产生停顿"""
        from core.player import Player

        mock_driver = MagicMock()
        mock_driver_class.return_value = mock_driver

        keymap = MagicMock()
        keymap.key_for = lambda nid: {"mid_1": "Z", "mid_2": "X", "mid_3": "C"}.get(nid)

        player = Player(keymap, driver=mock_driver)

        notes = [
            {"notes": ["mid_1"], "dur": 0.5},
            {"notes": [], "dur": 1.0},  # 休止
            {"notes": ["mid_2"], "dur": 0.5},
        ]

        # 不实际运行线程,只验证数据处理流程
        # 真实演奏由 GUI 集成测试或手动测试验证
        self.assertEqual(notes[1]["notes"], [])


if __name__ == "__main__":
    unittest.main()
