"""端到端集成测试(不真实按键,用 FakeDriver)。运行: python -m unittest tests.test_e2e -v"""

import os
import time
import unittest

from core.database import ScoreDB
from core.keymap import KeyMap
from core.parser import parse_jianpu
from core.player import Player

MAPPING = {
    "high": ["Q", "W", "E", "R", "T", "Y", "U"],
    "mid": ["A", "S", "D", "F", "G", "H", "J"],
    "low": ["Z", "X", "C", "V", "B", "N", "M"],
}


class FakeDriver:
    def __init__(self):
        self.events = []

    def press_key(self, k):
        self.events.append(("press", k))

    def release_key(self, k):
        self.events.append(("release", k))

    def press_chord(self, keys):
        self.events.append(("press_chord", tuple(keys)))

    def release_chord(self, keys):
        self.events.append(("release_chord", tuple(keys)))


def wait_stop(player, timeout=10.0):
    deadline = time.time() + timeout
    while player.is_playing and time.time() < deadline:
        time.sleep(0.02)
    return not player.is_playing


class TestKeyMap(unittest.TestCase):
    def test_mapping(self):
        km = KeyMap(MAPPING)
        self.assertEqual(km.key_for("high_1"), "Q")
        self.assertEqual(km.key_for("high_7"), "U")
        self.assertEqual(km.key_for("mid_1"), "A")
        self.assertEqual(km.key_for("mid_4"), "F")
        self.assertEqual(km.key_for("mid_5"), "G")
        self.assertEqual(km.key_for("low_1"), "Z")
        self.assertEqual(km.key_for("low_7"), "M")

    def test_invalid(self):
        km = KeyMap(MAPPING)
        self.assertIsNone(km.key_for("high_8"))
        self.assertIsNone(km.key_for("bogus"))
        self.assertIsNone(km.key_for("mid_x"))


class TestPipeline(unittest.TestCase):
    """粘贴识别路径端到端:外部 AI 输出格式的简谱 -> 解析 -> 入库 -> 读回。"""

    def setUp(self):
        self.db = ScoreDB("data/test_e2e.db")

    def tearDown(self):
        self.db.conn.close()
        if os.path.exists("data/test_e2e.db"):
            os.remove("data/test_e2e.db")

    def test_pasted_jianpu_parse_store_roundtrip(self):
        """模拟外部 AI 工具返回的规范化简谱,直接粘贴解析入库。"""
        text = "1 1 5, 5, 6 6 5'- 4 4 3 3 2 2 1- 0 0 [1' 3' 5']-"
        notes = parse_jianpu(text)
        self.assertGreater(len(notes), 0)
        self.assertEqual(notes[0], {"notes": ["mid_1"], "dur": 1.0})
        self.assertEqual(notes[2], {"notes": ["low_5"], "dur": 1.0})
        self.assertEqual(notes[6], {"notes": ["high_5"], "dur": 2.0})
        score_id = self.db.add_score(
            "测试曲", notes, raw_text=text, source_type="manual", bpm_default=120
        )
        score = self.db.get_score(score_id)
        self.assertEqual(score["notes"], notes)
        self.assertEqual(score["name"], "测试曲")
        self.assertEqual(score["bpm_default"], 120)
        self.assertEqual(score["source_type"], "manual")
        self.assertEqual(len(self.db.list_scores()), 1)
        self.db.delete_score(score_id)
        self.assertEqual(len(self.db.list_scores()), 0)


class TestPlayer(unittest.TestCase):
    def test_play_full_sequence_with_chord_and_rest(self):
        km = KeyMap(MAPPING)
        driver = FakeDriver()
        player = Player(km, driver=driver)
        notes = [
            {"notes": ["high_1"], "dur": 1.0},
            {"notes": ["mid_1", "mid_3", "mid_5"], "dur": 1.0},
            {"notes": [], "dur": 1.0},
        ]
        player.play(notes, bpm=600, hold_ratio=0.5, gap_ms=0)
        self.assertTrue(wait_stop(player))
        self.assertTrue(("press_chord", ("Q",)) in driver.events)
        self.assertTrue(("press_chord", ("A", "D", "G")) in driver.events)
        self.assertTrue(("release_chord", ("A", "D", "G")) in driver.events)

    def test_stop_releases_all_keys(self):
        km = KeyMap(MAPPING)
        driver = FakeDriver()
        player = Player(km, driver=driver)
        long_notes = [{"notes": ["mid_1"], "dur": 4.0}] * 50
        player.play(long_notes, bpm=60, hold_ratio=0.5, gap_ms=0)
        time.sleep(0.15)
        player.stop()
        self.assertTrue(wait_stop(player, timeout=12.0))
        self.assertFalse(player.is_playing)


if __name__ == "__main__":
    unittest.main()