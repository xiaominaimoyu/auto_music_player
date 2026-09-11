"""gui.player_tab 模块级纯函数测试(事件路径构建 + config 写回)。

不实例化 PlayerTab(避免 GUI),只测可脱离界面的事件路径构建与配置手术。
"""

import os
import tempfile
import unittest

from core.profile import load_profiles
from gui.player_tab import _update_active_profile, build_event_plan

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROFILES_DIR = os.path.join(REPO_ROOT, "profiles")

LEGACY_KEYMAP = {"high": list("QWERTYU"), "mid": list("ASDFGHJ"), "low": list("ZXCVBNM")}


def _delta():
    return [p for p in load_profiles(PROFILES_DIR, fallback_keymap=LEGACY_KEYMAP)
            if p.id == "delta_force_harmonica"][0]


def _sig(events):
    return [(e.device, e.key, e.action) for e in events]


KW = dict(bpm=100, settle_ms=30.0, release_settle_ms=20.0, hold_ratio=1.0, gap_ms=0.0)


class TestBuildEventPlan(unittest.TestCase):
    def test_high_note_uses_right_modifier(self):
        """非直达键的高音(如 high_3)走鼠标右键升调。"""
        plan, degs = build_event_plan([{"notes": ["high_3"], "dur": 0.01}],
                                      _delta(), "free_play", **KW)
        self.assertEqual(_sig(plan.events), [
            ("mouse", "right", "down"),
            ("kb", "C", "down"),
            ("kb", "C", "up"),
            ("mouse", "right", "up"),
        ])
        self.assertEqual(degs, [])

    def test_high_1_uses_comma_override(self):
        """D3:high_1 命中 `,` 物理直达键,不再按修饰键。"""
        plan, _ = build_event_plan([{"notes": ["high_1"], "dur": 0.01}],
                                   _delta(), "free_play", **KW)
        self.assertEqual(_sig(plan.events), [("kb", ",", "down"), ("kb", ",", "up")])

    def test_low_note_uses_left_modifier(self):
        plan, _ = build_event_plan([{"notes": ["low_5"], "dur": 0.01}],
                                   _delta(), "free_play", **KW)
        self.assertEqual([e.key for e in plan.events if e.device == "mouse"],
                         ["left", "left"])

    def test_npc_quest_appends_q_submit(self):
        plan, _ = build_event_plan([{"notes": ["mid_1"], "dur": 0.01}],
                                   _delta(), "npc_quest", **KW)
        self.assertEqual(_sig(plan.events)[-2:],
                         [("kb", "Q", "down"), ("kb", "Q", "up")])
        self.assertEqual(plan.interrupt_mode, "abort")

    def test_npc_quest_settle_is_slower(self):
        """识别友好:S2 的修饰→音键间隔(60ms)必须大于 S1(30ms)。"""
        notes = [{"notes": ["low_1"], "dur": 0.01}]
        free = build_event_plan(notes, _delta(), "free_play", **KW)[0]
        npc = build_event_plan(notes, _delta(), "npc_quest", **KW)[0]
        free_key_t = [e.t_ms for e in free.events if e.action == "down" and e.device == "kb"][0]
        npc_key_t = [e.t_ms for e in npc.events if e.action == "down" and e.device == "kb"][0]
        self.assertAlmostEqual(free_key_t, 30.0)
        self.assertAlmostEqual(npc_key_t, 60.0)

    def test_chord_degrades_with_list(self):
        plan, degs = build_event_plan([{"notes": ["mid_1", "mid_3"], "dur": 1.0}],
                                      _delta(), "free_play", **KW)
        self.assertEqual(len(degs), 1)
        self.assertIn("chord", degs[0].requested)
        # 降级为取首音,不产生多键同按
        kb = [e for e in plan.events if e.device == "kb"]
        self.assertEqual(len(kb), 2)

    def test_input_notes_not_mutated(self):
        notes = [{"notes": ["high_1"], "dur": 0.01}]
        before = [dict(n) for n in notes]
        build_event_plan(notes, _delta(), "free_play", **KW)
        self.assertEqual(notes, before)

    def test_free_play_events_have_no_submit(self):
        plan, _ = build_event_plan([{"notes": ["mid_1"], "dur": 0.01}],
                                   _delta(), "free_play", **KW)
        self.assertFalse(any(e.key == "Q" for e in plan.events))


class TestUpdateActiveProfile(unittest.TestCase):
    def _write(self, text):
        f = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8")
        f.write(text)
        f.close()
        self.addCleanup(os.unlink, f.name)
        return f.name

    def test_replace_existing_and_preserve_comments(self):
        path = self._write(
            "app:\n"
            "  data_dir: data\n"
            "  # 当前激活的游戏档位\n"
            "  active_profile: default\n"
            "recognizer:\n"
            "  provider: stub\n")
        _update_active_profile(path, "delta_force_harmonica")
        text = open(path, encoding="utf-8").read()
        self.assertIn("active_profile: delta_force_harmonica", text)
        self.assertIn("# 当前激活的游戏档位", text, "注释必须保留")
        self.assertIn("recognizer:", text)
        self.assertEqual(text.count("active_profile:"), 1, "不得重复写入")

    def test_insert_when_missing(self):
        path = self._write("app:\n  data_dir: data\nrecognizer:\n  provider: stub\n")
        _update_active_profile(path, "delta_force_harmonica")
        text = open(path, encoding="utf-8").read()
        self.assertIn("active_profile: delta_force_harmonica", text)
        # 应插入到 app: 段内,而不是文件末尾
        self.assertLess(text.index("active_profile"), text.index("recognizer:"))

    def test_append_when_no_app_section(self):
        path = self._write("recognizer:\n  provider: stub\n")
        _update_active_profile(path, "default")
        text = open(path, encoding="utf-8").read()
        self.assertIn("app:\n", text)
        self.assertIn("active_profile: default", text)


if __name__ == "__main__":
    unittest.main()
