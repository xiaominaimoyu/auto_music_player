"""core.scenario 单元测试。"""

import unittest

from core.compiler import CompileParams, InputEvent
from core.scenario import (
    FREE_PLAY_INTERVALS,
    NPC_QUEST_INTERVALS,
    FreePlayScenario,
    NpcQuestScenario,
    ScenarioPlan,
    get_scenario,
)

MUSIC = [
    InputEvent(0.0, "mouse", "right", "down"),
    InputEvent(30.0, "kb", "Z", "down"),
    InputEvent(80.0, "kb", "Z", "up"),
    InputEvent(100.0, "mouse", "right", "up"),
]


class TestFreePlay(unittest.TestCase):
    def setUp(self):
        self.plan = FreePlayScenario().plan(MUSIC)

    def test_interrupt_mode_pause(self):
        self.assertEqual(self.plan.interrupt_mode, "pause")

    def test_no_events_added(self):
        """自由演奏不加任何收尾:事件与输入完全一致。"""
        self.assertEqual(self.plan.events, MUSIC)

    def test_no_abort_hint(self):
        self.assertEqual(self.plan.abort_hint, "")

    def test_identity(self):
        self.assertEqual(self.plan.scenario_id, "free_play")


class TestNpcQuest(unittest.TestCase):
    def setUp(self):
        self.plan = NpcQuestScenario().plan(MUSIC)

    def test_interrupt_mode_abort(self):
        self.assertEqual(self.plan.interrupt_mode, "abort")

    def test_abort_hint_present(self):
        self.assertIn("重新听 NPC 示范", self.plan.abort_hint)

    def test_post_roll_is_q_down_up(self):
        tail = self.plan.events[len(MUSIC):]
        self.assertEqual([(e.device, e.key, e.action) for e in tail],
                         [("kb", "Q", "down"), ("kb", "Q", "up")])

    def test_post_roll_after_music(self):
        last_music_t = MUSIC[-1].t_ms
        q_down = self.plan.events[len(MUSIC)]
        self.assertGreaterEqual(q_down.t_ms, last_music_t)

    def test_post_roll_default_timing(self):
        tail = self.plan.events[len(MUSIC):]
        self.assertAlmostEqual(tail[0].t_ms, MUSIC[-1].t_ms + 200.0)
        self.assertAlmostEqual(tail[1].t_ms, MUSIC[-1].t_ms + 200.0 + 80.0)

    def test_custom_submit_key(self):
        plan = NpcQuestScenario(submit_key="Enter", submit_hold_ms=50).plan(MUSIC)
        tail = plan.events[len(MUSIC):]
        self.assertEqual(tail[0].key, "Enter")
        self.assertAlmostEqual(tail[1].t_ms - tail[0].t_ms, 50.0)

    def test_empty_music_still_submits(self):
        plan = NpcQuestScenario().plan([])
        self.assertEqual([(e.key, e.action) for e in plan.events],
                         [("Q", "down"), ("Q", "up")])

    def test_input_events_not_mutated(self):
        """plan 不得改动传入的音乐事件列表。"""
        before = list(MUSIC)
        NpcQuestScenario().plan(MUSIC)
        self.assertEqual(MUSIC, before)


class TestIntervals(unittest.TestCase):
    BASE = dict(bpm=100, settle_ms=30.0, release_settle_ms=20.0,
                hold_ratio=1.0, max_hold_ms=None, gap_ms=20.0)

    def test_free_play_intervals_match_global_default(self):
        self.assertEqual(FREE_PLAY_INTERVALS, {"settle_ms": 30.0, "gap_ms": 20.0})

    def test_npc_quest_intervals_are_slower(self):
        """识别友好:S2 间隔必须显著大于 S1。"""
        self.assertGreater(NPC_QUEST_INTERVALS["settle_ms"], FREE_PLAY_INTERVALS["settle_ms"])
        self.assertGreater(NPC_QUEST_INTERVALS["gap_ms"], FREE_PLAY_INTERVALS["gap_ms"])

    def test_apply_intervals_overrides_and_preserves_original(self):
        params = CompileParams(**self.BASE)
        new = NpcQuestScenario().apply_intervals(params)
        self.assertEqual(new.settle_ms, 60.0)
        self.assertEqual(new.gap_ms, 120.0)
        self.assertEqual(params.settle_ms, 30.0, "apply_intervals 不得改动原对象")
        self.assertEqual(params.gap_ms, 20.0)

    def test_apply_intervals_free_play_keeps_values(self):
        params = CompileParams(**self.BASE)
        new = FreePlayScenario().apply_intervals(params)
        self.assertEqual(new.settle_ms, 30.0)
        self.assertEqual(new.gap_ms, 20.0)


class TestRegistry(unittest.TestCase):
    def test_get_known(self):
        self.assertIsInstance(get_scenario("free_play"), FreePlayScenario)
        self.assertIsInstance(get_scenario("npc_quest"), NpcQuestScenario)

    def test_get_with_kwargs(self):
        s = get_scenario("npc_quest", submit_key="Enter")
        self.assertEqual(s.submit_key, "Enter")

    def test_unknown_raises(self):
        with self.assertRaises(ValueError):
            get_scenario("nope")

    def test_plan_is_scenario_plan(self):
        self.assertIsInstance(FreePlayScenario().plan(MUSIC), ScenarioPlan)


if __name__ == "__main__":
    unittest.main()
