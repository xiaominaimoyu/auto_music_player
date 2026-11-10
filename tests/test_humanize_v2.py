"""真人化 v2：抖动标准差、乐句速度弧回零、换气不推拍、扫弦。"""

import random
import statistics
import time
import unittest

from core.humanize import HumanizeParams, plan_timings
from core.keymap import KeyMap
from core.player import Player


def _note(name="mid_1", dur=1.0):
    return {"notes": [name], "dur": dur}


class TestHumanizeV2(unittest.TestCase):
    def test_ar1_stddev_matches_configured_jitter(self):
        params = HumanizeParams(
            jitter_ms=12.0,
            breath_ms=0.0,
            leap_ms=0.0,
            tempo_arc_ms=0.0,
            accent_ms=0.0,
            jitter_correlation=0.65,
        )
        notes = [_note(dur=0.5) for _ in range(800)]
        offsets = [item[0] for item in plan_timings(notes, params, seed=7)]
        deviation = statistics.pstdev(offsets)
        self.assertAlmostEqual(deviation, 12.0, delta=1.5)

    def test_phrase_arc_sums_near_zero(self):
        params = HumanizeParams(
            jitter_ms=0.0, breath_ms=0.0, leap_ms=0.0, tempo_arc_ms=10.0, accent_ms=0.0,
        )
        notes = [_note(dur=0.5) for _ in range(8)]
        offsets = [item[0] for item in plan_timings(notes, params, seed=1)]
        self.assertAlmostEqual(sum(offsets), 0.0, places=6)
        self.assertGreater(max(offsets), 0.0)
        self.assertLess(min(offsets), 0.0)

    def test_humanize_off_plan_is_exact_when_amplitudes_are_zero(self):
        params = HumanizeParams(
            jitter_ms=0.0, breath_ms=0.0, leap_ms=0.0, tempo_arc_ms=0.0, accent_ms=0.0,
        )
        offsets = [item[0] for item in plan_timings([_note(), _note(), _note()], params, seed=1)]
        self.assertEqual(offsets, [0.0, 0.0, 0.0])

    def test_same_seed_replays_the_same_plan(self):
        params = HumanizeParams()
        notes = [_note("mid_1"), _note("high_1", 0.5), {"notes": [], "dur": 1.0}, _note("mid_3")]
        self.assertEqual(
            plan_timings(notes, params, seed=99),
            plan_timings(notes, params, seed=99),
        )
        self.assertNotEqual(
            plan_timings(notes, params, seed=99),
            plan_timings(notes, params, seed=100),
        )


class _StrumDriver:
    def __init__(self):
        self.events = []

    def press_key(self, key):
        self.events.append(("down", key, time.perf_counter()))

    def press_chord(self, keys):
        now = time.perf_counter()
        for key in keys:
            self.events.append(("down", key, now))

    def release_chord(self, _keys):
        return None

    def release_key(self, _key):
        return None


class TestChordStrum(unittest.TestCase):
    def test_strum_spreads_a_chord_and_alternates_direction(self):
        driver = _StrumDriver()
        player = Player(
            KeyMap({
                "high": list("QWERTYU"),
                "mid": list("ASDFGHJ"),
                "low": list("ZXCVBNM"),
            }),
            driver=driver,
            humanize=HumanizeParams(
                jitter_ms=0.0,
                breath_ms=0.0,
                leap_ms=0.0,
                tempo_arc_ms=0.0,
                accent_ms=0.0,
                strum_ms=16.0,
                short_hold=(0.5, 0.5),
                mid_hold=(0.5, 0.5),
                long_hold=(0.5, 0.5),
            ),
        )
        player.play(
            [{"notes": ["mid_1", "mid_3", "mid_5"], "dur": 0.5}],
            bpm=120,
            hold_ratio=0.5,
            gap_ms=0,
        )
        deadline = time.perf_counter() + 2.0
        while player.is_playing and time.perf_counter() < deadline:
            time.sleep(0.01)
        downs = [event for event in driver.events if event[0] == "down"]
        self.assertEqual([event[1] for event in downs], ["A", "D", "G"])
        span = (downs[-1][2] - downs[0][2]) * 1000.0
        self.assertGreater(span, 8.0)
        self.assertLess(span, 30.0)


if __name__ == "__main__":
    unittest.main()
