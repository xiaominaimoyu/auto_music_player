"""节奏忠实调度：起音间隔保持谱面比例，修饰键预按，松开间隙从时值里扣。"""

import threading
import time
import unittest

from core.compiler import CompileParams, compile_score
from core.ir import IRNote
from core.keymap import KeyMap
from core.player import Player
from core.timing import slot_hold_and_advance, timing_config, wait_until

_KEYMAP = KeyMap({
    "high": list("QWERTYU"),
    "mid": list("ASDFGHJ"),
    "low": list("ZXCVBNM"),
})


def _kb_downs(result):
    return [event.t_ms for event in result.events if event.device == "kb" and event.action == "down"]


def _ioi(result):
    downs = _kb_downs(result)
    return [right - left for left, right in zip(downs, downs[1:])]


class TestReleaseGapCarvedFromDuration(unittest.TestCase):
    def test_quarter_notes_keep_ioi_and_leave_release_gap(self):
        params = CompileParams(
            bpm=120, settle_ms=0.0, release_settle_ms=0.0,
            hold_ratio=1.0, max_hold_ms=None, gap_ms=20.0,
        )
        result = compile_score([IRNote(1, 0, 0, 1.0), IRNote(2, 0, 0, 1.0)], params)
        downs = _kb_downs(result)
        ups = [event.t_ms for event in result.events if event.device == "kb" and event.action == "up"]
        self.assertAlmostEqual(downs[1] - downs[0], 500.0, places=3)
        self.assertAlmostEqual(downs[1] - ups[0], 20.0, places=3)

    def test_hold_ratio_does_not_stretch_ioi_when_gap_already_fits(self):
        params = CompileParams(
            bpm=120, settle_ms=0.0, release_settle_ms=0.0,
            hold_ratio=0.75, max_hold_ms=None, gap_ms=20.0,
        )
        result = compile_score([IRNote(1, 0, 0, 1.0), IRNote(2, 0, 0, 0.5)], params)
        self.assertAlmostEqual(_ioi(result)[0], 500.0, places=3)

    def test_note_shorter_than_gap_is_stretched_to_min_release(self):
        params = CompileParams(
            bpm=100, settle_ms=0.0, release_settle_ms=0.0,
            hold_ratio=1.0, max_hold_ms=None, gap_ms=20.0,
        )
        result = compile_score([IRNote(1, 0, 0, 0.01), IRNote(1, 0, 0, 0.01)], params)
        self.assertGreaterEqual(_ioi(result)[0], 20.0)

    def test_additive_gap_restores_old_stretch(self):
        params = CompileParams(
            bpm=100, settle_ms=0.0, release_settle_ms=0.0,
            hold_ratio=1.0, max_hold_ms=None, gap_ms=20.0, additive_gap=True,
        )
        result = compile_score([IRNote(1, 0, 0, 1.0), IRNote(2, 0, 0, 1.0)], params)
        self.assertAlmostEqual(_ioi(result)[0], 620.0, places=3)


class TestModifierPrepress(unittest.TestCase):
    def _eighths(self, **overrides):
        params = CompileParams(
            bpm=120, settle_ms=30.0, release_settle_ms=20.0,
            hold_ratio=1.0, max_hold_ms=None, gap_ms=0.0,
            **overrides,
        )
        notes = [
            IRNote(1, 0, 0, 0.5),
            IRNote(1, 1, 0, 0.5),
            IRNote(2, 0, 0, 0.5),
            IRNote(2, -1, 0, 0.5),
            IRNote(3, 0, 0, 0.5),
            IRNote(3, 1, 0, 0.5),
        ]
        return compile_score(notes, params)

    def test_prepress_preserves_eighth_note_ioi(self):
        result = self._eighths()
        for interval in _ioi(result):
            self.assertAlmostEqual(interval, 250.0, places=2)
        self.assertFalse(any(item.reason == "PREPRESS_DELAY" for item in result.degradations))

    def test_key_lands_on_grid_when_modifier_changes(self):
        result = self._eighths()
        downs = _kb_downs(result)
        self.assertAlmostEqual(downs[0], 0.0, places=2)
        self.assertAlmostEqual(downs[1], 250.0, places=2)
        mouse_downs = [
            event.t_ms for event in result.events
            if event.device == "mouse" and event.action == "down"
        ]
        # 高音修饰键在拍点前 30ms，而不是把音键推迟 30ms。
        self.assertAlmostEqual(mouse_downs[0], 220.0, places=2)

    def test_impossible_prepress_delays_and_logs(self):
        params = CompileParams(
            bpm=100, settle_ms=30.0, release_settle_ms=20.0,
            hold_ratio=1.0, max_hold_ms=None, gap_ms=0.0,
        )
        result = compile_score(
            [IRNote(1, 1, 0, 0.01), IRNote(2, -1, 0, 0.01)],
            params,
        )
        reasons = [item.reason for item in result.degradations]
        self.assertIn("PREPRESS_DELAY", reasons)
        downs = _kb_downs(result)
        self.assertGreater(downs[1] - downs[0], 6.0)

    def test_legacy_timing_reproduces_stretched_modifier_ioi(self):
        result = self._eighths(legacy_timing=True)
        intervals = [round(item, 1) for item in _ioi(result)]
        self.assertEqual(intervals, [280.0, 270.0, 280.0, 270.0, 280.0])

    def test_prepress_off_does_not_keep_every_key_on_the_unshifted_grid(self):
        faithful = self._eighths()
        delayed = self._eighths(modifier_prepress=False)
        self.assertAlmostEqual(_ioi(faithful)[1], 250.0, places=2)
        self.assertGreater(abs(_ioi(delayed)[1] - 250.0), 5.0)


class TestSlotMath(unittest.TestCase):
    def test_carved_gap_keeps_duration(self):
        hold, advance = slot_hold_and_advance(
            500.0, 1.0, 20.0, additive=False, followed_by_attack=True,
        )
        self.assertAlmostEqual(hold, 480.0)
        self.assertAlmostEqual(advance, 500.0)

    def test_partial_hold_keeps_duration_when_silence_already_covers_gap(self):
        hold, advance = slot_hold_and_advance(
            500.0, 0.75, 20.0, additive=False, followed_by_attack=True,
        )
        self.assertAlmostEqual(hold, 375.0)
        self.assertAlmostEqual(advance, 500.0)

    def test_additive_gap_adds_to_the_slot(self):
        _hold, advance = slot_hold_and_advance(
            500.0, 0.75, 20.0, additive=True, followed_by_attack=True,
        )
        self.assertAlmostEqual(advance, 520.0)

    def test_timing_config_legacy_bundle(self):
        legacy = timing_config({"legacy_timing": True, "modifier_prepress": True})
        self.assertTrue(legacy.legacy_timing)
        self.assertTrue(legacy.additive_gap)
        self.assertFalse(legacy.modifier_prepress)
        self.assertFalse(legacy.precise_wait)
        self.assertFalse(legacy.batch_chords)
        modern = timing_config({})
        self.assertTrue(modern.modifier_prepress)
        self.assertTrue(modern.precise_wait)
        self.assertFalse(modern.additive_gap)


class _StampDriver:
    def __init__(self):
        self.presses = []
        self.lock = threading.Lock()

    def press_chord(self, keys):
        with self.lock:
            self.presses.append(time.perf_counter())

    def release_chord(self, _keys):
        return None

    def release_key(self, _key):
        return None


class TestPlayerCarvesGap(unittest.TestCase):
    def test_press_intervals_follow_score_not_gap(self):
        driver = _StampDriver()
        player = Player(_KEYMAP, driver=driver, humanize=None)
        notes = [{"notes": ["mid_1"], "dur": 0.5}] * 4
        player.play(notes, bpm=600, hold_ratio=1.0, gap_ms=20)
        deadline = time.perf_counter() + 2.0
        while player.is_playing and time.perf_counter() < deadline:
            time.sleep(0.01)
        self.assertEqual(len(driver.presses), 4)
        intervals = [b - a for a, b in zip(driver.presses, driver.presses[1:])]
        # 600 BPM 的八分音符是 50ms。旧调度会变成 70ms。
        for interval in intervals:
            self.assertAlmostEqual(interval, 0.05, delta=0.015)


class TestPreciseWait(unittest.TestCase):
    def test_wait_hits_deadline_within_a_few_milliseconds(self):
        stop = threading.Event()
        start = time.perf_counter()
        deadline = start + 0.04
        stopped = wait_until(stop, deadline, precise=True)
        elapsed = time.perf_counter() - start
        self.assertFalse(stopped)
        self.assertAlmostEqual(elapsed, 0.04, delta=0.008)

    def test_stop_interrupts_long_wait(self):
        stop = threading.Event()
        def _trip():
            time.sleep(0.02)
            stop.set()
        threading.Thread(target=_trip, daemon=True).start()
        started = time.perf_counter()
        stopped = wait_until(stop, started + 2.0, precise=True)
        self.assertTrue(stopped)
        self.assertLess(time.perf_counter() - started, 0.2)


if __name__ == "__main__":
    unittest.main()
