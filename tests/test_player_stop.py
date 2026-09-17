"""P0-1 紧急停止机制测试:停止响应时间、按键释放保证、退出保护、断点语义。

运行: python -m unittest tests.test_player_stop -v
"""

import sys
import threading
import time
import unittest

from PyQt6.QtCore import Qt

from core.keymap import KeyMap
from core.player import Player

MAPPING = {
    "high": ["Q", "W", "E", "R", "T", "Y", "U"],
    "mid": ["A", "S", "D", "F", "G", "H", "J"],
    "low": ["Z", "X", "C", "V", "B", "N", "M"],
}

_ALL_KEYS = [KeyMap(MAPPING).key_for(f"{o}_{i}") for o in ("high", "mid", "low") for i in range(1, 8)]


class FakeDriver:
    def __init__(self):
        self.events = []
        self.lock = threading.Lock()

    def _log(self, kind, payload):
        with self.lock:
            self.events.append((kind, payload))

    def press_key(self, k):
        self._log("press", k)

    def release_key(self, k):
        self._log("release", k)

    def press_chord(self, keys):
        self._log("press_chord", tuple(keys))

    def release_chord(self, keys):
        self._log("release_chord", tuple(keys))


def wait_for_event(driver, kind, timeout=5.0):
    """等待某类事件出现(确保按键已真实发出,便于测中断与释放)。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        with driver.lock:
            if any(e[0] == kind for e in driver.events):
                return True
        time.sleep(0.002)
    return False


def make_player():
    driver = FakeDriver()
    player = Player(KeyMap(MAPPING), driver=driver)
    return player, driver


class TestStopLatency(unittest.TestCase):
    """核心指标:stop() 后演奏线程 100ms 内结束并释放按键。"""

    def _assert_stop_within_100ms(self, notes, bpm, wait_kind="press_chord"):
        player, driver = make_player()
        player.play(notes, bpm=bpm, hold_ratio=0.75, gap_ms=0)
        if wait_kind:
            self.assertTrue(wait_for_event(driver, wait_kind))
        else:
            time.sleep(0.05)  # 休止序列无驱动事件,稍等线程进入等待即可
        t0 = time.perf_counter()
        player.stop()
        while player.is_playing and time.perf_counter() - t0 < 2.0:
            time.sleep(0.001)
        elapsed = time.perf_counter() - t0
        self.assertFalse(player.is_playing)
        self.assertLess(elapsed, 0.1, f"停止响应耗时 {elapsed * 1000:.0f}ms,超出 100ms 要求")

    def test_stop_during_long_hold(self):
        # 30BPM 全音符:按住 3 秒,旧实现会等满 3 秒才响应
        notes = [{"notes": ["mid_1"], "dur": 4.0}] * 30
        self._assert_stop_within_100ms(notes, bpm=30)

    def test_stop_during_long_rest(self):
        notes = [{"notes": [], "dur": 4.0}] * 30
        self._assert_stop_within_100ms(notes, bpm=30, wait_kind=None)


class TestReleaseGuarantee(unittest.TestCase):
    def _released_keys(self, driver):
        with driver.lock:
            return {payload for kind, payload in driver.events if kind == "release"}

    def test_stop_releases_all_mapped_keys(self):
        player, driver = make_player()
        player.play([{"notes": ["mid_1"], "dur": 4.0}] * 30, bpm=30, gap_ms=0)
        self.assertTrue(wait_for_event(driver, "press_chord"))
        player.stop()
        deadline = time.time() + 5
        while player.is_playing and time.time() < deadline:
            time.sleep(0.005)
        self.assertTrue(set(_ALL_KEYS) <= self._released_keys(driver))

    def test_shutdown_releases_all_and_joins_thread(self):
        player, driver = make_player()
        player.play([{"notes": ["mid_1"], "dur": 4.0}] * 30, bpm=30, gap_ms=0)
        self.assertTrue(wait_for_event(driver, "press_chord"))
        t0 = time.perf_counter()
        player.shutdown()
        elapsed = time.perf_counter() - t0
        self.assertFalse(player.is_playing)
        self.assertLess(elapsed, 1.5)
        self.assertTrue(set(_ALL_KEYS) <= self._released_keys(driver))

    def test_shutdown_idempotent_when_idle(self):
        player, _ = make_player()
        player.shutdown()
        player.shutdown()
        self.assertFalse(player.is_playing)

    def test_shutdown_idle_performs_no_driver_calls(self):
        """闲置时 shutdown 不产生任何按键事件(避免进程收尾期的无谓 Win32 调用)。"""
        player, driver = make_player()
        player.shutdown()
        self.assertEqual(driver.events, [])

    def test_dispatch_guard_blocks_press_and_still_releases(self):
        player, driver = make_player()
        errors = []
        player.error_occurred.connect(
            errors.append, type=Qt.ConnectionType.DirectConnection
        )
        player.set_dispatch_guard(lambda: (False, "目标窗口已失焦"))
        player.play([{"notes": ["mid_1"], "dur": 0.1}], bpm=120, gap_ms=0)
        deadline = time.time() + 5
        while player.is_playing and time.time() < deadline:
            time.sleep(0.005)
        self.assertFalse(any(kind == "press_chord" for kind, _ in driver.events))
        self.assertEqual(errors, ["目标窗口已失焦"])
        self.assertTrue(set(_ALL_KEYS) <= self._released_keys(driver))


class TestPauseResumeSemantics(unittest.TestCase):
    def test_pause_mid_note_keeps_index_for_replay(self):
        """音符按住期间暂停:断点留在当前音符,续播时重放。"""
        player, driver = make_player()
        paused_events = []
        # DirectConnection:无事件循环的测试环境下直接在工作线程执行回调
        player.paused.connect(
            lambda d, t: paused_events.append((d, t)), type=Qt.ConnectionType.DirectConnection
        )
        notes = [{"notes": ["mid_1"], "dur": 1.0}] * 10
        player.play(notes, bpm=60, hold_ratio=0.75, gap_ms=0)  # 每音按住 750ms
        self.assertTrue(wait_for_event(driver, "press_chord"))
        player.stop()  # 立即停止,必然落在第一个音符按住期
        deadline = time.time() + 5
        while player.is_playing and time.time() < deadline:
            time.sleep(0.005)
        self.assertEqual(paused_events, [(0, 10)])

    def test_progress_index_starts_from_start_index(self):
        """从断点续播后立即暂停:断点不得回退到 0。"""
        player, driver = make_player()
        paused_events = []
        player.paused.connect(
            lambda d, t: paused_events.append((d, t)), type=Qt.ConnectionType.DirectConnection
        )
        notes = [{"notes": ["mid_1"], "dur": 4.0}] * 10
        player.play(notes, bpm=30, gap_ms=0, start_index=7)
        player.stop()  # 循环开始前即停止,断点应保持 7
        deadline = time.time() + 5
        while player.is_playing and time.time() < deadline:
            time.sleep(0.005)
        self.assertEqual(paused_events, [(7, 10)])


class TestDriverImportIsolation(unittest.TestCase):
    def test_module_importable_anywhere(self):
        """非 Windows 平台也必须能导入(测试隔离前提)。"""
        import importlib

        import core.keyboard_driver as kd

        importlib.reload(kd)
        self.assertTrue(callable(kd.KeyboardDriver))

    @unittest.skipIf(sys.platform == "win32", "Windows 上实例化必然成功")
    def test_instantiation_raises_off_windows(self):
        from core.keyboard_driver import KeyboardDriver

        with self.assertRaises(OSError):
            KeyboardDriver()


if __name__ == "__main__":
    unittest.main()
