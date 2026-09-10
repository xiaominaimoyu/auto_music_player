"""core.event_player 单元测试。

模式沿用 tests.test_player_stop:用桩驱动观察真实调用,
信号用 DirectConnection 在无事件循环的测试环境下于工作线程同步执行回调。
"""

import threading
import time
import unittest

from PyQt6.QtCore import Qt

from core.compiler import InputEvent
from core.event_player import EventPlayer


class StubDriver:
    def __init__(self):
        self.events = []
        self.lock = threading.Lock()

    def _log(self, kind, payload):
        with self.lock:
            self.events.append((kind, payload))

    def press_key(self, k):
        self._log("press_key", k)

    def release_key(self, k):
        self._log("release_key", k)

    def press_mouse(self, b):
        self._log("press_mouse", b)

    def release_mouse(self, b):
        self._log("release_mouse", b)

    def panic_release(self, keys, mouse):
        self._log("panic_release", (tuple(keys), tuple(mouse)))


def wait_for_event(driver, kind, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        with driver.lock:
            if any(e[0] == kind for e in driver.events):
                return True
        time.sleep(0.002)
    return False


def wait_for_n_events(driver, kind, n, timeout=5.0):
    """等待某类事件出现至少 n 次。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        with driver.lock:
            if sum(1 for e in driver.events if e[0] == kind) >= n:
                return True
        time.sleep(0.002)
    return False


def wait_idle(player, timeout=5.0):
    deadline = time.time() + timeout
    while player.is_playing and time.time() < deadline:
        time.sleep(0.005)


def make_player():
    driver = StubDriver()
    return EventPlayer(driver), driver


class _Capture:
    """连接信号并记录(参数)的辅助。"""

    def __init__(self, signal):
        self.records = []
        signal.connect(lambda *a: self.records.append(a),
                       type=Qt.ConnectionType.DirectConnection)


class TestDispatchOrder(unittest.TestCase):
    def test_events_dispatched_in_order(self):
        player, driver = make_player()
        events = [
            InputEvent(0, "mouse", "right", "down"),
            InputEvent(5, "kb", "Z", "down"),
            InputEvent(10, "kb", "Z", "up"),
            InputEvent(15, "mouse", "right", "up"),
        ]
        player.play(events)
        wait_idle(player)
        dispatched = [e for e in driver.events if e[0] != "panic_release"]
        self.assertEqual(dispatched, [
            ("press_mouse", "right"),
            ("press_key", "Z"),
            ("release_key", "Z"),
            ("release_mouse", "right"),
        ])

    def test_progress_counts_events(self):
        player, _ = make_player()
        cap = _Capture(player.progress)
        player.play([InputEvent(i * 5, "kb", "Z", "down") for i in range(3)])
        wait_idle(player)
        self.assertEqual(cap.records, [(1, 3), (2, 3), (3, 3)])

    def test_finish_releases_set_derived_from_events(self):
        """释放集合来自事件本身,不依赖外部键位清单。"""
        player, driver = make_player()
        events = [
            InputEvent(0, "mouse", "left", "down"),
            InputEvent(5, "kb", "B", "down"),
            InputEvent(10, "kb", "B", "up"),
            InputEvent(15, "mouse", "left", "up"),
        ]
        player.play(events)
        wait_idle(player)
        panic = [p for k, p in driver.events if k == "panic_release"]
        self.assertEqual(panic, [(("B",), ("left",))])


class TestResume(unittest.TestCase):
    def test_start_index_skips_prior_events(self):
        player, driver = make_player()
        events = [InputEvent(i * 5, "kb", "Z", "down") for i in range(4)]
        player.play(events, start_index=2)
        wait_idle(player)
        dispatched = [e for e in driver.events if e[0] == "press_key"]
        self.assertEqual(len(dispatched), 2, "应从索引 2 开始,只发后两个事件")

    def test_pause_mid_play_keeps_index(self):
        player, driver = make_player()
        paused = _Capture(player.paused)
        # 第二个事件在 5 秒后,留出充足时间让停止落在第一事件之后
        events = [
            InputEvent(0, "kb", "Z", "down"),
            InputEvent(10, "kb", "Z", "up"),
            InputEvent(5000, "kb", "X", "down"),
        ]
        player.play(events, interrupt_mode="pause")
        self.assertTrue(wait_for_event(driver, "release_key"))
        player.stop()
        wait_idle(player)
        self.assertEqual(paused.records, [(2, 3)])

    def test_resume_after_pause_does_not_replay(self):
        """续播不得重放已完成事件。前两个事件快速发出,把 done 钉死在 2,消除时序抖动。"""
        player, driver = make_player()
        paused = _Capture(player.paused)
        # 前两个事件紧邻,第三个在 5 秒后 → 停在确定的 done=2
        events = [
            InputEvent(0, "kb", "Z", "down"),
            InputEvent(10, "kb", "X", "down"),
            InputEvent(5000, "kb", "C", "down"),
        ]
        player.play(events, interrupt_mode="pause")
        self.assertTrue(wait_for_n_events(driver, "press_key", 2))
        player.stop()
        wait_idle(player)
        self.assertEqual(paused.records, [(2, 3)])
        driver.events.clear()
        # 续播:基时刻按 events[2].t_ms 折算,立即发出第三个,不重放前两个
        player.play(events, interrupt_mode="pause", start_index=2)
        wait_idle(player)
        replayed = [e for e in driver.events if e[0] == "press_key"]
        self.assertEqual(replayed, [("press_key", "C")])


class TestInterruptSemantics(unittest.TestCase):
    LONG = [
        InputEvent(0, "kb", "Z", "down"),
        InputEvent(5000, "kb", "Z", "up"),
    ]

    def test_pause_emits_paused_not_finished(self):
        player, driver = make_player()
        paused = _Capture(player.paused)
        finished = _Capture(player.finished)
        player.play(self.LONG, interrupt_mode="pause")
        self.assertTrue(wait_for_event(driver, "press_key"))
        player.stop()
        wait_idle(player)
        self.assertEqual(len(paused.records), 1)
        self.assertEqual(finished.records, [], "pause 与 finished 互斥")

    def test_abort_emits_aborted_not_paused(self):
        player, driver = make_player()
        aborted = _Capture(player.aborted)
        paused = _Capture(player.paused)
        player.play(self.LONG, interrupt_mode="abort")
        self.assertTrue(wait_for_event(driver, "press_key"))
        player.stop()
        wait_idle(player)
        self.assertEqual(len(aborted.records), 1)
        self.assertEqual(paused.records, [], "abort 不应发 paused")

    def test_abort_overrides_pause_mode(self):
        """显式 abort(reason) 即使在 pause 模式下也强制中止。"""
        player, driver = make_player()
        aborted = _Capture(player.aborted)
        paused = _Capture(player.paused)
        player.play(self.LONG, interrupt_mode="pause")
        self.assertTrue(wait_for_event(driver, "press_key"))
        player.abort("焦点丢失")
        wait_idle(player)
        self.assertEqual(aborted.records, [("焦点丢失",)])
        self.assertEqual(paused.records, [])

    def test_stop_releases_held_key_and_mouse(self):
        """中途停止必须释放事件中出现过的键与鼠标(无残留)。"""
        player, driver = make_player()
        events = [
            InputEvent(0, "mouse", "right", "down"),
            InputEvent(0, "kb", "Z", "down"),
            InputEvent(5000, "kb", "Z", "up"),
            InputEvent(5000, "mouse", "right", "up"),
        ]
        player.play(events, interrupt_mode="pause")
        self.assertTrue(wait_for_event(driver, "press_key"))
        player.stop()
        wait_idle(player)
        panic = [p for k, p in driver.events if k == "panic_release"]
        self.assertTrue(any("Z" in ks for ks, _ in panic))
        self.assertTrue(any("right" in ms for _, ms in panic))


class TestEdgeCases(unittest.TestCase):
    def test_empty_events_finishes_safely(self):
        player, driver = make_player()
        finished = _Capture(player.finished)
        player.play([])
        wait_idle(player)
        self.assertEqual(finished.records, [(True,)])
        self.assertEqual(driver.events, [])

    def test_invalid_interrupt_mode_raises(self):
        player, _ = make_player()
        with self.assertRaises(ValueError):
            player.play([InputEvent(0, "kb", "Z", "down")], interrupt_mode="bogus")

    def test_shutdown_releases_and_joins(self):
        player, driver = make_player()
        player.play(TestInterruptSemantics.LONG, interrupt_mode="pause")
        self.assertTrue(wait_for_event(driver, "press_key"))
        t0 = time.perf_counter()
        player.shutdown()
        self.assertFalse(player.is_playing)
        self.assertLess(time.perf_counter() - t0, 1.5)
        self.assertTrue(any(k == "panic_release" for k, _ in driver.events))

    def test_shutdown_idempotent_when_idle(self):
        player, driver = make_player()
        player.shutdown()
        player.shutdown()
        self.assertFalse(player.is_playing)
        self.assertEqual(driver.events, [], "闲置时 shutdown 不应产生驱动调用")

    def test_finished_false_on_driver_error(self):
        player, driver = make_player()
        finished = _Capture(player.finished)
        errors = _Capture(player.error_occurred)

        def boom(_k):
            raise OSError("SendInput 失败")

        driver.press_key = boom
        player.play([InputEvent(0, "kb", "Z", "down")], interrupt_mode="pause")
        wait_idle(player)
        self.assertEqual(finished.records, [(False,)])
        self.assertEqual(len(errors.records), 1)


if __name__ == "__main__":
    unittest.main()
