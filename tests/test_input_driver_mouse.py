"""输入驱动鼠标能力测试(M1)。

覆盖鼠标三键发送、组合按键顺序、settle 可配置、兜底释放与既有键盘行为回归。
不产生真实输入:通过替换模块级 _user32 为 stub 拦截 SendInput。
"""

import sys
import unittest
from unittest import mock

import core.keyboard_driver as kd
from core.keyboard_driver import KeyboardDriver


class _StubUser32:
    """拦截 SendInput,记录每次输入的设备类型与标志位。"""

    def __init__(self):
        self.calls = []
        self.send_result = 1

    def SendInput(self, n, ptr, size):
        # 生产代码用 ctypes.byref(...) 传参,得到的是 CArgObject;
        # 其 _obj 回指原 INPUT 实例,从而无需为测试改动生产代码。
        inp = getattr(ptr, "_obj", None)
        if inp is None:
            inp = ptr.contents
        if inp.type == kd.INPUT_KEYBOARD:
            self.calls.append({
                "device": "kb",
                "wScan": inp.ki.wScan,
                "flags": inp.ki.dwFlags,
            })
        else:
            self.calls.append({
                "device": "mouse",
                "flags": inp.mi.dwFlags,
            })
        return self.send_result

    def VkKeyScanW(self, ch):
        return ord(chr(ch).upper())

    def MapVirtualKeyW(self, vk, mode):
        return vk


class _MouseDriverTestCase(unittest.TestCase):
    def setUp(self):
        self._orig_user32 = kd._user32
        self.stub = _StubUser32()
        kd._user32 = self.stub
        self.driver = KeyboardDriver()

    def tearDown(self):
        kd._user32 = self._orig_user32

    def _sleeps(self):
        """返回 recorder:记录 time.sleep 被要求的毫秒数,且不真的睡眠。"""
        recorder = []
        return recorder, recorder

    def flags(self):
        return [c["flags"] for c in self.stub.calls]

    def devices(self):
        return [c["device"] for c in self.stub.calls]


class TestMouseFlags(_MouseDriverTestCase):
    def test_mouse_flags_map_complete(self):
        for button in kd.MOUSE_BUTTONS:
            for up in (False, True):
                self.assertIn((button, up), kd._MOUSE_FLAGS)
        values = list(kd._MOUSE_FLAGS.values())
        self.assertEqual(len(values), 6)
        self.assertEqual(len(set(values)), 6, "6 个标志位必须互不相同")

    def test_press_mouse_middle_flag(self):
        self.driver.press_mouse("middle")
        self.assertEqual(self.flags(), [kd.MOUSEEVENTF_MIDDLEDOWN])
        self.assertEqual(self.flags()[0], 0x0020)

    def test_release_mouse_middle_flag(self):
        self.driver.release_mouse("middle")
        self.assertEqual(self.flags(), [kd.MOUSEEVENTF_MIDDLEUP])
        self.assertEqual(self.flags()[0], 0x0040)

    def test_left_right_flag_values(self):
        self.driver.press_mouse("left")
        self.driver.release_mouse("left")
        self.driver.press_mouse("right")
        self.driver.release_mouse("right")
        self.assertEqual(self.flags(), [
            kd.MOUSEEVENTF_LEFTDOWN, kd.MOUSEEVENTF_LEFTUP,
            kd.MOUSEEVENTF_RIGHTDOWN, kd.MOUSEEVENTF_RIGHTUP,
        ])

    def test_unknown_button_raises(self):
        with self.assertRaises(ValueError):
            self.driver.press_mouse("xbutton")
        self.assertEqual(self.stub.calls, [], "非法按钮不得发出任何输入")


class TestSettleConfig(_MouseDriverTestCase):
    def test_negative_settle_raises(self):
        with self.assertRaises(ValueError):
            KeyboardDriver(settle_ms=-1)
        with self.assertRaises(ValueError):
            KeyboardDriver(release_settle_ms=-1)

    def test_non_numeric_settle_raises(self):
        with self.assertRaises(ValueError):
            KeyboardDriver(settle_ms="30")

    def test_defaults(self):
        d = KeyboardDriver()
        self.assertEqual(d.settle_ms, 30.0)
        self.assertEqual(d.release_settle_ms, 20.0)

    def test_no_arg_construction_still_works(self):
        """向后兼容:无参构造必须继续可用(生产代码与 selftest 依赖)。"""
        self.assertIsInstance(KeyboardDriver(), KeyboardDriver)


class TestComboOrder(_MouseDriverTestCase):
    def _run_with_sleep_record(self, fn):
        sleeps = []
        with mock.patch.object(kd.time, "sleep", side_effect=lambda s: sleeps.append(s * 1000.0)):
            fn()
        return sleeps

    def test_press_combo_order_and_settle(self):
        sleeps = self._run_with_sleep_record(
            lambda: self.driver.press_combo("Z", "right"))
        self.assertEqual(self.devices(), ["mouse", "kb"], "修饰键必须先于音键按下")
        self.assertEqual(self.flags()[0], kd.MOUSEEVENTF_RIGHTDOWN)
        self.assertEqual(len(sleeps), 1)
        self.assertAlmostEqual(sleeps[0], 30.0, places=3)

    def test_release_combo_order_and_settle(self):
        sleeps = self._run_with_sleep_record(
            lambda: self.driver.release_combo("Z", "right"))
        self.assertEqual(self.devices(), ["kb", "mouse"], "音键必须先于修饰键松开")
        self.assertEqual(self.flags()[0], kd.KEYEVENTF_KEYUP | kd.KEYEVENTF_SCANCODE)
        self.assertEqual(self.flags()[1], kd.MOUSEEVENTF_RIGHTUP)
        self.assertAlmostEqual(sleeps[0], 20.0, places=3)

    def test_combo_without_mouse(self):
        sleeps = self._run_with_sleep_record(lambda: self.driver.press_combo("Z"))
        self.driver.release_combo("Z")
        self.assertEqual(self.devices(), ["kb", "kb"], "无修饰键时不应发出鼠标事件")
        self.assertEqual(sleeps, [])

    def test_per_call_settle_override(self):
        sleeps = self._run_with_sleep_record(
            lambda: self.driver.press_combo("Z", "right", settle_ms=50))
        self.assertAlmostEqual(sleeps[0], 50.0, places=3)

    def test_full_roundtrip_order(self):
        """完整一轮:mouse down → key down → key up → mouse up。"""
        self._run_with_sleep_record(lambda: (
            self.driver.press_combo("Z", "left"),
            self.driver.release_combo("Z", "left"),
        ))
        self.assertEqual(self.flags(), [
            kd.MOUSEEVENTF_LEFTDOWN,
            kd.KEYEVENTF_SCANCODE,
            kd.KEYEVENTF_KEYUP | kd.KEYEVENTF_SCANCODE,
            kd.MOUSEEVENTF_LEFTUP,
        ])


class TestPanicRelease(_MouseDriverTestCase):
    def test_releases_all_and_never_raises(self):
        self.driver.panic_release(keys=["Z", None, "M"], mouse_buttons=["right", None, "left"])
        self.assertEqual(self.devices(), ["kb", "kb", "mouse", "mouse"])

    def test_swallows_individual_failure(self):
        def boom(_key, _up):
            raise OSError("SendInput 失败")

        with mock.patch.object(KeyboardDriver, "_send", side_effect=boom):
            self.driver.panic_release(keys=["Z", "M"], mouse_buttons=["right"])
        # 未抛出即通过;键盘失败不应阻止鼠标释放
        mouse_flags = [c["flags"] for c in self.stub.calls if c["device"] == "mouse"]
        self.assertEqual(mouse_flags, [kd.MOUSEEVENTF_RIGHTUP])

    def test_idempotent(self):
        self.driver.panic_release(keys=["Z"], mouse_buttons=["right"])
        first = len(self.stub.calls)
        self.driver.panic_release(keys=["Z"], mouse_buttons=["right"])
        self.assertEqual(len(self.stub.calls), first * 2)


class TestPlatformAndRegression(_MouseDriverTestCase):
    def test_non_windows_raises(self):
        kd._user32 = None
        with mock.patch.object(sys, "platform", "linux"):
            with self.assertRaises(OSError):
                self.driver.press_mouse("left")
        kd._user32 = self.stub

    def test_send_failure_raises(self):
        self.stub.send_result = 0
        with self.assertRaises(OSError):
            self.driver.press_mouse("left")

    def test_press_key_regression_unchanged(self):
        """既有键盘路径行为必须与改造前一致(Q → wScan=81, flags=SCANCODE)。"""
        self.driver.press_key("Q")
        self.assertEqual(self.stub.calls, [{"device": "kb", "wScan": 81, "flags": 8}])

    def test_release_key_regression_unchanged(self):
        self.driver.release_key("Q")
        self.assertEqual(self.stub.calls, [{"device": "kb", "wScan": 81, "flags": 8 | 2}])

    def test_chord_still_uses_chord_interval(self):
        sleeps = []
        with mock.patch.object(kd.time, "sleep", side_effect=lambda s: sleeps.append(s * 1000.0)):
            self.driver.press_chord(["A", "S", "D"])
        self.assertEqual(sleeps, [5.0, 5.0], "和弦内间隔仍为 CHORD_INTERVAL_MS=5")


if __name__ == "__main__":
    unittest.main()
