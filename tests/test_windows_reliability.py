import unittest
from unittest import mock

import core.windows_reliability as reliability
from core.windows_reliability import assess_elevation, inspect_target_elevation


class TestElevationPolicy(unittest.TestCase):
    @mock.patch.object(reliability.sys, "platform", "win32")
    @mock.patch.object(reliability.ctypes, "windll", create=True)
    def test_current_process_elevation_queries_shell32(self, windll):
        windll.shell32.IsUserAnAdmin.return_value = 1
        self.assertTrue(reliability.current_process_elevated())
        windll.shell32.IsUserAnAdmin.assert_called_once_with()

    @mock.patch.object(reliability.sys, "platform", "win32")
    @mock.patch.object(reliability.sys, "argv", ["app/main.py", "--safe"])
    @mock.patch.object(reliability.sys, "executable", "C:\\Python\\python.exe")
    @mock.patch.object(reliability.ctypes, "windll", create=True)
    def test_admin_restart_keeps_source_entry_script(self, windll):
        windll.shell32.ShellExecuteW.return_value = 42
        self.assertTrue(reliability.restart_as_admin())
        call = windll.shell32.ShellExecuteW.call_args.args
        self.assertEqual(call[2], "C:\\Python\\python.exe")
        self.assertIn("main.py", call[3])
        self.assertIn("--safe", call[3])

    def test_normal_app_cannot_drive_confirmed_elevated_target(self):
        result = assess_elevation(False, True)
        self.assertTrue(result.blocked)
        self.assertEqual(result.status, "blocked")
        self.assertIn("管理员", result.message)

    def test_equal_or_stronger_app_is_compatible(self):
        for app, target in ((False, False), (True, False), (True, True)):
            with self.subTest(app=app, target=target):
                result = assess_elevation(app, target)
                self.assertFalse(result.blocked)
                self.assertEqual(result.status, "compatible")

    def test_unknown_is_not_claimed_as_compatible(self):
        for app, target in ((None, True), (False, None), (None, None)):
            with self.subTest(app=app, target=target):
                result = assess_elevation(app, target)
                self.assertFalse(result.blocked)
                self.assertEqual(result.status, "unknown")

    @mock.patch("core.windows_reliability.process_elevated", return_value=True)
    @mock.patch("core.windows_reliability.current_process_elevated", return_value=False)
    @mock.patch("core.windows_reliability.window_process_id", return_value=321)
    def test_inspection_composes_read_only_queries(self, get_pid, get_app, get_target):
        result = inspect_target_elevation(123)
        self.assertTrue(result.blocked)
        get_pid.assert_called_once_with(123)
        get_app.assert_called_once_with()
        get_target.assert_called_once_with(321)


if __name__ == "__main__":
    unittest.main()
