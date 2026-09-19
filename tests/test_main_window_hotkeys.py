"""主窗口全局热键接线测试:默认 F6 播放、F8 停止。"""

import os
import tempfile
import unittest
from unittest import mock

from PyQt6.QtWidgets import QApplication

import main as app_main
from gui.main_window import MainWindow
from gui.player_tab import PlayerTab


_APP = None


def ensure_qapp():
    global _APP
    _APP = QApplication.instance() or _APP or QApplication([])
    return _APP


class _FakeGlobalHotKeys:
    instances = []

    def __init__(self, mapping):
        self.mapping = mapping
        self.started = False
        self.stopped = False
        self.__class__.instances.append(self)

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True


class TestMainWindowHotkeys(unittest.TestCase):
    def test_default_f6_play_and_f8_stop_are_forwarded(self):
        ensure_qapp()
        cfg = app_main.load_config()
        with tempfile.TemporaryDirectory() as tmp:
            db = app_main.ScoreDB(os.path.join(tmp, "scores.db"))
            keymap = app_main.KeyMap(cfg["keymap"])
            player = app_main.Player(keymap)
            _FakeGlobalHotKeys.instances.clear()
            win = None
            try:
                with mock.patch("gui.main_window.pk.GlobalHotKeys", _FakeGlobalHotKeys):
                    with mock.patch.object(
                        PlayerTab, "mini_play", autospec=True
                    ) as play:
                        with mock.patch.object(
                            PlayerTab, "mini_stop", autospec=True
                        ) as stop:
                            win = MainWindow(cfg, db, keymap, player)
                            listener = _FakeGlobalHotKeys.instances[-1]
                            self.assertEqual(set(listener.mapping), {"<f6>", "<f8>"})
                            self.assertTrue(listener.started)

                            listener.mapping["<f6>"]()
                            listener.mapping["<f8>"]()
                            play.assert_called_once_with(win.player_tab)
                            stop.assert_called_once_with(win.player_tab)
            finally:
                if win is not None:
                    win._cleanup_on_quit()
                    win.deleteLater()
                db.conn.close()


if __name__ == "__main__":
    unittest.main()
