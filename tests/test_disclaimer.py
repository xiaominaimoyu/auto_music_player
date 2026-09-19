"""gui.disclaimer 模块测试:文案逐字断言 + 对话框交互/防绕过断言。

offscreen 平台运行,不依赖真实显示器。运行: python -m unittest tests.test_disclaimer
"""

import os
import sys
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QLabel,
    QPushButton,
    QScrollArea,
    QWidget,
)

from gui.disclaimer import (
    BTN_CONTINUE_TEXT,
    BTN_QUIT_TEXT,
    COUNTDOWN_SECONDS,
    DISCLAIMER_BODY,
    DISCLAIMER_OPEN_SOURCE,
    DISCLAIMER_SKIP_KEY,
    DISCLAIMER_TUTORIAL,
    DISCLAIMER_TITLE,
    RiskDisclaimerDialog,
    SKIP_CHECKBOX_TEXT,
    confirm_risk_disclaimer,
)
from gui.theme import SURFACE_2

BANNED_TEXTS = ("稍后决定",)


class MemorySettings:
    def __init__(self, values=None):
        self.values = dict(values or {})
        self.sync_count = 0

    def value(self, key, default=False):
        return self.values.get(key, default)

    def setValue(self, key, value):
        self.values[key] = value

    def sync(self):
        self.sync_count += 1


def setUpModule():
    global _app
    _app = QApplication.instance() or QApplication(sys.argv)


class TestDisclaimerConstants(unittest.TestCase):
    def test_constants_match_required_values(self):
        self.assertEqual(DISCLAIMER_TITLE, "⚠️ 风险警示")
        self.assertIn("游戏账号封禁", DISCLAIMER_BODY)
        self.assertIn("免费开源", DISCLAIMER_OPEN_SOURCE)
        self.assertIn("作者联系方式", DISCLAIMER_OPEN_SOURCE)
        self.assertIn("试听当前乐谱", DISCLAIMER_TUTORIAL)
        self.assertEqual(BTN_CONTINUE_TEXT, "我已知晓并自愿承担全部风险，继续使用")
        self.assertEqual(BTN_QUIT_TEXT, "退出程序")

    def test_constants_non_empty(self):
        for value in (
            DISCLAIMER_TITLE,
            DISCLAIMER_BODY,
            DISCLAIMER_OPEN_SOURCE,
            DISCLAIMER_TUTORIAL,
            BTN_CONTINUE_TEXT,
            BTN_QUIT_TEXT,
            SKIP_CHECKBOX_TEXT,
        ):
            self.assertTrue(value)


class TestRiskDisclaimerDialog(unittest.TestCase):
    def setUp(self):
        self.dlg = RiskDisclaimerDialog()
        self.addCleanup(self.dlg._countdown_timer.stop)
        self.addCleanup(self.dlg.deleteLater)

    def _button(self, text):
        for btn in self.dlg.findChildren(QPushButton):
            if btn.text() == text:
                return btn
        raise AssertionError(f"按钮不存在: {text}")

    def _finish_countdown(self):
        self.dlg._countdown_timer.stop()
        for _ in range(COUNTDOWN_SECONDS):
            self.dlg._countdown_tick()

    def test_dialog_is_modal(self):
        self.assertTrue(self.dlg.isModal())

    def test_continue_button_accepts_after_countdown(self):
        self._finish_countdown()
        self._button(BTN_CONTINUE_TEXT).click()
        self.assertEqual(self.dlg.result(), RiskDisclaimerDialog.DialogCode.Accepted)
        self.assertFalse(self.dlg._countdown_timer.isActive())

    def test_continue_button_is_locked_during_countdown(self):
        button = self._button(BTN_CONTINUE_TEXT)
        self.assertFalse(button.isEnabled())
        self.dlg.accept()
        self.assertEqual(self.dlg.result(), RiskDisclaimerDialog.DialogCode.Rejected)

    def test_skip_checkbox_is_present(self):
        checkbox = self.dlg.findChild(QCheckBox, "DisclaimerSkipCheckbox")
        self.assertIsNotNone(checkbox)
        self.assertEqual(checkbox.text(), SKIP_CHECKBOX_TEXT)

    def test_quit_button_rejects_and_stops_timer(self):
        self._button(BTN_QUIT_TEXT).click()
        self.assertEqual(self.dlg.result(), RiskDisclaimerDialog.DialogCode.Rejected)
        self.assertFalse(self.dlg._countdown_timer.isActive())

    def test_close_event_normalizes_to_rejected(self):
        # 模拟标题栏 × / Alt+F4 路径:close() 触发 closeEvent
        self.dlg.close()
        self.assertEqual(self.dlg.result(), RiskDisclaimerDialog.DialogCode.Rejected)
        self.assertFalse(self.dlg._countdown_timer.isActive())

    def test_exactly_two_clickable_buttons(self):
        buttons = self.dlg.findChildren(QPushButton)
        self.assertEqual(len(buttons), 2)
        self.assertEqual(
            sorted(b.text() for b in buttons),
            sorted([BTN_CONTINUE_TEXT, BTN_QUIT_TEXT]),
        )

    def test_no_alternate_bypass_widget(self):
        texts = " ".join(lbl.text() for lbl in self.dlg.findChildren(QLabel))
        for banned in BANNED_TEXTS:
            self.assertNotIn(banned, texts)

    def test_body_label_shows_verbatim_text(self):
        body = self.dlg.findChild(QLabel, "DisclaimerBody")
        self.assertIsNotNone(body)
        self.assertEqual(body.text(), DISCLAIMER_BODY)

    def test_content_area_has_contrasting_theme_background(self):
        content = self.dlg.findChild(QWidget, "DisclaimerContent")
        scroll = self.dlg.findChild(QScrollArea, "DisclaimerScroll")
        body = self.dlg.findChild(QLabel, "DisclaimerBody")
        self.assertIsNotNone(content)
        self.assertIsNotNone(scroll)
        self.assertIsNotNone(body)
        self.assertIn(SURFACE_2, content.styleSheet())
        self.assertIn(SURFACE_2, scroll.viewport().styleSheet())
        self.assertIn("font-size: 14px", body.styleSheet())


class TestConfirmRiskDisclaimer(unittest.TestCase):
    def test_accepted_with_skip_persists_preference(self):
        settings = MemorySettings()
        with patch.object(
            RiskDisclaimerDialog,
            "exec",
            lambda self: (
                self._countdown_timer.stop(),
                self.skip_checkbox.setChecked(True),
                RiskDisclaimerDialog.DialogCode.Accepted,
            )[2],
        ):
            self.assertTrue(confirm_risk_disclaimer(settings=settings))
        self.assertTrue(settings.value(DISCLAIMER_SKIP_KEY))
        self.assertEqual(settings.sync_count, 1)

    def test_rejected_does_not_persist_preference(self):
        settings = MemorySettings()
        with patch.object(
            RiskDisclaimerDialog,
            "exec",
            lambda self: (
                self._countdown_timer.stop(),
                self.skip_checkbox.setChecked(True),
                RiskDisclaimerDialog.DialogCode.Rejected,
            )[2],
        ):
            self.assertFalse(confirm_risk_disclaimer(settings=settings))
        self.assertNotIn(DISCLAIMER_SKIP_KEY, settings.values)
        self.assertEqual(settings.sync_count, 0)

    def test_saved_skip_preference_skips_dialog(self):
        settings = MemorySettings({DISCLAIMER_SKIP_KEY: "true"})
        with patch.object(
            RiskDisclaimerDialog,
            "__init__",
            side_effect=AssertionError("不应构造免责声明对话框"),
        ):
            self.assertTrue(confirm_risk_disclaimer(settings=settings))

    def test_legacy_skip_preference_does_not_skip_current_release_dialog(self):
        """升级用户的旧版跳过设置不能绕过 v1.4.1 启动风险提示。"""
        legacy_key = "disclaimer/skip_on_next_start"
        settings = MemorySettings({legacy_key: "true"})
        with patch.object(
            RiskDisclaimerDialog,
            "exec",
            lambda self: (
                self._countdown_timer.stop(),
                RiskDisclaimerDialog.DialogCode.Rejected,
            )[1],
        ):
            self.assertFalse(confirm_risk_disclaimer(settings=settings))
        self.assertNotIn(DISCLAIMER_SKIP_KEY, settings.values)


if __name__ == "__main__":
    unittest.main()
