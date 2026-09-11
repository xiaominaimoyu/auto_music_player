"""gui.disclaimer 模块测试:文案逐字断言 + 对话框交互/防绕过断言。

offscreen 平台运行,不依赖真实显示器。运行: python -m unittest tests.test_disclaimer
"""

import os
import sys
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtWidgets import QApplication, QLabel, QPushButton

from gui.disclaimer import (
    BTN_CONTINUE_TEXT,
    BTN_QUIT_TEXT,
    DISCLAIMER_BODY,
    DISCLAIMER_TITLE,
    RiskDisclaimerDialog,
    confirm_risk_disclaimer,
)

# spec 6.1/6.2 锁定的字面值(测试断言基准,非渲染来源)
LOCKED_TITLE = "⚠️ 风险警示"
LOCKED_BODY = (
    "该软件目前尚不成熟，有游戏账号封禁的可能，因此下载使用前请务必知悉使用风险！"
    "作者及其共创者不负有任何责任。选择使用软件造成的一切后果由自己承担。"
)
LOCKED_CONTINUE = "我已知晓并自愿承担全部风险，继续使用"
LOCKED_QUIT = "退出程序"

BANNED_TEXTS = ("不再提示", "跳过", "稍后决定")


def setUpModule():
    global _app
    _app = QApplication.instance() or QApplication(sys.argv)


class TestDisclaimerConstants(unittest.TestCase):
    def test_constants_match_spec_locked_values(self):
        self.assertEqual(DISCLAIMER_TITLE, LOCKED_TITLE)
        self.assertEqual(DISCLAIMER_BODY, LOCKED_BODY)
        self.assertEqual(BTN_CONTINUE_TEXT, LOCKED_CONTINUE)
        self.assertEqual(BTN_QUIT_TEXT, LOCKED_QUIT)

    def test_constants_non_empty(self):
        for value in (DISCLAIMER_TITLE, DISCLAIMER_BODY, BTN_CONTINUE_TEXT, BTN_QUIT_TEXT):
            self.assertTrue(value)


class TestRiskDisclaimerDialog(unittest.TestCase):
    def setUp(self):
        self.dlg = RiskDisclaimerDialog()
        self.addCleanup(self.dlg.deleteLater)

    def _button(self, text):
        for btn in self.dlg.findChildren(QPushButton):
            if btn.text() == text:
                return btn
        raise AssertionError(f"按钮不存在: {text}")

    def test_dialog_is_modal(self):
        self.assertTrue(self.dlg.isModal())

    def test_continue_button_accepts(self):
        self._button(BTN_CONTINUE_TEXT).click()
        self.assertEqual(self.dlg.result(), RiskDisclaimerDialog.DialogCode.Accepted)

    def test_quit_button_rejects(self):
        self._button(BTN_QUIT_TEXT).click()
        self.assertEqual(self.dlg.result(), RiskDisclaimerDialog.DialogCode.Rejected)

    def test_close_event_normalizes_to_rejected(self):
        # 模拟标题栏 × / Alt+F4 路径:close() 触发 closeEvent
        self.dlg.close()
        self.assertEqual(self.dlg.result(), RiskDisclaimerDialog.DialogCode.Rejected)

    def test_exactly_two_clickable_buttons(self):
        buttons = self.dlg.findChildren(QPushButton)
        self.assertEqual(len(buttons), 2)
        self.assertEqual(
            sorted(b.text() for b in buttons),
            sorted([BTN_CONTINUE_TEXT, BTN_QUIT_TEXT]),
        )

    def test_no_bypass_widgets(self):
        texts = " ".join(lbl.text() for lbl in self.dlg.findChildren(QLabel))
        for banned in BANNED_TEXTS:
            self.assertNotIn(banned, texts)

    def test_body_label_shows_verbatim_text(self):
        body = self.dlg.findChild(QLabel, "DisclaimerBody")
        self.assertIsNotNone(body)
        self.assertEqual(body.text(), DISCLAIMER_BODY)


class TestConfirmRiskDisclaimer(unittest.TestCase):
    def test_returns_true_on_accepted(self):
        with patch.object(
            RiskDisclaimerDialog, "exec",
            lambda self: RiskDisclaimerDialog.DialogCode.Accepted,
        ):
            self.assertTrue(confirm_risk_disclaimer())

    def test_returns_false_on_rejected(self):
        with patch.object(
            RiskDisclaimerDialog, "exec",
            lambda self: RiskDisclaimerDialog.DialogCode.Rejected,
        ):
            self.assertFalse(confirm_risk_disclaimer())


if __name__ == "__main__":
    unittest.main()
