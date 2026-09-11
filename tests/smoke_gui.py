"""GUI 冒烟测试:启动主窗口 1.5 秒后自动退出,验证无崩溃。
运行: python tests/smoke_gui.py"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication

import main as m


def run():
    cfg = m.load_config()
    db = m.ScoreDB(os.path.join("data", "smoke.db"))
    keymap = m.KeyMap(cfg["keymap"])
    player = m.Player(keymap)
    app = QApplication(sys.argv)
    app.setStyleSheet(m.APP_QSS)
    win = m.MainWindow(cfg, db, keymap, player)
    win.show()
    # 演奏小窗冒烟:切为小窗 -> 还原主窗(验证原生样式/穿透过滤/状态镜像无崩溃)
    QTimer.singleShot(600, win._switch_to_mini)
    QTimer.singleShot(1100, win._restore_from_mini)
    # AppDialog 组件冒烟:五种类型可构造不崩溃
    from gui.widgets import AppDialog

    for dtype in ("success", "info", "warning", "error", "confirm"):
        dlg = AppDialog(win, dtype, "冒烟测试", "组件构建验证", [("知道了", "primary")])
        dlg.deleteLater()
    # 风险警示对话框冒烟:常量非空 + 可构造可布局不崩溃
    from gui.disclaimer import (
        BTN_CONTINUE_TEXT,
        BTN_QUIT_TEXT,
        DISCLAIMER_BODY,
        DISCLAIMER_TITLE,
        RiskDisclaimerDialog,
    )

    assert DISCLAIMER_TITLE and DISCLAIMER_BODY and BTN_CONTINUE_TEXT and BTN_QUIT_TEXT
    disclaimer = RiskDisclaimerDialog()
    disclaimer.deleteLater()
    QTimer.singleShot(1500, app.quit)
    rc = app.exec()
    db.conn.close()
    if os.path.exists("data/smoke.db"):
        os.remove("data/smoke.db")
    print("GUI smoke OK" if rc == 0 else f"GUI smoke rc={rc}")
    return rc


if __name__ == "__main__":
    sys.exit(run())