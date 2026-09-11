"""入口:加载配置,组装数据库/识别器/按键映射/演奏器,启动 GUI(暗色琥珀金主题)。"""

import atexit
import ctypes
import os
import shutil
import sys

import yaml
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QApplication

from core.database import ScoreDB
from core.event_player import EventPlayer
from core.keyboard_driver import KeyboardDriver
from core.keymap import KeyMap
from core.play_logger import PlayLogger
from core.player import Player
from core.profile import ensure_profiles, load_profiles, resolve_profile
from gui.main_window import MainWindow
from gui.theme import APP_QSS


def load_config(path="config.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def is_admin() -> bool:
    try:
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return False


def app_icon():
    """应用图标:任务栏与窗口图标。"""
    candidates = []
    if getattr(sys, "frozen", False):
        candidates.append(os.path.join(getattr(sys, "_MEIPASS", ""), "app.ico"))
        candidates.append(os.path.join(os.path.dirname(sys.executable), "app.ico"))
    candidates.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.ico"))
    for c in candidates:
        if c and os.path.exists(c):
            return QIcon(c)
    return None


def ensure_config() -> str:
    """切到数据目录并保证 config.yaml 可用。

    exe 运行:数据落在 exe 旁边;exe 旁没有 config.yaml 时,
    自动释放打包时内嵌的默认配置,保证单文件可运行。
    """
    if getattr(sys, "frozen", False):
        exe_dir = os.path.dirname(sys.executable)
        target = os.path.join(exe_dir, "config.yaml")
        if not os.path.exists(target):
            bundled = os.path.join(getattr(sys, "_MEIPASS", exe_dir), "config.yaml")
            shutil.copyfile(bundled, target)
        os.chdir(exe_dir)
        return target
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    return "config.yaml"


def resource_path(name: str) -> str:
    """打包后资源在 _MEIPASS 临时目录;开发时在项目根目录。"""
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, name)


def main():
    config_path = ensure_config()
    cfg = load_config(config_path)
    app_cfg = cfg.get("app", {})
    data_dir = app_cfg.get("data_dir", "data")
    player_cfg = cfg.get("player", {})
    db = ScoreDB(os.path.join(data_dir, app_cfg.get("db_file", "scores.db")))
    keymap = KeyMap(cfg["keymap"])
    # 游戏档位:profiles/ 目录优先,缺失时用 config.yaml 的 keymap 合成默认档位
    profiles = load_profiles(ensure_profiles("."), fallback_keymap=cfg.get("keymap"))
    profile = resolve_profile(profiles, app_cfg.get("active_profile"))
    # 修饰键与音键的间隔:配置缺失时回落到驱动默认值
    driver = KeyboardDriver(
        settle_ms=float(player_cfg.get("modifier_settle_ms", KeyboardDriver.DEFAULT_SETTLE_MS)),
        release_settle_ms=float(
            player_cfg.get("modifier_release_ms", KeyboardDriver.DEFAULT_RELEASE_SETTLE_MS)
        ),
    )
    player = Player(
        keymap,
        driver=driver,
        logger=PlayLogger(os.path.join(data_dir, "logs")),
        latency_compensation_ms=float(player_cfg.get("latency_compensation_ms", 0)),
    )
    # 事件演奏器(M4):三角洲档位走编译器 + EventPlayer,与默认 Player 共用同一驱动
    event_player = EventPlayer(
        driver=driver,
        latency_compensation_ms=float(player_cfg.get("latency_compensation_ms", 0)),
    )
    # 进程退出兜底:任何退出路径(atexit)都停止演奏并释放全部按键,防止键卡死
    atexit.register(player.shutdown)

    # Windows 任务栏分组图标:显式 AppUserModelID 让任务栏显示自定义图标而非 Python 默认图标
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("AutoMusicPlayer.App")
    except Exception:
        pass

    app = QApplication(sys.argv)
    # 显式声明高 DPI 缩放策略:125%/150% 等缩放下按逻辑像素平滑渲染,避免打包环境差异
    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    icon = app_icon()
    if icon:
        app.setWindowIcon(icon)
    app.setStyleSheet(APP_QSS)
    icon_path = resource_path("app.ico")
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))
    win = MainWindow(cfg, db, keymap, player,
                     config_path=config_path, profile=profile, profiles=profiles,
                     event_player=event_player)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()