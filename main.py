"""入口:加载配置,组装数据库/识别器/按键映射/演奏器,启动 GUI(暗色琥珀金主题)。"""

import atexit
import ctypes
import json
import os
import shutil
import sys
import time

import yaml
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QApplication

from core.database import ScoreDB
from core.app_identity import is_store_package, stable_app_root, store_local_state_dir
from core.data_migration import migrate_store_data
from core.event_logger import configure_event_logger
from core.event_player import EventPlayer
from core.humanize import HumanizeParams
from core.keyboard_driver import KeyboardDriver
from core.keymap import KeyMap
from core.play_logger import PlayLogger
from core.player import Player
from core.profile import ensure_profiles, load_profiles, resolve_profile
from gui.disclaimer import confirm_risk_disclaimer
from gui.main_window import MainWindow
from gui.theme import APP_QSS


def load_config(path="config.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        user = yaml.safe_load(f) or {}
    bundled_path = os.path.join(
        getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__))),
        "config.yaml",
    )
    defaults = {}
    if os.path.isfile(bundled_path) and os.path.abspath(bundled_path) != os.path.abspath(path):
        try:
            with open(bundled_path, encoding="utf-8") as f:
                defaults = yaml.safe_load(f) or {}
        except (OSError, yaml.YAMLError):
            defaults = {}

    def merge(base, override):
        if not isinstance(base, dict) or not isinstance(override, dict):
            return override if override is not None else base
        result = dict(base)
        for key, value in override.items():
            result[key] = merge(result[key], value) if key in result else value
        return result

    return merge(defaults, user)


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
    candidates.append(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.ico")
    )
    for c in candidates:
        if c and os.path.exists(c):
            return QIcon(c)
    return None


def ensure_config() -> str:
    """保证 config.yaml 可用，并避免 Store 包目录写入。

    便携 exe 延续原有“exe 旁配置”兼容行为。Store/MSIX 包安装目录只读，
    因此将可写配置放入 LocalState；包内默认配置仍通过 ``_MEIPASS`` 读取。
    """
    if getattr(sys, "frozen", False):
        if is_store_package():
            state_dir = store_local_state_dir()
            os.makedirs(state_dir, exist_ok=True)
            target = os.path.join(state_dir, "config.yaml")
            if not os.path.exists(target):
                bundled = os.path.join(getattr(sys, "_MEIPASS", state_dir), "config.yaml")
                if not os.path.isfile(bundled):
                    raise FileNotFoundError(f"缺少 MSIX 内置默认配置: {bundled}")
                shutil.copyfile(bundled, target)
            # 只把相对路径解析到用户数据位置；绝不把工作目录设到 WindowsApps。
            os.chdir(state_dir)
            return target
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


def ensure_omr_components(data_dir: str) -> str:
    """补充/升级应用管理的组件，不覆盖用户自行安装的同名目录。"""

    target = os.path.join(os.path.abspath(os.fspath(data_dir)), "omr")
    bundled = resource_path("omr")
    if os.path.isdir(bundled):
        os.makedirs(target, exist_ok=True)
        for name in os.listdir(bundled):
            source = os.path.join(bundled, name)
            destination = os.path.join(target, name)
            if name.lower() == "readme.md":
                continue
            if not os.path.exists(destination):
                if os.path.isdir(source):
                    shutil.copytree(source, destination)
                else:
                    shutil.copy2(source, destination)
                continue
            if not os.path.isdir(source) or not os.path.isdir(destination):
                continue
            source_manifest = _component_manifest(source)
            destination_manifest = _component_manifest(destination)
            if not source_manifest or not source_manifest.get("managed_by_app"):
                continue
            if not destination_manifest or not destination_manifest.get("managed_by_app"):
                # A user may have installed a component at this path manually;
                # leave it untouched and let the component detector report it.
                continue
            source_version = json.dumps(source_manifest.get("component"), sort_keys=True)
            destination_version = json.dumps(destination_manifest.get("component"), sort_keys=True)
            if source_version == destination_version:
                continue
            staging = f"{destination}.staging-{os.getpid()}"
            if os.path.exists(staging):
                shutil.rmtree(staging, ignore_errors=True)
            shutil.copytree(source, staging)
            backup = f"{destination}.previous-{int(time.time())}"
            shutil.move(destination, backup)
            shutil.move(staging, destination)
    return target


def _component_manifest(path: str):
    manifest_path = os.path.join(path, "component.json")
    try:
        with open(manifest_path, encoding="utf-8-sig") as stream:
            value = json.load(stream)
    except (OSError, ValueError, TypeError):
        return None
    return value if isinstance(value, dict) else None


def resolve_data_dir(config_path: str, configured_data_dir: str) -> str:
    """向后兼容地解析便携/安装双模式数据根。

    未显式传入 ``data_mode`` 时保留旧行为，便于现有测试和开发环境使用。
    frozen + auto 模式会优先保留已有 exe 旁 data，新的安装版则使用
    ``%LOCALAPPDATA%\\AutoMusicPlayer``，更新程序不会覆盖用户数据。
    """

    return _resolve_data_dir(config_path, configured_data_dir)


def _resolve_data_dir(config_path: str, configured_data_dir: str, data_mode="auto") -> str:
    """内部实现；``resolve_data_dir`` 保留原有两参数 API。"""

    config_path = os.path.abspath(config_path)
    path = os.fspath(configured_data_dir)
    mode = str(data_mode or "auto").strip().lower()
    if mode not in {"auto", "portable", "installed"}:
        raise ValueError("app.data_mode 必须是 auto、portable 或 installed")
    if is_store_package():
        # Store 通道不允许通过配置把数据写回只读包目录或任意绝对路径。
        # 保留相对 data 子目录，确保数据库/来源文件都在 LocalState 内。
        relative = "data" if os.path.isabs(path) or not path.strip() else path
        store_root = os.path.abspath(store_local_state_dir())
        candidate = os.path.abspath(os.path.join(store_root, relative))
        if os.path.commonpath([store_root, candidate]) != store_root:
            raise ValueError("Store 版 app.data_dir 必须位于 LocalState 内")
        return candidate
    if os.path.isabs(path):
        return os.path.abspath(path)
    config_dir = os.path.dirname(config_path)
    if mode == "portable" or not getattr(sys, "frozen", False):
        return os.path.abspath(os.path.join(config_dir, path))
    portable_flag = os.path.join(config_dir, "portable.flag")
    legacy_data = os.path.join(config_dir, path)
    if mode == "auto" and (os.path.exists(portable_flag) or os.path.exists(legacy_data)):
        return os.path.abspath(legacy_data)
    return os.path.abspath(os.path.join(stable_app_root(), path))


def main():
    config_path = ensure_config()
    config_path = os.path.abspath(config_path)
    cfg = load_config(config_path)
    app_cfg = cfg.get("app", {})
    configured_data_dir = app_cfg.get("data_dir", "data")
    data_dir = _resolve_data_dir(
        config_path,
        configured_data_dir,
        app_cfg.get("data_mode", "auto"),
    )
    if is_store_package():
        migration = migrate_store_data(data_dir)
        if migration.status == "error":
            print(f"Store 数据迁移失败，将保留现有数据并继续启动: {migration.error}", file=sys.stderr)
    ensure_omr_components(data_dir)
    player_cfg = cfg.get("player", {})
    db = ScoreDB(os.path.join(data_dir, app_cfg.get("db_file", "scores.db")))
    keymap = KeyMap(cfg["keymap"])
    # 游戏档位:profiles/ 目录优先,缺失时用 config.yaml 的 keymap 合成默认档位
    profile_base = data_dir if getattr(sys, "frozen", False) else "."
    legacy_profiles = os.path.join(os.path.dirname(config_path), "profiles")
    if (
        getattr(sys, "frozen", False)
        and app_cfg.get("data_mode", "auto") != "installed"
        and not is_store_package()
        and os.path.isdir(legacy_profiles)
    ):
        # 兼容旧便携版把用户档位放在 exe/profiles 的布局。
        profile_base = os.path.dirname(config_path)
    profiles = load_profiles(
        ensure_profiles(profile_base),
        fallback_keymap=cfg.get("keymap"),
    )
    profile = resolve_profile(profiles, app_cfg.get("active_profile"))
    # 修饰键与音键的间隔:配置缺失时回落到驱动默认值
    driver = KeyboardDriver(
        settle_ms=float(
            player_cfg.get("modifier_settle_ms", KeyboardDriver.DEFAULT_SETTLE_MS)
        ),
        release_settle_ms=float(
            player_cfg.get(
                "modifier_release_ms", KeyboardDriver.DEFAULT_RELEASE_SETTLE_MS
            )
        ),
    )
    # 真人化节奏参数:从 config.yaml 读取,默认启用
    humanize_cfg = player_cfg.get("humanize") or {}
    humanize_enabled = bool(humanize_cfg.get("enabled", True))
    humanize_params = None
    if humanize_enabled:
        humanize_params = HumanizeParams(
            jitter_ms=float(humanize_cfg.get("jitter_ms", 12.0)),
            breath_ms=float(humanize_cfg.get("breath_ms", 25.0)),
            jitter_correlation=float(humanize_cfg.get("jitter_correlation", 0.65)),
            # 保持默认值,不在 config 暴露过多旋钮
        )
    # 两条播放路径共用同一个事件日志目录/会话格式，记录页才能同时看到
    # 默认 21 键与三角洲口琴演奏；PlayLogger 继续保留本地统计兼容文件。
    event_log_dir = os.path.join(data_dir, "play_logs")
    event_logger = configure_event_logger(event_log_dir)
    player = Player(
        keymap,
        driver=driver,
        logger=PlayLogger(os.path.join(data_dir, "logs")),
        latency_compensation_ms=float(player_cfg.get("latency_compensation_ms", 0)),
        humanize=humanize_params,
        event_logger=event_logger,
    )
    # 事件演奏器(M4):三角洲档位走编译器 + EventPlayer,与默认 Player 共用同一驱动
    event_player = EventPlayer(
        driver=driver,
        latency_compensation_ms=float(player_cfg.get("latency_compensation_ms", 0)),
        logger=event_logger,
    )
    # 进程退出兜底:任何退出路径(atexit)都停止演奏并释放全部按键,防止键卡死
    atexit.register(player.shutdown)
    atexit.register(event_player.shutdown)
    atexit.register(db.close)

    # MSIX 由包身份管理 AUMID；便携版保留显式任务栏分组标识。
    if not is_store_package():
        try:
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                "AutoMusicPlayer.App"
            )
        except Exception:
            pass

    app = QApplication(sys.argv)
    # 显式声明高 DPI 缩放策略:125%/150% 等缩放下按逻辑像素平滑渲染,避免打包环境差异
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    icon = app_icon()
    if icon:
        app.setWindowIcon(icon)
    app.setStyleSheet(APP_QSS)
    icon_path = resource_path("app.ico")
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))

    # 风险警示关卡:启动后、主界面前的强制模态确认;异常=安全终止而非跳过
    try:
        confirmed = confirm_risk_disclaimer()
    except Exception as e:
        print(f"风险警示界面异常,程序终止: {e}", file=sys.stderr)
        sys.exit(1)
    if not confirmed:
        sys.exit(0)  # atexit 兜底自动释放虚拟按键后正常退出

    win = MainWindow(
        cfg,
        db,
        keymap,
        player,
        config_path=config_path,
        profile=profile,
        profiles=profiles,
        event_player=event_player,
        event_log_dir=event_log_dir,
        export_dir=os.path.join(data_dir, "exports"),
    )
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
