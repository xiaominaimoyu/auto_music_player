"""目标窗口焦点检测:演奏中周期检查前台窗口,丢失焦点即自动暂停。

- 目标窗口两种指定方式:
  1. 开始演奏瞬间快照当前前台窗口(句柄精确比对,默认)
  2. 配置 player.focus_check.target_window_title 指定标题子串(大小写不敏感,优先级更高)
- Win32 API 延迟加载;非 Windows 平台 is_available() 为 False,调用方自动降级(不检测)
- 快照失败/无法判定前台窗口时按"不打断演奏"处理(fail-open),避免误伤正常演奏
"""

import ctypes
import sys

_user32 = None


def _load_user32():
    """延迟加载 user32 并绑定 64 位安全原型;非 Windows 显式报错。"""
    global _user32
    if _user32 is None:
        if sys.platform != "win32":
            raise OSError("焦点检测仅支持 Windows")
        u32 = ctypes.WinDLL("user32", use_last_error=True)
        u32.GetForegroundWindow.restype = ctypes.c_void_p
        u32.GetWindowTextW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_int]
        u32.GetWindowTextW.restype = ctypes.c_int
        _user32 = u32
    return _user32


def _get_fg_hwnd() -> int | None:
    """当前前台窗口句柄;无前台窗口或平台不支持时返回 None。"""
    try:
        u32 = _load_user32()
    except OSError:
        return None
    hwnd = u32.GetForegroundWindow()
    return int(hwnd) if hwnd else None


def _get_window_title(hwnd: int) -> str:
    u32 = _load_user32()
    buf = ctypes.create_unicode_buffer(512)
    u32.GetWindowTextW(hwnd, buf, 512)
    return buf.value


class FocusLockPolicy:
    """焦点目标锁定策略(纯逻辑,无 Win32 依赖,可确定性单测)。

    自动模式(无标题覆盖):
    - 演奏开始后不预设目标;前台出现首个"非本应用"窗口(即用户切到的游戏)才锁定
    - 锁定前用户停留在本应用内绝不误停;锁定后离开目标窗口即判定失焦
    手动模式(title_override 非空):始终按标题子串匹配,无需锁定。
    无法判定前台窗口时一律不打断(fail-open)。
    """

    def __init__(self, own_hwnd: int | None = None, title_override: str = ""):
        self.own_hwnd = own_hwnd
        self.title_override = title_override
        self.locked = bool(title_override)
        self.target: dict | None = None

    def evaluate(self, current: dict | None) -> bool:
        """传入当前前台窗口快照 {"hwnd", "title"}(无法判定时为 None),返回是否应暂停。"""
        if self.title_override:
            if current is None:
                return False
            return self.title_override.lower() not in (current.get("title") or "").lower()
        if not self.locked:
            if current is None:
                return False
            if self.own_hwnd is not None and current.get("hwnd") == self.own_hwnd:
                return False   # 用户仍在本应用内,等待其切到游戏
            self.target = current
            self.locked = True
            return False       # 刚锁定,不打断
        if self.target is None:
            return False
        if current is None:
            return True
        if current.get("hwnd") == self.target.get("hwnd"):
            return False
        # 句柄变了但标题一致(游戏全屏切换/重创窗口) → 视为同一目标,更新句柄
        if (current.get("title") or "") and current.get("title") == self.target.get("title"):
            self.target = current
            return False
        return True


class ForegroundWatcher:
    """前台窗口快照与焦点比对;轮询节奏由调用方(GUI QTimer)驱动。"""

    def __init__(self, poll_interval: float = 0.4):
        self.poll_interval = poll_interval

    @staticmethod
    def is_available() -> bool:
        return sys.platform == "win32"

    def capture_current(self) -> dict | None:
        """快照当前前台窗口:{"hwnd": int, "title": str};快照失败返回 None。"""
        hwnd = _get_fg_hwnd()
        if not hwnd:
            return None
        try:
            title = _get_window_title(hwnd)
        except Exception:
            return None
        return {"hwnd": hwnd, "title": title}

    def is_target_foreground(self, target: dict | None, title_override: str = "") -> bool:
        """目标窗口是否仍在前台。

        - title_override 非空:对当前前台窗口标题做子串匹配(优先级最高)
        - 否则按快照句柄精确比对
        - 无有效目标/无法判定前台窗口:返回 True(不打断演奏)
        """
        try:
            hwnd = _get_fg_hwnd()
            if hwnd is None:
                return True
            if title_override:
                return title_override.lower() in _get_window_title(hwnd).lower()
            if not target:
                return True
            return hwnd == target.get("hwnd")
        except Exception:
            return True
