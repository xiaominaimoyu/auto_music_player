"""键盘/鼠标模拟驱动:Windows SendInput + scancode。

- KEYEVENTF_SCANCODE 方式发送物理扫描码,兼容 DirectInput / Raw Input 游戏
- INPUT 结构体与 winuser.h 完全一致(x64 下 sizeof=40),否则 SendInput 静默失败
- 发送失败时抛出异常,不静默吞掉

鼠标修饰键用于「音位键 + 修饰态」输入模型的游戏(如《三角洲行动》口琴):
按下顺序 = 修饰键 → 间隔 settle_ms → 音键;松开顺序 = 音键 → 间隔 release_settle_ms → 修饰键。
顺序颠倒会让最后一个音被错误转调,故固化在 press_combo/release_combo 内,业务层不得自行拼装。
"""

import ctypes
import sys
import time

INPUT_MOUSE = 0
INPUT_KEYBOARD = 1
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_SCANCODE = 0x0008
MAPVK_VK_TO_VSC = 0

MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
MOUSEEVENTF_MIDDLEDOWN = 0x0020
MOUSEEVENTF_MIDDLEUP = 0x0040

# 支持的鼠标修饰键;middle 能力默认提供,是否启用由 profile / 配置决定
MOUSE_BUTTONS = ("left", "middle", "right")

ULONG_PTR = ctypes.c_size_t

_user32 = None


def _load_user32():
    """延迟加载 user32:保证本模块在非 Windows 平台可安全导入(便于测试隔离)。"""
    global _user32
    if _user32 is None:
        if sys.platform != "win32":
            raise OSError("KeyboardDriver 仅支持 Windows(SendInput)")
        _user32 = ctypes.WinDLL("user32", use_last_error=True)
    return _user32


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.c_long),
        ("dy", ctypes.c_long),
        ("mouseData", ctypes.c_ulong),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", ULONG_PTR),
    ]


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", ctypes.c_ushort),
        ("wScan", ctypes.c_ushort),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", ULONG_PTR),
    ]


class _INPUTunion(ctypes.Union):
    _fields_ = [("mi", MOUSEINPUT), ("ki", KEYBDINPUT)]


class INPUT(ctypes.Structure):
    _anonymous_ = ("union",)
    _fields_ = [("type", ctypes.c_ulong), ("union", _INPUTunion)]


def _expected_input_size() -> int:
    return 40 if ctypes.sizeof(ctypes.c_void_p) == 8 else 28


_MOUSE_FLAGS = {
    ("left", False): MOUSEEVENTF_LEFTDOWN,
    ("left", True): MOUSEEVENTF_LEFTUP,
    ("right", False): MOUSEEVENTF_RIGHTDOWN,
    ("right", True): MOUSEEVENTF_RIGHTUP,
    ("middle", False): MOUSEEVENTF_MIDDLEDOWN,
    ("middle", True): MOUSEEVENTF_MIDDLEUP,
}


def _check_settle(name: str, value) -> float:
    """settle 类参数校验:必须是有限非负数。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} 必须是数字")
    v = float(value)
    if v != v or v in (float("inf"), float("-inf")):
        raise ValueError(f"{name} 必须是有限数字")
    if v < 0:
        raise ValueError(f"{name} 不能为负数")
    return v


class KeyboardDriver:
    """接口与 pynput 版保持一致:press/release 单键与和弦。"""

    CHORD_INTERVAL_MS = 5  # 和弦键之间的发送间隔,保证游戏按序收到

    # 修饰键按下 → 音键按下的默认间隔(毫秒)。
    # 取值依据:过短时游戏可能尚未注册修饰态,导致高/低音被弹成中音。
    # 30ms 为保守估计,实机可在 config.yaml 的 player.modifier_settle_ms 校准。
    DEFAULT_SETTLE_MS = 30.0
    # 音键松开 → 修饰键松开的默认间隔(毫秒)
    DEFAULT_RELEASE_SETTLE_MS = 20.0

    def __init__(self, settle_ms: float = DEFAULT_SETTLE_MS,
                 release_settle_ms: float = DEFAULT_RELEASE_SETTLE_MS):
        size = ctypes.sizeof(INPUT)
        if size != _expected_input_size():
            raise RuntimeError(
                f"INPUT 结构体大小异常: {size} != {_expected_input_size()},SendInput 将不可用"
            )
        self.settle_ms = _check_settle("settle_ms", settle_ms)
        self.release_settle_ms = _check_settle("release_settle_ms", release_settle_ms)

    def press_key(self, key: str):
        self._send(key, up=False)

    def release_key(self, key: str):
        self._send(key, up=True)

    def press_chord(self, keys: list[str]):
        for i, k in enumerate(keys):
            if i:
                time.sleep(self.CHORD_INTERVAL_MS / 1000.0)
            self._send(k, up=False)

    def release_chord(self, keys: list[str]):
        for i, k in enumerate(keys):
            if i:
                time.sleep(self.CHORD_INTERVAL_MS / 1000.0)
            self._send(k, up=True)

    # ---------- 鼠标修饰键 ----------
    def press_mouse(self, button: str):
        self._send_mouse(button, up=False)

    def release_mouse(self, button: str):
        self._send_mouse(button, up=True)

    # ---------- 组合(修饰键 + 音键) ----------
    def press_combo(self, key: str, mouse: str | None = None, settle_ms: float | None = None):
        """按下组合:修饰键 mouse 先按下,间隔 settle_ms 后再按音键 key。"""
        if mouse:
            self.press_mouse(mouse)
            time.sleep(self._settle(settle_ms) / 1000.0)
        self.press_key(key)

    def release_combo(self, key: str, mouse: str | None = None,
                      release_settle_ms: float | None = None):
        """松开组合:先松音键,间隔 release_settle_ms 后再松修饰键。

        顺序不可颠倒——修饰态是 latch-while-held,先松修饰键会让后续音被继续转调。
        """
        self.release_key(key)
        if mouse:
            time.sleep(self._release_settle(release_settle_ms) / 1000.0)
            self.release_mouse(mouse)

    # ---------- 兜底释放 ----------
    def panic_release(self, keys=(), mouse_buttons=()):
        """尽力释放指定键与鼠标键,单项失败不影响其余项,永不抛出异常。

        用于演奏中断 / 异常 / 进程退出的收尾,避免按键或修饰键残留。
        """
        for k in keys:
            if not k:
                continue
            try:
                self.release_key(k)
            except Exception:
                pass
        for mb in mouse_buttons:
            if not mb:
                continue
            try:
                self.release_mouse(mb)
            except Exception:
                pass

    # ---------- 参数回落 ----------
    def _settle(self, override) -> float:
        return _check_settle("settle_ms", override) if override is not None else self.settle_ms

    def _release_settle(self, override) -> float:
        if override is not None:
            return _check_settle("release_settle_ms", override)
        return self.release_settle_ms

    @classmethod
    def _scan_code(cls, key: str) -> int:
        u32 = _load_user32()
        ch = key.upper()
        if "A" <= ch <= "Z" or "0" <= ch <= "9":
            vk = ord(ch)
        else:
            vk = u32.VkKeyScanW(ord(ch[0])) & 0xFF
        return u32.MapVirtualKeyW(vk, MAPVK_VK_TO_VSC)

    @classmethod
    def _send(cls, key: str, up: bool):
        u32 = _load_user32()
        scan = cls._scan_code(key)
        flags = KEYEVENTF_SCANCODE | (KEYEVENTF_KEYUP if up else 0)
        inp = INPUT(type=INPUT_KEYBOARD)
        inp.ki = KEYBDINPUT(0, scan, flags, 0, 0)
        sent = u32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))
        if sent != 1:
            raise OSError(f"SendInput 发送失败(key={key}, winerror={ctypes.GetLastError()})")

    @classmethod
    def _send_mouse(cls, button: str, up: bool):
        u32 = _load_user32()
        flag = _MOUSE_FLAGS.get((button, up))
        if flag is None:
            raise ValueError(
                f"未知鼠标键: {button!r}(应为 {' / '.join(repr(b) for b in MOUSE_BUTTONS)})"
            )
        inp = INPUT(type=INPUT_MOUSE)
        inp.mi = MOUSEINPUT(0, 0, 0, flag, 0, 0)
        sent = u32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))
        if sent != 1:
            raise OSError(
                f"SendInput 鼠标发送失败(button={button}, winerror={ctypes.GetLastError()})"
            )