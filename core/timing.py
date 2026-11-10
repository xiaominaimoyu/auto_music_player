"""播放时序：节奏忠实的音槽、可打断的高精度等待、Windows 计时器分辨率。

``gap_ms`` 默认是音符时值内部的松开间隙，不起音间隔仍等于谱面时值。
只有音符短于这个间隙时，下一音才会被撑开。``legacy_timing`` 恢复旧行为
（间隙加在时值后面）。
"""

from __future__ import annotations

import ctypes
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass

# 松开间隙再短也不把按键削到这个长度以下；削不动就推迟下一音。
MIN_HOLD_MS = 1.0
# 高精度睡眠之后留下的空转尾巴。定时器即使准到 1ms，也不会睡过截止点。
_SPIN_TAIL_S = 0.002
# Windows 默认时间片约 15.625ms。Event.wait 最多晚一个时间片，
# 所以粗等待必须在截止点前提前不止 2ms，否则会直接睡过拍点。
_WIN_COARSE_MARGIN_S = 0.020
# 长等待分段，保证 stop() 在远小于 100ms 内被看到。
_STOP_POLL_S = 0.025
_HRT_FLAG = 0x00000002  # CREATE_WAITABLE_TIMER_HIGH_RESOLUTION
_TIMER_ACCESS = 0x001F0003  # TIMER_ALL_ACCESS

_tls = threading.local()
_hrt_lock = threading.Lock()
_hrt_supported = None


@dataclass(frozen=True)
class TimingConfig:
    """从 config.yaml 的 player 段读出的调度开关。"""

    legacy_timing: bool = False
    modifier_prepress: bool = True
    additive_gap: bool = False
    precise_wait: bool = True
    batch_chords: bool = True


def timing_config(player_cfg: dict | None) -> TimingConfig:
    """解析播放配置。``legacy_timing`` 打开时整段回到修正前的调度。"""

    cfg = player_cfg or {}
    legacy = bool(cfg.get("legacy_timing", False))
    if legacy:
        return TimingConfig(
            legacy_timing=True,
            modifier_prepress=False,
            additive_gap=True,
            precise_wait=False,
            batch_chords=False,
        )
    return TimingConfig(
        legacy_timing=False,
        modifier_prepress=bool(cfg.get("modifier_prepress", True)),
        additive_gap=bool(cfg.get("additive_gap", False)),
        precise_wait=bool(cfg.get("precise_wait", True)),
        batch_chords=bool(cfg.get("batch_chords", True)),
    )


def slot_hold_and_advance(
    dur_ms: float,
    hold_ratio: float,
    gap_ms: float,
    *,
    additive: bool,
    followed_by_attack: bool,
) -> tuple[float, float]:
    """返回 (按住毫秒, 时间轴推进毫秒)。

    非 additive 时，推进量等于谱面时值；若后面紧跟发音，按住被削到
    至少留出 ``gap_ms`` 的松开间隙。时值比间隙还短时无法削，推进量变为
    ``按住 + gap``，保证最小松开间隙。
    """

    dur = max(0.0, float(dur_ms))
    gap = max(0.0, float(gap_ms))
    ratio = float(hold_ratio)
    if ratio <= 0:
        advance = dur + gap if additive else dur
        return 0.0, advance
    desired = dur * ratio
    if additive:
        return desired, dur + gap
    if not followed_by_attack or gap <= 0:
        return desired, dur
    room = dur - gap
    if room >= desired:
        return desired, dur
    if room >= MIN_HOLD_MS:
        return room, dur
    return desired, max(dur, desired + gap)


class _WinHighResTimer:
    """不依赖 timeBeginPeriod 的高精度可等待定时器（Windows 10 1803+）。"""

    def __init__(self, kernel, handle):
        self._kernel = kernel
        self._handle = handle

    @classmethod
    def try_open(cls):
        if sys.platform != "win32":
            return None
        try:
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            create = kernel.CreateWaitableTimerExW
            create.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_uint, ctypes.c_uint]
            create.restype = ctypes.c_void_p
            kernel.SetWaitableTimer.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_longlong),
                ctypes.c_long,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_int,
            ]
            kernel.SetWaitableTimer.restype = ctypes.c_int
            kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint]
            kernel.WaitForSingleObject.restype = ctypes.c_uint
            handle = create(None, None, _HRT_FLAG, _TIMER_ACCESS)
        except (AttributeError, OSError):
            return None
        if not handle:
            return None
        return cls(kernel, handle)

    def wait(self, seconds: float) -> bool:
        """睡大约 ``seconds``。失败返回 False，调用方改走粗等待。"""

        due = ctypes.c_longlong(int(-max(seconds, 0.0001) * 10_000_000))
        if not self._kernel.SetWaitableTimer(self._handle, ctypes.byref(due), 0, None, None, False):
            return False
        # WAIT_OBJECT_0 == 0。定时器是自动复位的。
        return self._kernel.WaitForSingleObject(self._handle, 0xFFFFFFFF) == 0

    def close(self):
        close = getattr(self._kernel, "CloseHandle", None)
        if close is None or not self._handle:
            return
        close.argtypes = [ctypes.c_void_p]
        close.restype = ctypes.c_int
        close(self._handle)
        self._handle = None


def _high_res_timer():
    """当前线程的高精度定时器。尚未确认可用时返回 None。"""

    if _hrt_supported is not True:
        return None
    cached = getattr(_tls, "timer", None)
    if cached is False:
        return None
    if cached is not None:
        return cached
    timer = _WinHighResTimer.try_open()
    _tls.timer = timer if timer is not None else False
    return timer


def _disable_high_res_timer():
    global _hrt_supported
    _hrt_supported = False
    _tls.timer = False


def _maybe_calibrate(deadline: float):
    """有余量时测一次 20ms。误差超过 4ms 就当它仍是 15.6ms 时间片，不用它靠近拍点。"""

    global _hrt_supported
    if sys.platform != "win32" or _hrt_supported is not None:
        return
    if deadline - time.perf_counter() < 0.05:
        return
    with _hrt_lock:
        if _hrt_supported is not None:
            return
        timer = _WinHighResTimer.try_open()
        if timer is None:
            _hrt_supported = False
            return
        started = time.perf_counter()
        ok = timer.wait(0.020)
        elapsed = time.perf_counter() - started
        if ok and abs(elapsed - 0.020) <= 0.004:
            _hrt_supported = True
            _tls.timer = timer
            return
        timer.close()
        _hrt_supported = False


def _spin_until(stop_event, deadline: float) -> bool:
    while time.perf_counter() < deadline:
        if stop_event.is_set():
            return True
    return stop_event.is_set()


def wait_until(stop_event, deadline: float, *, precise: bool = True) -> bool:
    """等到 ``deadline``（perf_counter 时刻）。返回 True 表示 stop 已置位。

    精确模式不依赖 ``timeBeginPeriod``。Windows 上优先用
    ``CREATE_WAITABLE_TIMER_HIGH_RESOLUTION`` 睡到截止点前约 2ms，再空转补齐。
    没有该定时器时，粗等待提前一个完整时间片（20ms）醒来再空转，避免
    默认 15.6ms 时间片把 ``Event.wait`` 直接睡过拍点。
    长等待分段轮询，stop 仍会在几十毫秒内返回。
    """

    if not precise:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            return stop_event.is_set()
        return bool(stop_event.wait(remaining))

    _maybe_calibrate(deadline)
    while True:
        if stop_event.is_set():
            return True
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            return False
        if remaining <= _SPIN_TAIL_S:
            return _spin_until(stop_event, deadline)

        timer = _high_res_timer() if sys.platform == "win32" else None
        if timer is not None:
            sleep_for = min(remaining - _SPIN_TAIL_S, _STOP_POLL_S)
            started = time.perf_counter()
            ok = timer.wait(sleep_for)
            elapsed = time.perf_counter() - started
            # 定时器若立刻返回，说明不可用，改走带时间片余量的粗等待。
            if not ok or (sleep_for >= 0.005 and elapsed < sleep_for * 0.25):
                _disable_high_res_timer()
            continue

        # 没证明定时器是高精度时，Windows 上不能把 Event.wait 排进最后 20ms。
        margin = _WIN_COARSE_MARGIN_S if sys.platform == "win32" else _SPIN_TAIL_S
        if remaining <= margin:
            return _spin_until(stop_event, deadline)
        coarse = min(remaining - margin, _STOP_POLL_S)
        if coarse > 0 and stop_event.wait(coarse):
            return True


@contextmanager
def high_resolution_period(ms: int = 1):
    """播放期间把 Windows 多媒体定时器分辨率调到 1ms。其他系统为空操作。

    ``timeBeginPeriod`` 是进程内引用计数，必须与 ``timeEndPeriod`` 成对调用。
    """

    if sys.platform != "win32":
        yield
        return
    winmm = None
    period = int(ms)
    try:
        winmm = ctypes.WinDLL("winmm")
        winmm.timeBeginPeriod.argtypes = [ctypes.c_uint]
        winmm.timeBeginPeriod.restype = ctypes.c_uint
        winmm.timeEndPeriod.argtypes = [ctypes.c_uint]
        winmm.timeEndPeriod.restype = ctypes.c_uint
        if winmm.timeBeginPeriod(period) != 0:
            winmm = None
    except Exception:
        winmm = None
    try:
        yield
    finally:
        if winmm is not None:
            try:
                winmm.timeEndPeriod(period)
            except Exception:
                pass
