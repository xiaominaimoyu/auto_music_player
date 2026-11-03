"""播放时序：节奏忠实的音槽、可打断的高精度等待、Windows 计时器分辨率。

``gap_ms`` 默认是音符时值内部的松开间隙，不起音间隔仍等于谱面时值。
只有音符短于这个间隙时，下一音才会被撑开。``legacy_timing`` 恢复旧行为
（间隙加在时值后面）。
"""

from __future__ import annotations

import ctypes
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass

# 松开间隙再短也不把按键削到这个长度以下；削不动就推迟下一音。
MIN_HOLD_MS = 1.0
# 粗等待在截止点前提前醒来，剩下的时间用 perf_counter 空转补齐。
_SPIN_S = 0.002


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


def wait_until(stop_event, deadline: float, *, precise: bool = True) -> bool:
    """等到 ``deadline``（perf_counter 时刻）。返回 True 表示 stop 已置位。

    精确模式先用 Event.wait 睡到截止点前约 2ms，再用空转补齐，避免
    Windows 默认定时器 15.6ms 的随机迟到。stop 在粗等待和空转中都会立刻返回。
    """

    while True:
        if stop_event.is_set():
            return True
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            return False
        if not precise or remaining <= _SPIN_S:
            if not precise:
                return bool(stop_event.wait(remaining))
            while time.perf_counter() < deadline:
                if stop_event.is_set():
                    return True
            return False
        coarse = remaining - _SPIN_S
        if coarse >= 0.001 and stop_event.wait(coarse):
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
