"""Windows reliability checks for playback input.

The checks are intentionally small and side-effect-light: they report whether
the process is elevated and whether the SendInput ctypes structure matches the
Windows ABI. Real playback and panic release remain in KeyboardDriver/Player.
"""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys
from dataclasses import dataclass

from core.keyboard_driver import INPUT, _expected_input_size


@dataclass(frozen=True)
class WindowsReliabilityReport:
    is_windows: bool
    is_admin: bool | None
    input_size: int
    expected_input_size: int
    sendinput_struct_ok: bool
    status: str
    messages: tuple[str, ...]

    @property
    def summary(self) -> str:
        if not self.is_windows:
            return "非 Windows · 真实输入不可用"
        if self.status == "error":
            return "Windows 输入结构异常"
        if self.is_admin:
            return "管理员权限 · 输入链路就绪"
        return "普通权限 · 必要时管理员重启"


@dataclass(frozen=True)
class ElevationAssessment:
    """Read-only UIPI compatibility result for one target window."""

    app_elevated: bool | None
    target_elevated: bool | None
    status: str
    blocked: bool
    message: str
    process_id: int | None = None


def _process_elevated() -> bool | None:
    if sys.platform != "win32":
        return None
    try:
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return None


def current_process_elevated() -> bool | None:
    """Return this process' elevation state, or ``None`` when unavailable."""

    return _process_elevated()


def window_process_id(hwnd: int) -> int | None:
    """Resolve an HWND to a process id without opening or changing the target."""

    if sys.platform != "win32" or not hwnd:
        return None
    try:
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        user32.GetWindowThreadProcessId.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_ulong),
        ]
        user32.GetWindowThreadProcessId.restype = ctypes.c_ulong
        pid = ctypes.c_ulong(0)
        user32.GetWindowThreadProcessId(ctypes.c_void_p(int(hwnd)), ctypes.byref(pid))
        return int(pid.value) if pid.value else None
    except Exception:
        return None


def process_elevated(process_id: int) -> bool | None:
    """Read a process TokenElevation flag; failures are reported as unknown."""

    if sys.platform != "win32" or not process_id:
        return None
    process_handle = None
    token_handle = ctypes.c_void_p()
    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
        kernel32.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
        kernel32.OpenProcess.restype = ctypes.c_void_p
        kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
        kernel32.CloseHandle.restype = ctypes.c_int
        advapi32.OpenProcessToken.argtypes = [
            ctypes.c_void_p,
            ctypes.c_ulong,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        advapi32.OpenProcessToken.restype = ctypes.c_int
        advapi32.GetTokenInformation.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_ulong,
            ctypes.POINTER(ctypes.c_ulong),
        ]
        advapi32.GetTokenInformation.restype = ctypes.c_int

        # PROCESS_QUERY_LIMITED_INFORMATION and TOKEN_QUERY.
        process_handle = kernel32.OpenProcess(0x1000, 0, int(process_id))
        if not process_handle:
            return None
        if not advapi32.OpenProcessToken(process_handle, 0x0008, ctypes.byref(token_handle)):
            return None
        elevation = ctypes.c_ulong(0)
        returned = ctypes.c_ulong(0)
        # TOKEN_INFORMATION_CLASS.TokenElevation == 20.
        if not advapi32.GetTokenInformation(
            token_handle,
            20,
            ctypes.byref(elevation),
            ctypes.sizeof(elevation),
            ctypes.byref(returned),
        ):
            return None
        return bool(elevation.value)
    except Exception:
        return None
    finally:
        try:
            if token_handle.value:
                kernel32.CloseHandle(token_handle)
        except Exception:
            pass
        try:
            if process_handle:
                kernel32.CloseHandle(process_handle)
        except Exception:
            pass


def assess_elevation(
    app_elevated: bool | None,
    target_elevated: bool | None,
    *,
    process_id: int | None = None,
) -> ElevationAssessment:
    """Apply the Windows UIPI rule without guessing unknown privilege states."""

    if app_elevated is None or target_elevated is None:
        return ElevationAssessment(
            app_elevated,
            target_elevated,
            "unknown",
            False,
            "无法确认本程序或目标窗口权限；输入兼容性未知。",
            process_id,
        )
    if not app_elevated and target_elevated:
        return ElevationAssessment(
            app_elevated,
            target_elevated,
            "blocked",
            True,
            "目标窗口以管理员权限运行，普通权限程序会被 Windows UIPI 阻止输入。",
            process_id,
        )
    return ElevationAssessment(
        app_elevated,
        target_elevated,
        "compatible",
        False,
        "本程序与目标窗口权限级别兼容。",
        process_id,
    )


def inspect_target_elevation(hwnd: int) -> ElevationAssessment:
    """Compose the read-only HWND/PID/token checks for playback preflight."""

    pid = window_process_id(hwnd)
    app_state = current_process_elevated()
    target_state = process_elevated(pid) if pid is not None else None
    return assess_elevation(app_state, target_state, process_id=pid)


def check_windows_reliability(
    *,
    platform: str | None = None,
    is_admin: bool | None = None,
    input_size: int | None = None,
    expected_input_size: int | None = None,
) -> WindowsReliabilityReport:
    platform = platform or sys.platform
    is_windows = platform == "win32"
    if is_admin is None:
        is_admin = _process_elevated() if is_windows else None
    input_size = int(input_size if input_size is not None else ctypes.sizeof(INPUT))
    expected_input_size = int(
        expected_input_size
        if expected_input_size is not None
        else _expected_input_size()
    )
    struct_ok = input_size == expected_input_size

    messages = []
    status = "ok"
    if not is_windows:
        status = "warning"
        messages.append("当前平台不是 Windows,真实键鼠模拟会降级为不可用。")
    if not struct_ok:
        status = "error"
        messages.append(
            f"SendInput INPUT 结构体大小 {input_size} != {expected_input_size},真实输入不可用。"
        )
    if is_windows and is_admin is False:
        if status != "error":
            status = "warning"
        messages.append("当前为普通权限;若游戏以管理员运行,Windows 会阻止本程序注入输入。")
    if is_windows and is_admin is None:
        if status != "error":
            status = "warning"
        messages.append("无法确认管理员权限;启动前建议用一次真实游戏窗口自检。")
    if not messages:
        messages.append("Windows 输入结构与权限状态正常。")

    return WindowsReliabilityReport(
        is_windows=is_windows,
        is_admin=is_admin,
        input_size=input_size,
        expected_input_size=expected_input_size,
        sendinput_struct_ok=struct_ok,
        status=status,
        messages=tuple(messages),
    )


def restart_as_admin(extra_args: list[str] | None = None) -> bool:
    """Ask Windows ShellExecute to relaunch this process elevated."""

    if sys.platform != "win32":
        return False
    args = list(sys.argv[1:] if extra_args is None else extra_args)
    if not getattr(sys, "frozen", False):
        # Source launches must pass the script path back to python.exe.  A
        # frozen executable is already its own entry point and needs only the
        # original trailing arguments.
        args.insert(0, os.path.abspath(sys.argv[0]))
    params = subprocess.list2cmdline(args)
    try:
        rc = ctypes.windll.shell32.ShellExecuteW(
            None,
            "runas",
            sys.executable,
            params,
            os.getcwd(),
            1,
        )
    except Exception:
        return False
    return int(rc) > 32
