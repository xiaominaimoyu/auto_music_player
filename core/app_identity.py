"""应用分发通道和 Windows 包身份辅助函数。

便携版继续使用 GitHub/Gitee 的签名 manifest；Microsoft Store 版由 MSIX
包身份管理更新。这里不依赖第三方 WinRT 包，未打包运行时也能安全返回
``False``，便于单元测试和现有源码启动路径继续工作。
"""

from __future__ import annotations

import ctypes
import os
import sys


_ERROR_INSUFFICIENT_BUFFER = 122
_ERROR_NO_PACKAGE = 15700


def current_package_full_name() -> str | None:
    """返回当前进程的 MSIX full name，未打包时返回 ``None``。

    ``GetCurrentPackageFullName`` 是 Windows 自带 API；避免引入 WinRT
    运行时，使 PyInstaller 基础包保持轻量。测试可以通过
    ``PACKAGE_FULL_NAME`` 环境变量注入确定值。
    """

    for key in ("PACKAGE_FULL_NAME", "APP_PACKAGE_FULL_NAME"):
        value = str(os.environ.get(key) or "").strip()
        if value:
            return value
    if sys.platform != "win32":
        return None
    try:
        function = ctypes.windll.kernel32.GetCurrentPackageFullName
        function.argtypes = [
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_wchar),
        ]
        function.restype = ctypes.c_long
        length = ctypes.c_uint32(0)
        result = function(ctypes.byref(length), None)
        if result == _ERROR_NO_PACKAGE:
            return None
        if result != _ERROR_INSUFFICIENT_BUFFER or length.value <= 1:
            return None
        buffer = ctypes.create_unicode_buffer(length.value)
        result = function(ctypes.byref(length), buffer)
        if result != 0:
            return None
        value = buffer.value.strip()
        return value or None
    except (AttributeError, OSError, TypeError, ValueError):
        return None


def package_family_name() -> str | None:
    """返回 MSIX package family name（PFN）。"""

    for key in ("PACKAGE_FAMILY_NAME", "APP_PACKAGE_FAMILY_NAME"):
        value = str(os.environ.get(key) or "").strip()
        if value:
            return value
    # Windows normally exposes PACKAGE_FAMILY_NAME for packaged processes.
    # 不对 full name 做脆弱的字符串猜测；缺少 PFN 时由调用方使用安全回退目录。
    return None


def is_store_package() -> bool:
    """判断当前进程是否为 Store/MSIX 通道。

    ``AUTOMUSIC_STORE_PACKAGE`` 仅用于开发/测试强制选择通道，不作为生产
    身份依据。显式设为 false 可让本地打包探针模拟便携路径。
    """

    override = str(os.environ.get("AUTOMUSIC_STORE_PACKAGE") or "").strip().lower()
    if override in {"1", "true", "yes", "on"}:
        return True
    if override in {"0", "false", "no", "off"}:
        return False
    return bool(current_package_full_name() or package_family_name())


def local_app_data_dir() -> str:
    """返回当前用户 LocalAppData 根目录。"""

    value = os.environ.get("LOCALAPPDATA")
    if value:
        return os.path.abspath(os.path.expandvars(os.path.expanduser(value)))
    return os.path.abspath(os.path.expanduser("~\\AppData\\Local"))


def stable_app_root() -> str:
    """便携安装版现有的稳定用户根目录。"""

    return os.path.join(local_app_data_dir(), "AutoMusicPlayer")


def store_local_state_dir() -> str:
    """返回 Store 版可写的 LocalState 路径。

    正式 MSIX 进程会提供 PFN，路径为标准的
    ``%LOCALAPPDATA%\\Packages\\<PFN>\\LocalState``。没有包身份时的
    回退只用于开发/测试，不会与真实 Store 包混用。
    """

    family = package_family_name()
    if family:
        return os.path.join(local_app_data_dir(), "Packages", family, "LocalState")
    return os.path.join(stable_app_root(), "store-local-state")


def channel_name() -> str:
    """返回面向日志/UI 的稳定通道名称。"""

    return "store" if is_store_package() else "portable"


__all__ = [
    "channel_name",
    "current_package_full_name",
    "is_store_package",
    "local_app_data_dir",
    "package_family_name",
    "stable_app_root",
    "store_local_state_dir",
]
