# -*- mode: python ; coding: utf-8 -*-
# PyInstaller 打包配置:体积优化版
#   1. 排除用不到的 PyQt6 子模块与第三方库
#   2. 过滤 Qt 的无用 DLL(opengl32sw/d3dcompiler/Network/Pdf/Svg/Qml/Multimedia 等)
#      与插件(仅保留 platforms + styles)
# 打包: python -m PyInstaller --noconfirm --clean AutoMusicPlayer.spec

import fnmatch
import os

EXCLUDES = [
    # PyQt6 未使用的子模块
    "PyQt6.QtNetwork", "PyQt6.QtQml", "PyQt6.QtQuick", "PyQt6.QtQuickWidgets",
    "PyQt6.QtSvg", "PyQt6.QtSvgWidgets", "PyQt6.QtPdf", "PyQt6.QtPdfWidgets",
    "PyQt6.QtMultimedia", "PyQt6.QtMultimediaWidgets", "PyQt6.QtWebEngineCore",
    "PyQt6.QtWebEngineWidgets", "PyQt6.QtWebChannel", "PyQt6.QtPositioning",
    "PyQt6.QtBluetooth", "PyQt6.QtNfc", "PyQt6.QtSerialPort", "PyQt6.QtSensors",
    "PyQt6.QtTest", "PyQt6.QtXml", "PyQt6.QtDBus", "PyQt6.QtDesigner",
    "PyQt6.QtHelp", "PyQt6.QtOpenGL", "PyQt6.QtOpenGLWidgets", "PyQt6.Qt3DCore",
    "PyQt6.QtCharts", "PyQt6.QtDataVisualization", "PyQt6.QtSql",
    "PyQt6.QtTextToSpeech", "PyQt6.QtWebSockets", "PyQt6.QtRemoteObjects",
    "PyQt6.QtScxml", "PyQt6.QtStateMachine", "PyQt6.QtJsonRpc",
    "PyQt6.QtHttpServer", "PyQt6.QtGraphs", "PyQt6.QtGrpc",
    # 文本型 PDF/DOCX 离线导入使用 pypdf/python-docx；不再排除 docx/lxml。
    # 在线大模型识别已移除,requests/Pillow 不再需要
    "requests", "PIL",
    # 环境里其他包注册的 hook 拽进来的无关依赖(mitmproxy hook 引入)
    "numpy", "mitmproxy",
    # 标准库中用不到的大件
    "tkinter", "pydoc_data",
]

# Windows App Certification Kit checks the executable's embedded manifest for
# DPI awareness.  Qt's runtime scaling policy in main.py is still useful, but
# it does not replace this native declaration.  Keep the legacy dpiAware value
# as a fallback for older Windows versions and prefer PerMonitorV2 on newer
# systems without requesting elevation.
WINDOWS_MANIFEST = r'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<assembly xmlns="urn:schemas-microsoft-com:asm.v1" manifestVersion="1.0">
  <trustInfo xmlns="urn:schemas-microsoft-com:asm.v3">
    <security>
      <requestedPrivileges>
        <requestedExecutionLevel level="asInvoker" uiAccess="false" />
      </requestedPrivileges>
    </security>
  </trustInfo>
  <application xmlns="urn:schemas-microsoft-com:asm.v3">
    <windowsSettings>
      <dpiAware xmlns="http://schemas.microsoft.com/SMI/2005/WindowsSettings">true/pm</dpiAware>
      <dpiAwareness xmlns="http://schemas.microsoft.com/SMI/2016/WindowsSettings">PerMonitorV2</dpiAwareness>
    </windowsSettings>
  </application>
</assembly>'''

OPTIONAL_DATAS = []
OMR_BUNDLE_SOURCE = os.environ.get("AUTOMUSIC_OMR_BUNDLE_DIR", "").strip()
if OMR_BUNDLE_SOURCE and os.path.isdir(OMR_BUNDLE_SOURCE):
    # 发布构建可把 Audiveris/jianpu-omr 放入此目录；基础 EXE 没有时仍可构建。
    component_bytes = sum(
        os.path.getsize(os.path.join(root, name))
        for root, _dirs, files in os.walk(OMR_BUNDLE_SOURCE)
        for name in files
    )
    if component_bytes > 500 * 1024 * 1024:
        raise SystemExit("离线 OMR 组件总大小不能超过 500 MB")
    OPTIONAL_DATAS.append((OMR_BUNDLE_SOURCE, "omr"))


def _keep(entry_name: str) -> bool:
    n = entry_name.replace("\\", "/").lower()
    base = n.rsplit("/", 1)[-1]
    # Qt 6.11 链接 Windows 系统 ICU。构建机 PATH 中若存在 Poppler/Conda 的
    # 同名 ICU，PyInstaller 会误收集并遮蔽 System32，导致 QtCore 找不到导出过程。
    if fnmatch.fnmatch(base, "icu*.dll"):
        return False
    # Qt 插件:仅保留窗口平台与窗口样式
    if "/qt6/plugins/" in n:
        return "/platforms/" in n or "/styles/" in n
    # Qt DLL 目录:保留 Core/Gui/Widgets + 所有运行时依赖(修复 QtCore 加载失败)
    if "/qt6/bin/" in n and n.endswith(".dll"):
        # 保留 Qt6 核心模块
        if base.startswith(("qt6core", "qt6gui", "qt6widgets")):
            return True
        # 保留所有 MSVC 运行时库和 Windows API Set DLL
        if base.startswith(("msvcp", "vcruntime", "concrt", "ucrtbase", "api-ms-win-")):
            return True
        # 保留可能的 Qt 底层依赖(如 libgcc/libstdc++/libEGL)
        if any(x in base for x in ("libgcc", "libstdc++", "libegl", "libglesv2")):
            return True
        # 其他 Qt DLL 过滤掉
        return False
    # Qt 翻译文件(.qm)用不到
    if "/translations/" in n and n.endswith(".qm"):
        return False
    return True


a = Analysis(
    ["main.py"],
    pathex=[],
    binaries=[],
    datas=[
        ("config.yaml", "."),
        ("app.ico", "."),
        ("profiles", "profiles"),
        ("THIRD_PARTY_NOTICES.md", "."),
        ("version.txt", "."),
    ] + OPTIONAL_DATAS,
    hiddenimports=[
        "pynput",
        "mido",
        "pypdf",
        "docx",
        "docx.oxml",
        "cryptography",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=EXCLUDES,
    noarchive=False,
)

a.binaries = [b for b in a.binaries if _keep(b[0])]
a.datas = [d for d in a.datas if _keep(d[0])]

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="AutoMusicPlayer",
    icon="app.ico",
    manifest=WINDOWS_MANIFEST,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
