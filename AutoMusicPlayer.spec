# -*- mode: python ; coding: utf-8 -*-
# PyInstaller 打包配置:体积优化版
#   1. 排除用不到的 PyQt6 子模块与第三方库
#   2. 过滤 Qt 的无用 DLL(opengl32sw/d3dcompiler/Network/Pdf/Svg/Qml/Multimedia 等)
#      与插件(仅保留 platforms + styles)
# 打包: python -m PyInstaller --noconfirm --clean AutoMusicPlayer.spec

import fnmatch

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
    # 不再使用的库
    "docx", "lxml",
    # 在线大模型识别已移除,requests/Pillow 不再需要
    "requests", "PIL",
    # 环境里其他包注册的 hook 拽进来的无关依赖(mitmproxy hook 引入)
    "numpy", "cryptography", "mitmproxy",
    # 标准库中用不到的大件
    "tkinter", "pydoc_data",
]


def _keep(entry_name: str) -> bool:
    n = entry_name.replace("\\", "/").lower()
    # Qt 插件:仅保留窗口平台与窗口样式
    if "/qt6/plugins/" in n:
        return "/platforms/" in n or "/styles/" in n
    # Qt DLL 目录:仅保留 Core/Gui/Widgets 与 MSVC 运行库
    if "/qt6/bin/" in n and n.endswith(".dll"):
        base = n.rsplit("/", 1)[-1]
        if base.startswith(("qt6core", "qt6gui", "qt6widgets")):
            return True
        return base.startswith(("msvcp", "vcruntime", "concrt", "ucrtbase"))
    # Qt 翻译文件(.qm)用不到
    if "/translations/" in n and n.endswith(".qm"):
        return False
    return True


a = Analysis(
    ["main.py"],
    pathex=[],
    binaries=[],
    datas=[("config.yaml", "."), ("app.ico", "."), ("profiles", "profiles")],
    hiddenimports=["pynput"],
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