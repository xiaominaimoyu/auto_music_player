"""主窗口:自绘标题栏(标准 Windows 风格按钮) + 侧边栏导航 + 页面堆栈 + 状态栏。

无边框窗口支持:
- 标题栏拖动 / 双击最大化
- 八方向边缘拉伸(边缘 6px 命中,优先于标题栏拖动;最大化时禁用)
"""

import ctypes

from pynput import keyboard as pk
from PyQt6.QtCore import QEvent, Qt, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QCursor, QPainter, QPen
from PyQt6.QtWidgets import (
    QAbstractButton,
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QPushButton,
    QSizeGrip,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from gui.library_tab import LibraryTab
from gui.player_tab import PlayerTab
from gui.settings_tab import SettingsTab
from gui.theme import (
    BRAND,
    INK,
    INK_2,
    INK_3,
    STATE_ERROR,
    STATE_SUCCESS,
    STATE_WARNING,
    SURFACE_3,
)
from gui.upload_tab import UploadTab

NAV_ITEMS = [
    ("上传识别", "↑"),
    ("乐谱库", "♪"),
    ("演奏控制", "▶"),
    ("模型设置", "⚙"),
]

TITLE_BAR_HEIGHT = 32
SIDEBAR_WIDTH = 240
STATUS_BAR_HEIGHT = 28
RESIZE_MARGIN = 6

_RESIZE_CURSORS = {
    "l": Qt.CursorShape.SizeHorCursor,
    "r": Qt.CursorShape.SizeHorCursor,
    "t": Qt.CursorShape.SizeVerCursor,
    "b": Qt.CursorShape.SizeVerCursor,
    "tl": Qt.CursorShape.SizeFDiagCursor,
    "br": Qt.CursorShape.SizeFDiagCursor,
    "tr": Qt.CursorShape.SizeBDiagCursor,
    "bl": Qt.CursorShape.SizeBDiagCursor,
}


class TitleBarBtn(QAbstractButton):
    """标准 Windows 风格标题栏按钮(QPainter 自绘横线/方框/X/小窗)。"""

    def __init__(self, kind: str, parent=None):
        super().__init__(parent)
        self._kind = kind  # "min" / "max" / "mini" / "close"
        self._hover = False
        self._is_maximized = False
        self.setFixedSize(46, TITLE_BAR_HEIGHT)
        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.setToolTip({
            "min": "最小化",
            "max": "最大化",
            "mini": "切为演奏小窗(悬浮于游戏上方,点击不夺游戏焦点)",
            "close": "关闭",
        }[kind])

    def set_maximized(self, maximized: bool):
        if self._is_maximized != maximized:
            self._is_maximized = maximized
            self.update()

    def enterEvent(self, event):
        self._hover = True
        self.update()

    def leaveEvent(self, event):
        self._hover = False
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()

        if self._hover:
            if self._kind == "close":
                p.fillRect(0, 0, w, h, QColor(STATE_ERROR))
                fg = QColor("#FFFFFF")
            else:
                p.fillRect(0, 0, w, h, QColor(SURFACE_3))
                fg = QColor(INK)
        else:
            fg = QColor(INK_2)

        pen = QPen(fg, 1.0)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)

        cx, cy = w // 2, h // 2
        if self._kind == "min":
            p.drawLine(cx - 5, cy, cx + 5, cy)
        elif self._kind == "max":
            if self._is_maximized:
                p.drawRect(cx - 4, cy - 1, 8, 8)
                p.drawLine(cx - 2, cy - 1, cx - 2, cy - 3)
                p.drawLine(cx - 2, cy - 3, cx + 5, cy - 3)
                p.drawLine(cx + 5, cy - 3, cx + 5, cy + 3)
                p.drawLine(cx + 5, cy + 3, cx + 4, cy + 3)
            else:
                p.drawRect(cx - 5, cy - 5, 10, 10)
        elif self._kind == "mini":
            # 画中画:外框 + 右下角小实心块
            p.drawRect(cx - 6, cy - 5, 11, 9)
            p.setBrush(QBrush(fg))
            p.drawRect(cx, cy, 4, 3)
            p.setBrush(Qt.BrushStyle.NoBrush)
        elif self._kind == "close":
            p.setPen(QPen(fg, 1.2))
            p.drawLine(cx - 5, cy - 5, cx + 5, cy + 5)
            p.drawLine(cx + 5, cy - 5, cx - 5, cy + 5)


class TitleBar(QWidget):
    minimize_requested = pyqtSignal()
    maximize_requested = pyqtSignal()
    mini_requested = pyqtSignal()
    close_requested = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setObjectName("TitleBar")
        self.setFixedHeight(TITLE_BAR_HEIGHT)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 0, 0, 0)
        layout.setSpacing(8)

        self.title_label = QLabel("自动演奏器")
        self.title_label.setObjectName("TitleBarTitle")
        layout.addWidget(self.title_label)
        layout.addStretch(1)

        self.btn_min = TitleBarBtn("min")
        self.btn_max = TitleBarBtn("max")
        self.btn_mini = TitleBarBtn("mini")
        self.btn_close = TitleBarBtn("close")
        self.btn_min.clicked.connect(self.minimize_requested.emit)
        self.btn_max.clicked.connect(self.maximize_requested.emit)
        self.btn_mini.clicked.connect(self.mini_requested.emit)
        self.btn_close.clicked.connect(self.close_requested.emit)
        layout.addWidget(self.btn_min)
        layout.addWidget(self.btn_max)
        layout.addWidget(self.btn_mini)
        layout.addWidget(self.btn_close)


class MainWindow(QMainWindow):
    def __init__(self, cfg, db, keymap, recognizer, player, settings_store, config_path=None,
                 profile=None, profiles=None, event_player=None):
        super().__init__()
        self._cfg = cfg
        self._player = player
        self._settings_store = settings_store
        self._config_path = config_path
        # 游戏档位(M3 起由 main.py 注入):当前激活档位与全部候选档位。
        # M4 在演奏控制页消费它们(档位下拉 + 按档位组装 CompileParams)。
        self._profile = profile
        self._profiles = list(profiles or [])
        self._event_player = event_player
        self._drag_pos = None
        self._is_maximized = False
        self._normal_geometry = None
        self._mini_window = None
        # 边缘拉伸状态
        self._resize_dir = None
        self._press_pos = None
        self._press_geometry = None

        self.setWindowTitle("自动演奏器")
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        self.setMouseTracking(True)
        self.resize(1080, 720)
        self.setMinimumSize(900, 600)

        self.upload_tab = UploadTab(
            db,
            recognizer,
            keymap,
            advisor_fn=lambda: (self._cfg.get("advisor") or {}, self._settings_store.get_active()),
        )
        self.library_tab = LibraryTab(db)
        self.player_tab = PlayerTab(db, player, cfg.get("player", {}), config_path=config_path,
                                    profile=self._profile, profiles=self._profiles,
                                    event_player=self._event_player)
        self.settings_tab = SettingsTab(settings_store)

        self._build_ui()

        self.upload_tab.saved.connect(self.library_tab.refresh)
        self.upload_tab.saved.connect(self.player_tab.refresh)
        self.upload_tab.settings_requested.connect(lambda: self.nav.setCurrentRow(3))
        self.library_tab.go_play.connect(self._go_play)
        self.settings_tab.providers_saved.connect(self._on_providers_saved)

        hotkey = str(cfg.get("player", {}).get("stop_hotkey", "F8")).lower()
        self._hotkey_listener = pk.GlobalHotKeys({f"<{hotkey}>": self._on_hotkey_stop})
        self._hotkey_listener.start()

        # 退出清理挂在 aboutToQuit(事件循环仍存活)而非 closeEvent:
        # 窗口析构阶段的 closeEvent 已处于解释器收尾期,此时调用 Win32 会引发进程 fast-fail
        QApplication.instance().aboutToQuit.connect(self._cleanup_on_quit)

        self.player_tab.refresh()
        self.settings_tab.refresh_provider_status()

        # 应用级事件过滤器:子控件覆盖边缘时也能命中拉伸
        QApplication.instance().installEventFilter(self)

    def _on_hotkey_stop(self):
        """全局热键同时停止旧播放器和事件播放器。"""
        self._player.stop()
        if self._event_player is not None:
            self._event_player.stop()

    def _cleanup_on_quit(self):
        """退出清理:停全局热键监听与演奏线程,确保全部按键释放。"""
        try:
            self._hotkey_listener.stop()
        except Exception:
            pass
        self._player.shutdown()
        self.upload_tab.stop_preview()
        self.player_tab.stop_preview()

    def _build_ui(self):
        root = QWidget()
        root.setObjectName("AppRoot")
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        self.title_bar = TitleBar()
        self.title_bar.minimize_requested.connect(self.showMinimized)
        self.title_bar.maximize_requested.connect(self._toggle_maximize)
        self.title_bar.mini_requested.connect(self._switch_to_mini)
        self.title_bar.close_requested.connect(self.close)
        root_layout.addWidget(self.title_bar)

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)

        sidebar = QWidget()
        sidebar.setObjectName("Sidebar")
        sidebar.setFixedWidth(SIDEBAR_WIDTH)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.setSpacing(0)

        logo_row = QHBoxLayout()
        logo_row.setContentsMargins(20, 18, 20, 14)
        logo_text = QLabel("自动演奏器")
        logo_text.setObjectName("SidebarLogoText")
        logo_text.setStyleSheet(f"color: {BRAND};")
        logo_row.addWidget(logo_text)
        sidebar_layout.addLayout(logo_row)

        self.nav = QListWidget()
        self.nav.setObjectName("SidebarNav")
        self.nav.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        for text, icon in NAV_ITEMS:
            self.nav.addItem(QListWidgetItem(f"{icon}  {text}"))
        self.nav.currentRowChanged.connect(self._switch_page)
        sidebar_layout.addWidget(self.nav, 1)

        footer = QLabel("v1.1")
        footer.setObjectName("SidebarFooter")
        sidebar_layout.addWidget(footer)

        body.addWidget(sidebar)

        self.stack = QStackedWidget()
        self.stack.addWidget(self.upload_tab)
        self.stack.addWidget(self.library_tab)
        self.stack.addWidget(self.player_tab)
        self.stack.addWidget(self.settings_tab)
        body.addWidget(self.stack, 1)

        root_layout.addLayout(body, 1)

        status_bar = QWidget()
        status_bar.setObjectName("StatusBar")
        status_bar.setFixedHeight(STATUS_BAR_HEIGHT)
        status_layout = QHBoxLayout(status_bar)
        status_layout.setContentsMargins(16, 0, 4, 0)
        status_layout.setSpacing(12)
        dot = QLabel()
        dot.setObjectName("StatusDot")
        self.status_label = QLabel("就绪")
        self.status_label.setObjectName("StatusText")
        hotkey_label = QLabel("按 F8 可随时暂停演奏")
        hotkey_label.setObjectName("StatusText")
        hotkey_label.setToolTip("全局热键 F8:无论焦点在哪个窗口,按下 F8 会暂停当前演奏,可在演奏页「继续演奏」或「重置」")
        status_layout.addWidget(dot)
        status_layout.addWidget(self.status_label)
        status_layout.addStretch(1)
        try:
            admin = bool(ctypes.windll.shell32.IsUserAnAdmin())
        except Exception:
            admin = False
        perm_label = QLabel("管理员权限" if admin else "普通权限 · 游戏收不到按键时请以管理员运行")
        perm_label.setObjectName("StatusText")
        perm_label.setStyleSheet(f"color: {STATE_SUCCESS};" if admin else f"color: {STATE_WARNING};")
        status_layout.addWidget(perm_label)
        status_layout.addWidget(hotkey_label)
        status_layout.addWidget(QSizeGrip(status_bar))
        root_layout.addWidget(status_bar)

        self.setCentralWidget(root)
        self.nav.setCurrentRow(0)

    def _switch_page(self, row: int):
        self.stack.setCurrentIndex(row)

    def _go_play(self, score_id: int):
        """统一跳转入口:侧边栏选中态与页面同步。"""
        self.nav.setCurrentRow(2)
        if self.stack.currentIndex() != 2:
            self.stack.setCurrentIndex(2)
        self.player_tab.select_score(score_id)

    # ---------- 演奏小窗 ----------

    def _switch_to_mini(self):
        """切为小窗:隐藏主窗(演奏线程与状态不受影响),显示置顶小窗。"""
        from gui.mini_window import MiniPlayerWindow

        if self._mini_window is None:
            self._mini_window = MiniPlayerWindow(self.player_tab, on_restore=self._restore_from_mini)
        self.player_tab.disable_focus_watch()
        self._mini_window.show()
        self.hide()

    def _restore_from_mini(self):
        if self._mini_window is not None:
            self._mini_window.hide()
        self.show()
        self.raise_()
        self.activateWindow()
        self.player_tab.rearm_focus_watch()

    def set_status(self, text: str):
        self.status_label.setText(text)

    def _on_providers_saved(self):
        from core.recognizer import get_recognizer_from_provider

        provider = self._settings_store.get_active()
        recognizer = get_recognizer_from_provider(provider)
        self.upload_tab.set_recognizer(recognizer)
        self.set_status(
            f"识别模型: {provider['name']} · {provider['model']}" if recognizer else "识别模型: 未配置"
        )

    # ---------- 边缘拉伸 ----------

    def _hit_dir(self, gpos):
        """光标相对窗口边缘的位置 → 拉伸方向;不在边缘返回 None。"""
        if self._is_maximized:
            return None
        frame = self.frameGeometry()
        x = gpos.x() - frame.left()
        y = gpos.y() - frame.top()
        w, h, m = frame.width(), frame.height(), RESIZE_MARGIN
        if not (0 <= x < w and 0 <= y < h):
            return None
        left, right = x < m, x >= w - m
        top, bottom = y < m, y >= h - m
        if left and top:
            return "tl"
        if right and top:
            return "tr"
        if left and bottom:
            return "bl"
        if right and bottom:
            return "br"
        if left:
            return "l"
        if right:
            return "r"
        if top:
            return "t"
        if bottom:
            return "b"
        return None

    def eventFilter(self, obj, event):
        et = event.type()
        if et == QEvent.Type.MouseMove:
            gpos = event.globalPosition().toPoint()
            if self._resize_dir:
                self._do_resize(gpos)
                return False
            direction = self._hit_dir(gpos)
            if direction and obj in (self, self.title_bar, self.centralWidget()):
                self.setCursor(_RESIZE_CURSORS[direction])
            elif obj in (self, self.title_bar, self.centralWidget()):
                self.unsetCursor()
            return False
        if et == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
            direction = self._hit_dir(event.globalPosition().toPoint())
            if direction:
                self._resize_dir = direction
                self._press_pos = event.globalPosition().toPoint()
                self._press_geometry = self.frameGeometry()
                return True  # 吞掉,避免子控件响应误触
            return False
        if et == QEvent.Type.MouseButtonRelease and self._resize_dir:
            self._resize_dir = None
            self.unsetCursor()
            return False
        return super().eventFilter(obj, event)

    def _do_resize(self, gpos):
        d, g0 = self._resize_dir, self._press_geometry
        dx = gpos.x() - self._press_pos.x()
        dy = gpos.y() - self._press_pos.y()
        l, t = g0.left(), g0.top()
        r, b = g0.right(), g0.bottom()
        min_w, min_h = self.minimumWidth(), self.minimumHeight()
        if "l" in d:
            l = min(g0.left() + dx, r - min_w)
        if "r" in d:
            r = max(g0.right() + dx, l + min_w)
        if "t" in d:
            t = min(g0.top() + dy, b - min_h)
        if "b" in d:
            b = max(g0.bottom() + dy, t + min_h)
        self.setGeometry(l, t, r - l + 1, b - t + 1)

    # ---------- 标题栏拖动 / 最大化 ----------

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and event.position().y() <= TITLE_BAR_HEIGHT:
            self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        if self._drag_pos is not None and event.buttons() & Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_pos)
            event.accept()

    def mouseReleaseEvent(self, event):
        self._drag_pos = None

    def mouseDoubleClickEvent(self, event):
        if event.position().y() <= TITLE_BAR_HEIGHT:
            self._toggle_maximize()

    def _toggle_maximize(self):
        if self._is_maximized:
            self.showNormal()
            if self._normal_geometry:
                self.setGeometry(self._normal_geometry)
            self.title_bar.btn_max.set_maximized(False)
        else:
            self._normal_geometry = self.geometry()
            self.showMaximized()
            self.title_bar.btn_max.set_maximized(True)
        self._is_maximized = not self._is_maximized
