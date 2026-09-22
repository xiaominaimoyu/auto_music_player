"""启动风险警示(免责声明)与首次使用说明。

对话框在主界面前显示,首次启动必须完整观看 3 秒后才能继续。
文案常量集中在本文件顶部,后续修订文案只需修改常量字面值。
"""

from PyQt6.QtCore import QSettings, QTimer, Qt
from PyQt6.QtGui import QGuiApplication
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from gui.theme import INK, LINE_2, STATE_ERROR, SURFACE, SURFACE_2

# ---------- 文案(统一常量组:唯一存放处) ----------
DISCLAIMER_TITLE = "⚠️ 风险警示"
DISCLAIMER_BODY = (
    "该软件目前尚不成熟，有游戏账号封禁的可能，因此下载使用前请务必知悉使用风险！"
    "作者及其共创者不负有任何责任。选择使用软件造成的一切后果由自己承担。"
)
DISCLAIMER_OPEN_SOURCE = (
    "再次声明：本工具免费开源，只做编程的交流学习；凡是拿它收费的现象皆是骗子，请勿上当。\n"
    "欢迎测试反馈。作者联系方式：2441377859@qq.com（邮箱）；378031486（抖音号）。"
)
DISCLAIMER_TUTORIAL_TITLE = "简单使用教程"
DISCLAIMER_TUTORIAL = (
    "1. 在「上传识别」页复制提示词，交给外部 AI 工具并附上乐谱图片。\n"
    "2. 将 AI 返回的简谱粘贴回来，点击「解析到校对表格」。\n"
    "3. 仔细校对音符、时值、半音和休止符，确认无误后保存到乐谱库。\n"
    "4. 在「演奏控制」页先点击「试听当前乐谱」确认旋律；试听只使用电脑扬声器，不会向游戏发送按键。\n"
    "5. 确认后再点击「开始演奏」，在倒计时内切回游戏；演奏中可按 F8 暂停。"
)
BTN_CONTINUE_TEXT = "我已知晓并自愿承担全部风险，继续使用"
BTN_QUIT_TEXT = "退出程序"
SKIP_CHECKBOX_TEXT = "下次启动不再弹出此提示"
COUNTDOWN_SECONDS = 3
# 按发布版本隔离“下次不再弹出”选择。旧版本曾使用未带版本号的键，
# 若直接复用会让升级到 v1.4.2 的用户被旧设置静默跳过启动风险提示。
DISCLAIMER_SKIP_KEY = "disclaimer/skip_on_next_start/v1.4.2"
_SETTINGS_ORGANIZATION = "AutoMusicPlayer"
_SETTINGS_APPLICATION = "AutoMusicPlayer"

_DIALOG_WIDTH = 520
_MAX_HEIGHT_RATIO = 0.8


class RiskDisclaimerDialog(QDialog):
    """启动时强制模态的风险警示对话框。

    结果码语义:Accepted 仅可能来自倒计时结束后的继续按钮;退出按钮 / 标题栏 × / Esc /
    Alt+F4 全部归一为 Rejected。对话框只产出结果码,进程终止由调用方决定。
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(DISCLAIMER_TITLE)
        self.setModal(True)
        self.setFixedWidth(_DIALOG_WIDTH)
        self._countdown_left = COUNTDOWN_SECONDS
        self._build()
        self._fit_height()
        self._countdown_timer = QTimer(self)
        self._countdown_timer.setInterval(1000)
        self._countdown_timer.timeout.connect(self._countdown_tick)
        self._update_countdown()
        self._countdown_timer.start()

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 20, 20, 20)
        root.setSpacing(14)

        card = QFrame()
        card.setObjectName("DisclaimerCard")
        card.setStyleSheet(
            f"QFrame#DisclaimerCard {{ background: {SURFACE};"
            f" border: 1px solid {LINE_2}; border-radius: 12px; }}"
        )
        lay = QVBoxLayout(card)
        lay.setContentsMargins(24, 20, 24, 20)
        lay.setSpacing(12)

        title = QLabel(DISCLAIMER_TITLE)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet(
            f"color: {STATE_ERROR}; font-size: 22px; font-weight: 800;"
        )
        lay.addWidget(title)

        content = QWidget()
        content.setObjectName("DisclaimerContent")
        content.setStyleSheet(
            f"QWidget#DisclaimerContent {{ background: {SURFACE_2}; }}"
        )
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(14)

        body = QLabel(DISCLAIMER_BODY)
        body.setObjectName("DisclaimerBody")
        body.setWordWrap(True)
        body.setStyleSheet(
            f"QLabel#DisclaimerBody {{ color: {INK}; font-size: 14px;"
            " background: transparent; border: none; }"
        )
        content_layout.addWidget(body)

        tutorial_title = QLabel(DISCLAIMER_TUTORIAL_TITLE)
        tutorial_title.setObjectName("DisclaimerTutorialTitle")
        tutorial_title.setStyleSheet(
            f"color: {STATE_ERROR}; font-size: 15px; font-weight: 700;"
            " background: transparent; border: none;"
        )
        content_layout.addWidget(tutorial_title)

        tutorial = QLabel(DISCLAIMER_TUTORIAL)
        tutorial.setObjectName("DisclaimerTutorial")
        tutorial.setWordWrap(True)
        tutorial.setStyleSheet(
            f"QLabel#DisclaimerTutorial {{ color: {INK}; font-size: 14px;"
            " background: transparent; border: none; }"
        )
        content_layout.addWidget(tutorial)

        open_source = QLabel(DISCLAIMER_OPEN_SOURCE)
        open_source.setObjectName("DisclaimerOpenSource")
        open_source.setWordWrap(True)
        open_source.setStyleSheet(
            f"QLabel#DisclaimerOpenSource {{ color: {INK}; font-size: 14px;"
            " background: transparent; border: none; }"
        )
        content_layout.addWidget(open_source)

        scroll = QScrollArea()
        scroll.setObjectName("DisclaimerScroll")
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(
            f"QScrollArea#DisclaimerScroll {{ border: none; background: {SURFACE_2}; }}"
            f"QScrollArea#DisclaimerScroll > QWidget {{ background: {SURFACE_2}; }}"
        )
        scroll.viewport().setStyleSheet(
            f"background: {SURFACE_2}; border: none;"
        )
        scroll.setWidget(content)
        lay.addWidget(scroll, 1)

        root.addWidget(card, 1)

        self.countdown_label = QLabel()
        self.countdown_label.setObjectName("DisclaimerCountdown")
        self.countdown_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.countdown_label.setWordWrap(True)
        self.countdown_label.setStyleSheet(
            f"QLabel#DisclaimerCountdown {{ color: {STATE_ERROR}; font-size: 12px;"
            " background: transparent; border: none; }"
        )
        root.addWidget(self.countdown_label)

        self.skip_checkbox = QCheckBox(SKIP_CHECKBOX_TEXT)
        self.skip_checkbox.setObjectName("DisclaimerSkipCheckbox")
        self.skip_checkbox.setToolTip("仅在本次确认继续使用后保存此选择")
        root.addWidget(self.skip_checkbox)

        btn_row = QHBoxLayout()
        self.continue_btn = QPushButton(BTN_CONTINUE_TEXT)
        self.continue_btn.setObjectName("BtnPrimary")
        self.continue_btn.setDefault(True)
        self.continue_btn.setEnabled(False)
        self.continue_btn.clicked.connect(self.accept)
        quit_btn = QPushButton(BTN_QUIT_TEXT)
        quit_btn.setObjectName("BtnDanger")
        quit_btn.clicked.connect(self.reject)
        btn_row.addWidget(self.continue_btn, 1)
        btn_row.addWidget(quit_btn)
        root.addLayout(btn_row)
        quit_btn.setAutoDefault(False)
        self.continue_btn.setFocus()

    def _update_countdown(self):
        if self._countdown_left > 0:
            self.countdown_label.setText(
                f"请完整阅读以上内容，倒计时结束后才可以继续（还需 {self._countdown_left} 秒）"
            )
            self.continue_btn.setEnabled(False)
            self.continue_btn.setToolTip("请先完成 3 秒观看")
        else:
            self.countdown_label.setText("已完成 3 秒观看，可以继续使用")
            self.continue_btn.setEnabled(True)
            self.continue_btn.setToolTip("")

    def _countdown_tick(self):
        if self._countdown_left <= 0:
            self._countdown_timer.stop()
            return
        self._countdown_left -= 1
        self._update_countdown()
        if self._countdown_left <= 0:
            self._countdown_timer.stop()

    def accept(self):
        """倒计时未结束时拒绝任何继续操作,避免绕过强制观看。"""
        if self._countdown_left > 0:
            return
        self._countdown_timer.stop()
        super().accept()

    def reject(self):
        """退出路径立即停止倒计时,避免关闭后继续投递计时事件。"""
        if hasattr(self, "_countdown_timer"):
            self._countdown_timer.stop()
        super().reject()

    def _fit_height(self):
        """高度随内容自适应,并受屏幕可用高度上限约束;超出时正文区内部滚动。"""
        max_h = 600
        screen = QGuiApplication.primaryScreen()
        if screen is not None:
            max_h = int(screen.availableGeometry().height() * _MAX_HEIGHT_RATIO)
        self.adjustSize()
        if self.height() > max_h:
            self.setFixedHeight(max_h)

    def closeEvent(self, event):
        # 标题栏 × / Alt+F4 归一为"退出"结论,与退出按钮走同一结果码(Rejected)
        event.ignore()
        self.reject()


def _get_settings():
    """返回应用级设置存储,不把用户选择写入项目目录。"""
    return QSettings(_SETTINGS_ORGANIZATION, _SETTINGS_APPLICATION)


def _setting_is_true(settings, key: str) -> bool:
    value = settings.value(key, False)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def confirm_risk_disclaimer(settings=None) -> bool:
    """展示风险警示对话框并返回用户结论。True=继续使用;False=退出。

    用户勾选"下次启动不再弹出"并确认后,选择写入 QSettings;
    不内部吞对话框异常、不调用任何进程终止 API,进程终止由调用方决定。
    `settings` 仅用于测试或宿主注入,默认使用应用级 QSettings。
    """
    settings = _get_settings() if settings is None else settings
    if _setting_is_true(settings, DISCLAIMER_SKIP_KEY):
        return True
    dialog = RiskDisclaimerDialog()
    result = dialog.exec()
    accepted = result == QDialog.DialogCode.Accepted
    if accepted and dialog.skip_checkbox.isChecked():
        settings.setValue(DISCLAIMER_SKIP_KEY, True)
        settings.sync()
    return accepted
