"""启动风险警示(免责声明)对话框:程序启动后、主界面呈现前的强制模态关卡。

文案常量为本文件顶部的单一存放处(spec 4.4),警示正文逐字保留、禁止改写截断;
后续修订文案只需修改常量字面值,无需触碰布局代码。
"""

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QGuiApplication
from PyQt6.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
)

from gui.theme import INK, LINE_2, STATE_ERROR, SURFACE

# ---------- 警示文案(统一常量组:唯一存放处,spec 6.1/6.2 锁定字面值) ----------
DISCLAIMER_TITLE = "⚠️ 风险警示"
DISCLAIMER_BODY = (
    "该软件目前尚不成熟，有游戏账号封禁的可能，因此下载使用前请务必知悉使用风险！"
    "作者及其共创者不负有任何责任。选择使用软件造成的一切后果由自己承担。"
)
BTN_CONTINUE_TEXT = "我已知晓并自愿承担全部风险，继续使用"
BTN_QUIT_TEXT = "退出程序"

_DIALOG_WIDTH = 520
_MAX_HEIGHT_RATIO = 0.8


class RiskDisclaimerDialog(QDialog):
    """启动时强制模态的风险警示对话框。

    结果码语义:Accepted 仅可能来自继续按钮;退出按钮 / 标题栏 × / Esc / Alt+F4
    全部归一为 Rejected。对话框只产出结果码,进程终止由调用方决定。
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(DISCLAIMER_TITLE)
        self.setModal(True)
        self.setFixedWidth(_DIALOG_WIDTH)
        self._build()
        self._fit_height()

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

        body = QLabel(DISCLAIMER_BODY)
        body.setObjectName("DisclaimerBody")
        body.setWordWrap(True)
        # 正文可用宽度 = 对话框宽 - root 边距(20*2) - card 边距(24*2) - card 边框(1*2)
        body.setFixedWidth(_DIALOG_WIDTH - 40 - 48 - 2)
        body.setStyleSheet(
            f"QLabel#DisclaimerBody {{ color: {INK}; font-size: 13px;"
            " background: transparent; border: none; }"
        )
        scroll = QScrollArea()
        scroll.setObjectName("DisclaimerScroll")
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(
            "QScrollArea#DisclaimerScroll { border: none; background: transparent; }"
            "QScrollArea#DisclaimerScroll > QWidget { background: transparent; }"
        )
        scroll.setWidget(body)
        lay.addWidget(scroll, 1)

        root.addWidget(card, 1)

        btn_row = QHBoxLayout()
        continue_btn = QPushButton(BTN_CONTINUE_TEXT)
        continue_btn.setObjectName("BtnPrimary")
        continue_btn.setDefault(True)
        continue_btn.clicked.connect(self.accept)
        quit_btn = QPushButton(BTN_QUIT_TEXT)
        quit_btn.setObjectName("BtnDanger")
        quit_btn.clicked.connect(self.reject)
        btn_row.addWidget(continue_btn, 1)
        btn_row.addWidget(quit_btn)
        root.addLayout(btn_row)
        continue_btn.setFocus()

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


def confirm_risk_disclaimer() -> bool:
    """展示风险警示对话框并返回用户结论。True=继续使用;False=退出。

    不内部吞异常(异常向上抛给调用方兜底)、无磁盘/配置副作用、
    不调用任何进程终止 API;每进程调用语义为至多一次(由 main() 启动流程保证)。
    """
    dialog = RiskDisclaimerDialog()
    result = dialog.exec()
    return result == QDialog.DialogCode.Accepted