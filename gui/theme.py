"""主题样式:对齐 gui-redesign 设计稿(暗色 + 琥珀金品牌色)。"""

# 设计 token
BG = "#0E0E14"
SURFACE = "#16161E"
SURFACE_2 = "#1E1E28"
SURFACE_3 = "#262631"
INK = "#EDEDF2"
INK_2 = "#9494A2"
INK_3 = "#5E5E6E"
LINE = "#26262F"
LINE_2 = "#2E2E3A"
BRAND = "#D4A24C"
BRAND_2 = "#E8B85E"
BRAND_3 = "#B88638"
BRAND_SOFT = "rgba(212, 162, 76, 0.12)"
STATE_SUCCESS = "#4ADE80"
STATE_WARNING = "#FBBF24"
STATE_ERROR = "#F87171"
STATE_INFO = "#38BDF8"
RADIUS_MD = 8
RADIUS_LG = 8

FONT_SANS = '"Microsoft YaHei", "PingFang SC", "Segoe UI", sans-serif'
FONT_MONO = '"Consolas", "Cascadia Code", monospace'

APP_QSS = f"""
* {{
    font-family: {FONT_SANS};
}}
QMainWindow, QDialog, QWidget#AppRoot {{
    background: {BG};
    color: {INK};
}}

/* ---------- 标题栏 ---------- */
QWidget#TitleBar {{
    background: {SURFACE};
    border-bottom: 1px solid {LINE};
}}
QPushButton#TitleBarTitle {{
    color: {INK_2};
    font-size: 12px;
    font-weight: 500;
}}

/* ---------- 侧边栏 ---------- */
QWidget#Sidebar {{
    background: {SURFACE};
    border-right: 1px solid {LINE};
}}
QLabel#SidebarLogoText {{
    color: {INK};
    font-size: 15px;
    font-weight: 600;
}}
QListWidget#SidebarNav {{
    background: transparent;
    border: none;
    outline: none;
}}
QListWidget#SidebarNav::item {{
    color: {INK_2};
    padding: 10px 12px;
    margin: 2px 12px;
    border-radius: {RADIUS_MD}px;
    font-size: 14px;
    font-weight: 500;
}}
QListWidget#SidebarNav::item:hover {{
    background: {SURFACE_2};
    color: {INK};
}}
QListWidget#SidebarNav::item:selected {{
    background: {BRAND_SOFT};
    color: {BRAND};
    font-weight: 600;
}}
QLabel#SidebarFooter {{
    color: {INK_3};
    font-size: 12px;
    padding: 12px 20px;
    border-top: 1px solid {LINE};
}}

/* ---------- 主区域 ---------- */
QWidget#PageHeader {{
    background: {SURFACE};
    border-bottom: 1px solid {LINE};
}}
QLabel#PageTitle {{
    color: {INK};
    font-size: 18px;
    font-weight: 600;
}}
QLabel#PageSub {{
    color: {INK_2};
    font-size: 13px;
}}
QScrollArea#ContentScroll {{
    background: transparent;
    border: none;
}}
QWidget#ContentInner {{
    background: transparent;
}}

/* ---------- 卡片 ---------- */
QFrame#SectionCard {{
    background: {SURFACE};
    border: 1px solid {LINE};
    border-radius: {RADIUS_LG}px;
}}
QLabel#SectionTitle {{
    color: {INK};
    font-size: 14px;
    font-weight: 600;
}}
QLabel#SectionSubtitle {{
    color: {INK_2};
    font-size: 12px;
}}
QLabel#StepBadge {{
    background: {BRAND_SOFT};
    color: {BRAND};
    border-radius: 12px;
    min-width: 24px;
    max-width: 24px;
    min-height: 24px;
    max-height: 24px;
    font-size: 12px;
    font-weight: 700;
}}

/* ---------- 按钮 ---------- */
QPushButton {{
    border: none;
    border-radius: {RADIUS_MD}px;
    min-height: 36px;
    padding: 0 16px;
    font-size: 14px;
    font-weight: 600;
    background: transparent;
    color: {INK};
}}
QPushButton:disabled {{
    opacity: 0.4;
    color: {INK_3};
}}
QPushButton#BtnPrimary {{
    background: {BRAND};
    color: {BG};
}}
QPushButton#BtnPrimary:hover {{
    background: {BRAND_2};
}}
QPushButton#BtnPrimary:disabled {{
    background: {BRAND_3};
    color: rgba(14, 14, 20, 0.6);
}}
QPushButton#BtnSecondary {{
    background: transparent;
    border: 1px solid {LINE_2};
    color: {INK};
}}
QPushButton#BtnSecondary:hover {{
    border-color: {INK_3};
    background: {SURFACE_2};
}}
QPushButton#BtnSecondary:disabled {{
    border-color: {LINE};
    color: {INK_3};
}}
QPushButton#BtnDanger {{
    background: transparent;
    border: 1px solid rgba(248, 113, 113, 0.3);
    color: {STATE_ERROR};
}}
QPushButton#BtnDanger:hover {{
    background: rgba(248, 113, 113, 0.08);
}}
QPushButton#BtnDanger:disabled {{
    border-color: {LINE};
    color: {INK_3};
}}
QPushButton#BtnStop {{
    background: transparent;
    border: 1px solid rgba(248, 113, 113, 0.3);
    color: {STATE_ERROR};
}}
QPushButton#BtnStop:hover {{
    background: rgba(248, 113, 113, 0.08);
}}
QPushButton#BtnStop:disabled {{
    border-color: {LINE};
    color: {INK_3};
}}

/* ---------- 输入控件(单行: min-height + 水平 padding,防高 DPI 裁字) ---------- */
QLineEdit, QSpinBox, QComboBox {{
    background: {SURFACE_2};
    border: 1px solid {LINE_2};
    border-radius: {RADIUS_MD}px;
    min-height: 34px;
    padding: 0 12px;
    color: {INK};
    font-size: 14px;
    selection-background-color: {BRAND};
    selection-color: {BG};
}}
QPlainTextEdit, QTextEdit {{
    background: {SURFACE_2};
    border: 1px solid {LINE_2};
    border-radius: {RADIUS_MD}px;
    padding: 8px 12px;
    color: {INK};
    font-size: 14px;
    selection-background-color: {BRAND};
    selection-color: {BG};
}}
QComboBox::drop-down {{
    border: none;
    width: 32px;
}}
QComboBox QAbstractItemView {{
    background: {SURFACE_2};
    border: 1px solid {LINE_2};
    border-radius: {RADIUS_MD}px;
    outline: none;
    selection-background-color: {BRAND_SOFT};
    selection-color: {BRAND};
}}
QLineEdit:focus, QSpinBox:focus, QPlainTextEdit:focus, QTextEdit:focus, QComboBox:focus {{
    border-color: {BRAND_3};
}}
QLineEdit:disabled, QSpinBox:disabled, QPlainTextEdit:disabled, QComboBox:disabled {{
    color: {INK_3};
    border-color: {LINE};
}}
QPlainTextEdit {{ font-family: {FONT_MONO}; }}
QLabel#LabelMono {{ font-family: {FONT_MONO}; }}
QLabel#FieldLabel {{
    color: {INK_2};
    font-size: 13px;
}}
QLabel#HintText {{
    color: {INK_3};
    font-size: 12px;
}}

/* ---------- 识别与节奏状态 ---------- */
QFrame#RecognitionPanel {{
    background: {BRAND_SOFT};
    border: 1px solid rgba(212, 162, 76, 0.28);
    border-radius: {RADIUS_MD}px;
}}
QLabel#RecognitionTitle {{
    color: {INK};
    font-size: 14px;
    font-weight: 600;
}}
QLabel#RecognitionElapsed {{
    color: {BRAND};
    font-size: 12px;
    font-weight: 600;
}}
QLabel#BpmValue {{
    background: {SURFACE_2};
    border: 1px solid {LINE_2};
    border-radius: 6px;
    color: {BRAND};
    padding: 6px 10px;
    font-size: 12px;
    font-weight: 600;
}}
QLabel#BpmValue[invalid="true"] {{
    color: {STATE_ERROR};
    border-color: rgba(248, 113, 113, 0.35);
}}
QFrame#RhythmWarning {{
    background: rgba(251, 191, 36, 0.08);
    border: 1px solid rgba(251, 191, 36, 0.24);
    border-radius: {RADIUS_MD}px;
}}
QLabel#RhythmWarningText {{
    color: {STATE_WARNING};
    font-size: 12px;
}}
QPushButton#BtnWarning {{
    background: transparent;
    border: 1px solid rgba(251, 191, 36, 0.35);
    color: {STATE_WARNING};
}}
QPushButton#BtnWarning:hover {{
    background: rgba(251, 191, 36, 0.08);
}}

/* ---------- 表格 ---------- */
QTableWidget {{
    background: transparent;
    border: 1px solid {LINE};
    border-radius: {RADIUS_MD}px;
    gridline-color: {LINE};
    alternate-background-color: {SURFACE_2};
    font-size: 13px;
}}
QTableWidget::item {{
    padding: 6px 8px;
    color: {INK};
    border-bottom: 1px solid {LINE};
}}
QTableWidget::item:selected {{
    background: {BRAND_SOFT};
    color: {BRAND};
}}
QHeaderView::section {{
    background: {SURFACE_2};
    color: {INK_2};
    border: none;
    border-bottom: 1px solid {LINE_2};
    padding: 8px 8px;
    font-size: 12px;
    font-weight: 600;
}}
QTableCornerButton::section {{
    background: {SURFACE_2};
    border: none;
}}
QMenu#NoteActionMenu {{
    background: {SURFACE_2};
    border: 1px solid {LINE_2};
    border-radius: {RADIUS_MD}px;
    padding: 6px;
    color: {INK};
}}
QMenu#NoteActionMenu::item {{
    min-width: 230px;
    padding: 9px 14px;
    border-radius: 6px;
}}
QMenu#NoteActionMenu::item:selected {{
    background: {BRAND_SOFT};
    color: {BRAND};
}}
QMenu#NoteActionMenu::item:disabled {{
    color: {INK_2};
    font-weight: 600;
}}
QMenu#NoteActionMenu::separator {{
    height: 1px;
    margin: 5px 8px;
    background: {LINE};
}}
QScrollBar:vertical {{
    background: transparent;
    width: 10px;
}}
QScrollBar::handle:vertical {{
    background: {SURFACE_3};
    border-radius: 5px;
    min-height: 40px;
}}
QScrollBar::handle:vertical:hover {{
    background: {LINE_2};
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
    background: transparent;
}}
QScrollBar:horizontal {{
    background: transparent;
    height: 10px;
}}
QScrollBar::handle:horizontal {{
    background: {SURFACE_3};
    border-radius: 5px;
    min-width: 40px;
}}
QScrollBar::handle:horizontal:hover {{
    background: {LINE_2};
}}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0;
}}
QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{
    background: transparent;
}}

/* ---------- 分隔条(QSplitter,可拖动调整识别结果/校对表格比例) ---------- */
QSplitter::handle {{
    background: {LINE};
}}
QSplitter::handle:hover {{
    background: {BRAND_3};
}}
QSplitter::handle:vertical {{
    height: 6px;
}}
QSplitter::handle:horizontal {{
    width: 6px;
}}

/* ---------- 进度条 ---------- */
QProgressBar {{
    background: {SURFACE_3};
    border: none;
    border-radius: 5px;
    max-height: 10px;
    min-height: 10px;
    text-align: center;
    font-size: 10px;
    color: transparent;
}}
QProgressBar::chunk {{
    background: {BRAND};
    border-radius: 5px;
}}

/* ---------- 滑块 ---------- */
QSlider::groove:horizontal {{
    height: 6px;
    background: {SURFACE_3};
    border-radius: 3px;
}}
QSlider::handle:horizontal {{
    width: 18px;
    height: 18px;
    margin: -6px 0;
    border-radius: 9px;
    background: {BRAND};
    border: 2px solid {SURFACE};
}}
QSlider::sub-page:horizontal {{
    background: {BRAND_3};
    border-radius: 3px;
}}

/* ---------- 列表(供应商) ---------- */
QListWidget#ProviderList {{
    background: transparent;
    border: none;
    outline: none;
}}
QListWidget#ProviderList::item {{
    background: {SURFACE_2};
    border: 1px solid {LINE};
    border-radius: {RADIUS_MD}px;
    margin: 4px 0;
    padding: 10px 12px;
    color: {INK};
}}
QListWidget#ProviderList::item:hover {{
    border-color: {LINE_2};
}}
QListWidget#ProviderList::item:selected {{
    border-color: {BRAND_3};
    background: {BRAND_SOFT};
}}

/* ---------- 状态栏 ---------- */
QWidget#StatusBar {{
    background: {SURFACE};
    border-top: 1px solid {LINE};
}}
QLabel#StatusText {{
    color: {INK_3};
    font-size: 11px;
}}
QLabel#StatusDot {{
    background: {STATE_SUCCESS};
    border-radius: 3px;
    min-width: 6px;
    max-width: 6px;
    min-height: 6px;
    max-height: 6px;
}}
"""
