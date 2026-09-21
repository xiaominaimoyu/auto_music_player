"""演奏控制页(对齐 Web 设计稿):选谱 -> BPM -> 开始/停止/重置 -> 进度。

三态控制:开始(或暂停后"继续演奏") / 停止(= 暂停,进度保留) / 重置(仅停止后可用,进度归零)。
"""

import random
import threading

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from core import ir as ir_mod
from core.compiler import compile_score
from core.event_logger import get_event_logger
from core.humanize import HumanizeParams, plan_timings
from core.practice import PracticeSession, build_dual_rail_hint, build_practice_cues
from core.preview_player import PreviewPlayer
from core.profile import resolve_profile
from core.scenario import SCENARIOS, get_scenario
from core.transport import MIN_GAME_NOTE_MS, normalize_transport_preferences, prepare_score
from core.window_monitor import FocusLockPolicy, ForegroundWatcher
from core.windows_reliability import inspect_target_elevation
from gui.theme import BRAND, INK_2, INK_3, STATE_INFO
from gui.widgets import AppDialog


_WINDOW_PROFILE_HINTS = {
    "delta_force_harmonica": ("三角洲行动", "delta force"),
    "default": ("鸣潮", "wuthering waves", "原神", "genshin impact"),
}


def suggested_profile_for_window(title: str) -> str | None:
    """Return an unambiguous profile hint from a foreground game title."""

    normalized = str(title or "").casefold()
    for profile_id, aliases in _WINDOW_PROFILE_HINTS.items():
        if any(alias.casefold() in normalized for alias in aliases):
            return profile_id
    return None


def _is_midi_score(score: dict) -> bool:
    source_file = str(score.get("source_file") or "").lower()
    return score.get("source_type") == "import" and source_file.endswith((".mid", ".midi"))


def build_event_plan(
    notes,
    profile,
    scenario_id,
    *,
    bpm,
    settle_ms,
    release_settle_ms,
    hold_ratio,
    gap_ms,
    humanize=None,
    source_index_offset=0,
):
    """事件路径(三角洲档位):谱面 → (ScenarioPlan, 降级清单)。

    纯函数,不依赖 GUI/游戏/时钟,便于单测。音符层与编译逻辑不 fork——
    复用 M2 的 IR/编译器与 M3 的档位,差异只由 scenario 收敛到会话层。
    humanize:HumanizeParams 或 None;仅自由演奏等 humanize 类场景生效
    (识别类场景由 scenario.humanize 决定,见 core/scenario.py)。
    """
    scenario = get_scenario(scenario_id)
    semitones = [n.get("semitone", 0) for n in notes]
    elements = ir_mod.from_storage(notes, semitones=semitones)
    params = profile.build_compile_params(
        bpm=bpm,
        settle_ms=settle_ms,
        release_settle_ms=release_settle_ms,
        hold_ratio=hold_ratio,
        max_hold_ms=None,
        gap_ms=gap_ms,
    )
    params = scenario.apply_intervals(params)
    timings = None
    if humanize is not None and scenario.humanize:
        timings = plan_timings(notes, humanize, random.Random())
    result = compile_score(
        elements,
        params,
        timings=timings,
        source_index_offset=source_index_offset,
    )
    return scenario.plan(result.events), result.degradations


def _update_active_profile(path: str, profile_id: str):
    """手术式更新 config.yaml 中 app 段的 active_profile,保留注释与编码。

    与 gui/calibration_dialog.update_player_config_value 同款策略,
    但作用于 app: 段。
    """
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()
    out, in_app, done = [], False, False
    for line in lines:
        s = line.rstrip("\n")
        top = bool(s) and not s[0].isspace()
        if top:
            in_app = s.startswith("app:")
        if in_app and not done and s.strip().startswith("active_profile:"):
            indent = s[: len(s) - len(s.lstrip())]
            out.append(f"{indent}active_profile: {profile_id}\n")
            done = True
            continue
        out.append(line)
    if not done:
        for idx, line in enumerate(out):
            if line.rstrip("\n").startswith("app:"):
                out.insert(idx + 1, f"  active_profile: {profile_id}\n")
                done = True
                break
    if not done:
        out.append("app:\n")
        out.append(f"  active_profile: {profile_id}\n")
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(out)


class PlayerTab(QWidget):
    practice_input = pyqtSignal(object, object)
    practice_stop_requested = pyqtSignal()

    def __init__(
        self,
        db,
        player,
        player_cfg,
        config_path=None,
        profile=None,
        profiles=None,
        event_player=None,
        preview_player=None,
    ):
        super().__init__()
        self._db = db
        self._player = player
        self._config_path = config_path
        # 游戏档位(M3):当前激活档位 + 全部候选档位 + 事件演奏器(M4)
        self._profile = profile
        self._profiles = list(profiles or [])
        self._event_player = event_player
        self._preview_player = (
            preview_player if preview_player is not None else PreviewPlayer(self)
        )
        self._preview_active = False
        self._preview_error = False
        self._hold_ratio = float(player_cfg.get("hold_ratio", 0.75))
        self._gap_ms = float(player_cfg.get("gap_ms", 20))
        self._settle_ms = float(player_cfg.get("modifier_settle_ms", 30))
        self._release_settle_ms = float(player_cfg.get("modifier_release_ms", 20))
        # 真人化节奏参数:从配置读取,供事件路径使用
        humanize_cfg = player_cfg.get("humanize") or {}
        humanize_enabled = bool(humanize_cfg.get("enabled", True))
        self._humanize_params = None
        if humanize_enabled:
            self._humanize_params = HumanizeParams(
                jitter_ms=float(humanize_cfg.get("jitter_ms", 12.0)),
                breath_ms=float(humanize_cfg.get("breath_ms", 25.0)),
            )
        self._score_id = None
        self._had_error = False
        self._paused_done = 0
        self._paused_total = 0
        self._event_degradation_count = 0
        self._transport_degradation_count = 0
        self._transport_degradations = []
        self._loading_preferences = False
        self._active_notes = []
        self._active_source_start = 0
        self._active_source_type = ""
        self._playback_gap_ms = self._gap_ms
        self._playback_humanize = self._humanize_params
        self._practice_active = False
        self._practice_notes = []
        self._practice_index = 0
        self._practice_bpm = 100
        self._practice_session = None
        self._practice_keyboard_listener = None
        self._practice_mouse_listener = None
        self._practice_keys = set()
        self._practice_mouse = set()
        self._practice_lock = threading.Lock()
        practice_cfg = player_cfg.get("practice_input") or {}
        self._practice_input_enabled = bool(practice_cfg.get("enabled", True))
        # 焦点检测:丢失目标窗口焦点时自动暂停,恢复后由用户选择续播或从头
        focus_cfg = player_cfg.get("focus_check") or {}
        self._focus_enabled = bool(focus_cfg.get("enabled", True))
        self._poll_interval_ms = max(100, int(focus_cfg.get("poll_interval_ms", 400)))
        self._target_title = str(focus_cfg.get("target_window_title", "") or "")
        self._watcher = ForegroundWatcher(poll_interval=self._poll_interval_ms / 1000.0)
        self._policy = None
        self._guard_target = None
        self._target_elevation = None
        self._mini_mode = False
        self._focus_lost = False
        self._build_ui()
        self._focus_timer = QTimer(self)
        self._focus_timer.setInterval(self._poll_interval_ms)
        self._focus_timer.timeout.connect(self._check_focus)
        self._practice_timer = QTimer(self)
        self._practice_timer.setSingleShot(True)
        self._practice_timer.timeout.connect(self._practice_tick)
        self.practice_input.connect(self._on_practice_input)
        self.practice_stop_requested.connect(self._stop)
        if hasattr(self._player, "set_dispatch_guard"):
            self._player.set_dispatch_guard(self._dispatch_allowed)
        self._player.progress.connect(self._on_progress)
        self._player.finished.connect(self._on_finished)
        self._player.paused.connect(self._on_paused)
        self._player.error_occurred.connect(self._on_error)
        # 事件演奏器(M4)的信号接到同一组 UI 处理(任意时刻只有一条路径在跑)
        if self._event_player is not None:
            if hasattr(self._event_player, "set_dispatch_guard"):
                self._event_player.set_dispatch_guard(self._dispatch_allowed)
            self._event_player.progress.connect(self._on_progress)
            self._event_player.finished.connect(self._on_finished)
            self._event_player.paused.connect(self._on_paused)
            self._event_player.aborted.connect(self._on_aborted)
            self._event_player.error_occurred.connect(self._on_error)
        self._preview_player.finished.connect(self._on_preview_finished)
        self._preview_player.error_occurred.connect(self._on_preview_error)

    def _card(self, title):
        card = QFrame()
        card.setObjectName("SectionCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(14)
        t = QLabel(title)
        t.setObjectName("SectionTitle")
        layout.addWidget(t)
        return card, layout

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # 使用 QScrollArea 包裹所有内容,避免窗口缩小时重叠
        from PyQt6.QtWidgets import QScrollArea

        scroll = QScrollArea()
        scroll.setObjectName("ContentScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        inner = QWidget()
        inner.setObjectName("ContentInner")
        inner_layout = QVBoxLayout(inner)
        inner_layout.setContentsMargins(32, 24, 32, 24)
        inner_layout.setSpacing(16)
        scroll.setWidget(inner)
        root.addWidget(scroll)

        title = QLabel("演奏控制")
        title.setObjectName("PageTitle")
        self.state_label = QLabel("就绪")
        self.state_label.setObjectName("PageSub")
        header = QHBoxLayout()
        header.addWidget(title)
        header.addWidget(self.state_label)
        header.addStretch(1)
        inner_layout.addLayout(header)

        # 游戏档位(仅当注入了档位列表时显示)
        if self._profiles:
            card, lay = self._card("游戏档位")
            self.profile_combo = QComboBox()
            for p in self._profiles:
                self.profile_combo.addItem(f"{p.group} · {p.name}", p.id)
            if self._profile is not None:
                for i in range(self.profile_combo.count()):
                    if self.profile_combo.itemData(i) == self._profile.id:
                        self.profile_combo.setCurrentIndex(i)
                        break
            self.profile_combo.currentIndexChanged.connect(self._on_profile_change)
            lay.addWidget(self.profile_combo)

            self.scenario_label = QLabel("场景")
            self.scenario_label.setObjectName("FieldLabel")
            lay.addWidget(self.scenario_label)
            self.scenario_combo = QComboBox()
            for sid, cls in SCENARIOS.items():
                self.scenario_combo.addItem(getattr(cls, "name", sid), sid)
            lay.addWidget(self.scenario_combo)
            # 场景下拉仅在事件路径(三角洲档位)时可见
            self._update_scenario_visibility()
            inner_layout.addWidget(card)

        card, lay = self._card("演奏模式")
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("正式演奏", "perform")
        self.mode_combo.addItem("练习模式", "practice")
        self.mode_combo.currentIndexChanged.connect(self._on_mode_change)
        lay.addWidget(self.mode_combo)
        self.practice_strict_check = QCheckBox("严格校验鼠标修饰键")
        self.practice_strict_check.setChecked(False)
        self.practice_strict_check.setToolTip(
            "关闭后只校验音键；练习模式始终不会调用自动演奏输入驱动"
        )
        self.practice_strict_check.setVisible(False)
        self.practice_strict_check.toggled.connect(self._save_transport_preferences)
        lay.addWidget(self.practice_strict_check)
        inner_layout.addWidget(card)

        # 选择乐谱
        card, lay = self._card("选择乐谱")
        label = QLabel("乐谱")
        label.setObjectName("FieldLabel")
        lay.addWidget(label)
        self.combo = QComboBox()
        self.combo.currentIndexChanged.connect(self._on_select)
        lay.addWidget(self.combo)
        inner_layout.addWidget(card)

        # 演奏速度 + 信息卡
        card, lay = self._card("演奏速度")
        bpm_row = QHBoxLayout()
        bpm_row.setSpacing(16)
        self.bpm_spin = QSpinBox()
        self.bpm_spin.setRange(30, 300)
        self.bpm_spin.setValue(100)
        self.bpm_spin.setMinimumWidth(90)
        self.bpm_spin.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.bpm_slider = QSlider(Qt.Orientation.Horizontal)
        self.bpm_slider.setRange(30, 300)
        self.bpm_slider.setValue(100)
        self.bpm_spin.valueChanged.connect(self.bpm_slider.setValue)
        self.bpm_slider.valueChanged.connect(self.bpm_spin.setValue)
        bpm_row.addWidget(self.bpm_spin)
        bpm_row.addWidget(self.bpm_slider, 1)
        unit = QLabel("拍/分钟")
        unit.setObjectName("HintText")
        bpm_row.addWidget(unit)
        lay.addLayout(bpm_row)

        transport_row = QHBoxLayout()
        transport_row.setSpacing(10)
        transpose_label = QLabel("移调")
        transpose_label.setObjectName("FieldLabel")
        self.transpose_spin = QSpinBox()
        self.transpose_spin.setObjectName("TransposeSpin")
        self.transpose_spin.setRange(-24, 24)
        self.transpose_spin.setSuffix(" 半音")
        self.transpose_spin.setToolTip("仅事件档位支持半音移调；超出范围时按八度折回并报告")
        segment_label = QLabel("播放片段")
        segment_label.setObjectName("FieldLabel")
        self.segment_start_spin = QSpinBox()
        self.segment_start_spin.setObjectName("SegmentStartSpin")
        self.segment_start_spin.setRange(1, 1)
        self.segment_start_spin.setPrefix("第 ")
        self.segment_start_spin.setSuffix(" 项")
        separator = QLabel("至")
        separator.setObjectName("HintText")
        self.segment_end_spin = QSpinBox()
        self.segment_end_spin.setObjectName("SegmentEndSpin")
        self.segment_end_spin.setRange(1, 1)
        self.segment_end_spin.setPrefix("第 ")
        self.segment_end_spin.setSuffix(" 项")
        self.bpm_spin.valueChanged.connect(self._on_transport_changed)
        self.transpose_spin.valueChanged.connect(self._on_transport_changed)
        self.segment_start_spin.valueChanged.connect(self._on_segment_changed)
        self.segment_end_spin.valueChanged.connect(self._on_segment_changed)
        segment_reset = QPushButton("全曲")
        segment_reset.setObjectName("BtnSecondary")
        segment_reset.clicked.connect(self._reset_segment)
        transport_row.addWidget(transpose_label)
        transport_row.addWidget(self.transpose_spin)
        transport_row.addSpacing(12)
        transport_row.addWidget(segment_label)
        transport_row.addWidget(self.segment_start_spin)
        transport_row.addWidget(separator)
        transport_row.addWidget(self.segment_end_spin)
        transport_row.addWidget(segment_reset)
        transport_row.addStretch(1)
        lay.addLayout(transport_row)

        info = QFrame()
        info.setStyleSheet(f"background: #1E1E28; border-radius: 8px;")
        info_layout = QHBoxLayout(info)
        info_layout.setContentsMargins(16, 12, 16, 12)
        info_layout.setSpacing(24)
        self.info_name = self._info_item(info_layout, "乐谱名称", "-")
        self.info_count = self._info_item(info_layout, "音符数", "-")
        self.info_duration = self._info_item(info_layout, "预计时长", "-")
        lay.addWidget(info)
        inner_layout.addWidget(card)

        # 控制
        card, lay = self._card("演奏控制")
        ctrl_row = QHBoxLayout()
        self.play_btn = QPushButton("开始演奏")
        self.play_btn.setObjectName("BtnPrimary")
        self.play_btn.setMinimumWidth(120)
        self.play_btn.setEnabled(False)
        self.play_btn.clicked.connect(self._play)
        self.preview_btn = QPushButton("试听当前乐谱")
        self.preview_btn.setObjectName("BtnSecondary")
        self.preview_btn.setMinimumWidth(120)
        self.preview_btn.setEnabled(False)
        self.preview_btn.setToolTip(
            "使用系统大钢琴音色试听当前乐谱,不会向游戏发送键盘或鼠标按键"
        )
        self.preview_btn.clicked.connect(self._toggle_preview)
        self.stop_btn = QPushButton("停止")
        self.stop_btn.setObjectName("BtnStop")
        self.stop_btn.setMinimumWidth(80)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop)
        self.reset_btn = QPushButton("重置")
        self.reset_btn.setObjectName("BtnSecondary")
        self.reset_btn.setMinimumWidth(80)
        self.reset_btn.setEnabled(False)
        self.reset_btn.setToolTip("仅在停止(暂停)后可用:清空演奏进度,下次从头开始")
        self.reset_btn.clicked.connect(self._reset)
        self.calib_btn = QPushButton("延迟校准")
        self.calib_btn.setObjectName("BtnSecondary")
        self.calib_btn.setToolTip("测量本机输出延迟并计算补偿值,消除长曲演奏的节奏漂移")
        self.calib_btn.clicked.connect(self._open_calibration)
        ctrl_row.addWidget(self.play_btn)
        ctrl_row.addWidget(self.preview_btn)
        ctrl_row.addWidget(self.stop_btn)
        ctrl_row.addWidget(self.reset_btn)
        ctrl_row.addWidget(self.calib_btn)
        ctrl_row.addStretch(1)
        lay.addLayout(ctrl_row)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        lay.addWidget(self.progress_bar)
        self.seek_slider = QSlider(Qt.Orientation.Horizontal)
        self.seek_slider.setObjectName("PlaybackSeekSlider")
        self.seek_slider.setRange(0, 0)
        self.seek_slider.setTracking(False)
        self.seek_slider.setToolTip("暂停时拖动，下次从片段内所选乐谱项开始")
        self.seek_slider.sliderReleased.connect(self._on_seek_released)
        lay.addWidget(self.seek_slider)
        pos_row = QHBoxLayout()
        self.pos_label = QLabel("0 / 0")
        self.pos_label.setStyleSheet(
            f"font-family: Consolas; color: {BRAND}; font-weight: 600; font-size: 14px;"
        )
        pos_row.addWidget(self.pos_label)
        pos_row.addStretch(1)
        self.progress_state = QLabel("就绪")
        self.progress_state.setObjectName("HintText")
        pos_row.addWidget(self.progress_state)
        lay.addLayout(pos_row)

        hint = QLabel(
            "演奏时请将焦点切到游戏窗口 · 失去焦点将自动暂停 · 按 F8 可随时暂停演奏"
        )
        hint.setFixedHeight(40)
        hint.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        hint.setStyleSheet(
            f"color: #EDEDF2; font-size: 13px; padding: 0 14px; "
            f"background: #1E1E28; border-radius: 8px; border: 1px solid #26262F;"
        )
        lay.addWidget(hint)
        self.preview_status = QLabel(
            "试听使用系统大钢琴音色，不会操作游戏；建议确认旋律后再开始演奏。"
        )
        self.preview_status.setObjectName("HintText")
        self.preview_status.setWordWrap(True)
        lay.addWidget(self.preview_status)
        inner_layout.addWidget(card)

        card, lay = self._card("双轨提示")
        self.score_rail_label = QLabel("乐谱轨: 请选择乐谱")
        self.score_rail_label.setObjectName("HintText")
        self.score_rail_label.setWordWrap(True)
        self.input_rail_label = QLabel("输入轨: 请选择乐谱")
        self.input_rail_label.setObjectName("HintText")
        self.input_rail_label.setWordWrap(True)
        self.degradation_rail_label = QLabel("")
        self.degradation_rail_label.setObjectName("HintText")
        self.degradation_rail_label.setWordWrap(True)
        lay.addWidget(self.score_rail_label)
        lay.addWidget(self.input_rail_label)
        lay.addWidget(self.degradation_rail_label)
        inner_layout.addWidget(card)
        inner_layout.addStretch(1)

    def _info_item(self, layout, label_text, value_text):
        col = QVBoxLayout()
        col.setSpacing(2)
        label = QLabel(label_text)
        label.setStyleSheet(f"font-size: 12px; color: {INK_3};")
        value = QLabel(value_text)
        value.setStyleSheet(
            f"font-size: 15px; font-weight: 600; color: #EDEDF2; font-family: Consolas;"
        )
        col.addWidget(label)
        col.addWidget(value)
        layout.addLayout(col)
        return value

    def _profile_preference_id(self):
        return str(getattr(self._profile, "id", "legacy") or "legacy")

    def _segment_bounds(self):
        if not hasattr(self, "segment_start_spin"):
            return 0, 0
        return self.segment_start_spin.value() - 1, self.segment_end_spin.value()

    def _prepare_current_score(self, score=None):
        score = score or (
            self._db.get_score(self._score_id) if self._score_id is not None else None
        )
        if not score:
            return None
        start, end = self._segment_bounds()
        transpose = self.transpose_spin.value() if self._use_event_path() else 0
        imported_timeline = _is_midi_score(score)
        return prepare_score(
            score["notes"],
            start_index=start,
            end_index=end,
            transpose=transpose,
            fold_octaves=True,
            bpm=self.bpm_spin.value(),
            min_playable_ms=MIN_GAME_NOTE_MS if imported_timeline else 0.0,
        )

    def _load_transport_preferences(self, score):
        total = len(score["notes"])
        raw = {}
        if hasattr(self._db, "get_score_preferences"):
            raw = self._db.get_score_preferences(
                self._score_id, self._profile_preference_id()
            )
        defaults = {
            "bpm": score["bpm_default"],
            "transpose": 0,
            "segment": [0, total],
        }
        merged = dict(defaults)
        merged.update(raw or {})
        settings = normalize_transport_preferences(merged, total)

        self._loading_preferences = True
        widgets = (
            self.bpm_spin,
            self.bpm_slider,
            self.transpose_spin,
            self.segment_start_spin,
            self.segment_end_spin,
            self.mode_combo,
            self.practice_strict_check,
        )
        for widget in widgets:
            widget.blockSignals(True)
        try:
            self.segment_start_spin.setRange(1, max(1, total))
            self.segment_end_spin.setRange(1, max(1, total))
            self.bpm_spin.setValue(settings["bpm"])
            self.bpm_slider.setValue(settings["bpm"])
            self.transpose_spin.setValue(
                settings["transpose"] if self._use_event_path() else 0
            )
            self.segment_start_spin.setValue(settings["segment"][0] + 1)
            self.segment_end_spin.setValue(max(1, settings["segment"][1]))
            mode = raw.get("mode") if isinstance(raw, dict) else None
            if mode in ("perform", "practice"):
                index = self.mode_combo.findData(mode)
                if index >= 0:
                    self.mode_combo.setCurrentIndex(index)
            self.practice_strict_check.setChecked(
                bool((raw or {}).get("practice_strict", False))
            )
        finally:
            for widget in widgets:
                widget.blockSignals(False)
            self._loading_preferences = False
        self._update_transport_capabilities()
        self._refresh_transport_view(reset_progress=True)

    def _save_transport_preferences(self, *_args):
        if self._loading_preferences or self._score_id is None:
            return
        if not hasattr(self._db, "save_score_preferences"):
            return
        start, end = self._segment_bounds()
        settings = {
            "version": 1,
            "bpm": self.bpm_spin.value(),
            "transpose": self.transpose_spin.value() if self._use_event_path() else 0,
            "segment": [start, end],
            "mode": self._current_mode(),
            "practice_strict": self.practice_strict_check.isChecked(),
        }
        self._db.save_score_preferences(
            self._score_id, self._profile_preference_id(), settings
        )

    def _update_transport_capabilities(self):
        event_path = self._use_event_path()
        self.transpose_spin.setEnabled(event_path and not self._real_playback_active())
        if not event_path and self.transpose_spin.value() != 0:
            self.transpose_spin.blockSignals(True)
            self.transpose_spin.setValue(0)
            self.transpose_spin.blockSignals(False)
        self.practice_strict_check.setVisible(
            self._practice_mode() and self._use_event_path()
        )

    def _set_transport_enabled(self, enabled):
        self.bpm_spin.setEnabled(enabled)
        self.bpm_slider.setEnabled(enabled)
        self.segment_start_spin.setEnabled(enabled)
        self.segment_end_spin.setEnabled(enabled)
        self.transpose_spin.setEnabled(enabled and self._use_event_path())

    def _refresh_transport_view(self, *, reset_progress):
        if self._score_id is None:
            return
        score = self._db.get_score(self._score_id)
        if not score:
            return
        try:
            prepared = self._prepare_current_score(score)
        except ValueError as exc:
            self.progress_state.setText(f"播放参数无效: {exc}")
            return
        self._active_notes = list(prepared.notes)
        self._active_source_start = prepared.source_start
        self._active_source_type = "midi" if _is_midi_score(score) else (score.get("source_type") or "")
        self._transport_degradation_count = len(prepared.degradations)
        self._transport_degradations = list(prepared.degradations)
        self.info_count.setText(
            str(len(prepared.notes))
            if len(prepared.notes) == len(score["notes"])
            else f"{len(prepared.notes)} / {len(score['notes'])}"
        )
        self.info_duration.setText(
            f"{self._estimate_seconds(prepared.notes, self.bpm_spin.value())} 秒"
        )
        self.seek_slider.setRange(0, len(prepared.notes))
        if reset_progress:
            self._paused_done = 0
            self._paused_total = len(prepared.notes)
            self.seek_slider.setValue(0)
            self.progress_bar.setRange(0, max(1, len(prepared.notes)))
            self.progress_bar.setValue(0)
            self.pos_label.setText(f"0 / {len(prepared.notes)}")
            self.reset_btn.setEnabled(False)
        self._set_dual_hint(self._paused_done)

    def _on_segment_changed(self, _value=None):
        if self._loading_preferences:
            return
        sender = self.sender()
        start = self.segment_start_spin.value()
        end = self.segment_end_spin.value()
        if start > end:
            if sender is self.segment_start_spin:
                self.segment_end_spin.setValue(start)
            else:
                self.segment_start_spin.setValue(end)
        self._clear_pause()
        self._refresh_transport_view(reset_progress=True)
        self._save_transport_preferences()

    def _on_transport_changed(self, _value=None):
        if self._loading_preferences or self._score_id is None:
            return
        self._clear_pause()
        self._refresh_transport_view(reset_progress=True)
        self._save_transport_preferences()

    def _reset_segment(self):
        if self._score_id is None:
            return
        score = self._db.get_score(self._score_id)
        if not score:
            return
        total = max(1, len(score["notes"]))
        self._loading_preferences = True
        try:
            self.segment_start_spin.setValue(1)
            self.segment_end_spin.setValue(total)
        finally:
            self._loading_preferences = False
        self._clear_pause()
        self._refresh_transport_view(reset_progress=True)
        self._save_transport_preferences()

    def _on_seek_released(self):
        if self._real_playback_active():
            self.seek_slider.setValue(self._paused_done)
            return
        done = self.seek_slider.value()
        total = len(self._active_notes)
        self._paused_done = max(0, min(done, total))
        self._paused_total = total
        self.progress_bar.setRange(0, max(1, total))
        self.progress_bar.setValue(self._paused_done)
        self.pos_label.setText(f"{self._paused_done} / {total}")
        self.play_btn.setText(
            ("继续练习" if self._practice_mode() else "继续演奏")
            if self._paused_done > 0
            else ("开始练习" if self._practice_mode() else "开始演奏")
        )
        self.reset_btn.setEnabled(self._paused_done > 0)
        self.progress_state.setText(f"已定位到 {self._paused_done} / {total}")
        self._set_dual_hint(self._paused_done)

    def _current_mode(self):
        if not hasattr(self, "mode_combo"):
            return "perform"
        return self.mode_combo.currentData() or "perform"

    def _practice_mode(self):
        return self._current_mode() == "practice"

    def _on_mode_change(self):
        self._stop_preview()
        self._clear_pause()
        self._update_transport_capabilities()
        self._save_transport_preferences()
        self._set_dual_hint(0)

    def _set_dual_hint(self, index):
        if not hasattr(self, "score_rail_label"):
            return
        if self._score_id is None:
            self.score_rail_label.setText("乐谱轨: 请选择乐谱")
            self.input_rail_label.setText("输入轨: 请选择乐谱")
            self.degradation_rail_label.setText("")
            return
        score = self._db.get_score(self._score_id)
        if not score:
            self.score_rail_label.setText("乐谱轨: 请选择乐谱")
            self.input_rail_label.setText("输入轨: 请选择乐谱")
            self.degradation_rail_label.setText("")
            return
        notes = self._active_notes or list(score["notes"])
        try:
            hint = build_dual_rail_hint(
                notes,
                0 if index is None else index,
                keymap=self._player.keymap,
                profile=self._profile,
                use_event_path=self._use_event_path(),
                bpm=self.bpm_spin.value(),
                settle_ms=self._settle_ms,
                release_settle_ms=self._release_settle_ms,
                hold_ratio=self._hold_ratio,
                gap_ms=self._gap_ms,
            )
        except Exception as e:
            self.score_rail_label.setText("乐谱轨: 无法生成提示")
            self.input_rail_label.setText(f"输入轨: {e}")
            self.degradation_rail_label.setText("")
            return
        self.score_rail_label.setText(hint.score_text)
        self.input_rail_label.setText(hint.input_text)
        details = []
        if self._transport_degradation_count:
            details.append(f"播放控制转换 {self._transport_degradation_count} 处")
        if hint.degradation_text:
            details.append(hint.degradation_text)
        self.degradation_rail_label.setText(
            f"降级: {'; '.join(details)}" if details else ""
        )

    @staticmethod
    def _practice_key_name(key):
        char = getattr(key, "char", None)
        if char:
            return char.upper() if len(char) == 1 else str(char).lower()
        name = getattr(key, "name", None)
        if name in {
            "shift", "shift_l", "shift_r", "ctrl", "ctrl_l", "ctrl_r",
            "alt", "alt_l", "alt_r", "cmd", "cmd_l", "cmd_r",
        }:
            return None
        return str(name).lower() if name else None

    def _start_practice_listeners(self):
        self._stop_practice_listeners()
        if not self._practice_input_enabled:
            return False
        try:
            from pynput import keyboard as pk
            from pynput import mouse as pm

            def on_press(key):
                token = self._practice_key_name(key)
                if token == "esc":
                    self.practice_stop_requested.emit()
                    return False
                if not token:
                    return
                with self._practice_lock:
                    if token in self._practice_keys:
                        return
                    self._practice_keys.add(token)
                    keyboard = tuple(self._practice_keys)
                    mouse = tuple(self._practice_mouse)
                self.practice_input.emit(keyboard, mouse)

            def on_release(key):
                token = self._practice_key_name(key)
                if token:
                    with self._practice_lock:
                        self._practice_keys.discard(token)

            def on_click(_x, _y, button, pressed):
                token = str(getattr(button, "name", button)).lower()
                with self._practice_lock:
                    if pressed:
                        self._practice_mouse.add(token)
                    else:
                        self._practice_mouse.discard(token)

            self._practice_keyboard_listener = pk.Listener(
                on_press=on_press, on_release=on_release
            )
            self._practice_mouse_listener = pm.Listener(on_click=on_click)
            self._practice_keyboard_listener.start()
            self._practice_mouse_listener.start()
            return True
        except Exception as exc:
            self._stop_practice_listeners()
            self.progress_state.setText(f"练习监听启动失败: {exc}")
            return False

    def _stop_practice_listeners(self):
        for attr in ("_practice_keyboard_listener", "_practice_mouse_listener"):
            listener = getattr(self, attr, None)
            if listener is not None:
                try:
                    listener.stop()
                except Exception:
                    pass
            setattr(self, attr, None)
        with self._practice_lock:
            self._practice_keys.clear()
            self._practice_mouse.clear()

    def _start_practice(self, score, start_index):
        notes = list(score["notes"])
        cues = build_practice_cues(
            notes,
            keymap=self._player.keymap,
            profile=self._profile,
            use_event_path=self._use_event_path(),
            bpm=self.bpm_spin.value(),
            settle_ms=self._settle_ms,
            release_settle_ms=self._release_settle_ms,
            hold_ratio=self._hold_ratio,
            gap_ms=self._gap_ms,
        )
        self._practice_notes = notes
        self._practice_bpm = self.bpm_spin.value()
        self._practice_session = PracticeSession(cues, start_index)
        self._practice_index = self._practice_session.index
        if self._practice_session.completed:
            self._finish_practice()
            return
        self._practice_active = True
        self._had_error = False
        self.mode_combo.setEnabled(False)
        self.play_btn.setEnabled(False)
        self.play_btn.setText("练习中")
        self.preview_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.reset_btn.setEnabled(False)
        self._set_transport_enabled(False)
        self.state_label.setText(f"练习中 BPM {self._practice_bpm}")
        listening = self._start_practice_listeners()
        if listening:
            self.progress_state.setText("请按输入轨提示操作；按对后才会进入下一项")
        elif not self._practice_input_enabled:
            self.progress_state.setText("练习输入监听已禁用；不会发送任何键鼠输入")
        self._on_progress(self._practice_index, len(self._practice_notes))

    def _on_practice_input(self, keyboard, mouse):
        session = self._practice_session
        if not self._practice_active or session is None:
            return
        result = session.submit(
            keyboard,
            mouse,
            strict_modifiers=self.practice_strict_check.isChecked(),
        )
        self._practice_index = result.index
        self._paused_done = result.index
        if result.correct:
            self._on_progress(result.index, result.total)
            if result.completed:
                self._finish_practice()
                return
            self.progress_state.setText(
                f"命中 {result.hits} · 错误 {result.misses} · 继续按当前提示"
            )
        else:
            self.progress_state.setText(
                f"未命中 · 正确 {result.hits} · 错误 {result.misses} · 仍停留在当前项"
            )

    def _practice_tick(self):
        """Compatibility test hook: submit exactly the currently expected input.

        No timer calls this method in normal operation; real practice advances
        only through the read-only global input listener above.
        """
        session = self._practice_session
        cue = session.current if session is not None else None
        if self._practice_active and cue is not None:
            self._on_practice_input(tuple(cue.keyboard), tuple(cue.mouse))

    def _pause_practice(self):
        if not self._practice_active:
            return
        self._practice_timer.stop()
        self._stop_practice_listeners()
        self._practice_active = False
        self._paused_done = self._practice_index
        self._paused_total = len(self._practice_notes)
        self.mode_combo.setEnabled(True)
        self.play_btn.setEnabled(True)
        self.play_btn.setText("继续练习" if self._paused_done > 0 else "开始练习")
        self.stop_btn.setEnabled(False)
        self.reset_btn.setEnabled(True)
        self._set_transport_enabled(True)
        self.state_label.setText("练习已暂停")
        self.progress_state.setText(
            f"练习暂停于 {self._paused_done} / {self._paused_total}"
        )
        self._update_preview_button()

    def _finish_practice(self):
        self._practice_timer.stop()
        self._stop_practice_listeners()
        total = len(self._practice_notes)
        hits = self._practice_session.hits if self._practice_session is not None else 0
        misses = self._practice_session.misses if self._practice_session is not None else 0
        self._practice_active = False
        self._paused_done = 0
        self._paused_total = 0
        self.mode_combo.setEnabled(True)
        self.play_btn.setEnabled(True)
        self.play_btn.setText("开始练习" if self._practice_mode() else "开始演奏")
        self.stop_btn.setEnabled(False)
        self.reset_btn.setEnabled(False)
        self._set_transport_enabled(True)
        self.state_label.setText("就绪")
        self.progress_state.setText(f"练习完成 · 命中 {hits} · 错误 {misses}")
        self._on_progress(total, total)
        self._update_preview_button()

    def refresh(self):
        self._clear_pause()
        self.combo.blockSignals(True)
        self.combo.clear()
        for s in self._db.list_scores():
            self.combo.addItem(f"{s['name']}  (BPM {s['bpm_default']})", s["id"])
        self.combo.setCurrentIndex(-1)
        self.combo.blockSignals(False)
        if self.combo.count() > 0:
            self.combo.setCurrentIndex(0)
        else:
            self._score_id = None
            self._active_notes = []
            self.info_name.setText("-")
            self.info_count.setText("-")
            self.info_duration.setText("-")
            self.play_btn.setEnabled(False)
            self._update_preview_button()
            self._set_dual_hint(None)

    def select_score(self, score_id: int):
        for i in range(self.combo.count()):
            if self.combo.itemData(i) == score_id:
                self.combo.setCurrentIndex(i)
                return

    def _on_select(self):
        self._stop_preview()
        self._score_id = self.combo.currentData()
        self._clear_pause()
        if self._score_id is None:
            self.play_btn.setEnabled(False)
            self._update_preview_button()
            return
        score = self._db.get_score(self._score_id)
        if score is None:
            return
        self.info_name.setText(score["name"])
        self._load_transport_preferences(score)
        self.play_btn.setEnabled(not self._player.is_playing)
        self._update_preview_button()
        self._set_dual_hint(self._paused_done)

    def _clear_pause(self):
        """回到未开始态:清空暂停进度与按钮状态。"""
        self._paused_done = 0
        self._paused_total = 0
        self._event_degradation_count = 0
        self.play_btn.setText("开始练习" if self._practice_mode() else "开始演奏")
        self.reset_btn.setEnabled(False)

    def _estimate_seconds(self, notes, bpm):
        beat_ms = 60000.0 / max(1, bpm)
        total_ms = sum(n["dur"] * beat_ms for n in notes)
        total_ms += self._gap_ms * len(notes)
        return round(total_ms / 1000.0, 1)

    def _real_playback_active(self):
        return self._practice_active or self._player.is_playing or (
            self._event_player is not None and self._event_player.is_playing
        )

    def _update_preview_button(self):
        if not hasattr(self, "preview_btn"):
            return
        score = self._db.get_score(self._score_id) if self._score_id is not None else None
        has_notes = bool(score and score.get("notes"))
        self.preview_btn.setEnabled(
            (self._preview_active or has_notes) and not self._real_playback_active()
        )

    def _toggle_preview(self):
        if self._preview_active:
            self._stop_preview()
            return
        if self._real_playback_active():
            return
        if self._score_id is None:
            return
        score = self._db.get_score(self._score_id)
        if not score or not score["notes"]:
            AppDialog.show_warning(self, "提示", "该乐谱没有音符数据")
            return
        try:
            prepared = self._prepare_current_score(score)
        except ValueError as exc:
            AppDialog.show_error(self, "试听失败", str(exc))
            return
        if prepared is None or not prepared.notes:
            AppDialog.show_warning(self, "提示", "当前播放片段为空")
            return
        self._preview_error = False
        self._preview_active = True
        self.preview_btn.setText("停止试听")
        self.play_btn.setEnabled(False)
        self.preview_status.setText("大钢琴音色试听中…不会向游戏发送按键")
        try:
            started = self._preview_player.play(
                prepared.notes,
                bpm=self.bpm_spin.value(),
                score_name=score["name"],
            )
        except (TypeError, ValueError, RuntimeError) as e:
            self._preview_active = False
            self.preview_btn.setText("试听当前乐谱")
            self.preview_status.setText(f"试听失败: {e}")
            self.play_btn.setEnabled(not self._real_playback_active())
            self._update_preview_button()
            return
        if not started:
            self._preview_active = False
            self.preview_btn.setText("试听当前乐谱")
            self.preview_status.setText("已有试听正在播放,请先停止后再试")
            self.play_btn.setEnabled(not self._real_playback_active())
            self._update_preview_button()

    def _stop_preview(self):
        if not self._preview_active:
            return
        self._preview_active = False
        self._preview_player.stop()
        self.preview_btn.setText("试听当前乐谱")
        self.preview_status.setText("试听已停止；不会向游戏发送按键")
        self.play_btn.setEnabled(not self._real_playback_active())
        self._update_preview_button()

    def _on_preview_error(self, message):
        if self._preview_active:
            self._preview_error = True
            self.preview_status.setText(f"试听失败: {message}")

    def _on_preview_finished(self, normal):
        if not self._preview_active:
            return
        self._preview_active = False
        self.preview_btn.setText("试听当前乐谱")
        self.play_btn.setEnabled(not self._real_playback_active())
        if self._preview_error:
            self.preview_status.setText("试听失败,请检查系统扬声器设置")
        elif normal:
            self.preview_status.setText("试听完成；不会向游戏发送按键")
        else:
            self.preview_status.setText("试听已停止；不会向游戏发送按键")
        self._update_preview_button()

    def _play(self):
        if self._preview_active:
            return
        if self._score_id is None:
            return
        score = self._db.get_score(self._score_id)
        if not score or not score["notes"]:
            AppDialog.show_warning(self, "提示", "该乐谱没有音符数据")
            return
        try:
            prepared = self._prepare_current_score(score)
        except ValueError as exc:
            AppDialog.show_error(self, "播放参数无效", str(exc))
            return
        if prepared is None or not prepared.notes:
            AppDialog.show_warning(self, "提示", "当前播放片段为空")
            return
        self._active_notes = list(prepared.notes)
        self._active_source_start = prepared.source_start
        self._active_source_type = "midi" if _is_midi_score(score) else (score.get("source_type") or "")
        imported_timeline = self._active_source_type == "midi"
        # MIDI 已按绝对时间线入库；额外 gap 和真人化抖动会逐元素累积，
        # 对短音密集歌曲造成明显改拍。导入谱演奏时保持原始时间线。
        self._playback_gap_ms = 0.0 if imported_timeline else self._gap_ms
        self._playback_humanize = None if imported_timeline else self._humanize_params
        self._transport_degradations = list(prepared.degradations)
        self._transport_degradation_count = len(prepared.degradations)
        if not self._use_event_path():
            legacy_semitones = sum(
                1 for item in prepared.notes if int(item.get("semitone", 0) or 0)
            )
            if legacy_semitones:
                self._transport_degradation_count += legacy_semitones
        start_index = self._paused_done if self._paused_done > 0 else 0
        if self._practice_mode():
            self._start_practice(
                {"notes": prepared.notes, "name": score["name"]}, start_index
            )
            return
        self._pending = (
            prepared.notes,
            self.bpm_spin.value(),
            start_index,
            score["name"],
        )
        self._had_error = False
        self._countdown_left = 3
        self.play_btn.setEnabled(False)
        self.preview_btn.setEnabled(False)
        self.mode_combo.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.reset_btn.setEnabled(False)
        self._set_transport_enabled(False)
        resume_hint = f"从第 {start_index + 1} 项继续" if start_index > 0 else ""
        self.state_label.setText(
            f"3 秒后演奏: 《{score['name']}》 {resume_hint}".strip()
        )
        self.progress_state.setText(
            f"{self._countdown_left} 秒后开始演奏,请切换到游戏窗口..."
        )

        # 记录控制事件
        try:
            logger = get_event_logger()
            action = "resume" if start_index > 0 else "start"
            logger.log_control(
                action=action,
                score_name=score["name"],
                bpm=self.bpm_spin.value(),
                start_index=start_index,
                total_notes=len(prepared.notes),
                source_start=prepared.source_start,
                source_end=prepared.source_end,
                transport_degradation_count=self._transport_degradation_count,
            )
        except Exception:
            pass

        self._countdown_timer = QTimer(self)
        self._countdown_timer.timeout.connect(self._countdown_tick)
        self._countdown_timer.start(1000)

    # ---------- 档位与事件路径(M3/M4/M5) ----------

    def _use_event_path(self) -> bool:
        """是否走事件演奏路径(三角洲档位)。

        判据:有事件演奏器 + 当前档位存在 + 该档位没有 21 键旧模型
        (legacy_keymap 为空 → 只能走编译器 + EventPlayer;默认档位走旧 Player)。
        """
        return (
            self._event_player is not None
            and self._profile is not None
            and self._profile.legacy_keymap is None
        )

    def _update_scenario_visibility(self):
        vis = self._use_event_path()
        self.scenario_label.setVisible(vis)
        self.scenario_combo.setVisible(vis)

    def _on_profile_change(self, index):
        pid = self.profile_combo.itemData(index)
        if pid is None:
            return
        self._profile = resolve_profile(self._profiles, pid)

        # 记录控制事件
        try:
            logger = get_event_logger()
            logger.log_control(
                action="gear_change",
                gear_id=self._profile.id,
                gear_name=self._profile.name,
            )
        except Exception:
            pass

        if self._config_path:
            try:
                _update_active_profile(self._config_path, self._profile.id)
            except Exception as e:
                self.progress_state.setText(f"档位已切换(写回配置失败: {e})")
        self._update_scenario_visibility()
        score = self._db.get_score(self._score_id) if self._score_id is not None else None
        if score:
            self._load_transport_preferences(score)
        else:
            self._update_transport_capabilities()
        self._set_dual_hint(self._paused_done)

    def _play_event(self, notes, bpm, start_index, score_name):
        if start_index >= len(notes):
            self._clear_pause()
            self.play_btn.setEnabled(True)
            self.mode_combo.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.state_label.setText("就绪")
            self.progress_state.setText("演奏完成")
            return
        remaining_notes = notes[start_index:]
        scenario_id = self.scenario_combo.currentData() or "free_play"
        plan, degradations = build_event_plan(
            remaining_notes,
            self._profile,
            scenario_id,
            bpm=bpm,
            settle_ms=self._settle_ms,
            release_settle_ms=self._release_settle_ms,
            hold_ratio=self._hold_ratio,
            gap_ms=self._playback_gap_ms,
            humanize=self._playback_humanize,
            source_index_offset=start_index,
        )
        self._event_degradation_count = len(degradations)
        self._event_player.play(
            plan.events,
            plan.interrupt_mode,
            score_name=score_name,
            source_total=len(notes),
            source_start_index=start_index,
            source_notes=notes,
            bpm=bpm,
            trace_context={
                "profile_id": self._profile.id,
                "profile_name": self._profile.name,
                "scenario_id": scenario_id,
                "source_type": self._active_source_type,
                "gap_ms": self._playback_gap_ms,
                "humanize_enabled": self._playback_humanize is not None,
                "degradation_count": (
                    len(degradations) + self._transport_degradation_count
                ),
            },
        )
        n = len(degradations) + self._transport_degradation_count
        suffix = f" · 已降级 {n} 处" if n else ""
        self.state_label.setText(f"演奏中 BPM {bpm} · {self._profile.name}")
        self.progress_state.setText(f"演奏中...{suffix}")

    def _on_aborted(self, reason):
        """中止(S2):进度清空,不可续播。"""
        self._focus_timer.stop()
        self._paused_done = 0
        self._paused_total = 0
        self._event_degradation_count = 0
        self.play_btn.setEnabled(True)
        self.mode_combo.setEnabled(True)
        self._set_transport_enabled(True)
        self._update_preview_button()
        QTimer.singleShot(0, self._update_preview_button)
        self.play_btn.setText("开始演奏")
        self.stop_btn.setEnabled(False)
        self.reset_btn.setEnabled(False)
        self.state_label.setText("已中止")
        self.progress_state.setText(reason or "演奏已中止")
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.seek_slider.setValue(0)
        self.pos_label.setText("0 / 0")

    def _countdown_tick(self):
        self._countdown_left -= 1
        if self._countdown_left > 0:
            self.progress_state.setText(
                f"{self._countdown_left} 秒后开始演奏,请切换到游戏窗口..."
            )
            return
        self._countdown_timer.stop()
        notes, bpm, start_index, score_name = self._pending
        if not self._preflight_target():
            self.play_btn.setEnabled(True)
            self.mode_combo.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self._set_transport_enabled(True)
            self._update_preview_button()
            return
        self._start_focus_watch()

        # 记录控制事件:倒计时结束,实际开始演奏
        try:
            logger = get_event_logger()
            logger.log_control(
                action="play_started",
                score_name=score_name,
                bpm=bpm,
                start_index=start_index,
                path_type="event" if self._use_event_path() else "legacy",
            )
        except Exception:
            pass

        if self._use_event_path():
            self._play_event(notes, bpm, start_index, score_name)
        else:
            self._player.play(
                notes,
                bpm,
                self._hold_ratio,
                self._playback_gap_ms,
                start_index=start_index,
                score_name=score_name,
                humanize_override=self._playback_humanize,
                trace_context={
                    "path_type": "legacy",
                    "profile_id": getattr(self._profile, "id", "default"),
                    "profile_name": getattr(self._profile, "name", "默认"),
                    "source_type": self._active_source_type,
                    "transport_degradation_count": self._transport_degradation_count,
                },
            )
            self.state_label.setText(f"演奏中 BPM {bpm}")
            suffix = (
                f" · 已降级 {self._transport_degradation_count} 处"
                if self._transport_degradation_count
                else ""
            )
            self.progress_state.setText(f"演奏中...{suffix}")

    def _stop_all(self):
        """统一停止:同时停止 Player 和 EventPlayer(若存在)。

        stop() 对未播放的播放器是幂等空操作,双路径同时调用安全。
        """
        self._player.stop()
        if self._event_player is not None:
            self._event_player.stop()
        self.progress_state.setText("正在暂停...")

    def _stop(self):
        timer = getattr(self, "_countdown_timer", None)
        if timer is not None and timer.isActive():
            timer.stop()
            self.play_btn.setEnabled(True)
            self.mode_combo.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self._set_transport_enabled(True)
            self._update_preview_button()
            self.progress_state.setText("已取消")

            # 记录控制事件
            try:
                logger = get_event_logger()
                logger.log_control(action="cancel_countdown")
            except Exception:
                pass
            return

        if self._practice_active:
            self._pause_practice()
            return

        # 记录控制事件
        try:
            logger = get_event_logger()
            logger.log_control(
                action="pause",
                note_index=self._player.current_index if self._player else -1,
            )
        except Exception:
            pass

        self._stop_all()

    # ---------- 焦点检测 ----------

    def _own_hwnd(self):
        try:
            return int(self.window().winId())
        except Exception:
            return None

    def _preflight_target(self):
        """Lock and inspect the actual foreground target before any key-down."""

        self._guard_target = None
        self._target_elevation = None
        if not self._focus_enabled or not self._watcher.is_available():
            return True
        try:
            current = self._watcher.capture_current()
        except Exception:
            current = None
        if current is None:
            AppDialog.show_error(
                self,
                "无法确认目标窗口",
                "未能读取当前前台窗口。为避免把按键发送到错误程序，本次演奏未启动。",
            )
            self.progress_state.setText("未启动：无法确认目标窗口")
            return False
        title = current.get("title") or ""
        if current.get("hwnd") == self._own_hwnd():
            AppDialog.show_warning(
                self,
                "请切换到游戏窗口",
                "倒计时结束时仍是本程序在前台。为避免误输入，本次演奏未启动。",
            )
            self.progress_state.setText("未启动：倒计时内未切换到游戏窗口")
            return False
        if self._target_title and self._target_title.lower() not in title.lower():
            AppDialog.show_warning(
                self,
                "目标窗口不匹配",
                f"当前前台窗口《{title or '未命名窗口'}》不匹配配置的目标标题。",
            )
            self.progress_state.setText("未启动：目标窗口不匹配")
            return False
        suggested_profile = suggested_profile_for_window(title)
        current_profile = getattr(self._profile, "id", None)
        if suggested_profile and current_profile != suggested_profile:
            expected = next(
                (profile for profile in self._profiles if profile.id == suggested_profile),
                None,
            )
            expected_name = expected.name if expected is not None else suggested_profile
            current_name = getattr(self._profile, "name", current_profile or "未选择")
            AppDialog.show_warning(
                self,
                "游戏档位不匹配",
                f"当前前台窗口《{title}》应使用“{expected_name}”，"
                f"但当前选择的是“{current_name}”。\n"
                "本次已停止发送按键，请切换正确游戏档位后重新开始。",
            )
            self.progress_state.setText("未启动：游戏档位与目标窗口不匹配")
            return False
        assessment = inspect_target_elevation(current.get("hwnd"))
        self._target_elevation = assessment
        if assessment.blocked:
            AppDialog.show_error(self, "Windows 权限不兼容", assessment.message)
            self.progress_state.setText("未启动：目标窗口权限高于本程序")
            return False

        self._guard_target = dict(current)
        self._policy = FocusLockPolicy(
            own_hwnd=self._own_hwnd(), title_override=self._target_title
        )
        self._policy.target = dict(current)
        self._policy.locked = True
        if assessment.status == "unknown":
            self.progress_state.setToolTip(assessment.message)
        return True

    def _dispatch_allowed(self):
        """Worker-thread guard. Key-up/panic paths never call this callback."""

        if not self._focus_enabled or not self._watcher.is_available():
            return True
        target = self._guard_target
        if target is None:
            return False, "尚未锁定目标窗口，拒绝发送输入"
        try:
            current = self._watcher.capture_current()
        except Exception:
            current = None
        if current is None:
            return False, "无法确认当前前台窗口，拒绝发送输入"
        if self._target_title:
            allowed = self._target_title.lower() in (current.get("title") or "").lower()
        else:
            allowed = current.get("hwnd") == target.get("hwnd")
            if not allowed and (target.get("title") or ""):
                allowed = current.get("title") == target.get("title")
        return (
            (True, "")
            if allowed
            else (False, "目标窗口已失去前台焦点，输入已安全中止")
        )

    def _start_focus_watch(self):
        """启动焦点轮询;自动模式下不预设目标,用户切到的首个外部窗口被锁定为目标。

        小窗模式下焦点检测整体禁用(游戏全程保持前台,无切窗即无误停)。
        """
        self._focus_timer.stop()
        self._focus_lost = False
        if not self._focus_enabled or not self._watcher.is_available():
            self._policy = None
            return
        if self._policy is None:
            self._policy = FocusLockPolicy(
                own_hwnd=self._own_hwnd(), title_override=self._target_title
            )
        if not self._mini_mode:
            self._focus_timer.start()

    def _check_focus(self):
        if not self._player.is_playing and (
            self._event_player is None or not self._event_player.is_playing
        ):
            self._focus_timer.stop()
            return
        if self._policy is None:
            self._focus_timer.stop()
            return
        was_locked = self._policy.locked
        current = self._watcher.capture_current()
        if self._policy.evaluate(current):
            self._pause_for_focus()
            return
        if not was_locked and self._policy.locked and self._policy.target:
            self.progress_state.setText(
                f"已锁定目标窗口: {self._policy.target.get('title') or '未命名窗口'}"
            )
            # 记录焦点锁定事件
            try:
                logger = get_event_logger()
                if self._policy.target:
                    logger.log_focus(
                        event="regained",
                        window_title=self._policy.target.get("title", ""),
                        hwnd=self._policy.target.get("hwnd", 0),
                        note_index=self._player.current_index if self._player else -1,
                    )
            except Exception:
                pass

    def _pause_for_focus(self):
        self._focus_timer.stop()
        self._focus_lost = True

        # 记录焦点丢失事件
        try:
            logger = get_event_logger()
            current = self._watcher.capture_current()
            logger.log_focus(
                event="lost",
                window_title=current.get("title", "") if current else "",
                hwnd=current.get("hwnd", 0) if current else 0,
                note_index=self._player.current_index if self._player else -1,
            )
        except Exception:
            pass

        self._stop_all()
        self.progress_state.setText("目标窗口失去焦点,正在暂停...")

    def _prompt_focus_recover(self):
        """焦点恢复选择:断点续播 / 从头开始 / 保持暂停。"""
        choice = AppDialog._popup(
            self,
            "warning",
            "目标窗口失去焦点",
            "演奏已自动暂停,进度已保留。\n切回游戏窗口后,可选择从断点继续或从头开始。",
            [
                ("保持暂停", "secondary"),
                ("从头开始", "secondary"),
                ("断点续播", "primary"),
            ],
        )
        if choice == "断点续播":
            self._play()
        elif choice == "从头开始":
            self._reset()
            self._play()
        # 保持暂停 / 关闭弹窗:停留在暂停态,按钮由 _on_paused 控制

    def _open_calibration(self):
        from gui.calibration_dialog import CalibrationDialog

        CalibrationDialog(self, self._player, self._config_path).exec()

    # ---------- 演奏小窗联动 ----------

    def mini_play(self):
        # F6/小窗播放入口与按钮共用;演奏或倒计时中忽略重复触发。
        if self._real_playback_active():
            return
        timer = getattr(self, "_countdown_timer", None)
        if timer is not None and timer.isActive():
            return
        self._play()

    def mini_stop(self):
        self._stop()

    def mini_reset(self):
        self._reset()

    def disable_focus_watch(self):
        """小窗模式:游戏全程保持前台焦点,焦点自动暂停没有意义,禁用。

        必须用标志位而不是只清 policy:后续 _play -> _countdown_tick 会再次
        调用 _start_focus_watch,若不拦截会在小窗模式下重新武装焦点检测。
        """
        self._mini_mode = True
        self._focus_timer.stop()

    def rearm_focus_watch(self):
        """从小窗还原到主窗时恢复焦点检测(仅演奏中生效)。"""
        self._mini_mode = False
        if self._player.is_playing or (
            self._event_player is not None and self._event_player.is_playing
        ):
            self._start_focus_watch()

    def snapshot_for_mini(self) -> dict:
        """供演奏小窗 200ms 镜像同步的状态快照。"""
        return {
            "score": self.info_name.text(),
            "state": self.progress_state.text(),
            "pos": self.pos_label.text(),
            "value": self.progress_bar.value(),
            "max": self.progress_bar.maximum(),
            "play_text": self.play_btn.text(),
            "play_enabled": self.play_btn.isEnabled(),
            "stop_enabled": self.stop_btn.isEnabled(),
            "reset_enabled": self.reset_btn.isEnabled(),
        }

    def _on_progress(self, done, total):
        self.progress_bar.setRange(0, max(1, total))
        self.progress_bar.setValue(done)
        self.seek_slider.setRange(0, max(0, total))
        self.seek_slider.setValue(done)
        self.pos_label.setText(f"{done} / {total}")
        self._set_dual_hint(done)

    def _on_paused(self, done, total):
        """停止(暂停):进度保留,可继续或重置。"""
        self._focus_timer.stop()
        self._paused_done = done
        self._paused_total = total
        self.play_btn.setEnabled(True)
        self.mode_combo.setEnabled(True)
        self._set_transport_enabled(True)
        self.play_btn.setText("继续演奏" if done > 0 else "开始演奏")
        self.stop_btn.setEnabled(False)
        self.reset_btn.setEnabled(True)
        self._update_preview_button()
        self.state_label.setText("已暂停")
        resume_hint = f" · 下次从第 {done + 1} 项继续" if done < total else ""
        self.progress_state.setText(
            f"已暂停于 {done} / {total}{resume_hint} · 「继续演奏」或「重置」"
        )
        if self._focus_lost and not self._mini_mode:
            self._focus_lost = False
            self._prompt_focus_recover()

    def _reset(self):
        """重置:清空暂停进度,回到未开始态(仅在暂停态可点击)。"""
        if self._practice_active:
            self._practice_timer.stop()
            self._practice_active = False
        self._stop_practice_listeners()

        # 记录控制事件
        try:
            logger = get_event_logger()
            logger.log_control(action="reset")
        except Exception:
            pass

        self._clear_pause()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.seek_slider.setValue(0)
        self.pos_label.setText("0 / 0")
        self.state_label.setText("就绪")
        self.progress_state.setText("进度已重置")
        self.mode_combo.setEnabled(True)
        self._set_transport_enabled(True)
        self._set_dual_hint(0)
        self._update_preview_button()

    def shutdown(self):
        """Stop UI-owned listeners/timers before the playback engines exit."""

        self._stop_practice_listeners()
        self._practice_timer.stop()
        self._focus_timer.stop()
        timer = getattr(self, "_countdown_timer", None)
        if timer is not None:
            timer.stop()

    def _on_error(self, msg: str):
        self._had_error = True
        self.state_label.setText("演奏出错")
        self.progress_state.setText(f"出错: {msg}")

    def _on_finished(self, normal):
        self._focus_timer.stop()
        event_degradation_count = self._event_degradation_count
        self._clear_pause()
        self.play_btn.setEnabled(True)
        self.mode_combo.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self._set_transport_enabled(True)
        self._update_preview_button()
        is_event_path = self._use_event_path() and self._event_player is not None
        event_log_path = ""
        event_log_error = None
        if is_event_path:
            event_log_path = self._event_player.last_log_path
            event_log_error = self._event_player.last_log_error
        summary = None if is_event_path else getattr(self._player, "last_summary", None)
        if summary is not None:
            text = summary.format()
            if self._transport_degradation_count:
                text += f" · 已降级 {self._transport_degradation_count} 处"
            self.progress_state.setText(text)
            if summary.log_path:
                self.progress_state.setToolTip(f"演奏日志: {summary.log_path}")
        elif event_log_path:
            self.progress_state.setToolTip(f"演奏日志: {event_log_path}")
        elif event_log_error:
            self.progress_state.setToolTip(f"日志写入失败: {event_log_error}")
        if normal:
            self.state_label.setText("就绪")
            if event_log_error:
                self.progress_state.setText(f"演奏完成 · 日志写入失败: {event_log_error}")
            elif summary is None:
                total_degradations = (
                    event_degradation_count + self._transport_degradation_count
                )
                suffix = (
                    f" · 已降级 {total_degradations} 处"
                    if total_degradations
                    else ""
                )
                self.progress_state.setText(f"演奏完成{suffix}")
        elif not self._had_error:
            self.state_label.setText("就绪")
            self.progress_state.setText("已停止")
