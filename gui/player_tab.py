"""演奏控制页(对齐 Web 设计稿):选谱 -> BPM -> 开始/停止/重置 -> 进度。

三态控制:开始(或暂停后"继续演奏") / 停止(= 暂停,进度保留) / 重置(仅停止后可用,进度归零)。
"""

import random

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
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
from core.humanize import HumanizeParams, plan_timings
from core.profile import resolve_profile
from core.scenario import SCENARIOS, get_scenario
from core.window_monitor import FocusLockPolicy, ForegroundWatcher
from gui.theme import BRAND, INK_2, INK_3, STATE_INFO
from gui.widgets import AppDialog


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
):
    """事件路径(三角洲档位):谱面 → (ScenarioPlan, 降级清单)。

    纯函数,不依赖 GUI/游戏/时钟,便于单测。音符层与编译逻辑不 fork——
    复用 M2 的 IR/编译器与 M3 的档位,差异只由 scenario 收敛到会话层。
    humanize:HumanizeParams 或 None;仅自由演奏等 humanize 类场景生效
    (识别类场景由 scenario.humanize 决定,见 core/scenario.py)。
    """
    scenario = get_scenario(scenario_id)
    elements = ir_mod.from_storage(notes)
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
    result = compile_score(elements, params, timings=timings)
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
    def __init__(
        self,
        db,
        player,
        player_cfg,
        config_path=None,
        profile=None,
        profiles=None,
        event_player=None,
    ):
        super().__init__()
        self._db = db
        self._player = player
        self._config_path = config_path
        # 游戏档位(M3):当前激活档位 + 全部候选档位 + 事件演奏器(M4)
        self._profile = profile
        self._profiles = list(profiles or [])
        self._event_player = event_player
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
        # 焦点检测:丢失目标窗口焦点时自动暂停,恢复后由用户选择续播或从头
        focus_cfg = player_cfg.get("focus_check") or {}
        self._focus_enabled = bool(focus_cfg.get("enabled", True))
        self._poll_interval_ms = max(100, int(focus_cfg.get("poll_interval_ms", 400)))
        self._target_title = str(focus_cfg.get("target_window_title", "") or "")
        self._watcher = ForegroundWatcher(poll_interval=self._poll_interval_ms / 1000.0)
        self._policy = None
        self._mini_mode = False
        self._focus_lost = False
        self._build_ui()
        self._focus_timer = QTimer(self)
        self._focus_timer.setInterval(self._poll_interval_ms)
        self._focus_timer.timeout.connect(self._check_focus)
        self._player.progress.connect(self._on_progress)
        self._player.finished.connect(self._on_finished)
        self._player.paused.connect(self._on_paused)
        self._player.error_occurred.connect(self._on_error)
        # 事件演奏器(M4)的信号接到同一组 UI 处理(任意时刻只有一条路径在跑)
        if self._event_player is not None:
            self._event_player.progress.connect(self._on_progress)
            self._event_player.finished.connect(self._on_finished)
            self._event_player.paused.connect(self._on_paused)
            self._event_player.aborted.connect(self._on_aborted)
            self._event_player.error_occurred.connect(self._on_error)

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
        root.setContentsMargins(32, 24, 32, 24)
        root.setSpacing(16)

        title = QLabel("演奏控制")
        title.setObjectName("PageTitle")
        self.state_label = QLabel("就绪")
        self.state_label.setObjectName("PageSub")
        header = QHBoxLayout()
        header.addWidget(title)
        header.addWidget(self.state_label)
        header.addStretch(1)
        root.addLayout(header)

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
            root.addWidget(card)

        # 选择乐谱
        card, lay = self._card("选择乐谱")
        label = QLabel("乐谱")
        label.setObjectName("FieldLabel")
        lay.addWidget(label)
        self.combo = QComboBox()
        self.combo.currentIndexChanged.connect(self._on_select)
        lay.addWidget(self.combo)
        root.addWidget(card)

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

        info = QFrame()
        info.setStyleSheet(f"background: #1E1E28; border-radius: 8px;")
        info_layout = QHBoxLayout(info)
        info_layout.setContentsMargins(16, 12, 16, 12)
        info_layout.setSpacing(24)
        self.info_name = self._info_item(info_layout, "乐谱名称", "-")
        self.info_count = self._info_item(info_layout, "音符数", "-")
        self.info_duration = self._info_item(info_layout, "预计时长", "-")
        lay.addWidget(info)
        root.addWidget(card)

        # 控制
        card, lay = self._card("演奏控制")
        ctrl_row = QHBoxLayout()
        self.play_btn = QPushButton("开始演奏")
        self.play_btn.setObjectName("BtnPrimary")
        self.play_btn.setMinimumWidth(120)
        self.play_btn.setEnabled(False)
        self.play_btn.clicked.connect(self._play)
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
        ctrl_row.addWidget(self.stop_btn)
        ctrl_row.addWidget(self.reset_btn)
        ctrl_row.addWidget(self.calib_btn)
        ctrl_row.addStretch(1)
        lay.addLayout(ctrl_row)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        lay.addWidget(self.progress_bar)
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
        root.addWidget(card)
        root.addStretch(1)

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
            self.info_name.setText("-")
            self.info_count.setText("-")
            self.info_duration.setText("-")
            self.play_btn.setEnabled(False)

    def select_score(self, score_id: int):
        for i in range(self.combo.count()):
            if self.combo.itemData(i) == score_id:
                self.combo.setCurrentIndex(i)
                return

    def _on_select(self):
        self._score_id = self.combo.currentData()
        self._clear_pause()
        if self._score_id is None:
            self.play_btn.setEnabled(False)
            return
        score = self._db.get_score(self._score_id)
        if score is None:
            return
        self.bpm_spin.setValue(score["bpm_default"])
        self.info_name.setText(score["name"])
        self.info_count.setText(str(len(score["notes"])))
        self.info_duration.setText(
            f"{self._estimate_seconds(score['notes'], self.bpm_spin.value())} 秒"
        )
        self.play_btn.setEnabled(not self._player.is_playing)

    def _clear_pause(self):
        """回到未开始态:清空暂停进度与按钮状态。"""
        self._paused_done = 0
        self._paused_total = 0
        self.play_btn.setText("开始演奏")
        self.reset_btn.setEnabled(False)

    def _estimate_seconds(self, notes, bpm):
        beat_ms = 60000.0 / max(1, bpm)
        total_ms = sum(n["dur"] * beat_ms for n in notes)
        total_ms += self._gap_ms * len(notes)
        return round(total_ms / 1000.0, 1)

    def _play(self):
        if self._score_id is None:
            return
        score = self._db.get_score(self._score_id)
        if not score or not score["notes"]:
            AppDialog.show_warning(self, "提示", "该乐谱没有音符数据")
            return
        start_index = self._paused_done if self._paused_done > 0 else 0
        self._pending = (
            score["notes"],
            self.bpm_spin.value(),
            start_index,
            score["name"],
        )
        self._had_error = False
        self._countdown_left = 3
        self.play_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.reset_btn.setEnabled(False)
        resume_hint = f"从第 {start_index} 音符继续" if start_index > 0 else ""
        self.state_label.setText(
            f"3 秒后演奏: 《{score['name']}》 {resume_hint}".strip()
        )
        self.progress_state.setText(
            f"{self._countdown_left} 秒后开始演奏,请切换到游戏窗口..."
        )
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
        if self._config_path:
            try:
                _update_active_profile(self._config_path, self._profile.id)
            except Exception as e:
                self.progress_state.setText(f"档位已切换(写回配置失败: {e})")
        self._update_scenario_visibility()

    def _play_event(self, notes, bpm, start_index, score_name):
        plan, degradations = build_event_plan(
            notes,
            self._profile,
            self.scenario_combo.currentData() or "free_play",
            bpm=bpm,
            settle_ms=self._settle_ms,
            release_settle_ms=self._release_settle_ms,
            hold_ratio=self._hold_ratio,
            gap_ms=self._gap_ms,
            humanize=self._humanize_params,
        )
        self._event_player.play(
            plan.events,
            plan.interrupt_mode,
            start_index=start_index,
            score_name=score_name,
        )
        n = len(degradations)
        suffix = f" · 已降级 {n} 处" if n else ""
        self.state_label.setText(f"演奏中 BPM {bpm} · {self._profile.name}")
        self.progress_state.setText(f"演奏中...{suffix}")

    def _on_aborted(self, reason):
        """中止(S2):进度清空,不可续播。"""
        self._focus_timer.stop()
        self._paused_done = 0
        self._paused_total = 0
        self.play_btn.setEnabled(True)
        self.play_btn.setText("开始演奏")
        self.stop_btn.setEnabled(False)
        self.reset_btn.setEnabled(False)
        self.state_label.setText("已中止")
        self.progress_state.setText(reason or "演奏已中止")
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
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
        self._start_focus_watch()
        if self._use_event_path():
            self._play_event(notes, bpm, start_index, score_name)
        else:
            self._player.play(
                notes,
                bpm,
                self._hold_ratio,
                self._gap_ms,
                start_index=start_index,
                score_name=score_name,
            )
            self.state_label.setText(f"演奏中 BPM {bpm}")
            self.progress_state.setText("演奏中...")

    def _stop(self):
        timer = getattr(self, "_countdown_timer", None)
        if timer is not None and timer.isActive():
            timer.stop()
            self.play_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.progress_state.setText("已取消")
            return
        self._player.stop()
        self.progress_state.setText("正在暂停...")

    # ---------- 焦点检测 ----------

    def _own_hwnd(self):
        try:
            return int(self.window().winId())
        except Exception:
            return None

    def _start_focus_watch(self):
        """启动焦点轮询;自动模式下不预设目标,用户切到的首个外部窗口被锁定为目标。

        小窗模式下焦点检测整体禁用(游戏全程保持前台,无切窗即无误停)。
        """
        self._focus_timer.stop()
        self._focus_lost = False
        if (
            self._mini_mode
            or not self._focus_enabled
            or not self._watcher.is_available()
        ):
            self._policy = None
            return
        self._policy = FocusLockPolicy(
            own_hwnd=self._own_hwnd(), title_override=self._target_title
        )
        self._focus_timer.start()

    def _check_focus(self):
        if not self._player.is_playing:
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

    def _pause_for_focus(self):
        self._focus_timer.stop()
        self._focus_lost = True
        self._player.stop()
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
        self._policy = None

    def rearm_focus_watch(self):
        """从小窗还原到主窗时恢复焦点检测(仅演奏中生效)。"""
        self._mini_mode = False
        if self._player.is_playing:
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
        self.pos_label.setText(f"{done} / {total}")

    def _on_paused(self, done, total):
        """停止(暂停):进度保留,可继续或重置。"""
        self._focus_timer.stop()
        self._paused_done = done
        self._paused_total = total
        self.play_btn.setEnabled(True)
        self.play_btn.setText("继续演奏" if done > 0 else "开始演奏")
        self.stop_btn.setEnabled(False)
        self.reset_btn.setEnabled(True)
        self.state_label.setText("已暂停")
        self.progress_state.setText(
            f"已暂停于 {done} / {total} · 「继续演奏」或「重置」"
        )
        if self._focus_lost and not self._mini_mode:
            self._focus_lost = False
            self._prompt_focus_recover()

    def _reset(self):
        """重置:清空暂停进度,回到未开始态(仅在暂停态可点击)。"""
        self._clear_pause()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.pos_label.setText("0 / 0")
        self.state_label.setText("就绪")
        self.progress_state.setText("进度已重置")

    def _on_error(self, msg: str):
        self._had_error = True
        self.state_label.setText("演奏出错")
        self.progress_state.setText(f"出错: {msg}")

    def _on_finished(self, normal):
        self._focus_timer.stop()
        self._clear_pause()
        self.play_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        summary = getattr(self._player, "last_summary", None)
        if summary is not None:
            self.progress_state.setText(summary.format())
            if summary.log_path:
                self.progress_state.setToolTip(f"演奏日志: {summary.log_path}")
        if normal:
            self.state_label.setText("就绪")
            if summary is None:
                self.progress_state.setText("演奏完成")
        elif not self._had_error:
            self.state_label.setText("就绪")
            self.progress_state.setText("已停止")
