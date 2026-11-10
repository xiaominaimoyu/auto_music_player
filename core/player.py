"""演奏引擎:按音符序列的节奏模拟键盘按键(支持和弦)。

三态控制:
- 开始/继续:play(start_index) 从指定进度开始
- 停止:stop() 暂停演奏,进度经 paused(done, total) 信号带出,可继续
- 重置:由调用方丢弃 paused 进度即可(下次 play 从头)

停止响应性:全部等待使用 Event.wait(可立即打断),stop() 后 100ms 内释放按键。
退出保护:shutdown() 供主窗口 closeEvent 与进程 atexit 调用,
任何退出路径都保证不残留按下状态的按键。
断点语义:音符被中途截断时断点留在该音符,续播时重放;音符间歇期停止则记为已完成。
"""

import threading
import time

from PyQt6.QtCore import QObject, pyqtSignal

from core.direct_timing import direct_slots, strum_order
from core.event_logger import NoteEvent
from core.humanize import make_seed, plan_timings
from core.keyboard_driver import KeyboardDriver
from core.timing import high_resolution_period, wait_until


_USE_CONFIGURED_HUMANIZE = object()


class Player(QObject):
    progress = pyqtSignal(int, int)      # 已完成音符数, 总数(全局索引)
    finished = pyqtSignal(bool)          # 演奏自然结束(True)/出错(False)时发出
    paused = pyqtSignal(int, int)        # 停止(暂停)时发出: 已完成音符数, 总数
    error_occurred = pyqtSignal(str)     # 演奏过程中的错误信息

    def __init__(self, keymap, driver=None, logger=None, latency_compensation_ms=0,
                 humanize=None, parent=None, *, event_logger=None,
                 precise_wait=True, additive_gap=False):
        super().__init__(parent)
        self._keymap = keymap
        self._driver = driver or KeyboardDriver()
        self._logger = logger
        self.latency_compensation_ms = max(0.0, float(latency_compensation_ms))
        self._humanize = humanize      # HumanizeParams;None = 关闭真人化
        self._precise_wait = bool(precise_wait)
        self._additive_gap = bool(additive_gap)
        self._event_logger = event_logger
        self._stop_event = threading.Event()
        self._thread = None
        self._dispatch_guard = None
        # 最近一次演奏的统计(P0-4 可观测性);GUI 在 finished 后读取
        self.last_summary = None
        self.last_log_path = ""
        self.last_event_log_path = ""
        self.last_event_log_error = None
        self.last_humanize_seed = None

    @property
    def keymap(self):
        return self._keymap

    @property
    def driver(self):
        return self._driver

    @property
    def is_playing(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def set_dispatch_guard(self, guard):
        """设置每个和弦按下前执行的只读目标窗口保护回调。"""
        self._dispatch_guard = guard

    def _require_dispatch_allowed(self):
        if self._dispatch_guard is None:
            return
        result = self._dispatch_guard()
        if isinstance(result, tuple):
            allowed, reason = bool(result[0]), str(result[1] or "")
        else:
            allowed, reason = bool(result), ""
        if not allowed:
            raise RuntimeError(reason or "目标窗口不再允许发送输入")

    def play(
        self,
        notes,
        bpm,
        hold_ratio=0.75,
        gap_ms=20,
        start_index=0,
        score_name="",
        *,
        humanize_override=_USE_CONFIGURED_HUMANIZE,
        humanize_seed=None,
        trace_context=None,
    ):
        if self.is_playing:
            return
        self.last_summary = None
        self.last_log_path = ""
        self.last_event_log_path = ""
        self.last_event_log_error = None
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            args=(
                list(notes),
                int(bpm),
                float(hold_ratio),
                float(gap_ms),
                int(start_index),
                str(score_name),
                humanize_override,
                humanize_seed,
                dict(trace_context or {}),
            ),
            daemon=True,
        )
        self._thread.start()

    def stop(self):
        self._stop_event.set()

    def shutdown(self, join_timeout=1.0):
        """退出保护:停止演奏线程并确保全部按键释放。可重复调用。

        仅在线程存活时才做按键释放:闲置时跳过,避免进程收尾阶段的无谓 Win32 调用。
        """
        self.stop()
        t = self._thread
        if t is not None and t.is_alive():
            t.join(join_timeout)
            self._release_all()

    def _strum_chord(self, keys, strum_ms, index):
        """和弦在 8–25ms 内扫过，方向逐个和弦交替，避免永远从低到高。"""
        order, step_ms = strum_order(keys, strum_ms, index)
        step_s = step_ms / 1000.0
        pressed = []
        try:
            for nth, key in enumerate(order):
                if nth and wait_until(
                    self._stop_event,
                    time.perf_counter() + step_s,
                    precise=self._precise_wait,
                ):
                    break
                self._driver.press_key(key)
                pressed.append(key)
        except Exception:
            if pressed and hasattr(self._driver, "release_chord"):
                self._driver.release_chord(pressed)
            raise

    def _release_all(self):
        """保险:演奏结束/暂停/被停止/进程退出时松开所有可能按住的键。"""
        for octave in ("high", "mid", "low"):
            for i in range(1, 8):
                key = self._keymap.key_for(f"{octave}_{i}")
                if key:
                    try:
                        self._driver.release_key(key)
                    except Exception:
                        pass

    def _run(
        self,
        notes,
        bpm,
        hold_ratio,
        gap_ms,
        start_index,
        score_name="",
        humanize_override=_USE_CONFIGURED_HUMANIZE,
        humanize_seed=None,
        trace_context=None,
    ):
        total = len(notes)
        done = min(start_index, total)
        normal = False
        error = None
        comp_s = max(0.0, min(self.latency_compensation_ms, 200.0)) / 1000.0
        # 真人化节奏:一次会话只生成一枚 seed，供日志、Legacy 和 Delta 重放。
        # None 时全部按 0 偏移 + 固定 hold_ratio,与历史机械行为完全一致。
        humanize = (
            self._humanize
            if humanize_override is _USE_CONFIGURED_HUMANIZE
            else humanize_override
        )
        if humanize is not None:
            self.last_humanize_seed = int(
                humanize_seed if humanize_seed is not None else make_seed()
            )
            timings = plan_timings(notes, humanize, seed=self.last_humanize_seed)
            min_gap_ms = humanize.min_gap_ms
        else:
            timings = None
            min_gap_ms = 0.0
            self.last_humanize_seed = None
        if self._logger:
            self._logger.start(score_name, bpm, total)
        event_log_started = False
        if self._event_logger is not None:
            self._event_logger.start_session(score_name, bpm, total)
            event_log_started = self._event_logger.is_active
            self.last_event_log_path = str(self._event_logger.log_path or "")
            self.last_event_log_error = self._event_logger.last_error
            if event_log_started:
                details = {
                    "stage": "legacy_player_started",
                    "start_index": start_index,
                    "gap_ms": gap_ms,
                    "humanize_enabled": humanize is not None,
                    "humanize_seed": self.last_humanize_seed,
                }
                details.update(trace_context or {})
                self._event_logger.log_env("playback_pipeline", **details)
                self._event_logger.log_control(
                    "resume" if start_index > 0 else "start",
                    bpm=bpm,
                    start_index=start_index,
                    total_notes=total,
                )

        # 绝对时钟调度:每个音符的目标开始时刻 = t0 + 前置音符时值槽之和 - 补偿量。
        # 到点即发,节奏不随音符数累积漂移;start_index 之前的音符同样计入时间轴。
        # gap 默认从时值里扣成松开间隙,起音间隔仍等于谱面时值。
        slots = direct_slots(notes, bpm, hold_ratio, gap_ms, timings,
                             additive=self._additive_gap, min_gap_ms=min_gap_ms,
                             keymap=self._keymap, start_index=start_index)
        pre_ms = slots[start_index].cursor_ms if start_index < total else 0.0
        t0 = time.perf_counter() - pre_ms / 1000.0
        prev_start_t = None    # 上一音符实际开始时刻(测量实际间隔)
        prev_sched_ms = None   # 上一音符到本音符的理论间隔
        try:
            with high_resolution_period(1):
                for i in range(start_index, total):
                    if self._stop_event.is_set():
                        break
                    note = notes[i]
                    slot = slots[i]
                    keys = [self._keymap.key_for(nid) for nid in note["notes"]]
                    keys = [k for k in keys if k]
                    cursor_s = slot.cursor_ms / 1000.0
                    hold_s = slot.hold_ms / 1000.0
                    target_s = t0 + slot.onset_ms / 1000.0
                    interrupted = wait_until(
                        self._stop_event,
                        target_s - comp_s,
                        precise=self._precise_wait,
                    )
                    if interrupted:
                        break  # 停止信号在音符开始前到达:断点留在 i,续播重放
                    start_t = time.perf_counter()
                    actual_ms = (start_t - prev_start_t) * 1000.0 if prev_start_t is not None else None
                    dev_ms = (actual_ms - prev_sched_ms) if actual_ms is not None and prev_sched_ms is not None else None
                    error_msg = None
                    try:
                        if keys:
                            self._require_dispatch_allowed()
                            strum_ms = 0.0
                            if humanize is not None and len(keys) > 1:
                                strum_ms = float(getattr(humanize, "strum_ms", 0.0) or 0.0)
                            if strum_ms > 0:
                                self._strum_chord(keys, strum_ms, i)
                            else:
                                self._driver.press_chord(keys)
                            hold_interrupted = wait_until(
                                self._stop_event,
                                time.perf_counter() + hold_s,
                                precise=self._precise_wait,
                            )
                            self._driver.release_chord(keys)
                            if hold_interrupted:
                                break  # 按住期被截断:断点留在 i,续播时重放该音符(不记日志)
                        # 休止:时值由下一音符的绝对开始点体现,无需单独等待
                    except Exception as e:
                        error_msg = str(e)
                        if self._logger:
                            self._logger.log_note(i, note["notes"], keys, ok=False,
                                                  sched_ms=prev_sched_ms, actual_ms=actual_ms,
                                                  dev_ms=dev_ms, error=error_msg)
                        if event_log_started:
                            actual_s = max(0.0, start_t - t0)
                            self._event_logger.log_note(
                                NoteEvent(
                                    index=i,
                                    notes=list(note["notes"]),
                                    keys=list(keys),
                                    expected_time=max(0.0, cursor_s),
                                    actual_time=actual_s,
                                    deviation_ms=(actual_s - cursor_s) * 1000.0,
                                    success=False,
                                    error=error_msg,
                                )
                            )
                        raise
                    done = i + 1
                    if self._logger:
                        self._logger.log_note(i, note["notes"], keys, ok=True,
                                              sched_ms=prev_sched_ms, actual_ms=actual_ms, dev_ms=dev_ms)
                    if event_log_started:
                        actual_s = max(0.0, start_t - t0)
                        self._event_logger.log_note(
                            NoteEvent(
                                index=i,
                                notes=list(note["notes"]),
                                keys=list(keys),
                                expected_time=max(0.0, cursor_s),
                                actual_time=actual_s,
                                deviation_ms=(actual_s - cursor_s) * 1000.0,
                                success=True,
                            )
                        )
                    self.progress.emit(done, total)
                    prev_start_t = start_t
                    if keys:
                        prev_sched_ms = slot.advance_ms
                if self._stop_event.is_set():
                    # 停止 = 暂停:进度经 paused 信号带出,调用方可继续或重置
                    self.paused.emit(done, total)
                    return
                normal = True
        except Exception as e:
            error = str(e)
        finally:
            self._release_all()
            if self._logger:
                self.last_summary = self._logger.finish(
                    done, total, stopped_early=self._stop_event.is_set(), error=error
                )
                self.last_log_path = self._logger.path
            if event_log_started:
                self._event_logger.log_env(
                    "playback_pipeline",
                    stage="session_closing",
                    completed=done,
                    total=total,
                    stopped=self._stop_event.is_set(),
                    error=error,
                )
                self._event_logger.end_session(
                    completed=done,
                    total=total,
                    stopped_early=self._stop_event.is_set() or error is not None,
                    error=error,
                )
                self.last_event_log_error = self._event_logger.last_error
                self.last_log_path = self.last_event_log_path
                if self.last_summary is not None:
                    self.last_summary.log_path = self.last_event_log_path
        if error:
            self.error_occurred.emit(error)
        self.finished.emit(normal)
