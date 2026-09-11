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

import random
import threading
import time

from PyQt6.QtCore import QObject, pyqtSignal

from core.humanize import HumanizeParams, plan_timings
from core.keyboard_driver import KeyboardDriver


class Player(QObject):
    progress = pyqtSignal(int, int)      # 已完成音符数, 总数(全局索引)
    finished = pyqtSignal(bool)          # 演奏自然结束(True)/出错(False)时发出
    paused = pyqtSignal(int, int)        # 停止(暂停)时发出: 已完成音符数, 总数
    error_occurred = pyqtSignal(str)     # 演奏过程中的错误信息

    def __init__(self, keymap, driver=None, logger=None, latency_compensation_ms=0,
                 humanize=None, parent=None):
        super().__init__(parent)
        self._keymap = keymap
        self._driver = driver or KeyboardDriver()
        self._logger = logger
        self.latency_compensation_ms = max(0.0, float(latency_compensation_ms))
        self._humanize = humanize      # HumanizeParams;None = 关闭真人化
        self._stop_event = threading.Event()
        self._thread = None
        # 最近一次演奏的统计(P0-4 可观测性);GUI 在 finished 后读取
        self.last_summary = None
        self.last_log_path = ""

    @property
    def keymap(self):
        return self._keymap

    @property
    def driver(self):
        return self._driver

    @property
    def is_playing(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def play(self, notes, bpm, hold_ratio=0.75, gap_ms=20, start_index=0, score_name=""):
        if self.is_playing:
            return
        self.last_summary = None
        self.last_log_path = ""
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            args=(list(notes), int(bpm), float(hold_ratio), float(gap_ms), int(start_index), str(score_name)),
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

    def _run(self, notes, bpm, hold_ratio, gap_ms, start_index, score_name=""):
        beat_ms = 60000.0 / max(1, bpm)
        total = len(notes)
        done = min(start_index, total)
        normal = False
        error = None
        comp_s = max(0.0, min(self.latency_compensation_ms, 200.0)) / 1000.0
        gap_s = (gap_ms if gap_ms > 0 else 0.0) / 1000.0
        # 真人化节奏:每次演奏独立随机(同一谱每次演奏都有细微差异,像真人);
        # None 时全部按 0 偏移 + 固定 hold_ratio,与历史机械行为完全一致
        if self._humanize is not None:
            timings = plan_timings(notes, self._humanize, random.Random())
            min_gap_s = self._humanize.min_gap_ms / 1000.0
        else:
            timings = None
            min_gap_s = 0.0
        if self._logger:
            self._logger.start(score_name, bpm, total)

        # 绝对时钟调度:每个音符的目标开始时刻 = t0 + 前置音符时值槽之和 - 补偿量。
        # 到点即发,节奏不随音符数累积漂移;start_index 之前的音符同样计入时间轴。
        pre_s = sum(n["dur"] * beat_ms / 1000.0 + gap_s for n in notes[:start_index])
        t0 = time.perf_counter() - pre_s
        cursor_s = pre_s
        prev_release_s = None    # 上一发音音符的实际释放时刻(链式防叠键下限)
        prev_start_t = None    # 上一音符实际开始时刻(测量实际间隔)
        prev_sched_ms = None   # 上一音符到本音符的理论间隔(dur + gap)
        try:
            for i in range(start_index, total):
                if self._stop_event.is_set():
                    break
                note = notes[i]
                dur_ms = note["dur"] * beat_ms
                keys = [self._keymap.key_for(nid) for nid in note["notes"]]
                keys = [k for k in keys if k]
                if timings is not None:
                    off_s, ratio_i = timings[i]
                else:
                    off_s, ratio_i = 0.0, hold_ratio
                # 目标起音 = 理想时刻 + 人性化偏移;链式保护:不早于上一音完全释放
                target_s = t0 + cursor_s + off_s
                if prev_release_s is not None and keys:
                    target_s = max(target_s, prev_release_s + min_gap_s)
                wait_s = target_s - comp_s - time.perf_counter()
                if wait_s > 0:
                    interrupted = self._stop_event.wait(wait_s)
                else:
                    # 已到点(或落后于进度):直接发送;停止信号则在此退出
                    interrupted = self._stop_event.is_set()
                if interrupted:
                    break  # 停止信号在音符开始前到达:断点留在 i,续播重放
                start_t = time.perf_counter()
                actual_ms = (start_t - prev_start_t) * 1000.0 if prev_start_t is not None else None
                dev_ms = (actual_ms - prev_sched_ms) if actual_ms is not None and prev_sched_ms is not None else None
                ok = True
                error_msg = None
                try:
                    if keys:
                        self._driver.press_chord(keys)
                        hold_interrupted = self._stop_event.wait(dur_ms * ratio_i / 1000.0)
                        self._driver.release_chord(keys)
                        prev_release_s = target_s + dur_ms * ratio_i / 1000.0
                        if hold_interrupted:
                            break  # 按住期被截断:断点留在 i,续播时重放该音符(不记日志)
                    # 休止:时值由下一音符的绝对开始点体现,无需单独等待
                except Exception as e:
                    # 先记录失败再抛出,保持原有的中止行为
                    ok = False
                    error_msg = str(e)
                    if self._logger:
                        self._logger.log_note(i, note["notes"], keys, ok=False,
                                              sched_ms=prev_sched_ms, actual_ms=actual_ms,
                                              dev_ms=dev_ms, error=error_msg)
                    raise
                done = i + 1
                if self._logger:
                    self._logger.log_note(i, note["notes"], keys, ok=True,
                                          sched_ms=prev_sched_ms, actual_ms=actual_ms, dev_ms=dev_ms)
                self.progress.emit(done, total)
                prev_start_t = start_t
                if keys:
                    prev_sched_ms = dur_ms + (gap_ms if gap_ms > 0 else 0.0)
                cursor_s += dur_ms / 1000.0 + gap_s
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
        if error:
            self.error_occurred.emit(error)
        self.finished.emit(normal)
