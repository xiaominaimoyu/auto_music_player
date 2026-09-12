"""事件演奏器(EventPlayer):按绝对时刻消费编译器产出的输入事件序列。

与 `Player`(21 键简谱路径)的区别:
- 输入是 `InputEvent(t_ms, device, key, action)`,**不再查 keymap**——
  修饰态时序与降级已由编译器决定,本类只负责"到点即发"
- 中断语义除**可恢复暂停(pause)**外,新增**不可恢复中止(abort)**
- 释放集合**直接从本次播放的事件推导**,覆盖键盘与鼠标,不依赖外部键位清单

**默认档位继续走旧 `Player`(一字节未改);三角洲档位走本类,双路径由档位分发。**
这样 M4 对鸣潮/原神的回归风险为零。
"""

import threading
import time

from PyQt6.QtCore import QObject, pyqtSignal

_INTERRUPT_MODES = ("pause", "abort")


class EventPlayer(QObject):
    progress = pyqtSignal(int, int)      # 已完成音符数(无元数据时退化为事件数), 总数
    finished = pyqtSignal(bool)          # 自然结束(True)/出错(False)
    paused = pyqtSignal(int, int)        # 可恢复暂停: 下次起始音符索引, 总数
    aborted = pyqtSignal(str)            # 不可恢复中止: 原因
    error_occurred = pyqtSignal(str)

    def __init__(self, driver, latency_compensation_ms=0, parent=None):
        super().__init__(parent)
        self._driver = driver
        self.latency_compensation_ms = max(0.0, float(latency_compensation_ms))
        self._stop_event = threading.Event()
        self._thread = None
        self._force_abort = False
        self._abort_reason = "演奏被中断"
        self._resume_source_index = 0

    @property
    def driver(self):
        return self._driver

    @property
    def is_playing(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def resume_source_index(self):
        """最近一次暂停后应从哪个原始谱面元素重新编译。"""
        return self._resume_source_index

    def play(
        self,
        events,
        interrupt_mode="pause",
        start_index=0,
        score_name="",
        *,
        source_total=None,
        source_start_index=None,
    ):
        """播放事件序列。interrupt_mode 决定 stop() 的语义:
        "pause" = 可恢复暂停;"abort" = 不可恢复中止(清空进度)。
        """
        if self.is_playing:
            return
        if interrupt_mode not in _INTERRUPT_MODES:
            raise ValueError(
                f"interrupt_mode 必须是 {'/'.join(_INTERRUPT_MODES)}: {interrupt_mode!r}")
        self._force_abort = False
        self._abort_reason = "演奏被中断"
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            args=(
                list(events),
                interrupt_mode,
                int(start_index),
                str(score_name),
                source_total,
                source_start_index,
            ),
            daemon=True,
        )
        self._thread.start()

    def stop(self):
        self._stop_event.set()

    def abort(self, reason=""):
        """强制中止(无视 interrupt_mode):清空进度,发 aborted。"""
        self._force_abort = True
        self._abort_reason = reason or "演奏被中断"
        self._stop_event.set()

    def shutdown(self, join_timeout=1.0):
        """退出兜底:停止线程并确保全部键鼠释放。可重复调用。"""
        self.stop()
        t = self._thread
        if t is not None and t.is_alive():
            t.join(join_timeout)

    # ---------- 事件分发 ----------
    def _dispatch(self, e):
        d = self._driver
        if e.device == "kb":
            (d.press_key if e.action == "down" else d.release_key)(e.key)
        else:
            (d.press_mouse if e.action == "down" else d.release_mouse)(e.key)

    def _release_all(self, events):
        """释放集合直接从本次播放的事件推导:覆盖键盘与鼠标。

        不依赖外部键位清单——只要事件在这一把出现过,就保证被释放,
        因此天然覆盖修饰键(彻底解决"右键卡住会持续顶在高八度"的坑)。
        """
        keys = sorted({e.key for e in events if e.device == "kb"})
        mouse = sorted({e.key for e in events if e.device == "mouse"})
        if keys or mouse:
            self._driver.panic_release(keys, mouse)

    def _run(self, events, interrupt_mode, start_index, score_name, source_total, source_start_index):
        event_total = len(events)
        done = min(start_index, event_total)
        tagged = any(e.source_index is not None for e in events)
        if tagged:
            derived_total = max(e.source_index for e in events if e.source_index is not None) + 1
            total = max(derived_total, int(source_total) if source_total is not None else derived_total)
            if source_start_index is None:
                remaining = [e.source_index for e in events[start_index:] if e.source_index is not None]
                source_start_index = remaining[0] if remaining else total
            resume_source_index = max(0, min(int(source_start_index), total))
            self._resume_source_index = resume_source_index
            if resume_source_index:
                self.progress.emit(resume_source_index, total)
        else:
            total = event_total
            resume_source_index = done
        normal = False
        error = None
        comp_s = max(0.0, min(self.latency_compensation_ms, 200.0)) / 1000.0
        # 绝对时钟:事件 i 的目标时刻 = t0 + events[i].t_ms/1000 - 补偿。
        # 与 Player 同款调度,节奏不随事件数累积漂移。
        base_ms = events[start_index].t_ms if 0 <= start_index < event_total else 0.0
        t0 = time.perf_counter() - base_ms / 1000.0
        try:
            for i in range(start_index, event_total):
                if self._stop_event.is_set():
                    break
                e = events[i]
                wait_s = t0 + e.t_ms / 1000.0 - comp_s - time.perf_counter()
                if wait_s > 0:
                    if self._stop_event.wait(wait_s):
                        break  # 停止信号在事件到达前到来:断点留在 i
                elif self._stop_event.is_set():
                    break
                self._dispatch(e)
                done = i + 1
                if tagged:
                    if e.source_end and e.source_index is not None:
                        resume_source_index = max(resume_source_index, e.source_index + 1)
                        resume_source_index = min(resume_source_index, total)
                        self._resume_source_index = resume_source_index
                        self.progress.emit(resume_source_index, total)
                else:
                    self.progress.emit(done, total)
            if self._stop_event.is_set():
                # 停止 = pause / abort,互斥于 finished(不发 finished)
                if self._force_abort or interrupt_mode == "abort":
                    self.aborted.emit(self._abort_reason)
                else:
                    self.paused.emit(resume_source_index if tagged else done, total)
                return
            normal = True
        except Exception as e:
            error = str(e)
        finally:
            self._release_all(events)
        if normal and tagged and resume_source_index < total:
            # 尾部休止没有输入事件，也应在完成态反映为完整乐谱。
            self._resume_source_index = total
            self.progress.emit(total, total)
        if error:
            self.error_occurred.emit(error)
        self.finished.emit(normal)
