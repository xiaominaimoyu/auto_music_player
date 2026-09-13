"""多类型事件流日志系统。

扩展现有 note 日志，新增 focus、control、external_input、modifier_anomaly、env 五类事件。
所有事件写入同一 JSONL 文件，通过 type 字段区分。

事件类型：
- note: 音符演奏事件（保持现有格式兼容）
- focus: 焦点变化事件（失焦/回焦）
- control: 用户控制操作（开始/暂停/停止/参数调整）
- external_input: 外部键盘输入（区分 self/user/injected 三种来源）
- modifier_anomaly: 修饰键状态异常（物理状态与预期不符）
- env: 系统环境事件（UAC/锁屏/切桌面）
"""

import json
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

# ==================== 事件类型定义 ====================

EventType = Literal[
    "note", "focus", "control", "external_input", "modifier_anomaly", "env"
]


@dataclass
class BaseEvent:
    """所有事件的公共字段"""

    type: EventType
    timestamp: float = field(default_factory=time.time)
    session_id: str = ""
    score_name: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # 保留两位小数的时间戳
        d["timestamp"] = round(self.timestamp, 2)
        return d


@dataclass
class NoteEvent(BaseEvent):
    """音符演奏事件（保持向后兼容）"""

    type: EventType = "note"
    index: int = 0
    notes: list[str] = field(default_factory=list)
    keys: list[str] = field(default_factory=list)
    expected_time: float = 0.0
    actual_time: float = 0.0
    deviation_ms: float = 0.0
    success: bool = True
    error: str | None = None


@dataclass
class FocusEvent(BaseEvent):
    """焦点变化事件"""

    type: EventType = "focus"
    event: Literal["lost", "regained"] = "lost"
    window_title: str = ""
    hwnd: int = 0
    note_index: int = -1  # 失焦时已完成的音符索引


@dataclass
class ControlEvent(BaseEvent):
    """用户控制操作事件"""

    type: EventType = "control"
    action: str = ""  # start/pause/resume/stop/panic/bpm_change/hold_toggle/gear_change/countdown_focus
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExternalInputEvent(BaseEvent):
    """外部键盘输入事件"""

    type: EventType = "external_input"
    vk_code: int = 0
    scan_code: int = 0
    source: Literal["self", "user", "injected"] = "user"
    injected_flag: bool = False


@dataclass
class ModifierAnomalyEvent(BaseEvent):
    """修饰键状态异常事件"""

    type: EventType = "modifier_anomaly"
    modifier: Literal["LButton", "RButton", "Shift"] = "LButton"
    expected: bool = False
    actual: bool = False


@dataclass
class EnvEvent(BaseEvent):
    """系统环境事件"""

    type: EventType = "env"
    event: str = ""  # uac_change/lock_screen/show_desktop/uac_dialog
    details: dict[str, Any] = field(default_factory=dict)


# ==================== 日志写入器 ====================


class EventLogger:
    """事件流日志写入器，管理单次演奏会话的所有事件"""

    def __init__(self, log_dir: str = "data/play_logs"):
        self.log_dir = Path(log_dir)
        self.session_id: str = ""
        self.score_name: str = ""
        self.log_path: Path | None = None
        self._fh = None
        self._events_count = 0

    def start_session(self, score_name: str, bpm: int, note_count: int) -> str:
        """开始新的演奏会话，返回 session_id"""
        self.session_id = uuid.uuid4().hex[:16]
        self.score_name = score_name
        self._events_count = 0

        try:
            # 创建日志文件：YYYYMMDD_HHMMSS_曲谱名.jsonl
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_name = "".join(c for c in score_name if c.isalnum() or c in "_ -")[:50]
            filename = f"{timestamp}_{safe_name}.jsonl"
            self.log_path = self.log_dir / filename

            self.log_dir.mkdir(parents=True, exist_ok=True)
            self._fh = open(self.log_path, "w", encoding="utf-8")

            # 写入会话头
            session_header = {
                "type": "session_start",
                "timestamp": time.time(),
                "session_id": self.session_id,
                "score_name": score_name,
                "bpm": bpm,
                "note_count": note_count,
                "started_at": datetime.now().isoformat(),
            }
            self._write_line(session_header)
        except Exception as e:
            # 降级：仅内存统计，不影响演奏
            self._fh = None
            self.log_path = None
            print(f"[EventLogger] 无法创建日志文件: {e}")

        return self.session_id

    def end_session(
        self,
        completed: int,
        total: int,
        stopped_early: bool = False,
        error: str | None = None,
    ):
        """结束会话，写入统计摘要"""
        if self._fh is None:
            return

        try:
            session_footer = {
                "type": "session_end",
                "timestamp": time.time(),
                "session_id": self.session_id,
                "completed": completed,
                "total": total,
                "stopped_early": stopped_early,
                "error": error,
                "events_count": self._events_count,
                "ended_at": datetime.now().isoformat(),
            }
            self._write_line(session_footer)
        except Exception:
            pass
        finally:
            if self._fh:
                try:
                    self._fh.close()
                except Exception:
                    pass
                self._fh = None

    def log_note(self, note_event: NoteEvent):
        """记录音符事件"""
        note_event.session_id = self.session_id
        note_event.score_name = self.score_name
        self._write_event(note_event)

    def log_focus(
        self,
        event: Literal["lost", "regained"],
        window_title: str = "",
        hwnd: int = 0,
        note_index: int = -1,
    ):
        """记录焦点事件"""
        focus_event = FocusEvent(
            session_id=self.session_id,
            score_name=self.score_name,
            event=event,
            window_title=window_title,
            hwnd=hwnd,
            note_index=note_index,
        )
        self._write_event(focus_event)

    def log_control(self, action: str, **params):
        """记录控制事件"""
        control_event = ControlEvent(
            session_id=self.session_id,
            score_name=self.score_name,
            action=action,
            params=params,
        )
        self._write_event(control_event)

    def log_external_input(
        self,
        vk_code: int,
        scan_code: int,
        source: Literal["self", "user", "injected"],
        injected_flag: bool,
    ):
        """记录外部输入事件"""
        input_event = ExternalInputEvent(
            session_id=self.session_id,
            score_name=self.score_name,
            vk_code=vk_code,
            scan_code=scan_code,
            source=source,
            injected_flag=injected_flag,
        )
        self._write_event(input_event)

    def log_modifier_anomaly(
        self,
        modifier: Literal["LButton", "RButton", "Shift"],
        expected: bool,
        actual: bool,
    ):
        """记录修饰键异常事件"""
        anomaly_event = ModifierAnomalyEvent(
            session_id=self.session_id,
            score_name=self.score_name,
            modifier=modifier,
            expected=expected,
            actual=actual,
        )
        self._write_event(anomaly_event)

    def log_env(self, event: str, **details):
        """记录环境事件"""
        env_event = EnvEvent(
            session_id=self.session_id,
            score_name=self.score_name,
            event=event,
            details=details,
        )
        self._write_event(env_event)

    def _write_event(self, event: BaseEvent):
        """写入事件到日志文件"""
        if self._fh is None:
            return

        try:
            self._write_line(event.to_dict())
            self._events_count += 1
        except Exception as e:
            # 写入失败时降级，关闭文件句柄
            print(f"[EventLogger] 写入事件失败: {e}")
            try:
                self._fh.close()
            except Exception:
                pass
            self._fh = None

    def _write_line(self, obj: dict):
        """写入一行 JSON"""
        if self._fh:
            self._fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
            self._fh.flush()


# ==================== 操作审计日志 ====================


class AuditLogger:
    """操作审计日志，记录配置变更、谱面导入导出、管理员启动等操作"""

    def __init__(self, log_dir: str = "data/audit_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def log(self, action: str, **details):
        """记录审计事件"""
        try:
            # 每天一个审计日志文件
            date_str = datetime.now().strftime("%Y%m%d")
            log_path = self.log_dir / f"audit_{date_str}.jsonl"

            audit_entry = {
                "timestamp": time.time(),
                "datetime": datetime.now().isoformat(),
                "action": action,
                **details,
            }

            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(audit_entry, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"[AuditLogger] 写入审计日志失败: {e}")


# ==================== 全局实例 ====================

# 单例：演奏事件日志
_event_logger_instance: EventLogger | None = None


def get_event_logger() -> EventLogger:
    """获取全局 EventLogger 实例"""
    global _event_logger_instance
    if _event_logger_instance is None:
        _event_logger_instance = EventLogger()
    return _event_logger_instance


# 单例：操作审计日志
_audit_logger_instance: AuditLogger | None = None


def get_audit_logger() -> AuditLogger:
    """获取全局 AuditLogger 实例"""
    global _audit_logger_instance
    if _audit_logger_instance is None:
        _audit_logger_instance = AuditLogger()
    return _audit_logger_instance
