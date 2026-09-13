"""日志导出和历史管理功能。

提供三种导出格式：
1. JSONL - 原始事件流
2. CSV - 音符演奏数据表格
3. Markdown - 人类可读的摘要报告

历史日志管理：
- 列表显示所有历史日志
- 批量导出和删除
- 自动清理策略（7 天或 50MB）
"""

import json
import os
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

# ==================== JSONL 读取和解析 ====================


def read_jsonl_log(log_path: str | Path) -> list[dict]:
    """读取 JSONL 日志文件，返回所有事件"""
    events = []
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    events.append(json.loads(line))
    except Exception as e:
        print(f"读取日志失败 {log_path}: {e}")
    return events


def get_session_info(events: list[dict]) -> dict:
    """从事件流中提取会话信息"""
    session_start = next((e for e in events if e.get("type") == "session_start"), {})
    session_end = next((e for e in events if e.get("type") == "session_end"), {})

    return {
        "session_id": session_start.get("session_id", ""),
        "score_name": session_start.get("score_name", "未知曲目"),
        "bpm": session_start.get("bpm", 0),
        "note_count": session_start.get("note_count", 0),
        "started_at": session_start.get("started_at", ""),
        "ended_at": session_end.get("ended_at", ""),
        "completed": session_end.get("completed", 0),
        "total": session_end.get("total", 0),
        "stopped_early": session_end.get("stopped_early", False),
        "events_count": session_end.get("events_count", 0),
    }


# ==================== CSV 导出 ====================


def export_to_csv(log_path: str | Path, output_path: str | Path) -> bool:
    """导出为 CSV 格式（仅包含音符事件）"""
    try:
        events = read_jsonl_log(log_path)
        note_events = [e for e in events if e.get("type") == "note"]

        with open(output_path, "w", encoding="utf-8-sig") as f:
            # CSV 表头
            f.write("音符索引,音符,按键,预期时间,实际时间,偏差(ms),是否成功\n")

            for event in note_events:
                index = event.get("index", 0)
                notes = ",".join(event.get("notes", []))
                keys = ",".join(event.get("keys", []))
                expected = round(event.get("expected_time", 0.0), 3)
                actual = round(event.get("actual_time", 0.0), 3)
                deviation = round(event.get("deviation_ms", 0.0), 1)
                success = "是" if event.get("success", True) else "否"

                f.write(
                    f"{index},{notes},{keys},{expected},{actual},{deviation},{success}\n"
                )

        return True
    except Exception as e:
        print(f"导出 CSV 失败: {e}")
        return False


# ==================== Markdown 导出 ====================


def export_to_markdown(log_path: str | Path, output_path: str | Path) -> bool:
    """导出为 Markdown 摘要报告"""
    try:
        events = read_jsonl_log(log_path)
        session_info = get_session_info(events)

        note_events = [e for e in events if e.get("type") == "note"]
        focus_events = [e for e in events if e.get("type") == "focus"]
        control_events = [e for e in events if e.get("type") == "control"]
        anomaly_events = [e for e in events if e.get("type") == "modifier_anomaly"]
        env_events = [e for e in events if e.get("type") == "env"]
        external_events = [e for e in events if e.get("type") == "external_input"]

        with open(output_path, "w", encoding="utf-8") as f:
            # 标题
            score_name = session_info.get("score_name", "未知曲目")
            started_at = session_info.get("started_at", "")
            f.write(f"# 演奏报告 - {score_name}\n\n")
            f.write(f"**演奏时间**: {started_at}\n\n")
            f.write(f"**会话 ID**: {session_info.get('session_id', '')}\n\n")
            f.write("---\n\n")

            # 基本信息
            f.write("## 📊 基本信息\n\n")
            f.write(f"- **BPM**: {session_info.get('bpm', 0)}\n")
            f.write(f"- **总音符数**: {session_info.get('note_count', 0)}\n")
            f.write(f"- **完成音符**: {session_info.get('completed', 0)}\n")
            f.write(
                f"- **是否提前停止**: {'是' if session_info.get('stopped_early', False) else '否'}\n"
            )
            f.write(f"- **总事件数**: {session_info.get('events_count', 0)}\n\n")

            # 成功率统计
            if note_events:
                f.write("## 📈 成功率统计\n\n")
                total_notes = len(note_events)
                success_notes = sum(1 for e in note_events if e.get("success", True))
                success_rate = (
                    success_notes / total_notes * 100 if total_notes > 0 else 0
                )

                f.write(f"- **总音符数**: {total_notes}\n")
                f.write(f"- **成功**: {success_notes} ({success_rate:.2f}%)\n")
                f.write(
                    f"- **失败**: {total_notes - success_notes} ({100 - success_rate:.2f}%)\n\n"
                )

                # 偏差分布
                f.write("## 📉 偏差分布\n\n")
                deviations = [e.get("deviation_ms", 0.0) for e in note_events]
                if deviations:
                    avg_dev = sum(abs(d) for d in deviations) / len(deviations)
                    max_dev = max(abs(d) for d in deviations)

                    f.write(f"- **平均偏差**: {avg_dev:.1f} ms\n")
                    f.write(f"- **最大偏差**: {max_dev:.1f} ms\n\n")

                    # 分布区间统计
                    ranges = [
                        (-float("inf"), -50),
                        (-50, -20),
                        (-20, -10),
                        (-10, 0),
                        (0, 10),
                        (10, 20),
                        (20, 50),
                        (50, float("inf")),
                    ]

                    f.write("### 偏差区间分布\n\n")
                    f.write("| 区间 | 数量 | 占比 |\n")
                    f.write("|------|------|------|\n")

                    for start, end in ranges:
                        if start == -float("inf"):
                            count = sum(1 for d in deviations if d < end)
                            label = f"< {end}ms"
                        elif end == float("inf"):
                            count = sum(1 for d in deviations if d >= start)
                            label = f">= {start}ms"
                        else:
                            count = sum(1 for d in deviations if start <= d < end)
                            label = f"[{start}, {end})ms"

                        ratio = count / len(deviations) * 100 if deviations else 0
                        f.write(f"| {label} | {count} | {ratio:.1f}% |\n")

                    f.write("\n")

            # 异常事件时间线
            anomalies = (
                focus_events
                + anomaly_events
                + env_events
                + [e for e in external_events if e.get("source") == "injected"]
            )

            if anomalies:
                f.write("## ⚠️ 异常事件时间线\n\n")

                # 按时间排序
                anomalies.sort(key=lambda e: e.get("timestamp", 0))

                for event in anomalies:
                    timestamp = event.get("timestamp", 0)
                    time_str = datetime.fromtimestamp(timestamp).strftime("%H:%M:%S")
                    event_type = event.get("type", "")

                    if event_type == "focus":
                        focus_event = event.get("event", "")
                        window_title = event.get("window_title", "")
                        if focus_event == "lost":
                            f.write(
                                f"- **{time_str}** - [焦点丢失] 窗口切换至 `{window_title}`\n"
                            )
                        else:
                            f.write(f"- **{time_str}** - [焦点恢复] 回到目标窗口\n")

                    elif event_type == "modifier_anomaly":
                        modifier = event.get("modifier", "")
                        expected = event.get("expected", False)
                        actual = event.get("actual", False)
                        f.write(
                            f"- **{time_str}** - [修饰键异常] {modifier} 预期={expected}, 实际={actual}\n"
                        )

                    elif event_type == "env":
                        env_event = event.get("event", "")
                        details = event.get("details", {})
                        f.write(
                            f"- **{time_str}** - [环境事件] {env_event} {details}\n"
                        )

                    elif event_type == "external_input":
                        vk_code = event.get("vk_code", 0)
                        f.write(
                            f"- **{time_str}** - [第三方注入] 检测到外部程序注入按键 VK={vk_code}\n"
                        )

                f.write("\n")

            # 控制操作记录
            if control_events:
                f.write("## 🎮 控制操作记录\n\n")
                for event in control_events:
                    timestamp = event.get("timestamp", 0)
                    time_str = datetime.fromtimestamp(timestamp).strftime("%H:%M:%S")
                    action = event.get("action", "")
                    params = event.get("params", {})

                    f.write(f"- **{time_str}** - {action}")
                    if params:
                        f.write(f" {params}")
                    f.write("\n")

                f.write("\n")

            f.write("---\n\n")
            f.write("*此报告由 AutoMusicPlayer 自动生成*\n")

        return True
    except Exception as e:
        print(f"导出 Markdown 失败: {e}")
        return False


# ==================== 历史日志管理 ====================


class LogManager:
    """历史日志管理器"""

    def __init__(
        self, log_dir: str = "data/play_logs", export_dir: str = "data/exports"
    ):
        self.log_dir = Path(log_dir)
        self.export_dir = Path(export_dir)
        self.export_dir.mkdir(parents=True, exist_ok=True)

    def list_logs(self) -> list[dict]:
        """列出所有历史日志，返回元信息列表"""
        logs = []

        if not self.log_dir.exists():
            return logs

        for log_file in self.log_dir.glob("*.jsonl"):
            try:
                events = read_jsonl_log(log_file)
                session_info = get_session_info(events)

                # 统计异常事件
                anomaly_count = sum(
                    1
                    for e in events
                    if e.get("type") in ["focus", "modifier_anomaly", "env"]
                )
                has_anomaly = anomaly_count > 0

                logs.append(
                    {
                        "filename": log_file.name,
                        "path": str(log_file),
                        "score_name": session_info.get("score_name", "未知曲目"),
                        "started_at": session_info.get("started_at", ""),
                        "duration": self._calculate_duration(session_info),
                        "completed": session_info.get("completed", 0),
                        "total": session_info.get("total", 0),
                        "has_anomaly": has_anomaly,
                        "anomaly_count": anomaly_count,
                        "file_size": log_file.stat().st_size,
                        "modified_time": log_file.stat().st_mtime,
                    }
                )
            except Exception as e:
                print(f"解析日志失败 {log_file}: {e}")

        # 按修改时间倒序排序
        logs.sort(key=lambda x: x["modified_time"], reverse=True)
        return logs

    def _calculate_duration(self, session_info: dict) -> str:
        """计算演奏时长"""
        try:
            started = datetime.fromisoformat(session_info.get("started_at", ""))
            ended = datetime.fromisoformat(session_info.get("ended_at", ""))
            duration = (ended - started).total_seconds()

            minutes = int(duration // 60)
            seconds = int(duration % 60)
            return f"{minutes}:{seconds:02d}"
        except Exception:
            return "未知"

    def export_log(
        self, log_path: str, format: Literal["jsonl", "csv", "markdown"]
    ) -> str | None:
        """导出单个日志，返回导出文件路径"""
        try:
            log_path = Path(log_path)
            base_name = log_path.stem

            if format == "jsonl":
                # 直接复制原文件
                output_path = self.export_dir / f"{base_name}.jsonl"
                import shutil

                shutil.copy2(log_path, output_path)

            elif format == "csv":
                output_path = self.export_dir / f"{base_name}.csv"
                if not export_to_csv(log_path, output_path):
                    return None

            elif format == "markdown":
                output_path = self.export_dir / f"{base_name}.md"
                if not export_to_markdown(log_path, output_path):
                    return None

            else:
                return None

            return str(output_path)
        except Exception as e:
            print(f"导出失败: {e}")
            return None

    def delete_logs(self, log_paths: list[str]) -> int:
        """批量删除日志，返回成功删除的数量"""
        deleted = 0
        for log_path in log_paths:
            try:
                Path(log_path).unlink()
                deleted += 1
            except Exception as e:
                print(f"删除失败 {log_path}: {e}")
        return deleted

    def auto_cleanup(
        self, max_age_days: int = 7, max_total_size_mb: int = 50
    ) -> tuple[int, int]:
        """自动清理日志，返回 (删除文件数, 释放空间 MB)"""
        if not self.log_dir.exists():
            return 0, 0

        logs = list(self.log_dir.glob("*.jsonl"))
        if not logs:
            return 0, 0

        # 按修改时间排序
        logs.sort(key=lambda x: x.stat().st_mtime, reverse=True)

        # 计算总大小
        total_size = sum(log.stat().st_size for log in logs)
        total_size_mb = total_size / (1024 * 1024)

        # 计算时间阈值
        cutoff_time = (datetime.now() - timedelta(days=max_age_days)).timestamp()

        deleted_count = 0
        freed_size = 0

        # 删除超过时间阈值的日志
        for log in logs:
            if log.stat().st_mtime < cutoff_time:
                size = log.stat().st_size
                try:
                    log.unlink()
                    deleted_count += 1
                    freed_size += size
                except Exception as e:
                    print(f"删除失败 {log}: {e}")

        # 如果总大小仍然超标，删除最旧的日志
        if total_size_mb > max_total_size_mb:
            # 重新获取日志列表
            logs = list(self.log_dir.glob("*.jsonl"))
            logs.sort(key=lambda x: x.stat().st_mtime)

            current_size = sum(log.stat().st_size for log in logs)

            for log in logs:
                if current_size / (1024 * 1024) <= max_total_size_mb:
                    break

                size = log.stat().st_size
                try:
                    log.unlink()
                    deleted_count += 1
                    freed_size += size
                    current_size -= size
                except Exception as e:
                    print(f"删除失败 {log}: {e}")

        freed_size_mb = freed_size / (1024 * 1024)
        return deleted_count, int(freed_size_mb)
