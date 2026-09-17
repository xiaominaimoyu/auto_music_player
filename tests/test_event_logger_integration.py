"""事件日志系统集成测试。

验证 EventLogger, LogManager, 导出功能的集成工作流程。
"""

import os
import sys
from pathlib import Path

import pytest

# 修复 Windows 控制台编码问题
if sys.platform == "win32":
    import codecs

    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, "strict")
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.buffer, "strict")

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import core.event_logger as event_logger_mod
from core.event_logger import EventLogger, AuditLogger, get_audit_logger, get_event_logger
from core.log_export import LogManager


def _prepare_loggers(tmp_path, monkeypatch):
    play_dir = tmp_path / "play_logs"
    audit_dir = tmp_path / "audit_logs"
    export_dir = tmp_path / "exports"
    event_logger = EventLogger(str(play_dir))
    audit_logger = AuditLogger(str(audit_dir))
    monkeypatch.setattr(event_logger_mod, "_event_logger_instance", event_logger)
    monkeypatch.setattr(event_logger_mod, "_audit_logger_instance", audit_logger)
    return play_dir, audit_dir, export_dir


def _make_sample_log(tmp_path, monkeypatch):
    play_dir, _audit_dir, export_dir = _prepare_loggers(tmp_path, monkeypatch)
    logger = get_event_logger()
    logger.start_session("测试曲目", bpm=120, note_count=20)
    logger.log_control("start", bpm=120, score_name="测试曲目")
    logger.log_focus("lost", window_title="Chrome", hwnd=12345, note_index=5)
    logger.log_focus("regained", window_title="Game", hwnd=67890, note_index=5)
    logger.log_control("pause", note_index=10)
    logger.end_session(completed=10, total=20, stopped_early=True)
    return logger.log_path, LogManager(str(play_dir), str(export_dir))


@pytest.fixture
def log_bundle(tmp_path, monkeypatch):
    return _make_sample_log(tmp_path, monkeypatch)


@pytest.fixture
def log_path(log_bundle):
    return log_bundle[0]


@pytest.fixture
def manager(log_bundle):
    return log_bundle[1]


def test_event_logging(tmp_path, monkeypatch):
    """测试事件日志记录"""
    print("=" * 60)
    print("测试 1: 事件日志记录")
    print("=" * 60)

    _prepare_loggers(tmp_path, monkeypatch)
    logger = get_event_logger()

    # 开始会话
    session_id = logger.start_session("测试曲目", bpm=120, note_count=20)
    print(f"✓ 会话已创建: {session_id}")

    # 记录各类事件
    logger.log_control("start", bpm=120, score_name="测试曲目")
    print("✓ 控制事件已记录")

    logger.log_focus("lost", window_title="Chrome", hwnd=12345, note_index=5)
    print("✓ 焦点事件已记录")

    logger.log_focus("regained", window_title="Game", hwnd=67890, note_index=5)
    print("✓ 焦点恢复事件已记录")

    logger.log_control("pause", note_index=10)
    print("✓ 暂停事件已记录")

    # 结束会话
    logger.end_session(completed=10, total=20, stopped_early=True)
    print(f"✓ 会话已结束")
    print(f"✓ 日志文件: {logger.log_path}")
    assert logger.log_path is not None
    assert logger.log_path.exists()


def test_log_manager(manager, log_path):
    """测试日志管理器"""
    print("\n" + "=" * 60)
    print("测试 2: 日志管理器")
    print("=" * 60)

    # 列出所有日志
    logs = manager.list_logs()
    print(f"✓ 找到 {len(logs)} 个日志文件")

    if logs:
        latest = logs[0]
        print(f"  - 最新日志: {latest['filename']}")
        print(f"  - 曲目: {latest['score_name']}")
        print(f"  - 时间: {latest['started_at']}")
        print(f"  - 异常数: {latest['anomaly_count']}")
        print(f"  - 文件大小: {latest['file_size']} bytes")

    assert isinstance(manager, LogManager)
    assert logs


def test_export_csv(manager, log_path):
    """测试 CSV 导出"""
    print("\n" + "=" * 60)
    print("测试 3: CSV 导出")
    print("=" * 60)

    csv_path = manager.export_log(str(log_path), "csv")
    if csv_path:
        print(f"✓ CSV 已导出: {csv_path}")

        # 读取前几行验证
        with open(csv_path, "r", encoding="utf-8-sig") as f:
            lines = f.readlines()[:5]
            print(f"  前 {len(lines)} 行:")
            for i, line in enumerate(lines, 1):
                print(f"    {i}. {line.strip()}")
    else:
        print("✗ CSV 导出失败")

    assert csv_path
    assert Path(csv_path).exists()


def test_export_markdown(manager, log_path):
    """测试 Markdown 导出"""
    print("\n" + "=" * 60)
    print("测试 4: Markdown 导出")
    print("=" * 60)

    md_path = manager.export_log(str(log_path), "markdown")
    if md_path:
        print(f"✓ Markdown 已导出: {md_path}")

        # 读取前几行验证
        with open(md_path, "r", encoding="utf-8") as f:
            lines = f.readlines()[:15]
            print(f"  前 {len(lines)} 行:")
            for i, line in enumerate(lines, 1):
                print(f"    {i}. {line.rstrip()}")
    else:
        print("✗ Markdown 导出失败")

    assert md_path
    assert Path(md_path).exists()


def test_audit_logger(tmp_path, monkeypatch):
    """测试审计日志"""
    print("\n" + "=" * 60)
    print("测试 5: 操作审计日志")
    print("=" * 60)

    _play_dir, audit_dir, _export_dir = _prepare_loggers(tmp_path, monkeypatch)
    audit_logger = get_audit_logger()

    # 记录几个审计事件
    audit_logger.log("config_change", setting="bpm", old_value=100, new_value=120)
    print("✓ 配置变更已记录")

    audit_logger.log("score_import", filename="test.txt", source="upload")
    print("✓ 谱面导入已记录")

    audit_logger.log("admin_mode_start", user="current_user")
    print("✓ 管理员模式启动已记录")

    # 查找审计日志文件
    if audit_dir.exists():
        audit_files = list(audit_dir.glob("audit_*.jsonl"))
        if audit_files:
            latest_audit = audit_files[-1]
            print(f"✓ 审计日志文件: {latest_audit}")

            # 读取内容
            with open(latest_audit, "r", encoding="utf-8") as f:
                lines = f.readlines()
                print(f"  共 {len(lines)} 条审计记录")
                assert len(lines) >= 3


def test_auto_cleanup(manager):
    """测试自动清理"""
    print("\n" + "=" * 60)
    print("测试 6: 自动清理策略")
    print("=" * 60)

    # 执行自动清理（7天或50MB）
    deleted_count, freed_mb = manager.auto_cleanup(max_age_days=7, max_total_size_mb=50)
    print(f"✓ 自动清理完成")
    print(f"  - 删除文件数: {deleted_count}")
    print(f"  - 释放空间: {freed_mb} MB")
    assert deleted_count >= 0
    assert freed_mb >= 0


def main():
    """兼容旧的脚本入口,实际仍走 pytest fixture。"""
    return pytest.main([__file__])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
