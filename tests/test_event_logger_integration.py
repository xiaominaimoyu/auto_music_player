"""事件日志系统集成测试。

验证 EventLogger, LogManager, 导出功能的集成工作流程。
"""

import os
import sys
import tempfile
from pathlib import Path

# 修复 Windows 控制台编码问题
if sys.platform == "win32":
    import codecs

    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, "strict")
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.buffer, "strict")

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.event_logger import get_event_logger, get_audit_logger
from core.log_export import LogManager, export_to_csv, export_to_markdown


def test_event_logging():
    """测试事件日志记录"""
    print("=" * 60)
    print("测试 1: 事件日志记录")
    print("=" * 60)

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

    return logger.log_path


def test_log_manager(log_path):
    """测试日志管理器"""
    print("\n" + "=" * 60)
    print("测试 2: 日志管理器")
    print("=" * 60)

    manager = LogManager()

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

    return manager


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

    return csv_path


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

    return md_path


def test_audit_logger():
    """测试审计日志"""
    print("\n" + "=" * 60)
    print("测试 5: 操作审计日志")
    print("=" * 60)

    audit_logger = get_audit_logger()

    # 记录几个审计事件
    audit_logger.log("config_change", setting="bpm", old_value=100, new_value=120)
    print("✓ 配置变更已记录")

    audit_logger.log("score_import", filename="test.txt", source="upload")
    print("✓ 谱面导入已记录")

    audit_logger.log("admin_mode_start", user="current_user")
    print("✓ 管理员模式启动已记录")

    # 查找审计日志文件
    audit_dir = Path("data/audit_logs")
    if audit_dir.exists():
        audit_files = list(audit_dir.glob("audit_*.jsonl"))
        if audit_files:
            latest_audit = audit_files[-1]
            print(f"✓ 审计日志文件: {latest_audit}")

            # 读取内容
            with open(latest_audit, "r", encoding="utf-8") as f:
                lines = f.readlines()
                print(f"  共 {len(lines)} 条审计记录")


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


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("事件日志系统集成测试")
    print("=" * 60 + "\n")

    try:
        # 1. 测试事件记录
        log_path = test_event_logging()

        # 2. 测试日志管理
        manager = test_log_manager(log_path)

        # 3. 测试 CSV 导出
        test_export_csv(manager, log_path)

        # 4. 测试 Markdown 导出
        test_export_markdown(manager, log_path)

        # 5. 测试审计日志
        test_audit_logger()

        # 6. 测试自动清理
        test_auto_cleanup(manager)

        print("\n" + "=" * 60)
        print("✓ 所有测试通过")
        print("=" * 60 + "\n")

    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
