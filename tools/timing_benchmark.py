"""干跑节奏基准：对比修正前后的起音间隔，并测量等待误差。

不发送按键。在 Windows 上可直接看出 15ms 定时器抖动和修饰键预按的差别。

    python tools/timing_benchmark.py
    python tools/timing_benchmark.py --legacy
"""

from __future__ import annotations

import argparse
import statistics
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.compiler import CompileParams, compile_score
from core.ir import IRNote
from core.timing import slot_hold_and_advance, wait_until


def _kb_downs(events):
    return [event.t_ms for event in events if event.device == "kb" and event.action == "down"]


def _summarize(name, intervals, expected):
    if not intervals:
        print(f"{name}: no intervals")
        return
    ratios = [item / expected for item in intervals]
    print(f"{name}")
    print(f"  n={len(intervals)} expected_ioi_ms={expected:.1f}")
    print(
        "  ioi_ms mean={:.2f} stdev={:.2f} min={:.2f} max={:.2f}".format(
            statistics.mean(intervals),
            statistics.pstdev(intervals) if len(intervals) > 1 else 0.0,
            min(intervals),
            max(intervals),
        )
    )
    print(
        "  ioi_ratio min={:.4f} max={:.4f}".format(min(ratios), max(ratios))
    )
    print("  ioi_ms=" + ", ".join(f"{item:.1f}" for item in intervals))


def _eighth_pattern():
    # 自然音 / 高音 / 自然音 / 低音，专门踩修饰键切换。
    octaves = (0, 1, 0, -1)
    return [IRNote((index % 7) + 1, octaves[index % 4], 0, 0.5) for index in range(8)]


def schedule_report(legacy: bool):
    params = CompileParams(
        bpm=120,
        settle_ms=30.0,
        release_settle_ms=20.0,
        hold_ratio=1.0,
        max_hold_ms=None,
        gap_ms=20.0,
        legacy_timing=legacy,
        modifier_prepress=not legacy,
        additive_gap=legacy,
    )
    compiled = compile_score(_eighth_pattern(), params)
    intervals = []
    downs = _kb_downs(compiled.events)
    intervals = [right - left for left, right in zip(downs, downs[1:])]
    label = "legacy scheduler" if legacy else "faithful scheduler"
    _summarize(f"{label} 120BPM eighths + modifier switches, gap 20ms", intervals, 250.0)
    delays = [item for item in compiled.degradations if item.reason == "PREPRESS_DELAY"]
    print(f"  prepress_delays={len(delays)} note_count={compiled.note_count}")

    holds = []
    for dur, ratio, followed in ((500.0, 0.75, True), (250.0, 1.0, True), (6.0, 1.0, True)):
        hold, advance = slot_hold_and_advance(
            dur, ratio, 20.0, additive=legacy, followed_by_attack=followed,
        )
        holds.append((dur, ratio, hold, advance))
    print("  21-key slots (dur_ms, hold_ratio -> hold_ms, advance_ms):")
    for dur, ratio, hold, advance in holds:
        print(f"    {dur:.0f} x {ratio:.2f} -> hold {hold:.1f}, advance {advance:.1f}")


def wait_report(precise: bool, count: int, interval_ms: float):
    stop = threading.Event()
    interval_s = interval_ms / 1000.0
    start = time.perf_counter()
    actual = []
    for index in range(count):
        deadline = start + (index + 1) * interval_s
        if wait_until(stop, deadline, precise=precise):
            break
        actual.append((time.perf_counter() - start) * 1000.0)
    deviations = [stamp - (index + 1) * interval_ms for index, stamp in enumerate(actual)]
    intervals = [right - left for left, right in zip(actual, actual[1:])]
    mode = "precise" if precise else "coarse Event.wait"
    print(f"wait dry-run {mode} n={len(actual)} interval_ms={interval_ms:.1f}")
    if not deviations:
        print("  no samples")
        return
    print(
        "  dev_ms mean={:.3f} stdev={:.3f} max_abs={:.3f}".format(
            statistics.mean(deviations),
            statistics.pstdev(deviations) if len(deviations) > 1 else 0.0,
            max(abs(item) for item in deviations),
        )
    )
    if intervals:
        print(
            "  measured_ioi_ms mean={:.3f} stdev={:.3f}".format(
                statistics.mean(intervals),
                statistics.pstdev(intervals) if len(intervals) > 1 else 0.0,
            )
        )


def main():
    parser = argparse.ArgumentParser(description="干跑节奏基准，不发送按键")
    parser.add_argument("--legacy", action="store_true", help="只打印修正前的调度")
    parser.add_argument("--wait-ms", type=float, default=40.0, help="等待干跑的间隔毫秒")
    parser.add_argument("--count", type=int, default=16, help="等待干跑的次数")
    parser.add_argument("--skip-wait", action="store_true", help="只打印编译时间轴")
    args = parser.parse_args()

    if args.legacy:
        schedule_report(True)
    else:
        schedule_report(False)
        print()
        schedule_report(True)
    if args.skip_wait:
        return
    print()
    wait_report(not args.legacy, args.count, args.wait_ms)
    if not args.legacy:
        print()
        wait_report(False, args.count, args.wait_ms)


if __name__ == "__main__":
    main()
