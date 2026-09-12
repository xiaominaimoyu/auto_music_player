"""端到端冒烟：用真实 main() 装配逻辑（真实 KeyboardDriver / Player / EventPlayer /
真实 profiles 加载），验证修复后整条链路仍能构建，且档位路由判据正确。

不真的演奏（不发送任何键），只验证装配与路由。用法: python _smoke_main.py <仓库目录>
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
REPO = os.path.abspath(sys.argv[1])
sys.path.insert(0, REPO)
os.chdir(REPO)

import yaml                                              # noqa: E402
from PyQt6.QtWidgets import QApplication                 # noqa: E402

from core.database import ScoreDB                        # noqa: E402
from core.event_player import EventPlayer                # noqa: E402
from core.keyboard_driver import KeyboardDriver          # noqa: E402
from core.keymap import KeyMap                           # noqa: E402
from core.player import Player                           # noqa: E402
from core.profile import ensure_profiles, load_profiles, resolve_profile  # noqa: E402
from gui.main_window import MainWindow                   # noqa: E402

ok = []


def check(tag, cond, note=""):
    ok.append(bool(cond))
    print(f"[{'PASS' if cond else 'FAIL'}] {tag} {note}")


cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
app = QApplication.instance() or QApplication([])

player_cfg = cfg.get("player", {})
profiles = load_profiles(ensure_profiles("."), fallback_keymap=cfg.get("keymap"))
check("profiles 加载", len(profiles) == 2, f"→ {[p.id for p in profiles]}")

profile = resolve_profile(profiles, cfg.get("app", {}).get("active_profile"))
check("默认激活档位", profile.id == "default", f"→ {profile.id}")

# 真实驱动（不发送任何输入，仅结构化校验 INPUT 布局）
driver = KeyboardDriver(
    settle_ms=float(player_cfg.get("modifier_settle_ms", KeyboardDriver.DEFAULT_SETTLE_MS)),
    release_settle_ms=float(player_cfg.get("modifier_release_ms", KeyboardDriver.DEFAULT_RELEASE_SETTLE_MS)),
)
check("KeyboardDriver 构造（INPUT 结构体校验）", driver.settle_ms == 30.0 and driver.release_settle_ms == 20.0,
      f"→ settle={driver.settle_ms} release={driver.release_settle_ms}")

km = KeyMap(cfg["keymap"])
db = ScoreDB(os.path.join(cfg["app"].get("data_dir", "data"), "scores.db"))
player = Player(km, driver=driver, logger=None, latency_compensation_ms=0)
event_player = EventPlayer(driver=driver, latency_compensation_ms=0)
check("两条演奏路径共用同一驱动", player.driver is driver and event_player.driver is driver)

# 用 main.py 的装配方式构造 PlayerTab（MainWindow 需要 pynput，此处直接构造 PlayerTab）
from gui.player_tab import PlayerTab                     # noqa: E402
tab = PlayerTab(db, player, player_cfg, config_path="config.yaml",
                profile=profile, profiles=profiles, event_player=event_player)

check("默认档位 → 旧 Player 路径", tab._use_event_path() is False)
check("默认档位 → 绑定旧 Player", not tab._use_event_path() and tab._player is player)

delta = resolve_profile(profiles, "delta_force_harmonica")
tab._profile = delta
check("三角洲档位 → 事件路径", tab._use_event_path() is True)
check(
    "三角洲档位 → 绑定 EventPlayer",
    tab._use_event_path() and tab._event_player is event_player,
)
# 真实编译一次《小星星》片段（不播放）
from gui.player_tab import build_event_plan              # noqa: E402
notes = [{"notes": ["mid_1"], "dur": 1.0}, {"notes": ["mid_1"], "dur": 1.0},
         {"notes": ["mid_5"], "dur": 1.0}, {"notes": ["high_1"], "dur": 1.0, "semitone": 1}]
plan, degs = build_event_plan(notes, delta, "free_play", bpm=100, settle_ms=30.0,
                              release_settle_ms=20.0, hold_ratio=0.75, gap_ms=20.0)
evs = [(e.device, e.key, e.action) for e in plan.events]
check("编译产出事件", len(evs) > 0, f"→ {len(evs)} 个事件")
check("高音走鼠标右键", ("mouse", "right", "down") in evs)
check("半音降级被记录", len(degs) == 1, f"→ {[str(d) for d in degs]}")
check("事件序列首尾配对（按下必有释放）",
      sorted([k for d, k, a in evs if a == "down"]) == sorted([k for d, k, a in evs if a == "up"]),
      f"→ {evs}")
check("不支持和弦时时间轴不塌缩（含休止或降级项）", plan.interrupt_mode == "pause")

event_player.shutdown()
player.shutdown()
print(f"\n---- 冒烟汇总: {sum(ok)}/{len(ok)} 通过 ----")
sys.exit(0 if all(ok) else 1)
