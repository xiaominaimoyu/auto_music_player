"""运行时探针（只读验证，不修改被测仓库）：
在真实 delta_force_harmonica 档位下加载真实 PlayerTab，验证：

  P1  _stop() / F8 / 焦点暂停 是否作用到正在演奏的 EventPlayer
  P2  _check_focus() 在事件路径下是否自我解除
  P3  build_event_plan() 是否把 semitones 注入 from_storage（半音链路是否有源）
  P4  profiles/delta_force_harmonica.yaml 的真实键位与档位绑定
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
# 用法: python _probe_runtime.py [仓库根目录]
# 默认跑原始快照;传入 amp_fixed 可对修复版做回归。
if len(sys.argv) > 1:
    REPO = os.path.abspath(sys.argv[1])
else:
    REPO = r"D:\DeepSeek Harness\amp_main\auto_music_player-main"
sys.path.insert(0, REPO)
print(f"### 被测仓库: {REPO}")
FIXED = "fixed" in os.path.basename(REPO).lower()
print(f"### 副本判定: {'修复版 amp_fixed(断言契约)' if FIXED else '原始快照(断言缺陷)'}")

# 期望值随被测副本切换:同一份探针既能复现缺陷,也能验证修复。
# "正确契约" = 修复后应有的行为;原始快照在这些点上必然相反。
EXP_HOTKEY_TARGET = "player_tab.play_stop" if FIXED else "player.stop"
EXP_CLEANUP_EVENT = FIXED          # 退出清理应调用 EventPlayer.shutdown
EXP_ATEXIT_EVENT = FIXED           # atexit 应注册 EventPlayer.shutdown
EXP_SEMITONES = "有值" if FIXED else None   # build_event_plan 应注入半音
EXP_IR_ACCEPTS_HASH = FIXED        # IR 应接受 'low_1#' 形态(经 semitone 字段)
EXP_STATUS = "契约通过" if FIXED else "缺陷复现"

from PyQt6.QtCore import QObject, pyqtSignal          # noqa: E402
from PyQt6.QtWidgets import QApplication              # noqa: E402

from core import ir as ir_mod                         # noqa: E402
from core.profile import load_profiles, resolve_profile  # noqa: E402
from gui import player_tab as pt                      # noqa: E402


class SpyPlayer(QObject):
    """冒充演奏器：记录被调用情况。旧 Player 与 EventPlayer 共用此形状。"""
    progress = pyqtSignal(int, int)
    finished = pyqtSignal(bool)
    paused = pyqtSignal(int, int)
    aborted = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self, name):
        super().__init__()
        self.name = name
        self.stopped = 0
        self.aborted_calls = []
        self.shutdowns = 0
        self.playing = False

    def play(self, *a, **k):
        self.playing = True

    def stop(self):
        self.stopped += 1

    def abort(self, reason=""):
        self.aborted_calls.append(reason)

    def shutdown(self, *a, **k):
        self.shutdowns += 1

    @property
    def is_playing(self):
        return self.playing


class FakeDB:
    def list_scores(self):
        return []

    def get_score(self, sid):
        return None


app = QApplication(sys.argv)
results = []


def check(tag, actual, expected, note=""):
    ok = actual == expected
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {tag}: 实际={actual!r} 期望={expected!r} {note}")


# ---------- 装配真实档位 ----------
profiles = load_profiles(os.path.join(REPO, "profiles"), fallback_keymap=None)
delta = resolve_profile(profiles, "delta_force_harmonica")
legacy = SpyPlayer("legacy")
event = SpyPlayer("event")
tab = pt.PlayerTab(FakeDB(), legacy, {}, config_path=None,
                   profile=delta, profiles=profiles, event_player=event)

print("=== P4 档位真实取值 ===")
check("档位 id", delta.id, "delta_force_harmonica")
check("档位分组", delta.group, "三角洲行动")
check("pitch_keys", tuple(delta.pitch_keys), ("Z", "X", "C", "V", "B", "N", "M"))
check("low/high 修饰键", (delta.modifier_buttons.get("lower"), delta.modifier_buttons.get("higher")),
      ("left", "right"))
check("直达键", delta.pitch_direct_overrides.get("high_1"), ",")
check("legacy_keymap(None→走事件路径)", delta.legacy_keymap, None)
check("_use_event_path()", tab._use_event_path(), True)

print()
print("=== P1 停止路由：演奏中调用 _stop() ===")
event.playing = True
legacy.playing = False
tab._stop()
check("EventPlayer.stop 被调用次数", event.stopped, 1, "← 期望停掉正在演奏的 EventPlayer")
check("旧 Player.stop 被调用次数", legacy.stopped, 0, "← 不应误伤旧 Path")

print()
print("=== P1b 焦点暂停路由 ===")
event.playing = True
legacy.playing = False
tab._pause_for_focus()
check("焦点暂停后 EventPlayer.stop 次数", event.stopped, 2)
check("焦点暂停后 旧 Player.stop 次数", legacy.stopped, 0)

print()
print("=== P2 焦点检测守卫：事件路径演奏中 ===")


class FakePolicy:
    """伪造已武装的焦点锁定策略:目标已锁定,且判定为"焦点已丢失"。"""
    locked = True
    target = {"title": "假游戏窗口"}

    def evaluate(self, current):
        return True          # 触发 _pause_for_focus


class FakeWatcher:
    def is_available(self):
        return True

    def capture_current(self):
        return {"title": "别的窗口"}


event.playing = True
legacy.playing = False
tab._watcher = FakeWatcher()
tab._focus_timer.stop()
tab._policy = FakePolicy()
tab._check_focus()
check("焦点检测未被自我解除（policy 仍生效）", isinstance(tab._policy, FakePolicy), True,
      "← 若守卫用旧 Player.is_playing,policy 会被清空/计时器停止")
check("焦点丢失已触发 EventPlayer 暂停", event.stopped > 2, True,
      f"← 实际 stop 次数={event.stopped},期望继续增长即证明事件路径能被焦点暂停")

print()
print("=== P1c F8 热键绑定的目标（源码文本检查，避免依赖 pynput） ===")
import re  # noqa: E402
mw_src = open(os.path.join(REPO, "gui", "main_window.py"), encoding="utf-8").read()
hotkey_lines = [l.strip() for l in mw_src.splitlines()
                if "GlobalHotKeys" in l or "stop_hotkey" in l]
joined = " ".join(hotkey_lines)
check(f"F8 回调目标（期望 {EXP_HOTKEY_TARGET}）",
      EXP_HOTKEY_TARGET in joined, True, f"实际源码: {hotkey_lines}")
def _method_body(src, name, next_names):
    """粗略截取某个方法的源码正文。"""
    seg = src.split(f"def {name}")[1]
    for n in next_names:
        if f"def {n}" in seg:
            seg = seg.split(f"def {n}")[0]
    return seg


cleanup_src = _method_body(mw_src, "_cleanup_on_quit", ["_build_ui", "def "])
check(f"退出清理调用 EventPlayer.shutdown（期望 {EXP_CLEANUP_EVENT}）",
      ("event_player" in cleanup_src), EXP_CLEANUP_EVENT,
      "← 未调用则播放中退出时，鼠标修饰键可能残留在游戏里")
main_src = open(os.path.join(REPO, "main.py"), encoding="utf-8").read()
atexit_lines = [l.strip() for l in main_src.splitlines() if "atexit.register" in l]
check(f"atexit 注册 EventPlayer.shutdown（期望 {EXP_ATEXIT_EVENT}）",
      any("event" in l for l in atexit_lines), EXP_ATEXIT_EVENT,
      f"实际源码: {atexit_lines}")

print()
print("=== P3 半音链路：存储项带 semitone 时是否真的进到 IR ===")
captured = {}
orig = ir_mod.from_storage


def spy_from_storage(items, semitones=None):
    """记录调用形状与解析结果——契约是"半音进了 IR",不限定走参数还是字段。"""
    out = orig(items, semitones)
    captured["semitones_arg"] = semitones
    captured["ir_semitones"] = [getattr(el, "semitone", None) for el in out]
    return out


pt.ir_mod.from_storage = spy_from_storage
try:
    pt.build_event_plan([{"notes": ["low_1"], "dur": 1.0, "semitone": 1},
                         {"notes": ["mid_2"], "dur": 1.0}],
                        delta, "free_play", bpm=100, settle_ms=30.0,
                        release_settle_ms=20.0, hold_ratio=0.75, gap_ms=20.0)
finally:
    pt.ir_mod.from_storage = orig
got = captured.get("ir_semitones")
check(f"半音进入 IR（期望 {EXP_SEMITONES!r}，实际 {got!r}）",
      got, [1, 0] if FIXED else [0, 0],
      f"← 参数={captured.get('semitones_arg')!r};None 且 IR=[0,0] 表示半音链路是死的")

print()
print("=== P3b 存储项携带 semitone 字段时 IR 是否接受 ===")
try:
    els = ir_mod.from_storage([{"notes": ["low_1"], "dur": 1.0, "semitone": 1}])
    got = (els[0].octave, els[0].pitch, els[0].semitone)
    check(f"IR 读出 semitone（期望 {EXP_IR_ACCEPTS_HASH}）",
          els[0].semitone == 1, EXP_IR_ACCEPTS_HASH,
          f"← 实际解析为 octave/pitch/semitone={got}")
except Exception as e:
    check(f"IR 读出 semitone（期望 {EXP_IR_ACCEPTS_HASH}）",
          f"{type(e).__name__}: {e}", EXP_IR_ACCEPTS_HASH)

print()
print("=== P3c low_1 + 半音 应降级为纯低音（octave_first）且不再走直达键 ===")
try:
    plan, degs = pt.build_event_plan(
        [{"notes": ["high_1"], "dur": 0.5, "semitone": 1}],
        delta, "free_play", bpm=100, settle_ms=30.0,
        release_settle_ms=20.0, hold_ratio=0.75, gap_ms=20.0)
    evs = [(e.device, e.key, e.action) for e in plan.events]
    reasons = [d.reason for d in degs]
    if FIXED:
        # 有半音时不得命中 "," 直达键,必须走 右键 + M → 实际应是 high_1 键 Z? 不:high_1 用 pitch_keys[0]=Z 加右键
        check("带半音的 high_1 未命中 ',' 直达键", (",", "down") not in [(e.key, e.action) for e in plan.events], True,
              f"事件={evs} 降级={reasons}")
        check("半音被记录为降级", len(degs) >= 1, True, f"降级={[str(d) for d in degs]}")
    else:
        check("原始快照:带半音的 high_1 仍命中 ',' 且无降级（缺陷）",
              (",", "down") in [(e.key, e.action) for e in plan.events] and not degs, True,
              f"事件={evs} 降级={reasons}")
except Exception as e:
    check("P3c 执行", f"{type(e).__name__}: {e}", "无异常")

print()
print(f"---- 汇总: {sum(results)}/{len(results)} 通过 ----")
