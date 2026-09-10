"""场景适配器(ScenarioAdapter):把编译器产出的事件序列包装成可执行的演奏计划。

**音符层不 fork**:两个场景共用同一份 `InputEvent[]`(由 `compiler.compile_score` 产出),
差异只收敛在会话层——进入方式、退出(submit)、中断语义、音符间隔(决策 D6)。

    FreePlayScenario   S1 局内自由演奏:口琴由用户手动启用,无自动切换,pause 可续播
    NpcQuestScenario   S2 NPC 听旋律复刻任务:post-roll 为 Q 提交,abort 不可续播

**S2 中断规则(重要)**:任务被打断后手动重新掏出口琴也不能提交曲谱,必须重新听 NPC 示范,
故 S2 任何中断都 abort 并提示重来,"断点续播"在 S2 没有意义。
"""

from dataclasses import dataclass, field, replace

from core.compiler import CompileParams, InputEvent

# 场景各自的音符间隔(识别友好:S2 宁可慢、不可叠键,见调研「不要多点」)
FREE_PLAY_INTERVALS = {"settle_ms": 30.0, "gap_ms": 20.0}
NPC_QUEST_INTERVALS = {"settle_ms": 60.0, "gap_ms": 120.0}

DEFAULT_SUBMIT_KEY = "Q"
DEFAULT_SUBMIT_HOLD_MS = 80.0
DEFAULT_TAIL_MS = 200.0      # 最后一个音符落完后,留出收尾再提交


@dataclass(frozen=True)
class ScenarioPlan:
    """一个可直接交给 EventPlayer 的演奏计划。"""
    events: list
    interrupt_mode: str          # "pause" | "abort"
    abort_hint: str              # 中止后给用户的中文提示(空 = 不适用)
    scenario_id: str = ""
    scenario_name: str = ""


def _post_roll(submit_key: str, start_ms: float, hold_ms: float) -> list:
    """提交键按下再松开的收尾序列。"""
    return [
        InputEvent(start_ms, "kb", submit_key, "down"),
        InputEvent(start_ms + float(hold_ms), "kb", submit_key, "up"),
    ]


class Scenario:
    """场景适配器基类。"""
    id = "scenario"
    name = "场景"
    interrupt_mode = "pause"
    abort_hint = ""

    def intervals(self) -> dict:
        """本场景对 CompileParams 的间隔覆盖项。"""
        return dict(FREE_PLAY_INTERVALS)

    def apply_intervals(self, params: CompileParams) -> CompileParams:
        """在组装好的 CompileParams 上覆盖本场景的间隔(不改原对象)。"""
        return replace(params, **self.intervals())

    def plan(self, music_events) -> ScenarioPlan:
        raise NotImplementedError


class FreePlayScenario(Scenario):
    """S1 局内自由演奏。

    口琴由**用户手动从背包/EDC 启用**(自动 pre-roll 已移除),程序不做任何切换;
    事件就是编译器产出的音符序列,不加收尾。
    """

    id = "free_play"
    name = "自由演奏"
    interrupt_mode = "pause"
    abort_hint = ""

    def intervals(self) -> dict:
        return dict(FREE_PLAY_INTERVALS)

    def plan(self, music_events) -> ScenarioPlan:
        return ScenarioPlan(
            events=list(music_events),
            interrupt_mode=self.interrupt_mode,
            abort_hint=self.abort_hint,
            scenario_id=self.id,
            scenario_name=self.name,
        )


class NpcQuestScenario(Scenario):
    """S2 NPC 听旋律复刻任务。

    无 pre-roll(NPC 示范后自动装备);post-roll 为 `Q` 提交;
    中断即 abort 并提示重听 NPC 示范。
    """

    id = "npc_quest"
    name = "NPC 听旋律任务"
    interrupt_mode = "abort"
    abort_hint = "演奏被中断,请重新听 NPC 示范后再演奏"

    def __init__(self, submit_key: str = DEFAULT_SUBMIT_KEY,
                 submit_hold_ms: float = DEFAULT_SUBMIT_HOLD_MS,
                 tail_ms: float = DEFAULT_TAIL_MS):
        self.submit_key = submit_key
        self.submit_hold_ms = submit_hold_ms
        self.tail_ms = tail_ms

    def intervals(self) -> dict:
        return dict(NPC_QUEST_INTERVALS)

    def plan(self, music_events) -> ScenarioPlan:
        events = list(music_events)
        start_ms = (events[-1].t_ms + self.tail_ms) if events else self.tail_ms
        events.extend(_post_roll(self.submit_key, start_ms, self.submit_hold_ms))
        return ScenarioPlan(
            events=events,
            interrupt_mode=self.interrupt_mode,
            abort_hint=self.abort_hint,
            scenario_id=self.id,
            scenario_name=self.name,
        )


# ---------------------------------------------------------------- 注册表

SCENARIOS = {
    FreePlayScenario.id: FreePlayScenario,
    NpcQuestScenario.id: NpcQuestScenario,
}


def get_scenario(scenario_id: str, **kwargs) -> Scenario:
    """按 id 取场景适配器;未知 id 报错。"""
    cls = SCENARIOS.get(scenario_id)
    if cls is None:
        raise ValueError(f"未知场景 id: {scenario_id!r}(可选: {sorted(SCENARIOS)})")
    return cls(**kwargs)
