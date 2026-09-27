"""真人化节奏塑形(humanize):把机械等间隔的音符时间轴调成贴近真人演奏的自然表现。

三条塑形规则(纯函数、注入式随机源,完全可测试):
1. 起音微偏移(jitter):每个音符的起始时刻叠加零均值正态偏移——
   真人不可能每个音都精准踩在拍点上;
2. 动态按住(articulation):短音按得轻快(较低 hold),长音按得饱满(较高 hold),
   替代全局固定 hold_ratio,让颗粒感随音符密度自然变化;
3. 乐句呼吸(breath):休止之后、长音结尾之后的下一音稍作停顿再进入,
   模拟换气与重新起手;音高大跳进附加少量换指时间。

防叠键:本模块只产出"相对理想起音时刻的偏移",不直接移动时间轴;
调用方(Player / 编译器)负责链式下限保护——本音符实际起音不早于
上一音符完全释放 + min_gap_ms,因此任何参数下都不会叠键。

输入统一为 storage 格式音符序列 [{"notes": [...], "dur": 拍}, ...],
Player(21 键路径)与 build_event_plan(编译器路径)共用同一份塑形结果。
"""

import random
import secrets
from dataclasses import dataclass

# ≥ 该拍数的音符视为长音,其后的音符按乐句开头处理(换气)
PHRASE_LONG_DUR = 2.0
# 音高差 ≥ 该值视为大跳进(换指需要额外时间)
LEAP_THRESHOLD = 4

_OCTAVE_VAL = {"low": 0, "mid": 7, "high": 14}


@dataclass(frozen=True)
class HumanizeParams:
    """真人化参数。所有幅度都以毫秒计,保持克制——目标是"像人",不是"摇晃"。"""

    jitter_ms: float = 12.0  # 起音正态微偏移 σ(增强:8→12,更自然的时序抖动)
    breath_ms: float = 25.0  # 乐句边界后的呼吸停顿(增强:18→25,更明显的换气感)
    short_hold: tuple = (0.58, 0.70)  # 短音符(≤0.5 拍)hold 区间:轻快断奏(更短促)
    mid_hold: tuple = (0.70, 0.85)  # 中等音符 hold 区间(扩大上限,增加变化)
    long_hold: tuple = (0.86, 0.98)  # 长音符(≥2 拍)hold 区间:绵延饱满(更接近满按)
    leap_ms: float = 5.0  # 大跳进附加换指时间(增强:3→5,大跳更明显停顿)
    leap_threshold: int = LEAP_THRESHOLD
    min_gap_ms: float = 1.5  # 塑形后保证的最小不叠键间隙(放宽:2.0→1.5,允许更连贯)
    # 相邻起音抖动的相关系数。独立白噪声听起来像时钟抖动，适度相关更接近
    # 人在一个乐句内的连续动作；休止会重置该状态。
    jitter_correlation: float = 0.65

    def __post_init__(self):
        if self.jitter_ms < 0 or self.breath_ms < 0 or self.leap_ms < 0:
            raise ValueError("humanize 幅度参数不能为负数")
        if self.min_gap_ms < 0:
            raise ValueError("min_gap_ms 不能为负数")
        if not 0.0 <= float(self.jitter_correlation) < 1.0:
            raise ValueError("jitter_correlation 必须在 [0, 1) 之间")
        for name in ("short_hold", "mid_hold", "long_hold"):
            lo, hi = getattr(self, name)
            if not 0 < lo <= hi <= 1.0:
                raise ValueError(f"{name} 必须满足 0 < lo <= hi <= 1: {(lo, hi)}")


def _pitch_value(note_id: str):
    """note_id → 音高量值(用于跳进判断);无法解析返回 None。

    low=0..6, mid=7..13, high=14..20,跨八度连续。
    """
    name, _, num = str(note_id).partition("_")
    if name not in _OCTAVE_VAL or not num.isdigit() or not 1 <= int(num) <= 7:
        return None
    return _OCTAVE_VAL[name] + int(num)


@dataclass(frozen=True)
class HumanizePlan:
    """一次完整、可追溯的真人化计划。

    ``timings`` 保留旧的 ``[(offset_ms, hold_ratio), ...]`` 结构，方便旧
    播放器和测试继续消费；``seed`` 让两条播放路径能够重放同一份计划。
    """

    timings: tuple
    seed: int


def make_seed() -> int:
    """生成一枚会话级 seed；调用方应把它写入演奏日志。"""

    return secrets.randbits(64)


def build_plan(
    elements,
    params: HumanizeParams | None = None,
    *,
    seed: int | None = None,
    rng: random.Random | None = None,
) -> HumanizePlan:
    """storage 格式音符序列 → 与元素等长的 (offset_ms, hold_ratio) 列表。

    休止元素(notes 为空)的返回项为 (0.0, None)——休止不发声,由调用方跳过;
    但休止参与"乐句边界"判定:休止之后的音符带呼吸停顿。
    """
    p = params or HumanizeParams()
    if rng is None:
        actual_seed = int(seed if seed is not None else make_seed())
        rng = random.Random(actual_seed)
    else:
        # 外部注入 RNG 主要用于旧测试；没有可回读的内部状态时仍提供一个
        # 稳定的日志字段，调用方若需要跨路径一致性应传 seed 而不是 rng。
        actual_seed = int(seed if seed is not None else 0)
    timings = []
    prev_pitch = None  # 上一个发音元素的音高量值
    prev_dur = None  # 上一个元素的时值(拍)
    prev_was_rest = False  # 上一个元素是否为休止
    correlated_jitter = 0.0
    for el in elements:
        dur = float(el.get("dur") or 0)
        ids = list(el.get("notes") or [])
        if not ids:
            # 休止:占位,不发声;重置跳进链(下一音的呼吸已覆盖换指)
            timings.append((0.0, None))
            prev_pitch = None
            prev_dur = dur
            prev_was_rest = True
            correlated_jitter = 0.0
            continue

        independent_jitter = rng.gauss(0.0, p.jitter_ms)
        correlated_jitter = (
            float(p.jitter_correlation) * correlated_jitter
            + (1.0 - float(p.jitter_correlation)) * independent_jitter
        )
        offset = correlated_jitter
        # 乐句边界:前一元素是休止(换气),或长音结尾(重新起手)
        if prev_was_rest or (prev_dur is not None and prev_dur >= PHRASE_LONG_DUR):
            offset += p.breath_ms * rng.uniform(0.6, 1.0)
        # 大跳进:换指需要一点额外时间
        pitch = _pitch_value(ids[0])
        if pitch is not None and prev_pitch is not None:
            if abs(pitch - prev_pitch) >= p.leap_threshold:
                offset += p.leap_ms * rng.uniform(0.5, 1.0)

        # 动态按住:按音符长度分档,同档内随机
        if dur <= 0.5:
            lo, hi = p.short_hold
        elif dur < PHRASE_LONG_DUR:
            lo, hi = p.mid_hold
        else:
            lo, hi = p.long_hold
        hold_ratio = rng.uniform(lo, hi)

        timings.append((offset, hold_ratio))
        prev_pitch = pitch if pitch is not None else prev_pitch
        prev_dur = dur
        prev_was_rest = False
    return HumanizePlan(tuple(timings), actual_seed)


def plan_timings(
    elements,
    params: HumanizeParams | None = None,
    rng: random.Random | None = None,
    *,
    seed: int | None = None,
) -> list:
    """兼容入口：返回旧的 timings 列表。

    新代码应优先使用 :func:`build_plan`，以便保留 seed。
    """

    return list(build_plan(elements, params, seed=seed, rng=rng).timings)
