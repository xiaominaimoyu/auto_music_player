"""曲谱 IR → 输入事件序列编译器。

设计要点
- **纯函数**:不依赖 GUI / 游戏 / 时钟,单测可完全确定。
- **顺序固化**:`修饰键 down → settle → 音键 down → hold → 音键 up → release_settle → 修饰键 up`。
  顺序颠倒会让最后一个音被错误转调(修饰态是 latch-while-held),故只在此处实现一次。
- **策略参数化**:修饰态冲突与和弦各一组枚举策略,降级决策不散落在业务代码里。
- **不叠键**:下一音起点取 `max(音乐时值终点, 上一音完全释放)`,对应调研「不要多点」。
- **降级可追溯**:每一次降级都产出 `Degradation` 条目,供 GUI 汇总与导出。
"""

from dataclasses import dataclass, field, replace
from enum import Enum

from core.ir import IRChord, IRNote, IRRest, OCTAVE_TO_NAME

MAX_BPM = 300


class Modifier(Enum):
    """修饰态。四态互斥(决策 D4),不做 bitmask。"""
    NATURAL = "natural"
    LOWER = "lower"       # 鼠标左键:降调(低八度)
    SEMITONE = "semitone"  # 鼠标中键:升半音
    HIGHER = "higher"     # 鼠标右键:升调(高八度)


class ModifierPolicy(Enum):
    """八度与半音同时被请求时的冲突策略。"""
    OCTAVE_FIRST = "octave_first"        # 保八度、丢半音(默认)
    REJECT_NOTE = "reject_note"          # 拒绝该音
    KEEP_ACCIDENTAL = "keep_accidental"  # 保半音、丢八度


class ChordPolicy(Enum):
    """和弦降级策略(三角洲倾向不支持和弦)。"""
    CHORD_FIRST = "first"            # 取首个音(默认)
    CHORD_HIGHEST = "highest"        # 取最高音(旋律优先)
    CHORD_REJECT = "reject"          # 丢弃整个和弦
    CHORD_ARPEGGIATE = "arpeggiate"  # 拆成琶音,时值均分


DEFAULT_MODIFIER_BUTTONS = {
    Modifier.LOWER: "left",
    Modifier.SEMITONE: "middle",
    Modifier.HIGHER: "right",
}


@dataclass(frozen=True)
class InputEvent:
    t_ms: float
    device: str   # "kb" | "mouse"
    key: str
    action: str   # "down" | "up"
    # 对应原始谱面元素。None 表示场景收尾等非谱面事件。
    source_index: int | None = None
    # True 表示该谱面元素已经完整发声；EventPlayer 据此在暂停时从完整音符边界续播。
    source_end: bool = False


@dataclass(frozen=True)
class Degradation:
    """一次降级记录。index 为元素序号(从 0 计)。"""
    index: int
    requested: str
    actual: str
    reason: str

    def __str__(self):
        return f"元素 {self.index + 1}: {self.requested} → {self.actual}({self.reason})"


@dataclass(frozen=True)
class CompileParams:
    bpm: int = 100
    # 音位键,从左到右对应 1-7
    pitch_keys: tuple = ("Z", "X", "C", "V", "B", "N", "M")
    modifier_buttons: dict = field(default_factory=lambda: dict(DEFAULT_MODIFIER_BUTTONS))
    # 修饰键按下 → 音键按下的间隔
    settle_ms: float = 30.0
    # 音键松开 → 修饰键松开的间隔
    release_settle_ms: float = 20.0
    # 按住时长占时值的比例
    hold_ratio: float = 1.0
    # 单音按住上限(毫秒);None 表示不限制
    max_hold_ms: float | None = 3000.0
    gap_ms: float = 0.0
    # 真人化计划的最小键间隔。旧调用未传入时保持 1ms 的安全下限。
    min_gap_ms: float = 1.0
    # 物理直达键覆盖表,如 {"high_1": ","};命中后不再按修饰键(决策 D3)
    pitch_direct_overrides: dict = field(default_factory=dict)
    modifier_policy: ModifierPolicy = ModifierPolicy.OCTAVE_FIRST
    chord_policy: ChordPolicy = ChordPolicy.CHORD_FIRST
    # True: 修饰键提前到拍点之前按下，音键落在拍上。装不下时才推迟该音。
    modifier_prepress: bool = True
    # True: gap 加在时值后面（旧行为，节奏被拉长）。False: gap 从时值里扣出。
    additive_gap: bool = False
    # True: 整段使用修正前的调度循环（间隙外加 + 修饰键推迟并推后时间轴）。
    legacy_timing: bool = False

    def __post_init__(self):
        if not 1 <= int(self.bpm) <= MAX_BPM:
            raise ValueError(f"bpm 必须在 1-{MAX_BPM}: {self.bpm}")
        if len(self.pitch_keys) != 7:
            raise ValueError(f"pitch_keys 必须包含 7 个按键: {self.pitch_keys}")
        if not 0 < float(self.hold_ratio) <= 1.0:
            raise ValueError(f"hold_ratio 必须在 (0, 1]: {self.hold_ratio}")
        if self.max_hold_ms is not None and float(self.max_hold_ms) <= 0:
            raise ValueError(f"max_hold_ms 必须为正数或 None: {self.max_hold_ms}")
        for k in ("settle_ms", "release_settle_ms", "gap_ms", "min_gap_ms"):
            if float(getattr(self, k)) < 0:
                raise ValueError(f"{k} 不能为负数: {getattr(self, k)}")


@dataclass
class CompileResult:
    events: list = field(default_factory=list)
    degradations: list = field(default_factory=list)
    note_count: int = 0     # 实际发声的元素数
    skipped: int = 0        # 被丢弃的元素数
    duration_ms: float = 0.0


# ---------------------------------------------------------------- 冲突解析

def resolve_modifier(octave: int, semitone: int,
                     policy: ModifierPolicy = ModifierPolicy.OCTAVE_FIRST):
    """(八度位移, 半音) → (修饰态 或 None, 降级记录 或 None)。

    降级记录的 index 为 -1,由 `compile_score` 填入真实序号。
    None 修饰态表示该音被丢弃(REJECT_NOTE)。
    """
    if semitone and octave != 0:
        requested = f"{OCTAVE_TO_NAME[octave]}#"
        if policy is ModifierPolicy.OCTAVE_FIRST:
            mod = Modifier.LOWER if octave < 0 else Modifier.HIGHER
            return mod, Degradation(-1, requested, OCTAVE_TO_NAME[octave], policy.value)
        if policy is ModifierPolicy.KEEP_ACCIDENTAL:
            return Modifier.SEMITONE, Degradation(-1, requested, "semitone", policy.value)
        return None, Degradation(-1, requested, "skipped", policy.value)

    if semitone:
        return Modifier.SEMITONE, None
    if octave < 0:
        return Modifier.LOWER, None
    if octave > 0:
        return Modifier.HIGHER, None
    return Modifier.NATURAL, None


def _chord_desc(chord: IRChord) -> str:
    return "chord(" + ",".join(n.note_id for n in chord.notes) + ")"


def resolve_chord(chord: IRChord, policy: ChordPolicy = ChordPolicy.CHORD_FIRST):
    """和弦 → (单音列表, 降级记录或 None)。空列表表示该和弦被丢弃。"""
    if policy is ChordPolicy.CHORD_FIRST:
        return [chord.notes[0]], Degradation(-1, _chord_desc(chord),
                                             chord.notes[0].note_id, policy.value)
    if policy is ChordPolicy.CHORD_HIGHEST:
        note = max(chord.notes, key=lambda item: (item.octave, item.pitch, item.semitone))
        return [note], Degradation(-1, _chord_desc(chord), note.note_id, policy.value)
    if policy is ChordPolicy.CHORD_REJECT:
        return [], Degradation(-1, _chord_desc(chord), "skipped", policy.value)
    n = len(chord.notes)
    sub = [IRNote(p.pitch, p.octave, p.semitone, chord.dur / n) for p in chord.notes]
    return sub, Degradation(-1, _chord_desc(chord), f"arpeggio×{n}", policy.value)


# ---------------------------------------------------------------- 编译

def _expand(el, params: CompileParams):
    """元素 → (单音列表, 降级记录或 None, 休止时长或 None)。"""
    if isinstance(el, IRRest):
        return [], None, el.dur
    if isinstance(el, IRNote):
        return [el], None, None
    if isinstance(el, IRChord):
        notes, deg = resolve_chord(el, params.chord_policy)
        return notes, deg, None
    raise TypeError(f"未知 IR 元素类型: {type(el)!r}")


def _flatten(elements, params: CompileParams, source_index_offset: int = 0):
    """IR 序列 → 扁平计划(plan)。

    plan 项:{"kind": "note"|"rest", "dur", "button", "key", "index", "source_index"}
    被丢弃的元素(和弦 / 冲突拒绝)转为等长休止,保证时间轴不塌缩。
    """
    plan, degradations, skipped = [], [], 0
    for idx, el in enumerate(elements):
        notes, deg, rest_dur = _expand(el, params)
        if deg is not None:
            degradations.append(replace(deg, index=idx))

        source_index = idx + source_index_offset
        if rest_dur is not None:
            plan.append({"kind": "rest", "dur": rest_dur, "index": idx,
                         "source_index": source_index})
            continue
        if not notes:
            skipped += 1
            plan.append({"kind": "rest", "dur": el.dur, "index": idx,
                         "source_index": source_index})
            continue

        for note in notes:
            mod, mod_deg = resolve_modifier(note.octave, note.semitone,
                                            params.modifier_policy)
            if mod_deg is not None:
                degradations.append(replace(mod_deg, index=idx))
            if mod is None:
                skipped += 1
                plan.append({
                    "kind": "rest",
                    "dur": note.dur,
                    "index": idx,
                    "source_index": source_index,
                })
                continue
            # 物理直达键命中则不再按修饰键(决策 D3)
            # 半音守卫:带 semitone 的音不能走直达键(直达键不含半音修饰态),
            # 必须走修饰键路径以正确按下 SEMITONE 修饰键
            override = None
            if not note.semitone:
                override = params.pitch_direct_overrides.get(note.note_id)
            if override:
                key, button = override, None
            else:
                key = params.pitch_keys[note.pitch - 1]
                button = params.modifier_buttons.get(mod)
            plan.append({"kind": "note", "dur": note.dur, "button": button,
                         "key": key, "index": idx, "source_index": source_index})
    return plan, degradations, skipped


def compile_score(elements, params: CompileParams, timings=None, *, source_index_offset: int = 0) -> CompileResult:
    """把 IR 序列编译成输入事件序列。

    timings(可选):与 elements 等长的真人化塑形结果,由
    `core.humanize.plan_timings` 产出——每元素 (offset_ms, hold_ratio);
    休止元素与 None 时保持机械时序(hold 用 params.hold_ratio)。

    默认调度把 gap 从音符时值里扣成松开间隙，并在拍点之前预按修饰键，
    使音键落在拍上。``legacy_timing`` 回到修正前的循环。
    """
    if int(source_index_offset) < 0:
        raise ValueError("source_index_offset 不能为负数")
    source_index_offset = int(source_index_offset)
    plan, degradations, skipped = _flatten(elements, params, source_index_offset)
    if params.legacy_timing:
        return _schedule_legacy(plan, params, timings, degradations, skipped)
    return _schedule_faithful(plan, params, timings, degradations, skipped)


def _schedule_legacy(plan, params, timings, degradations, skipped) -> CompileResult:
    """修正前的调度：gap 加在时值后，修饰键切换把 nominal 时间轴往后推。"""
    beat_ms = 60000.0 / max(1, int(params.bpm))
    gap = float(params.gap_ms)
    settle = float(params.settle_ms)
    release_settle = float(params.release_settle_ms)
    result = CompileResult(degradations=degradations, skipped=skipped)
    # cursor 是未加 jitter 的逻辑时间轴。真人化只改变当前音的实际起音，
    # 不能反向推移下一拍，否则随机偏移会在长曲中累计漂移。
    cursor = 0.0
    held = None
    prev_kb_up = None   # 上一音符音键抬起时刻(链式防叠键下限)

    for i, item in enumerate(plan):
        if item["kind"] == "rest":
            if held:
                result.events.append(InputEvent(
                    cursor, "mouse", held, "up", item["source_index"]
                ))
                cursor += release_settle
                held = None
            cursor += item["dur"] * beat_ms + gap
            continue

        if timings is not None:
            off_ms, ratio_i = timings[item["index"]]
        else:
            off_ms, ratio_i = 0.0, params.hold_ratio
        nominal_t = cursor
        t = max(0.0, nominal_t + float(off_ms))
        # 链式防叠键:仅真人化模式需要——偏移可能把起音提前到上一音释放之前;
        # 机械模式下 cursor 已保证 t ≥ 上一音释放,保持精确时序不变
        if timings is not None and prev_kb_up is not None:
            t = max(t, prev_kb_up + float(params.min_gap_ms))

        nxt = plan[i + 1] if i + 1 < len(plan) else None
        source_end = nxt is None or nxt["source_index"] != item["source_index"]
        button = item["button"]
        if button and button != held:
            if held:                      # 切换修饰键:先松旧的,留足间隔再按新的
                result.events.append(InputEvent(t, "mouse", held, "up", item["source_index"]))
                t += release_settle
                nominal_t += release_settle
            result.events.append(InputEvent(t, "mouse", button, "down", item["source_index"]))
            t += settle
            nominal_t += settle
            held = button

        dur_ms = item["dur"] * beat_ms
        hold = dur_ms * float(ratio_i)
        if params.max_hold_ms is not None and hold > float(params.max_hold_ms):
            hold = float(params.max_hold_ms)
            result.degradations.append(Degradation(
                item["index"], f"hold {dur_ms:.0f}ms", f"{hold:.0f}ms", "MAX_HOLD"))

        t_up = t + hold
        nominal_t_up = nominal_t + hold
        result.events.append(InputEvent(t, "kb", item["key"], "down", item["source_index"]))
        result.events.append(InputEvent(
            t_up, "kb", item["key"], "up", item["source_index"], source_end=source_end
        ))
        prev_kb_up = t_up

        release_end = t_up
        nominal_release_end = nominal_t_up
        next_button = nxt["button"] if (nxt and nxt["kind"] == "note") else None
        if held and held != next_button:
            release_end = t_up + release_settle
            nominal_release_end = nominal_t_up + release_settle
            result.events.append(InputEvent(
                release_end, "mouse", held, "up", item["source_index"]
            ))
            held = None

        result.note_count += 1
        # 保持理想拍点；只有真实的键释放会撞到下一拍时，才以物理下限延后。
        cursor = max(
            nominal_t + dur_ms + gap,
            nominal_release_end + gap,
            release_end + gap,
        )

    if held:                              # 收尾兜底
        result.events.append(InputEvent(cursor, "mouse", held, "up"))
        held = None

    result.events.sort(key=lambda e: (e.t_ms, 0 if e.action == "up" else 1))
    if result.events:
        result.duration_ms = result.events[-1].t_ms
    return result


_MIN_HOLD_MS = 1.0


class _Placed:
    """一个已经落到时间轴上的发音音符，修饰键时刻在循环结束后再生成。"""

    def __init__(self, item, musical, key_down, key_up, button, source_end):
        self.item = item
        self.musical = musical
        self.key_down = key_down
        self.key_up = key_up
        self.button = button
        self.source_end = source_end
        self.mod_down = None
        self.release_modifier = False

    @property
    def hold_ms(self):
        return self.key_up - self.key_down


def _offset_and_ratio(item, timings, params):
    if timings is None:
        return 0.0, float(params.hold_ratio)
    off, ratio = timings[item["index"]]
    if ratio is None:
        ratio = params.hold_ratio
    return float(off), float(ratio)


def _transition_lead(held, button, settle, release_settle):
    """上一音键松开到本音键按下之间，修饰态切换所需要的时间。

    同态（含都是自然音）为 0。预按成功时这段时间从上一音的按住里扣出，
    音键仍落在拍点上。
    """
    if held == button:
        return 0.0
    if button and not held:
        return settle
    if held and not button:
        return release_settle
    if held and button:
        return release_settle + settle
    return 0.0


def _try_shorten(note: _Placed, needed_up: float) -> bool:
    """把上一音的松开时刻提前到 ``needed_up``。提前不到最短按住就放弃。"""
    if needed_up >= note.key_up - 1e-9:
        return True
    if note.hold_ms <= _MIN_HOLD_MS:
        return False
    min_up = note.key_down + _MIN_HOLD_MS
    if needed_up < min_up - 1e-6:
        return False
    note.key_up = needed_up
    return True


def _schedule_faithful(plan, params, timings, degradations, skipped) -> CompileResult:
    """节奏忠实调度。

    音键默认落在谱面拍点上。gap 是拍点前的最小松开间隙，从上一音的按住里扣。
    修饰键在拍点前 ``settle_ms`` 预按；只有上一音短到削不出这段时间时，
    才推迟当前音并记一条 PREPRESS_DELAY。
    """
    beat_ms = 60000.0 / max(1, int(params.bpm))
    gap = float(params.gap_ms)
    settle = float(params.settle_ms)
    release_settle = float(params.release_settle_ms)
    prepress = bool(params.modifier_prepress)
    additive = bool(params.additive_gap)
    min_gap = float(params.min_gap_ms) if timings is not None else 0.0

    musical = []
    cursor = 0.0
    for item in plan:
        musical.append(cursor)
        cursor += float(item["dur"]) * beat_ms
        if additive:
            cursor += gap

    if prepress:
        for index, item in enumerate(plan):
            if item["kind"] != "note" or not item["button"]:
                if item["kind"] == "note":
                    break
                continue
            off, _ = _offset_and_ratio(item, timings, params)
            earliest_key = musical[index] + off
            if earliest_key < settle:
                shift = settle - earliest_key
                musical = [value + shift for value in musical]
            break

    result = CompileResult(degradations=degradations, skipped=skipped)
    placed: list[_Placed] = []
    prev: _Placed | None = None
    held = None

    for index, item in enumerate(plan):
        if item["kind"] == "rest":
            if held and prev is not None:
                prev.release_modifier = True
            held = None
            continue

        off, ratio = _offset_and_ratio(item, timings, params)
        dur_ms = float(item["dur"]) * beat_ms
        desired = dur_ms * ratio
        if params.max_hold_ms is not None and desired > float(params.max_hold_ms):
            desired = float(params.max_hold_ms)
            result.degradations.append(Degradation(
                item["index"], f"hold {dur_ms:.0f}ms", f"{desired:.0f}ms", "MAX_HOLD"))

        onset = musical[index]
        ideal = onset + off
        if ideal < 0.0:
            ideal = 0.0
        target = ideal
        button = item["button"]
        lead = _transition_lead(held, button, settle, release_settle)
        next_item = plan[index + 1] if index + 1 < len(plan) else None
        next_is_attack = next_item is not None and next_item["kind"] == "note"
        # 松开间隙只加在两个相邻发音之间；休止本身就是间隔，不再额外加一层。
        prev_is_attack = index > 0 and plan[index - 1]["kind"] == "note"
        floor_gap = gap if (not additive and prev_is_attack) else 0.0

        if prev is not None and prepress:
            shorten_need = max(floor_gap, lead)
            if shorten_need > 0.0:
                _try_shorten(prev, target - shorten_need)

        if prev is not None:
            floor = prev.key_up + max(floor_gap, min_gap, lead)
            if target < floor:
                if prepress and lead > 0.0 and ideal < floor - 0.5:
                    result.degradations.append(Degradation(
                        item["index"],
                        "on-beat",
                        f"+{floor - ideal:.0f}ms",
                        "PREPRESS_DELAY",
                    ))
                target = floor

        if not prepress and lead > 0.0 and prev is None:
            # 曲首没有可以预按的空白，音键只能落在修饰键 settle 之后。
            target = max(target, ideal + lead)

        if not additive and gap > 0.0 and next_is_attack:
            room = (onset + dur_ms - gap) - target
            if room >= desired:
                hold = desired
            elif room >= _MIN_HOLD_MS:
                hold = room
            else:
                hold = desired
        else:
            hold = desired
        if hold < 0.0:
            hold = 0.0

        nxt = next_item
        source_end = nxt is None or nxt["source_index"] != item["source_index"]
        note = _Placed(item, onset, target, target + hold, button, source_end)
        if button and button != held:
            note.mod_down = target - settle
        if held and held != button and prev is not None:
            prev.release_modifier = True
        held = button
        placed.append(note)
        prev = note
        result.note_count += 1

    if held and prev is not None:
        prev.release_modifier = True

    for note in placed:
        if note.mod_down is not None:
            result.events.append(InputEvent(
                note.mod_down, "mouse", note.button, "down", note.item["source_index"]
            ))
        result.events.append(InputEvent(
            note.key_down, "kb", note.item["key"], "down", note.item["source_index"]
        ))
        result.events.append(InputEvent(
            note.key_up, "kb", note.item["key"], "up",
            note.item["source_index"], source_end=note.source_end,
        ))
        if note.release_modifier and note.button:
            result.events.append(InputEvent(
                note.key_up + release_settle, "mouse", note.button, "up",
                note.item["source_index"],
            ))

    _shift_nonnegative(result.events)
    result.events.sort(key=lambda e: (e.t_ms, 0 if e.action == "up" else 1))
    if result.events:
        result.duration_ms = result.events[-1].t_ms
    return result


def _shift_nonnegative(events):
    """预按若仍落到 0 之前，整段平移，起音间隔保持不变。"""
    if not events:
        return
    for index, event in enumerate(events):
        if -1e-4 < event.t_ms < 0.0:
            events[index] = replace(event, t_ms=0.0)
    earliest = min(event.t_ms for event in events)
    if earliest < -1e-4:
        events[:] = [replace(event, t_ms=event.t_ms - earliest) for event in events]
