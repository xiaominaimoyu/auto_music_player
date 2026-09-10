# M2 IR 与编译器规格（音符 → 输入事件）

> 版本：v1 · 2026-09-10（**已实施完成**）
> 上游：[`三角洲行动口琴-需求分析与集成计划.md`](./三角洲行动口琴-需求分析与集成计划.md) §9 M2
> 前置：M1（`docs/M1-输入驱动改造规格.md`）已完成
> 产出：`core/ir.py`、`core/compiler.py`、`tests/test_ir.py`、`tests/test_compiler.py`

---

## 0. 目标与边界

**做**
- `core/ir.py`：内部音符 IR，`semitone` 从第一天就存在（决策 D1「schema 冻结、内部 IR 不冻结」）
- `core/compiler.py`：IR → `InputEvent` 序列；`resolve_modifier` / `resolve_chord` 纯函数；降级清单
- 全策略枚举 + 单测全覆盖

**不做**
- ❌ 不改持久层 `note_id` 正则（D1）
- ❌ 不改 `core/player.py`（M4）、不引入 profile（M3）、不做 GUI（M4）
- ❌ 不接 `raw_text` 的 `#` 解析（V2，见 §6 已预留通道）

---

## 1. IR 设计（`core/ir.py`）

```python
IRNote(pitch: 1-7, octave: -1|0|+1, semitone: 0|1, dur: 拍)
IRChord(notes: tuple[IRNote], dur)
IRRest(dur)
```

| 关键约束 | 说明 |
|---|---|
| `IRNote.note_id` **不含 semitone** | D1：持久层冻结，`to_storage()` 只写 `high/mid/low_1~7` |
| `from_storage(items, semitones=None)` | `semitones` 为 D8 注入通道；**长度不符直接 `ValueError`，绝不静默错位回填**（D9） |
| 构造即校验 | pitch / octave / semitone / dur 越界立即报错，非法 IR 无法进入编译器 |

---

## 2. 编译器设计（`core/compiler.py`）

### 2.1 修饰态：四态互斥枚举（D4）

```python
Modifier.NATURAL / LOWER(左键) / SEMITONE(中键) / HIGHER(右键)
```

不按 bitmask 建模——互斥模型下 `low + high` 在类型层就不可构造（C6）。

### 2.2 冲突策略（D2，`resolve_modifier`）

| 请求 | `OCTAVE_FIRST`（默认） | `KEEP_ACCIDENTAL` | `REJECT_NOTE` |
|---|---|---|---|
| `low + #` | `LOWER` + 告警 | `SEMITONE` + 告警 | 丢弃 + 告警 |
| `high + #` | `HIGHER` + 告警 | `SEMITONE` + 告警 | 丢弃 + 告警 |

**注意**：`mid + #`（自然音升半音）**不是冲突**，直接取 `SEMITONE`，无告警。

### 2.3 和弦策略（C7，`resolve_chord`）

`CHORD_FIRST`（取首音，默认）/ `CHORD_REJECT`（丢弃）/ `CHORD_ARPEGGIATE`（均分时长拆琶音）
三种均产出 `Degradation` 条目。

### 2.4 编译时序

```
修饢态 down → settle_ms(30) → 音键 down → hold → 音键 up → release_settle_ms(20) → 修饢态 up
```

**关键实现（与原设计的重要差异）**：修饢键采用「**相邻同态保持按住**」。

> 原设计每个音符都完整 down/up 一次。实测推演发现：连续两个高音会在
> **0ms 内 release 再 press 同一个鼠标键** —— 这正是我在 M1 里批评参考实现
> 5ms settle 的**同类风险**，游戏极可能漏掉重新按下。
> 现改为：只有相邻音符修饰键**不同**时才释放再按下；休止前一率释放（避免长时间按住右键）。

下一音起点取 `max(音乐时值终点, 上一音完全释放)`，保证**不叠键**（调研「不要多点」）。

### 2.5 物理直达键（D3）

`pitch_direct_overrides: {"high_1": ","}` —— 命中后 `button=None`，不再按修饰键。
MVP 只启用 `,`；`.` `/` 证据不足，留待 M0 自检。

---

## 3. 降级清单

```python
Degradation(index, requested, actual, reason)
```

覆盖四类：修饢态冲突、和弦降级、超长截断（`MAX_HOLD`）。`__str__` 形如
`元素 3: low# → low(octave_first)`，供 GUI 汇总计数与导出（UC-12）。

---

## 4. 测试（47 个，全绿）

| 文件 | 用例数 | 覆盖 |
|---|---|---|
| `tests/test_ir.py` | 12 | note_id 不含 semitone、往返稳定、semitone 注入但不落盘、长度不符报错、各类非法入参 |
| `tests/test_compiler.py` | 35 | `resolve_modifier` 6 组冲突全矩阵 + 4 组无冲突；`resolve_chord` 3 策略；事件顺序（自然/高/低/半音）、音位映射 Z-M、同态保持按住、切换释放、休止释放、同音 gap、hold_ratio、max_hold 截断、直达键、降级 index、参数校验 |

---

## 5. 验收结果

- [x] 47 个新用例全绿
- [x] 全量 `Ran 154 tests`（107 + 47），**1 failure + 8 errors 与 M1 后基线完全一致** → 零新增失败
- [x] `py_compile` 通过
- [x] 未改动 `player.py` / `keymap.py` / `gui/` / 既有测试

**环境限制（同 M1）**：本机无 PyQt6，`test_e2e` / `test_latency_calibration` /
`test_play_logger` / `test_player_stop` 四个模块无法导入（改造前即如此）。
`core/ir.py` 与 `core/compiler.py` 均不依赖 PyQt6，可独立验证。

---

## 6. 留给 V2 的开关（D8 / D9）

半音注入通道已就位：`from_storage(items, semitones=[...])`。
`semitone_source="raw_text"` 的**回填解析**依赖 parser 支持 `#` 记号，归 V2；
但 **D9 的长度校验已实现并可测**——长度不符直接拒绝，不会静默错位。

V2 放开 `note_id` 正则时，只需改 `ir.from_storage` / `ir.to_storage`，
**编译器与 Player 一行不动**——这正是 D1 想避免的"三层齐动"。

---

## 7. 与 M4 的接口约定

| M4 需要 | M2 提供 |
|---|---|
| 消费事件序列 | `CompileResult.events: list[InputEvent(t_ms, device, key, action)]`，按时刻升序 |
| 释放兜底 | `KeyboardDriver.panic_release(keys, mouse_buttons)`（M1 已备） |
| 降级提示 | `CompileResult.degradations`，GUI 汇总计数 |
| 参数来源 | `CompileParams`（bpm / settle / release_settle / hold_ratio / max_hold / gap / 策略）由 M3 的 profile 组装 |
