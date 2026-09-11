# 半音(# 升半音)链路补齐:parser → IR → 编译器 + 倒计时可配

> 本文件是给上游 PR 用的说明草稿。补丁文件:`semitone-feature.patch`
> 基线:`main` @ `14c3565`(v1.1.3)

## 背景

`profiles/delta_force_harmonica.yaml` 定义了四态输入模型,其中 `modifier_buttons` 含
`semitone: middle`(鼠标中键 = 升半音),`core/ir.py` 的 `IRNote` 也从第一天就带
`semitone` 维度、`core/compiler.py` 的 `resolve_modifier` 也实现了半音与八度冲突时的
`OCTAVE_FIRST` 降级——**但这三者之间没有数据源**:

- `gui/player_tab.py` 的 `build_event_plan()` 调 `from_storage(notes)` 时**未传 `semitones`**,
  于是 `core/ir.py` 里 `semi = semitones[i] if semitones is not None else 0` 恒为 0;
- `to_storage()` 不落盘 semitone,存储格式里也没有承载字段;
- `core/parser.py` 的音高记号只认 `'` / `,`,`#` 会被判为"无法识别的内容"而静默丢弃。

结果是:鼠标中键(升半音)与 `Modifier.SEMITONE` 分支、以及"低/高音升半音会被降级"这条
行为**永远不会触发**,README 里相关表述与实现不一致。

另有一处**潜伏缺陷**:`note_id` 结构上不含 semitone,所以一旦半音真被注入,
`high_1#` 会与 `high_1` 命中同一条 `pitch_direct_overrides`(`,`),从而**绕过高八度
修饰键被弹成自然高音 1**。这条在主分支上因为"没有数据源"而暂未暴露,补齐链路后会立刻踩到。

## 改动

| 文件 | 改动 |
|---|---|
| `core/parser.py` | 支持 `#` 升半音标记(`1#` / `1'#` / `1,#`);落在该项的可选字段 `semitone=1`;和弦内任一音带 `#` 即标记整个和弦项 |
| `core/ir.py` | `from_storage` 支持从存储项的 `semitone` 字段读取;`to_storage` 在确有半音时写回该字段。**`note_id` 仍不含半音(D1 不破)** |
| `core/compiler.py` | 物理直达键加半音守卫:带半音的音不做 override,必须走"修饰键 + 音位键" |
| `gui/player_tab.py` | 倒计时秒数改由 `player.countdown_seconds` 配置;降级计数在演奏结束后仍可见(悬停见逐条明细) |
| `config.yaml` | 新增 `player.countdown_seconds: 3` |
| `tests/test_ir.py` | 原 D1 契约测试更新为"半音走可选字段"的新契约 |
| `tests/test_semitone.py` | **新增** 14 项回归用例(parser/IR/直达键守卫/倒计时) |

### 兼容性(重点)

存储 schema **没有破坏性变更**:

- `note_id` 仍是 `high|mid|low_1~7`,安卓端与旧库读到的 JSON 完全不变;
- 新增的 `semitone` 是**可选字段**,只在确有升半音时出现;自然音不写该字段;
- `score_model.validate_notes()` 只校验 `notes`/`dur`、忽略额外字段,因此带该字段的数据
  能正常入库;旧数据缺省 `semitone=0`。

### 关于「D1 冻结」的一处契约调整

原 `tests/test_ir.py::test_semitone_injected_but_not_persisted` 断言
`to_storage()` 丢弃 semitone。本补丁改为"以可选字段承载",故该用例改名为
`test_semitone_roundtrip_via_optional_field` 并断言新行为;同时新增
`test_semitone_read_from_storage_field` / `test_semitone_field_invalid_raises`。

如果维护者更倾向保持"绝不落盘",替代方案是只保留 parser 产出与 IR 读取,让
`to_storage()` 继续丢弃(即不改 D1),但那样半音在"入库→再演奏"后会丢失。
取舍请维护者定夺。

## 验证

- 上游原有测试:**267 → 通过**(打补丁后全套 283 项,含新增 14 项、跳过 2 项)
- 新增 `tests/test_semitone.py`:**14/14 通过**
- 补丁可应用性:18 个 hunk 全部逐字节重建一致(对 `main` @ `14c3565` 干净工作树)
- 端到端:delta 档位下 `high_1#` 不再命中 `,` 直达键,改为走 `mouse right down → kb Z down/up → mouse right up`,
  并产出降级记录 `元素 N: high# → high(octave_first)`

## 复现命令

```bash
git checkout main
git apply -p1 semitone-feature.patch
python -m unittest discover -s tests
```
