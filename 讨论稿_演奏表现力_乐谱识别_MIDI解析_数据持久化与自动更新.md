# 统一讨论稿：演奏表现力 · 乐谱识别 · MIDI 解析 · 数据持久化与自动更新

> 调研方式：主 agent 全库只读扫描（未改动任何代码）+ 三个独立视角并行 subagent 调研（音乐表现力 / 乐谱输入识别 / MIDI 解析），议题 4 由主 agent 调研。
> 生成时间：2026-09-26 · 代码基线：v1.4.2（version.txt，commit a9b8649）

---

## 全局现状认知（主 agent 扫描结论）

**技术栈**：Python 3.10+ / PyQt6 / SQLite（mido、pynput、PyInstaller 单文件打包）。
**核心数据流**：

```
简谱文本 ──parser.py──▶ storage [{notes,dur}] ──score_model 校验──▶ SQLite(scores.db)
MIDI 文件 ──midi_reader.py──▶ SourceSong(秒) ──source_score.adapt──▶ storage(同上)
                                    ▲                                    │
录制(recording.py) ────────────────┘                            ir.from_storage
                                                                      │
                     Player(21键,keymap 直查) ◀── humanize.plan_timings ─┤
                     或 compiler.py(IR→InputEvent) → EventPlayer(口琴) ◀┘
                                                                      │
                                              KeyboardDriver(SendInput scancode)
```

**关键架构事实**（影响全部四个议题）：
- 游戏输入层只有 `down/up/按住时长` 三元语义，**无 velocity、无踏板**（`compiler.py:50-59`、`event_player.py:148-155`）。
- 存储 schema 冻结（`ir.py:3-9` 决策 D1）：`high|mid|low_1~7`，semitone 为条目级附加字段；`MAX_DUR_BEATS=16`。
- 两条播放路径共用 `humanize.plan_timings`（`player.py:165`、`gui/player_tab.py:94`）。
- 打包为 PyInstaller **单文件 exe**（`AutoMusicPlayer.spec`），`QtNetwork`/`requests` 均被排除——联网只能用标准库 `urllib`。
- **version.txt 未列入 spec 的 datas**——当前 exe 无法获知自身版本号，这是自动更新的前置缺口。
- 双远端已就绪：GitHub（xiaominaimoyu）+ Gitee（muming04yue）；GitHub 有 releases 实践（`dist/v142-github-verify`）。
- 数据目录解析：`data/` 相对 **config.yaml 所在目录**（`main.py:79-85`），exe 版落在 exe 旁边；`ensure_config` 只在缺失时释放默认 config，不覆盖已有文件。

---

## 议题 1：自动演奏机械化

### 现状说明
表现力参数仅一个维度——按住时长（hold）：`humanize.py:100-107` 按时值分三档随机（短 0.58-0.70 / 中 0.70-0.85 / 长 0.86-0.98）；起音为高斯零均值抖动 σ=12ms（`:90`），乐句边界 +25ms 呼吸（`:92-93`），大跳进 +5ms（`:96-98`）。BPM 全局标量恒定。力度、踏板、声部平衡概念不存在。随机源每次新建无 seed（`player.py:165`）。

### 问题清单
1. **无系统性速度弹性**：零均值 jitter 无乐句级 accelerando/ritardando，无 tempo map。
2. **力度维度完全缺失**：hold 分档与音乐结构（重音、高潮、节拍相位）无关。
3. **jitter 无结构**：纯 iid 高斯，不随节拍层级/乐句位置变化——"白噪声感"来源（真人 timing deviation 与节拍相、乐句边界强相关，Repp/Friberg 研究）。
4. **hold 档位硬编码**：yaml 仅暴露 enabled/jitter/breath 三个旋钮。
5. **随机不可复现**：无 seed，无法 AB 对比与回归测试。
6. **链式钳制不对称**：防叠键下限只挡负偏移，无法"抢拍"（`player.py:220-221`）。

### 候选方案对比

| 方向 | 方案 | 优劣 | 风险 | 改动范围 | 优先级 |
|---|---|---|---|---|---|
| 速度弹性 | A1 乐句级 tempo warp：plan_timings 输出逐音符 tempo_scale（句头+2~4%、句尾 rit.），时间轴按 warp 累积 | 最贴近真人乐句拱形；纯函数可单测 | 与"cursor 不漂移"约束冲突（compiler.py:250 注释）；断点续播起始时刻需同步 warp | humanize + compiler + player 双路径 + 测试 | **P1** |
| 速度弹性 | A2 预设速度曲线：yaml 锚点 [进度%,倍率] 插值 | 用户可编辑、实现简单、运行期零改动 | 逐曲手调；与片段切割交互复杂 | humanize 新增 + config + player_tab UI | P2 |
| 力度近似 | B1 结构感知 hold/accent：节拍强拍、高潮长音 hold↑，经过音 hold↓ | 单音模型下唯一"音量"代理；改动小 | 游戏端音量是否随按住时长变化未经实测 | humanize + compiler:291 消费 + 测试 | **P0** |
| 力度近似 | B2 legato 微重叠 / staccato 断奏（hold>1 或 <0.5） | 连断对比鲜明 | 重叠需多键同按，仅 21 键路径可行，需按 profile 分流 | humanize + 双路径 + profile | P2 |
| 声部平衡 | C2 琶音化+轻短伴奏（chord_policy 已有 ARPEGGIATE，compiler.py:40） | 复用现有机制 | 仅近似 | profile yaml + accent 规则 | P2 |
| 随机可控 | D1 注入持久 seed（plan_timings 已收 rng，humanize.py:66） | 复现/ABX/回归前提；一行级 | 每次演奏相同会弱化卖点——加"每次随机/固定"开关 | player_tab + player + config | **P0** |
| 随机可控 | D2 幅度调制：σ 随密度/BPM/乐句位置缩放 | 消除白噪声感 | 参数标定成本 | humanize + 测试 | P1 |

### 推荐方案与实施顺序
**D1 → B1（P0）→ A1+D2（P1）→ A2/C2/B2（P2）**。

### 评估指标（可验证）
1. **Timing deviation 分布**：event_logger 已逐音记录 deviation_ms（player.py:253-263）；对照真人研究标定 σ、偏度、与节拍相的相关性。
2. **IOI 变异系数**：CV(相邻起音间隔)，机械 ≈0，真人 0.02-0.06。
3. **N-gram 重复度**：两次演奏 timing 序列 3-gram 重合率。
4. **数据集参照**：MAESTRO 子集映射到 21 键域，对比 IOI 分布/rubato 拱形/articulation 比例（KL/DTW）。
5. **ABX 主观协议**：机械 vs v1 vs 改进版，3 档曲目，≥15 人 2AFC + Likert。
6. **硬回归**：叠键率=0、命中率≥99.9%、humanize 结构测试全绿。

### 🔴 需拍板的决策点
- **D-1.1** 是否接受"固定 seed 模式"作为可选开关（默认仍每次随机）？
- **D-1.2** B1 的 accent 规则需要一次**实机实测**：按住时长是否影响游戏内音量——若无效果，力度议题整体降级为纯 articulation。
- **D-1.3** A1 时间轴 warp 会触碰断点续播语义（paused/resume 的时间轴换算），是否接受 P1 阶段动 compiler/player 双路径？

---

## 议题 2：图片/文档乐谱导入（本地/离线，禁外部大模型 API）

### 现状说明
当前为零自识别"人工搬运"链路：`prompt.py` 内置提示词 → 用户粘贴到外部 AI 并上传图片 → AI 输出简谱文本 → `upload_tab.py:424-474` 严格校验（粘连记号致命拒绝）→ `parser.py` 解析入库。`import_any` 只认 `.json/.mid`，不接图片/PDF（`score_io.py:457`）。**注意：已存在一条高质量 MIDI→乐谱适配链路**（选轨/降级报告），这是集成 OMR 的最大资产。

### 问题清单
1. 输入摩擦大，移动端不友好；2. 私有文本协议脆，外部 AI 常违反（strict_ai 致命错误即证据）；3. 外部 AI 格式漂移不可控、无法回归测试；4. 无 PDF/图片输入路径；5. **简谱图无现成主流 OMR 引擎**（国际 OMR 全部面向五线谱）。

### 候选方案对比

| 方案 | 准确率 | 环境要求 | 许可证 | 工程成本 | 集成路径 | 主要风险 |
|---|---|---|---|---|---|---|
| 维持外部 AI 现状 | 取决于外部模型（简谱图实测约 80-95%/曲） | 无 | — | 0 | prompt→parser（现状） | 摩擦与漂移长期存在 |
| **Audiveris 混合方案**（推荐） | 印刷体五线谱中等（官方：手写<40%，复杂管弦 60-75%） | Java 17+，CPU | AGPLv3 | 3-5 人日 | OMR→MusicXML→music21(BSD)→MIDI→**现有 import_midi 全套 UI** | 仅五线谱、不认简谱图；AGPL 需动态调用隔离；+300MB 分发体积 |
| 本地多模态 VLM（Qwen2-VL-2B/MiniCPM-V 端侧） | 无公开 OMR 基准背书；简谱点位/减时线恰是 VLM 结构化输出弱项 | 4-12GB 显存，纯 CPU 过慢 | Apache / 商用受限条款 | 5-8 人日 | VLM→简谱文本→现有 parser（零协议改动） | 准确率不可保证；显存门槛排除部分用户 |
| 简谱专用 OMR（OrpheusNet/音乐123/JPeditor） | OrpheusNet 自报旋律级最高 99%（清晰图）；音乐123 需手动标定、仅宋体 | CPU 可跑 | 学术/混合，需逐个核实 | 5-10 人日（均需二开为库级调用） | OMR→MIDI 或简谱文本→现有链路 | 学术原型泛化差（拍照/手写/多声部崩溃）；维护风险 |
| PDF/Word 文字版简谱 | — | — | — | 1-2 人日 | 文档抽取→parser.py | 版式自由度大，错误率高 |

### 推荐方案与实施顺序
**推荐：Audiveris 混合方案（P1）**——乐谱库页新增"图片/PDF 导入"入口，后台 `Audiveris CLI → MusicXML → music21 → MIDI → import_midi`。理由：①走已测试的既有链路而非新造协议；②信息损失可接受（prompt 本来就丢弃调号/表情/指法）；③music21 转换约 20 行代码。实施顺序：混合方案（P1，3-5 人日）→ 可选 VLM 插件（P2，面向高配用户）→ 跟踪专用简谱 OMR（观察）。

### 🔴 需拍板的决策点
- **D-2.1** 用户谱面中**简谱图 vs 五线谱图**的占比？（决定专用简谱 OMR 的优先级——该方向当前几乎无可用产品）
- **D-2.2** 是否接受分发体积 +300MB（捆绑 Java/MuseScore），还是改为"检测到未安装则引导用户装 MuseScore"？
- **D-2.3** AGPL（Audiveris）动态调用合规性是否接受？（独立进程调用通常合规，但需确认发行方式）
- **D-2.4** 是否维持"外部 AI 路径"作为一等公民？（调研显示外部通用多模态对**简谱图**的实际识别质量目前优于所有本地专用引擎，现状路径不应被低估）

---

## 议题 3：MIDI 导入解析准确性

### 现状说明
调用链：`.mid` → `midi_reader.read_midi_source`（mido，Type 0/1 PPQ，全轨 set_tempo 合并 tempo map，note_on/off 按 (轨,channel,note) deque 配对，绝对秒）→ `source_score.adapt_source_song`（选轨/最高活动音单声部化/80ms 短缝桥接/音高映射→storage）→ 校验入库。**tempo map 合并与 Type 1 tick 处理经验证是正确的**（各轨共享 PPQ，按 tick 全局排序合并）。

### 问题清单（按严重度）
1. **S1（P0 级损失）速度拍平**：首 tempo 定全局 BPM，秒→拍用单一 BPM 换算，`tempo_flattened` 只记录不补偿（source_score.py:554-558）。变速段时值系统性错误（首 120 后 60 的曲子后段误差 100%）。根因是 storage 单 BPM 拍数制的**结构限制**。
2. **S2 零时值装饰音静默删除**：`midi_reader.py:163` `if tick > start_tick` 丢弃且无 warning——倚音直接消失。
3. **S3 CC64 踏板完全忽略**：只处理 track_name/set_tempo/note 三类消息（midi_reader.py:132-146），踏板记谱的钢琴曲时值被截断。
4. **S4 velocity 全链路丢弃**：不进 SourceNote（source_score.py:36-43），存储层无载体（schema 冻结），表现力只能走 humanize 静态塑形。
5. **S5 选轨启发式中等**：density 目标 2.5/s 为拍脑袋值；高音伴奏轨可战胜旋律轨（所幸 UI 可改选）。
6. **S6 轨名解码仅 UTF-8/GB18030**，日文轨名乱码（仅影响显示）。
7. **S7 最高活动音旋律断裂**：声部交叉处旋律低于内声部被丢（polyphony_cleanup 计数）。

### 候选方案对比

| 方案 | 解决问题 | 优劣 | 风险 | 改动范围 | 优先级 |
|---|---|---|---|---|---|
| B 零时值音并入相邻音（记 degradation） | S2 | 小改动、保住时间轴 | 可能误吞真断奏 | midi_reader.py:163 + 1 测试 | **P0** |
| C 多 tempo 显式确认（拒绝/强制单 BPM 确认） | S1 | 诚实、无静默失真 | UX 摩擦 | source_score.py:554 + import_dialog | P1 |
| E CC64 延音：pedal on 时 note_off 延至下一 pedal off | S3 | 钢琴曲时值还原收益最大 | 与 80ms 桥接/MIN_GAME_NOTE 交互需回归 | midi_reader 配对段 | P1 |
| G 选轨加权改进（音符数×时长、density 区间放宽） | S5 | 低成本提命中 | 仍启发式 | source_score.py:193-222 | P2 |
| F velocity 传入 SourceNote→humanize 输入（不落盘） | S4 | 不碰 schema | frozen dataclass 签名变更波及 tests | source_score.py:36 + humanize | P2 |
| D 分段 BPM 入库（schema 扩展） | S1 | 保真最高 | 存储 schema 不支持，架构级工程 | score_model + compiler + DB | P2（单独立项） |
| H 声部跟踪（F0 连续性）替代最高音 | S7 | 旋律连贯性质变 | 算法复杂，需 MIDI 专用测试集 | source_score.py:296-327 | P2 |

**导出链路核对**：Type 0、单 tempo、velocity=100、dur 向上取整——对**已入库曲谱**保真；对"再导回 MIDI 应还原原文件"不成立（导入降维不可逆投影，属设计使然），建议在导出 tooltip 注明。

### 推荐方案与实施顺序
**B（P0）→ C+E（P1）→ G（P2）→ F/D/H（P2，视产品方向）**。F 是议题 1 力度工作的前置依赖，若表现力立项应提前。

### 🔴 需拍板的决策点
- **D-3.1** S1 的取舍：变速曲是"显式拒绝导入"（C）还是接受"能导但时值有误+强提示"？
- **D-3.2** 分段 BPM（方案 D）是否值得单独立项？（根治 S1/S4，但触碰冻结 schema，与议题 1 的 tempo warp 有复用空间）
- **D-3.3** CC64 延音的回归测试基准曲目从哪来（需收集含踏板的公开 MIDI 作 fixtures）？

---

## 议题 4：数据持久化与应用内自动更新

### 现状说明
- **持久化**：SQLite `scores` + `score_preferences`（版本化 JSON，`database.py:130-166`），schema 无 user_version 迁移机制（`CREATE TABLE IF NOT EXISTS`，database.py:21-48）。数据落在 exe 旁 `data/`（`main.py:79-94`）。
- **更新现状**：完全手动——用户需自行下载 zip/exe 覆盖。**没有任何更新检查代码**（全库 grep 确认）。
- **发布设施**：GitHub + Gitee 双远端；GitHub 已有 release 资产（`dist/v142-github-verify`）；version.txt=1.4.2。
- **关键缺口**：①version.txt 未打包进 exe（spec 的 datas 无此项）→ exe 无法自述版本；②QtNetwork/requests 被排除（spec:10-31）→ 联网须用标准库 urllib；③单文件 exe 运行中不可自替换；④config.yaml/settings.json 与用户数据混在 exe 目录，更新流程必须不触碰它们（`ensure_config` 已保证不覆盖，但用户手动解压 zip 会覆盖 config.yaml）。

### 问题清单
1. 更新即覆盖安装，用户数据与程序同目录，存在误删/覆盖风险路径。
2. 无版本自述能力，无法做更新检查。
3. DB 无迁移框架，未来 schema 演进（如议题 3 方案 D 分段 BPM）无安全路径。
4. Gitee 国内可达性好于 GitHub，但 release API 能力弱于 GitHub——需双源策略。

### 候选方案对比

| 方案 | 数据安全 | 工程成本 | 用户体验 | 主要风险 |
|---|---|---|---|---|
| A 维持手动更新 | 依赖用户自律 | 0 | 差 | 数据丢失风险长期存在 |
| B 仅"更新提醒"：启动时 urllib 拉 release API 比对版本，弹窗引导浏览器下载 | 高（不动文件） | 1-2 人日 | 中（仍要手动装） | GitHub 国内可达性——用 Gitee 镜像或双源回退 |
| **C 全量自动更新（推荐）**：后台下载新 exe→临时目录→校验 SHA256→派生 updater 脚本（PyInstaller 单文件不可自替换，需主程序退出后由外部脚本替换并重启） | 高（替换前备份 data/ 清单+替换仅限 exe 本体） | 5-8 人日 | 最好 | 更新失败中途态——需原子替换+失败回滚到旧 exe |
| D 差分/增量更新（bsdiff 或按 zip 资源分块） | 高 | +3-5 人日 | 省流量 | exe 差分脆弱（PyInstaller 单文件几乎整体变化），**不建议** |

### 推荐方案与实施顺序
**推荐：C（全量自动更新）+ B 的检查逻辑作为前置**。
1. **P0 前置**：version 进打包（spec datas 加 version.txt，代码内置 fallback）；DB 迁移框架（`PRAGMA user_version` + 逐级迁移函数 + 启动时迁移，兼容旧版数据，迁移前自动备份 scores.db）。
2. **P1**：更新检查（Gitee release 为主源、GitHub 为回退；urllib 标准库；semver 比对；跳过提示）。
3. **P2**：下载+SHA256 校验（release 资产附 checksum）+ 退出自替换 updater（`cmd /c` 或打包一个小 `_updater.exe` 资源释放）+ 失败回滚 + 更新后首次启动的迁移确认。
4. **数据兼容策略**：schema 只增不改（新增列给默认值）；notes_json 为 schema 冻结格式，永不改变其含义；任何破坏性变更走"迁移函数+备份+失败即回滚"。

### 🔴 需拍板的决策点
- **D-4.1** 更新主源选 **Gitee 还是 GitHub**？（Gitee 国内可达性好、但 release 附件 API 与稳定性弱；建议 Gitee 主源 + GitHub 回退，双源代码量约 +30%）
- **D-4.2** 自动更新是否需要"用户确认后才下载"（推荐，且更新前强制提示备份 data/）还是静默下载+确认安装？
- **D-4.3** 是否把 data/ 迁到 `%LOCALAPPDATA%`（彻底隔离程序与数据，根治覆盖风险）？代价是老用户数据需一次性迁移。
- **D-4.4** 单文件 exe 不可自替换——接受"退出后由临时脚本替换并自动重启"的方案吗？（替代方案是改 onedir 打包，分发形态变化）

---

## 全局实施顺序总览

| 阶段 | 事项 | 依赖 |
|---|---|---|
| P0 | 议题3-B 零时值音；议题1-D1 seed + B1 accent（若 D-1.2 实测成立）；议题4 版本自述 + DB 迁移框架 | — |
| P1 | 议题1-A1 tempo warp；议题3-C+E 变速确认+踏板；议题2-Audiveris 混合导入；议题4 更新检查 | P0 迁移框架 |
| P2 | 议题1-A2/C2/B2/D2；议题3-F/G/H；议题4 全量自动更新落地；议题2 可选 VLM 插件 | P1 |

## 全部决策点速览（需你拍板）
1. **D-1.2** 按住时长是否影响游戏音量（需一次实机实测，决定力度议题价值）
2. **D-2.1** 用户谱面简谱图 vs 五线谱图占比
3. **D-2.2/2.3** 是否接受 +300MB Java 运行时捆绑；AGPL 合规接受度
4. **D-3.1** 变速曲：拒绝导入 vs 强提示放行
5. **D-3.2** 分段 BPM 是否单独立项（触碰冻结 schema）
6. **D-4.1** 更新主源 Gitee/GitHub/双源
7. **D-4.3** data/ 是否迁出程序目录到 %LOCALAPPDATA%
8. **D-4.4** 接受"退出后外部脚本替换 exe 再重启"的更新形态
