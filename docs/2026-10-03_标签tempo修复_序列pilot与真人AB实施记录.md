# 标签/tempo、序列 pilot 与真人 A/B：实施记录

更新日期：2026-10-03。范围为离线研究工具与小规模训练；未改 Android、正式演奏引擎、生产模型或 EXE。下文将实现、实测结果、推断与未通过的采用门槛分开记录。

## 1. 已批准的 Git 提交

本地提交 `4d535ec548828d0936bb222a2ee4b4ed85a1e567`，备注 `Add multiseed quality-mask diagnostics and conservative policy`。没有推送远端。提交前读取原 `.gitignore`、敏感内容检查、核对显式暂存清单及 whitespace 检查；原 `.gitignore` 未修改。

仅提交用户确认的六个文件：

| 文件 | 内容 |
| --- | --- |
| docs/2026-10-03_训练模型能力评估与下一步计划.md | P1/v3 结果及证据边界 |
| tests/test_training_quality_mask.py | mask/审计回归 |
| training/README.md | 复现与回收说明 |
| training/audit_expression.py | 分作品审计 |
| training/review_quality_mask.py | 保守 v3 策略 |
| training/compare_quality_multiseed.py | 多 seed 配对比较 |

操作记录在 `<ML_ROOT>/data/evaluations/20261003_capability/git_preflight_P1_20261003.md`。其余既有未跟踪文件保留。**本次后续实现尚未提交，需另列清单并确认。**

## 2. 标签/tempo 修复：完成条件补齐和物理时间验证

问题定位：旧 note15/chord11 输入包含拍位置与拍长，却不包含谱面 MIDI tempo；而 IOI 与按键保持标签依赖积分后的谱面秒时间。因此“相同旧输入、不同实际标签”确实可能发生。不是把 MIDI 中的所有 tempo 变化当成错误，更不能直接抹平速度数据。

新增 `training/tempo_reference.py`，原 refined ZIP、v2 shard、labels 和 native masks 不改。针对已锁定的 train244/validation24 样本建立哈希绑定参考层，加入：谱面秒/拍基准、和弦局部 tempo 比、音符跨 tempo 变化积分后的时长条件。新 note17/chord13 输入只用谱面，不读 performance、全曲拟合速度、真人 onset 或可信标签比例。130/244 个 train 与 11/24 个 validation 谱面存在 tempo 变化。

实际产物：

```text
<ML_ROOT>/data/evaluations/20261003_capability/tempo_reference_244_24_20261003
```

268/268 个参考样本通过源成员、shard、offset、mask/label 对齐及 hash 验证。原标签还原公式为：

```text
performance_IOI = score_IOI_seconds * global_scale * exp(log_ioi_ratio)
```

原标签 IOI 还原最大误差：train 0.0005153 ms、validation 0.0002556 ms。误差是浮点数回构误差，**不是模型预测误差或上游 MIDI 转录准确率**。

验证集 1/4/16 小节有效窗口为 3456/2792/1784，覆盖 24/24/17 个演奏样本。只计算有确定边界且中间 IOI 全可信的窗口；无窗口返回 null，不计为零误差。小节根据 MIDI 拍号、tick 和默认 tick0 小节起点推导，变拍的 partial bar 排除，无法恢复 MIDI 未提供的弱起编号。

两种评估严格区分：`oracle_global` 使用真人演奏拟合的整体速度，仅用于残差诊断；`fixed_score` 固定 scale1，是只用谱面的部署式诊断，但仍不是游戏实测。标签已减去真人整体速度，当前模型**没有学会从谱面选择真人整体 BPM**；这个能力仍需独立目标、可靠整体速度条件或用户选择。

## 3. 小型序列模型：9 组实际 CUDA 对照

新增匹配当前输入的 MLP 与小型 BiGRU、训练 CLI、逐作品原始预测与秒级评估。所有模型使用同一 train244/validation24、native masks，无 v2/v3 quality mask；作品内 core128/context16，core 不重叠、halo 不监督，padding 经 packed sequence 排除；不跨演奏样本或 split。输入归一化只拟合选中 train；目标尺度沿用 v2 的 train-only normalizer。标签打乱仅在 train 内把 target 与 native mask 一起置换，validation 保持原样。

GPU 为 RTX 5060 Laptop，Torch 2.11.0+cu128。共9组，每组5 epoch、batch32个窗口：无 tempo MLP 单 seed；tempo MLP/GRU 各3 seed（20261003/4/5）；GRU 标签打乱负对照；MLP128 容量对照单 seed。各组训练加审计约17–36秒，Torch 峰值 allocated 显存约39–70 MiB（不是显卡总占用）。

MLP64/GRU64 参数量 7,816/52,872，额外 MLP128 为23,816；这是同输入/训练预算比较，**不是等参数或纯序列因果比较**。

实际产物：

```text
<ML_ROOT>/data/training_runs/v3/tempo_sequence_pilot_20261003
```

9 个 run 与9个 audit 全部完成，子目录 inventory 与文件 hash 复核；父 `_SUCCESS` 绑定 `pilot_summary.json`，摘要绑定所有子 inventory。无 tempo MLP 的原始预测另做物理时间评估，只追加父摘要，不改变其训练或子产物。本轮不重新读取 test shard/MIDI；历史研究 test 曾被接触，不将其包装成全新最终盲测。

### 3.1 按作品等权、三 seed 平均结果

24个验证演奏对应23个 `performance_dataset::piece_group` 单元，不能称24首不同作品。

| 目标 MAE | tempo MLP | tempo GRU | GRU−MLP | 配对作品 bootstrap 95% 区间 |
| --- | ---: | ---: | ---: | --- |
| 微时序 ms | 6.49576 | 6.53838 | +0.04263 | [+0.02097, +0.06417] |
| log 按键时长比 | 0.65042 | 0.65374 | +0.00332 | [−0.01190, +0.01902] |
| log IOI 比 | 0.17478 | 0.17412 | −0.00066 | [−0.00281, +0.00175] |
| 力度 0–1 | 0.09685 | 0.09428 | −0.00257 | [−0.00543, +0.00028] |
| 力度 robust-z | 0.50289 | 0.49960 | −0.00329 | [−0.00749, +0.00073] |
| 踏板覆盖 0–1 | 0.39443 | 0.37156 | −0.02287 | [−0.04409, −0.00403] |

负值更好。区间是对这三个固定 seed 的作品 bootstrap，不包含全部 seed 不确定性。总 validation loss 三 seed 平均 MLP0.33054、GRU0.32213；下降不能替代逐目标门槛。

tempo 消融单 seed：补 tempo 后 log IOI 误差下降0.001789，但时长误差上升0.006009；只能证明条件已补齐，不能声称全部目标改善。GRU 相对 shuffled 的力度 robust-z 降0.072625、时长降0.034506，微时序反而高0.141679 ms；负对照不支持微时序学习有效，更不是全面排除泄漏的单一证明。

### 3.2 真实秒和小节漂移

以下为“先对每个演奏样本求平均、再跨样本/三seed平均”，**不是上表的作品配对区间**：

| 秒级指标 | tempo MLP | tempo GRU |
| --- | ---: | ---: |
| oracle_global IOI MAE ms | 35.395 | 34.930 |
| fixed_score IOI MAE ms | 60.793 | 60.820 |
| fixed_score 1小节漂移 MAE ms | 649.508 | 656.936 |
| fixed_score 4小节漂移 MAE ms | 2479.281 | 2538.991 |
| fixed_score 16小节漂移 MAE ms | 3682.478 | 3591.444 |

identity 在 fixed_score 下 IOI MAE为60.107 ms，当前两种模型都未优于它。GRU虽然在16小节摘要上较好，1/4小节却变差，且尚无物理漂移的作品配对/来源非劣性门槛。

identity 的 fixed_score 1/4/16小节漂移分别为620.240/2312.298/3386.515 ms，两种模型在三个跨度都仍差于identity。不能把“GRU比MLP的16小节摘要好一些”改写成“已超过原谱基线”。

结论：**暂不全量重训、不替换正式 checkpoint、不接入游戏。** 首轮序列 context pilot 完成，但慢 tempo 曲线/局部残差分解、累计节奏训练 loss、部署漂移约束尚未实现。持续的正确小改善不等于已有人类听感提升。

## 4. 真人 A/B：已具备离线试听、导出与回收闭环

新增配对渲染、匿名试听页、严格答卷 loader 与回收统计；音频为本项目自编三段8小节短练习，3题/6 WAV，每段约17–21秒。不分发 PianoCoRe 原谱、真人 MIDI、SoundFont 或权重。候选为固定 seed20261003 GRU，不挑“最好听”的 seed；identity 对照与候选共用钢琴音源、固定力度76、关闭踏板及每对共同增益，**仅测 timing/key-hold，不代表力度/踏板已通过听感验收**。

候选采用固定 score scale1、strength0.5、micro±30 ms、raw log IOI±0.25、raw duration ratio[0.4,1.6]，披露 clipping/同音 retrigger 与40ms最小保持约束。这不是游戏档位适配器；真实游戏还需单独验证。

正式收集包（仅此 ZIP 可发给参与者；v1为集成中间产物，不再使用）：

```text
<ML_ROOT>/data/evaluations/20261003_capability/preference_collection_v2_20261003/listening_bundle.zip
SHA256: 4d48433e858404664f0e028d98cc39624e662bd6cffa11dcf2c2f2fb37bb36de
```

只分发 ZIP，不发私有映射目录。参与者完整解压、打开 `index.html`，同一音量听完两段后选择A/B/平局或跳过，主动勾选研究分享同意后下载 JSON，再通过双方约定渠道手动交回。分享同意与应用中“本地保存录制”不同，亦非公开数据授权。匿名UUID可由参与者保留，无姓名/IP/设备指纹/自由文本；hash及播放完成标记不证明真人身份或实际听完。

答卷原样保留，研究端核验版本、同意、题目/音频 hash、左右映射及重复/冲突；错误文件进入拒收清单，不修原件。数据不混入 PianoCoRe test。bootstrap 按参与者×作品 cell 汇总，少于5人或5作品不报区间；**这个3作品启动包无论收多少人，都不够作品维度的区间门槛。** 后续应另建至少5个经许可核验的独立作品版本；不同版本不自动混并。

正式回收目录：

```text
<ML_ROOT>/data/evaluations/20261003_capability/preference_collection_v2_20261003/responses
```

验收：独立 `--synthetic` 测试包在 Chromium/file:// 实际播放6个WAV（静音16倍速，仅工程测试）、验证未同意阻断、实际点击导出并保存2293-byte JSON，通过严格loader和回收审计；无HTTP/HTTPS资源请求或页面错误。命令行浏览器下载封装曾超时/取消，改用 Playwright `expect_download/save_as` 完成真正落盘验证。自动化答卷不计真人。证据在 `preference_browser_evidence_v2_20261003`，审计在 `preference_browser_v2_response_audit_20261003`。

当前正式包 **0个真人参与者、0条真人偏好，training_ready=false**。未上传到外部、未替用户招募或发送信息；研究数据仍需实际回收及分组审核。

## 5. 复现与回收命令

输出必须使用新目录；现有产物不覆盖。ML Python 为 `<ML_ROOT>/venv/Scripts/python.exe`。

```powershell
$MlPython = '<ML_ROOT>\venv\Scripts\python.exe'
$Capability = '<ML_ROOT>\data\evaluations\20261003_capability'

& $MlPython -m training.run_sequence_pilot `
  --dataset '<ML_ROOT>\data\processed\pianocore_expression\v2\full_20261003_w8' `
  --output '<ML_ROOT>\data\training_runs\v3\tempo_sequence_pilot_repeat' `
  --timing-reference "$Capability\tempo_reference_244_24_20261003" `
  --train-sample-id-file "$Capability\quality_pilot_500_20261003\train_sample_ids.txt" `
  --validation-sample-id-file "$Capability\quality_pilot_500_20261003\validation_sample_ids.txt" `
  --validation-cache "$Capability\quality_pilot_20261003\dataset_validation_cache.json" `
  --epochs 5 --device cuda

& $MlPython -m training.preference_dataset `
  --bundle-root "$Capability\preference_collection_v2_20261003" `
  --response '<ML_ROOT>\received_answers\returned.json'

& $MlPython -m training.audit_preferences `
  --bundle-root "$Capability\preference_collection_v2_20261003" `
  --responses "$Capability\preference_collection_v2_20261003\responses" `
  --output "$Capability\preference_collection_v2_received_audit_001"
```

本机回归：桌面 Python `pytest -q --ignore=tests/test_training_expression_options.py`：640 passed、4 skipped、14 subtests passed；该历史测试依赖 Torch，改在独立 ML 环境覆盖。独立 ML 环境10个训练/采集/偏好测试文件：115 passed。两组有重叠，不相加宣称755个独立测试。浏览器导出另外实测，未证明声学听感、用户回收或实机游戏验收。

## 6. 下一步顺序与采用门槛

1. 优先修“累计节奏”：将平滑低频 tempo 曲线与有界局部 timing 分离，加入1/4/16小节累计误差训练项；先以同样本/同输入MLP、序列模型及 identity 比较，不直接扩大模型。微时序增加 singleton/多音和弦分组审计，保留zero-offset备用。
2. 时长与力度分开看：核查谱面键长、声学延音、CC64监督含义；建立duration专门head或目标权重的小消融。用MLP128/更接近参数量模型、更多seed作容量控；来源BPM条件可能充当来源代理，只是待检验假说。
3. 在选中validation开发域冻结更明确的门槛：实际秒IOI/1/4/16小节漂移相对identity/规则基线改善、micro/duration非劣、每来源无明显退化，报告作品配对区间与seed离散性；持续验证负对照。门槛未过不做全量训练。
4. 分发启动包，回收真实匿名答卷；再扩展至少5个许可清晰的独立作品、更多演奏者。偏好只用于评测/后续排序阶段，不能把自编谱面或自动化答案称真人表演监督。确定回收渠道和参与者授权范围需组织者实际协调。
5. 上述门槛与真人偏好达标后，才开展扩大pilot、全量训练、独立最终holdout及桌面离线候选适配；正式游戏播放、Android与Release均另行验收。过拟合采用较小模型/dropout/weight decay/early-stop与作品/演奏者隔离；欠拟合先排查label、尺度、mask、baseline，再增加上下文，不靠反复使用最终test调参。

参考资料仅用于实现 API 与许可理解，指标均来自本机产物：[PyTorch GRU](https://docs.pytorch.org/docs/main/generated/torch.nn.GRU.html)、[packed sequences](https://docs.pytorch.org/docs/main/generated/torch.nn.utils.rnn.pack_padded_sequence.html)、[Mido MIDI](https://mido.readthedocs.io/en/latest/files/)、[Playwright downloads](https://playwright.dev/python/docs/downloads)。音源为S. Christian Collins的[GeneralUser GS](https://www.schristiancollins.com/generaluser)，本地保留上游[许可](https://github.com/mrbumpy409/GeneralUser-GS/blob/684543d5e5efaef08d02be50dcda8d552478fa60/documentation/LICENSE.txt)及sample历史来源不确定性说明；不做法律保证。渲染器[tinysoundfont](https://github.com/amberwhitehead/tinysoundfont-pybind)为独立开发依赖，不加入生产EXE。
