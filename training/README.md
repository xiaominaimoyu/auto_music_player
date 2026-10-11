# 深度学习数据采集与预处理

本目录只负责“把原始演奏数据变成可审计的训练输入”。它不会修改桌面端演奏器、不会调用外部大模型 API，也不处理 Android 客户端。

## 已建立的两条数据入口

### 1. PianoCoRe-A* 主训练数据

`pianocore_dataset.py` 读取 refined ZIP、A* 元数据和每条演奏的 alignment NPZ，生成按演奏样本组织的 ragged NPZ shard。原始 ZIP 始终只读，不先解压整包。

使用的输入三元组是：

- refined score MIDI：模型输入的乐谱结构；
- refined performance MIDI：真人/演奏来源的起音、键时长、力度和 CC64；
- refined alignment NPZ：`perf_idx` 与 `interpolated`，用于建立逐音符对应关系。

数据切分遵守以下硬规则：

- 官方 test 始终锁定为 test；
- 官方 train 内，把作品名、refined score 路径和规范化谱面语义指纹建成连通分组，再整体划入 train 或 validation；
- 谱面语义指纹由拍点归一化的音高、起音、时值和每个音符实际得到的 bar phase 构成，忽略不改变输入的重复拍号事件、PPQ、tempo、轨道、channel、velocity 与 MIDI 序列化差异；
- 同一作品、同一路径或同语义 score 都不能跨 split，构建结束再次执行三重零交叉检查；
- performer 保留在 manifest 并统计跨 split 数；锁定 test 的主评估仍定义为 unseen-piece，不宣称 unseen-performer；
- normalizer 只能读取 train 标签；
- interpolated 音符保留为上下文，但其全部 note target mask 为 false；和弦中真实匹配音符不足 50% 时，不监督和弦目标。

当前特征和标签：

| 层级 | 输入特征 | 训练标签 |
|---|---|---|
| note | 音高/音级、谱面位置与时值、拍内/小节相位、和弦大小与音域、局部密度、前后 IOI | 和弦内微偏移、键时长比例、两种力度表示、踏板起始值、踏板覆盖率 |
| chord | 谱面位置、和弦大小/均值/音域、拍内/小节相位、局部密度、前后 IOI | 有界起音偏差辅助目标、IOI 比例主目标 |

力度和踏板目前是保留的研究标签；现有游戏键盘输出链路还不能据此产生真实 velocity/CC64。第一版接入播放器时只应消费 timing 和 key duration。

长曲中的停顿和整体速度漂移可能达到数秒，因此 `onset_delta_ms` 只在可信和弦且绝对值不超过 2000 ms 时作为辅助监督；超出部分保留在原始 target 中但 mask 为 false。长程 rubato 主要由稳定得多的 `log_ioi_ratio` 学习，避免极端段落支配 loss。

v1 pilot 的 train 中有约 38.6% 和弦起音偏差超出该辅助通道范围，因此首版模型应忽略它或赋予很低权重，不能把它当作主要 rubato 标签。训练验收必须额外报告重建后的 onset MAE/p95、1/4/16 小节累计漂移、非正 IOI 比例及分来源指标。当前踏板标签只描述按键期间的起始值与覆盖率，还不是完整 CC64 曲线或 key-up 后释放时刻模型。

## 环境

已验证环境位于：

```text
E:\auto_music_player_ml\venv
Python 3.12.11
NumPy 2.5.3
mido 1.3.3
```

最小预处理依赖记录在 `training/requirements-preprocess.txt`。GPU 不是预处理的必要条件；训练使用单独的 CUDA PyTorch 环境。本机已验证 Torch 2.11.0+cu128 可识别 RTX 5060 Laptop GPU。

本机已锁定的主数据输入位于 E 盘：

- refined ZIP：`E:\auto_music_player_ml\data\datasets\pianocore\1.0\downloads\PianoCoRe-1.0-refined.zip`，3,581,508,325 bytes，SHA-256 `68f1b182741387eb8c74d98a5805bc382435136e8c9228dabe351627cc74ef0f`；
- A* metadata：`E:\auto_music_player_ml\data\datasets\pianocore\1.0\manifests\metadata_a_star.csv`，127,583,589 bytes，SHA-256 `221e4ce015081f1e0f9325cbccf024ff4fa558edea04912206c52b2fb70e697f`；
- 当前 A* 候选共 130,275 条。正式构建仍会把输入大小与 hash 写入 `build_config.json`，不能只依赖此处记录。

## 运行 PianoCoRe 预处理

PowerShell 示例：

```powershell
$MlPython = 'E:\auto_music_player_ml\venv\Scripts\python.exe'
$PianoRoot = 'E:\auto_music_player_ml\data\datasets\pianocore\1.0'

& $MlPython -m training.pianocore_dataset `
  --metadata "$PianoRoot\manifests\metadata_a_star.csv" `
  --archive "$PianoRoot\downloads\PianoCoRe-1.0-refined.zip" `
  --output 'E:\auto_music_player_ml\data\processed\pianocore_expression\v2\smoke' `
  --mode smoke `
  --smoke-per-group 1 `
  --strict
```

扩大到平衡 pilot 时可把 `--smoke-per-group` 改为 `10`。全量构建使用新的空输出目录并改成 `--mode full`；建议全量时不加 `--strict`，让程序收集所有 reject 原因，而不是首错退出。

构建器默认单进程，也支持 `--workers N` 的有界进程池。主进程始终按 metadata 顺序消费结果，因此 shard、manifest 与 normalizer 不受 worker 完成顺序影响；`--max-score-cache-entries` 限制每个进程缓存的 score 数，防止全量数据让内存持续增长。全量构建还会在选样后释放原始宽表，在每个 `(split, source)` 分组结束时立即写出尾部 shard，并把 `manifest.jsonl` / `rejects.jsonl` 按确定顺序流式落盘，不再把全部完整记录留在内存中。异常或严格模式首错退出时会保留 `_INCOMPLETE`，且不会产生 `_SUCCESS`。

不要在另一个训练任务大量占用 GPU 和系统内存时启动 full build。当前机器建议等物理空闲内存达到 6 GB 后再使用双 worker；4-6 GB 只使用单 worker 和更小缓存，低于 4 GB 仅执行测试和清单审计。这些是保守的运维门槛，full 峰值尚未完成实测。资源恢复到推荐档后使用：

```powershell
& $MlPython -m training.pianocore_dataset `
  --metadata "$PianoRoot\manifests\metadata_a_star.csv" `
  --archive "$PianoRoot\downloads\PianoCoRe-1.0-refined.zip" `
  --output 'E:\auto_music_player_ml\data\processed\pianocore_expression\v2\full' `
  --mode full `
  --workers 2 `
  --max-score-cache-entries 8
```

若启动前空闲内存只有 4-6 GB，改用 `--workers 1 --max-score-cache-entries 4`。不要按 CPU 核数盲目增大 worker：每个 worker 都会持有 MIDI 解析对象。这里的 4/6 GB 门槛指物理空闲内存，不是磁盘余量。预处理不会调用 CUDA，但内存争用仍可能拖慢或中断正在运行的 GPU 任务。

每次成功构建会产生：

- `build_config.json`：参数、输入 hash、预处理源码 hash 和运行时版本；
- `schema.json`：特征/标签顺序与 mask 规则；
- `manifest.jsonl`：每个样本的来源、split、原始文件 hash、QC 和 shard 定位；
- `rejects.jsonl`：拒绝样本与原因；
- `normalizer.json`：只由 train 拟合的统计量；
- `qc_summary.json`：通过率、来源、split、音符/和弦数量和泄漏审计；
- `integrity.json`：所有输出文件的大小与 SHA-256；
- `_SUCCESS`：`integrity.json` 的 SHA-256。存在 `_INCOMPLETE` 表示构建没有完成，该目录不得进入训练；所有候选样本都被拒绝时不会写 `_SUCCESS`。

输出目录必须为空，构建器不会覆盖既有结果。

成功后必须再执行独立的只读验收；验证器不会修复或重写任何产物：

```powershell
& $MlPython -m training.validate_pianocore_dataset `
  --dataset 'E:\auto_music_player_ml\data\processed\pianocore_expression\v2\full'
```

它会复核 `_SUCCESS`、完整性清单及每个文件 hash，使用 `allow_pickle=False` 回读全部 shard，并核对准确的 v2 schema/参数/provenance、数组维度与 dtype、finite、offset、`note_to_chord`、mask 语义、标签范围、manifest 诊断与 shard 定位、完整 train-only normalizer、汇总计数与 split 泄漏。manifest 与 reject 按行读取，normalizer 按 train shard 汇总，避免按样本反复分配大型直方图。验证命令应使用上述 `python -m training.validate_pianocore_dataset` 模块入口。

本机当前已完成正式 v2 full 构建并通过独立 validator：输出为
`E:\auto_music_player_ml\data\processed\pianocore_expression\v2\full_20261003_w8`，
130,275 条候选中接受 130,238、拒绝 37（均为 `MidiFeatureError`），共 302,620,399 个 note、161,629,130 个 chord；train/validation/test 为 108,649/9,667/11,922，作品、score 路径和 score 语义三类 split 泄漏均为 0。目录约 6.16 GB，`_SUCCESS` 与 `integrity.json` hash 已复核一致。旧的 2-worker 中止部分保留在 `full_20261003_partial_workers2`，不得进入训练。

## 第一阶段可执行训练

`train_expression.py` 是当前的离线基线训练入口，不修改播放器，也不把 velocity/CC64 直接写回游戏输出。它有独立的 note/chord MLP head，使用 train-only normalizer、masked Smooth-L1/Huber loss，并把 onset_delta_ms 作为低权重辅助目标。每次运行都会先调用独立 validator；输出目录为空时才允许启动，运行中保留 `_INCOMPLETE`，完成后才写 `_SUCCESS`、`checkpoint_best.pt`、`checkpoint_last.pt`、`metrics.jsonl` 和 `integrity.json`。

当前训练入口还支持可审计的受控消融参数：

- `--input-normalize`：仅用 train split 拟合输入特征均值/标准差，并把统计写入 `run_config.json`；
- `--relative-position`：把每个作品内的绝对 `score_onset_beats` 转为 `[0, 1]` 相对位置，必须和输入统计一起在同一实验中登记；
- `--shuffle-shards`：每个 epoch 随机化 shard 顺序，用于排除固定来源顺序偏差；
- `--duration-weight`、`--velocity-weight`、`--pedal-weight`、`--ioi-weight`：逐目标权重，不能把加权后的总 loss 与不同权重的运行直接横向比较；
- `--target-shuffle`：仅用于输入泄漏负对照，打乱训练目标行，不能用于模型候选。
- `--quality-mask`：加载 JSONL sidecar；只对 train split 应用质量决策，并与原生 note/chord mask 做交集。软权重进入 loss 分母，run config 记录 sidecar 路径、策略版本和 SHA-256；validation/test 默认保持原始口径。
- `--train-sample-id-file`、`--validation-sample-id-file`：只供诊断 pilot 使用的 newline-delimited sample ID 子集；训练器会重建 ragged offsets，并记录清单 hash。
- `--validation-cache`：复用同一数据目录已经通过的 validator 报告，缓存同时校验数据目录 `_SUCCESS` hash，仅用于重复的诊断 pilot。

评估已分为两层：`evaluate_expression.py` 保持快速复核；`audit_expression.py` 负责全量 raw-target MAE/RMSE、确定性 p95、均值/原谱基线、来源统计和作品等权 bootstrap，也支持 `--sample-id-file` + `--validation-cache` 对锁定的诊断子集输出逐作品误差。原始 refined 三元组抽检由 `audit_source_quality.py` 完成，至少按每个来源抽 10 个作品，输出时长偏差、对齐、tempo scale、插值率、召回率和踏板尾音。两者均只读，不会改写数据集或 checkpoint。

对来源审计标记的样本由 `review_quality_mask.py` 做二次只读复解析。它重新读取 refined score/performance/alignment，并再次通过 `build_aligned_features` 的对齐完整性检查；原始 shard、MIDI 和 alignment 不会被改写。v1 sidecar 保留历史硬 mask 候选；校准后的 `quality-mask-v2-calibrated` 把聚合异常当作复核证据，保留原生逐事件 mask，只对对应目标软降权，不把整曲异常直接当作标签错误；`quality-mask-v3-target-specific` 进一步把低对齐降权收窄到 timing targets，velocity/pedal 不再因低对齐连带降权（pedal 尾音仍按独立证据处理）。`train_expression.py --quality-mask` 已支持显式加载这些 sidecar，训练时只对 train 行应用，validation/test 不参与 mask 调参。

复核命令（输出目录须使用新的空目录）：

```powershell
& 'E:\auto_music_player_ml\venv\Scripts\python.exe' `
  -m training.review_quality_mask `
  --dataset-root 'E:\auto_music_player_ml\data\processed\pianocore_expression\v2\full_20261003_w8' `
  --archive 'E:\auto_music_player_ml\data\datasets\pianocore\1.0\downloads\PianoCoRe-1.0-refined.zip' `
  --source-audit 'E:\auto_music_player_ml\data\evaluations\20261003_capability\run02_source_quality.json' `
  --output-dir 'E:\auto_music_player_ml\data\evaluations\20261003_capability\quality_mask_v2_calibrated_20261003' `
  --policy quality-mask-v2-calibrated
```

`summary.json` 会记录复核范围、策略、每个目标的禁用计数和 sidecar SHA-256；`flagged_samples_review.json` 保留每个样本的原始指标及 `decision.findings`，便于人工复核和后续审计。校准 v2/v3 的硬禁用计数均为 0；P0 产物为 12 个软降权候选，P1 扩展产物为 147 个候选中的 141 个实际降权决策。v3 中低对齐直接影响 velocity 的行数为 0，pedal 仅由 45 条明确尾音证据降权。
其中若标记样本落在锁定 test split，复核结论只作为审计记录，不得用于选择 epoch、阈值或模型方案；将 sidecar 接入训练时只对 train 行应用监督 mask，并对 validation/test 同时保留原始口径和质量 mask 口径。

P1 扩展抽检示例：

```powershell
& 'E:\auto_music_player_ml\venv\Scripts\python.exe' `
  -m training.audit_source_quality `
  --dataset-root 'E:\auto_music_player_ml\data\processed\pianocore_expression\v2\full_20261003_w8' `
  --archive 'E:\auto_music_player_ml\data\datasets\pianocore\1.0\downloads\PianoCoRe-1.0-refined.zip' `
  --output 'E:\auto_music_player_ml\data\evaluations\20261003_capability\run03_source_quality_100.json' `
  --per-source 100
```

选定 validation 子集后，可用 `audit_expression.py --sample-id-file ... --validation-cache ...` 分别审计多个 checkpoint，再用 `compare_quality_multiseed.py --pair SEED BASELINE_AUDIT SOFT_MASK_AUDIT` 做作品级 paired bootstrap。该比较只用于 pilot 门禁，不替代锁定 test 或最终盲测。

v3 的同配置复核只需将训练命令中的 sidecar、ID 清单和输出目录切换到 `E:\auto_music_player_ml\data\evaluations\20261003_capability\quality_mask_v3_target_specific_500_20261003` 与 `quality_pilot_v3_target_specific_500_20261003`；已完成的三 seed 审计结果位于对应 `audits`，配对结果由 `multiseed_paired_comparison.json` 固化。v3 仍未达到全量采用门槛，不得替换正式 checkpoint。

诊断 pilot 的样本选择由 `build_quality_pilot.py` 固化：每个触发样本匹配同来源、不同作品、相近音符数且无审计警告的对照。P0 原始产物为 9 个 train 触发样本 + 9 个 train 对照，以及 2 个 validation 触发样本 + 2 个 validation 对照；P1 扩展产物为 122+122 个 train、12+12 个 validation，v2/v3 使用同一组 ID。对应训练命令示例：

```powershell
& 'E:\auto_music_player_ml\venv\Scripts\python.exe' `
  -m training.train_expression `
  --dataset 'E:\auto_music_player_ml\data\processed\pianocore_expression\v2\full_20261003_w8' `
  --output 'E:\auto_music_player_ml\data\training_runs\v2\quality_pilot_softmask_a_seed20261003' `
  --device cuda --epochs 5 --batch-size 8192 `
  --input-normalize --relative-position --shuffle-shards `
  --train-sample-id-file 'E:\auto_music_player_ml\data\evaluations\20261003_capability\quality_pilot_20261003\train_sample_ids.txt' `
  --validation-sample-id-file 'E:\auto_music_player_ml\data\evaluations\20261003_capability\quality_pilot_20261003\validation_sample_ids.txt' `
  --validation-cache 'E:\auto_music_player_ml\data\evaluations\20261003_capability\quality_pilot_20261003\dataset_validation_cache.json' `
  --quality-mask 'E:\auto_music_player_ml\data\evaluations\20261003_capability\quality_mask_v2_calibrated_20261003\quality_mask.jsonl'
```

smoke 数据链路已实际完成一次 CUDA tiny baseline：

```powershell
$MlPython = 'E:\auto_music_player_ml\venv\Scripts\python.exe'
$Dataset = 'E:\auto_music_player_ml\data\processed\pianocore_expression\v2\smoke_v1'

& $MlPython -m training.train_expression `
  --dataset $Dataset `
  --output 'E:\auto_music_player_ml\data\training_runs\v2\smoke_tiny_overfit' `
  --device cuda `
  --epochs 30 `
  --batch-size 2048 `
  --weight-decay 0 `
  --patience 10
```

要验证反向传播确实能拟合单个 shard，可增加 `--max-shards 1`、减小 batch 并提高 epoch；这类结果只用于数据/损失门禁，不能作为泛化能力结论。当前已完成的 smoke baseline 最优验证 loss 为 0.3359（epoch 4）；单 shard overfit 的 train loss 下降而 validation 变差，符合“能拟合、会过拟合”的预期。正式 pilot/full 训练前仍需按来源报告 MAE/p95、重建 onset 漂移、正 IOI 比例、duration/velocity/pedal 分布，并与均值/规则基线对比。

在上述 full 数据上已实际执行 CUDA 训练（RTX 5060 Laptop GPU，Torch 2.11.0+cu128，batch size 32768，最多 5 epoch，early stopping patience 2）：

- baseline：`E:\auto_music_player_ml\data\training_runs\v2\full_expression_baseline_b32768`，最佳 epoch 5，validation loss 0.39222；锁定 test loss 0.38295；
- regularized backup（dropout 0.2、weight decay 5e-4、hidden 256）：`E:\auto_music_player_ml\data\training_runs\v2\full_expression_regularized_b32768`，最佳 epoch 3，validation loss 0.39072，锁定 test loss 0.38115；
- 同一 test split 的训练集均值常数基线 loss 约 0.39126，因此正则化模型总损失有所改善；但 `log_key_duration_ratio` test MAE 仍为 0.92017，高于常数基线约 0.78248，暂不把任何 checkpoint 直接接入播放器；下一轮应先做目标加权和序列上下文实验。

两套训练目录均已通过 `_SUCCESS`/`integrity.json` 一致性检查；test 评估只读执行，没有参与调参。

## 本地真人采集数据

桌面端只有在用户主动勾选“保存为本地表现研究样本”时才写入：

```text
<应用数据目录>\performance_captures\completed\*.jsonl
```

程序当前不会自动上传这些文件。回收数据必须由用户明确同意后，手动提交 `completed` 中选定的 JSONL；不要收集 `pending`、`rejected`、日志、数据库或整个应用数据目录。

严格校验并生成不含原始按键/窗口信息的安全清单：

```powershell
$MlPython = 'E:\auto_music_player_ml\venv\Scripts\python.exe'
$CaptureRoot = "$env:LOCALAPPDATA\AutoMusicPlayer\data\performance_captures"

& $MlPython -m training.local_capture_manifest `
  --capture-root $CaptureRoot `
  --output 'E:\auto_music_player_ml\data\processed\local_captures\audit-001'
```

校验包含 schema、JSONL 顺序和时序、原始内容 hash、press 配对、synthetic close、从事件重算的完整 quality、canonical score hash，以及规范谱与演奏音高多重集合。失败样本只进入 `rejects.jsonl`，源文件不会被移动、修复或删除。

`_SUCCESS` 在这里仅表示“清单审计完整结束”，不代表已经有训练数据；`summary.json` 的 `training_ready` 才表示是否至少有一个合格样本。没有合格样本时还会生成 `_NO_ACCEPTED_SAMPLES`。

当前本地采集先作为独立的 `local_dev` / 偏好评测域，不混入 PianoCoRe 锁定 test。收集到足够样本后再按 `piece_group` 切分，并在存在可靠匿名演奏者标识时进一步做 performer 级隔离。

## A/B 偏好数据的启动条件

已生成3题/6 WAV的离线盲听启动包，正式真人偏好仍为0；不能用自动化测试或规则生成的伪偏好代替真人选择。启动包位于 `E:\auto_music_player_ml\data\evaluations\20261003_capability\preference_collection_v2_20261003\listening_bundle.zip`，只分发该 ZIP，私有模型映射留在研究端。完整解压后打开 `index.html`，主动同意分享后手动导出/交回答卷；程序不上传。切分时按匿名参与者和 `piece_group` 同时隔离，A/B 数据只用于偏好评测或后续排序阶段，不混入锁定 test。3作品启动包不足作品维度统计区间门槛，需要后续扩展到至少5个许可明确的独立作品。

## Tempo 条件与小型序列 pilot（2026-10-03）

`tempo_reference.py` 不修改 v2 数据；为选中244 train/24 validation补齐谱面秒/拍、local tempo与时长条件，并恢复真实IOI和1/4/16小节漂移。模型输入只能使用score-only条件，不能输入performance、拟合global_scale或可信标签比例。`oracle_global`仅为借用真人整体速度的诊断，`fixed_score`才是固定谱面速度诊断，两者均非游戏验收。

`run_sequence_pilot.py` 串行完成9组5epoch：无tempo MLP；tempo MLP/GRU三seed；GRU train-only标签打乱；MLP128容量对照。MLP/GRU匹配条件与预算而非参数量；sample内core128/context16、native masks、packed padding，core只写一次。产物与权重外置E盘，不随Git/EXE分发。

结果：GRU总loss与踏板覆盖改善，但micro误差增加，IOI作品区间跨零，fixed-score实际秒IOI没有优于identity。**目前不采用任何新checkpoint，不做全量重训。** 记录、命令和下一轮累计节奏/目标分解门槛见 `docs/2026-10-03_标签tempo修复_序列pilot与真人AB实施记录.md`。

试听链路使用 `render_expression_candidates.py` 与可选 `requirements-render.txt`（tinysoundfont），输出本项目自编短练习，无上游数据再分发；velocity76/pedal-off共同控制，仅评估timing/key-hold。`preference_bundle.py` 生成匿名离线包；`preference_dataset.py`严格核验；`audit_preferences.py`去重、显式排除冲突并统计。`smoke_preference_browser.py`（另需Playwright）拒绝真实收集包，只对`--synthetic`测试包实际播放/保存JSON，永不计入真人统计。

## 已完成的真实数据验收

2026-10-03 后续已完成累计IOI损失、时值/CC64独立参考审计、12组matched CUDA timing/hold pilot、identity/规则/零模型/历史MLP-GRU的raw→safe→compiler对照及CPU重复推理。所有候选仍未通过接入门禁；未修改实际演奏、GUI或Android，不开放实验入口。新工具为 `audit_duration_reference.py`、`run_playback_adoption_pilot.py`、`shadow_expression.py`、`evaluate_playback_adoption.py`、`benchmark_sequence_inference.py`；完整结果与复现见 [累计节奏与演奏接入门禁实施记录](../docs/2026-10-03_累计节奏与演奏接入门禁实施记录.md)。该阶段仅train/validation开发，权重/数据/日志外置E盘。累计1/8/32间隔训练并非真实1/4/16小节监督；CPU跑通也不是游戏或GUI验收。

2026-09-28 在本机 refined 包上完成的 v1 历史验收：

- 最小 smoke：15/15 通过，5 个来源 × 3 个 split；两次输出目录的 23/23 个文件逐字节一致；
- 平衡 pilot：150/150 通过，train/validation/test 各 50，五个来源各 30；
- 最终平衡 pilot 共 447,315 个 note、226,697 个 chord，作品/score 路径/score 内容三类跨 split 数均为 0；
- 21 个受完整性清单覆盖的文件和 15 个 shard 均通过 hash、offset、shape、dtype、finite 检查；产物约 10.53 MB。按音符数线性外推，全量 shard 约为 7-10 GB，但这只是容量规划估计，正式数值以 full build 为准。
- 单进程与 2-worker 在真实 smoke/pilot 的 `schema/manifest/rejects/normalizer` 和全部 shard 上逐字节一致；worker 数只写入 build config，不改变训练数据内容。
- v2 已加入语义谱面指纹、performer 审计、宽表释放、分组尾 shard 及时落盘和独立验收器；因此上述 v1 pilot 只能证明旧 revision。资源恢复后必须先重跑 v2 smoke/repeat/pilot，再启动 full。
- 全量串行基准曾在 1,300/130,275 条时主动中止，以便先加入有界并行和内存收敛；该历史输出保留为 `_INCOMPLETE`，不是可训练数据。随后使用 8-worker 有界并行在新目录完成了上面的 v2 full，并经过独立 validator 验收。

本地 refined 包里已经包含 ATEPP 来源；当前没有单独的 Vienna 4x22 数据。数据集的许可证/NOTICE 尚未在本地下载资产中找到，因此原始包和生成 shard 只保留在本机，不随软件或 Git 仓库分发，正式使用前需补齐上游许可核验。

## 真实小节窗口与 tempo/local 辅助监督（2026-10-06，已接线、未训练）

`bar_windows.py` 只读 score 几何（由 `chord_bar_phase` 得到 bar boundary，配合 `bar_supervision.BarSpans`）来规划窗口：core 只在真实小节边界处切、并在容量内切得尽量晚；`max_window_chords=512` 约束的是**含左右 halo 的整窗**，所以 core 容量是 `512 − 2 × context_size`；每个 chord 只属于一个 core，halo 可以压进邻居 core 但不会因此产生第二次 core 或辅助 loss。span 的起止端点必须属于同一套 boundary，否则直接报错，混用两份几何无法静默通过。超过 core 容量的长小节会强制 mid-bar 切分，`bar_window_coverage` 按 `capacity_forced_split_window_count` 与 `crosses_bar_aligned_cut_window_count` 分类报告，绝不截断冒充完整。规划函数的签名里没有 `chord_masks`、`chord_trusted`、target 或 start trust。

`bar_objective.py` 的 `true_bar_objective(...)` 输出 elapsed/tempo/local 三路：`b = log(sum(s·e^r)/sum(s))`、`local = r − b`；先按 1/4/16 horizon 各自求均值，再对 active horizon 等权（否则数量占优的 1 拍跨度会淹没 16 拍）；先 mask 再 `exp`/`cumsum`，所以 padding 里的 NaN 进不了跨度求和；预测在 `exp` 前 clamp，可信标签不 clamp 且非有限或非正直接报错；`bar_ioi_mask` 只过滤 local 行，tempo 与 elapsed 仍覆盖完整跨度。形参 `score_ioi_seconds` 必须是**间隔秒**（首元素为 0），不是绝对谱面时钟——写测试时就把绝对秒传进去过，靠与 `decompose_bar_targets` 的对照才暴露。

覆盖证据（合成 400 个 8 弦均匀小节）：16 拍跨度 384 个，固定 128 core 的 `same_core_window_count` 为 0，bar 对齐 core 保留 288、其余 96 个因跨越容量切分而被排除。真实均衡集上 394/2221 的对照要等 P1 用 audit 侧车复算，这里不提前宣称。

接线状态（2026-10-06 P1）：`sequence_expression.py` 的 `SequenceSample` 增加 score-only 的 `bar_span_rows/bar_span_horizons/bar_boundary_indices/bar_requested_horizons`（都能由 audit NPZ 重建，不依赖 target），`attach_bar_geometry()` 只读 timing reference 里的谱面几何，`make_windows_from_config()` 按 `window_policy` 切窗，`collate_windows()` 只在三个 bar 权重任一 > 0 时才输出窗口局部 span 张量与 `bar_ioi_mask`；`predict_sample()` 改走 `make_windows_from_config`，所以辅助 loss 关掉也不会退回固定 128。`train_sequence_expression.py` 新增 `--window-policy/--max-window-chords/--bar-horizons/--bar-elapsed-weight/--bar-tempo-weight/--bar-local-weight`，默认 `fixed` + 权重 0，即与历史 run 逐字同行为；`bar_*` 权重启用时强制 `window_policy=bar`、拒绝 `target_shuffle`、要求纯谱面 timing reference，并把 `bar_windows.py/bar_objective.py/bar_supervision.py` 的实现 hash 一起写进 `run_config.json`。`audit_sequence_expression.py` 在预测 NPZ 里落四个 score-only 几何数组并在 report 写 `bar_layout` 覆盖计数；`benchmark_sequence_inference.py` 缺几何就报错，并记录 zero-target 下实际规划出的 core 数与固定 128 应得的核数。**仍未启动任何训练，没有改动播放链路、GUI 或 Android。**

真实覆盖对比（只读诊断，`tools/compare_bar_window_coverage.py`，产物 `E:\auto_music_player_ml\data\evaluations\20261003_capability\bar_window_coverage_balanced_250_50_20261006`，summary sha256 `5a32f79402b100e37099c7f938b50e3a57d56ec9de7ccf43a2bbed7b7f37e2b2`；口径为 eligible 跨度中两端同属一个 core 的数量）：

| split | horizon | eligible | fixed 128 core | bar 对齐 core | 倍数 |
|---|---|---|---|---|---|
| validation | 1 | 5679 | 5312 | 5597 | 1.05 |
| validation | 4 | 4435 | 3322 | 4196 | 1.26 |
| validation | 16 | 2221 | 394 | 1758 | 4.46 |
| train | 1 | 23637 | 22325 | 23349 | 1.05 |
| train | 4 | 17876 | 14104 | 17090 | 1.21 |
| train | 16 | 8214 | 2405 | 6969 | 2.90 |

`capacity_forced_split` 在两个 split、三个 horizon 上都是 0，说明 512 上限在真实均衡集上从不强制切开小节；bar 对齐丢掉的跨度全部归类为 `crosses_bar_aligned_cut`（validation h=16 为 1391 个，含恰好终止于 core 边界的跨度，按定义排除而不截断）。这些是覆盖计数，不是听感或接入证据。

端到端管线检查（`tools/check_bar_inference_plumbing.py`，产物 `E:\auto_music_player_ml\data\evaluations\20261003_capability\bar_inference_plumbing_20261006`，summary sha256 `b501459562359507a571c5106a05037bc6194eceb51f0328b62a9b6a7f9b3e62`）：把 sequence pilot `gru_tempo_seed20261003` 的权重与 train-only normalizer 原样拷进两个 fixture run（`window_policy` 分别为 fixed 和 bar，脚本本身不做任何训练，summary 里 `trained=false`、`model_quality_evaluated=false`），再在锁定的 50 个 balanced validation 样本上真跑 `audit_sequence_expression` 与 `benchmark_sequence_inference`。结果：bar audit 的 `bar_layout` 报出 159 个 core / 63997 chord，与独立诊断工具逐字一致；只用预测 NPZ 里的四个 score-only 几何数组重新规划，也能重建出同样的 159 core / 63997 chord（`audit_and_npz_layout_match=true`，最大窗宽 512），而同一批样本在固定 128 core 下需要 523 个 core；两个策略的 zero-target 重复推理都 bitwise 相同。CPU 4 线程下 bar 的 median/p95/max = 0.0257/0.1126/0.2331 秒，fixed = 0.0368/0.1458/0.2955 秒；前一次独立运行是 0.0288 对 0.0350 median，即 bar 因前向次数降到 1/3.29 而略快，但单次抖动约 10%，不要当成精确性能结论。以上只证明几何侧车、integrity 绑定与推理布局可复现，不证明模型质量、听感偏好或可以接入播放器。

权重标定探针（`tools/compare_bar_loss_scales.py`，产物 `E:\auto_music_player_ml\data\evaluations\20261003_capability\bar_loss_scale_comparison_20261011`，summary sha256 `48afe3753ab863bc0eb0e5fd6372d2be7f4a3a3b0ddf6794e704396e7411be23`）：取均衡集 train 侧排序前 12 个样本（10654 chord），加载 `gru_tempo_seed20261003` 的权重后只跑评估 epoch（`optimizer=None`，本脚本不训练），三个 bar 权重全置 1.0 以观察各路原始量级。结果：bar_elapsed 0.008158、bar_tempo 0.008841、bar_local 0.006311，三者同在一个量级（最大最小差约 40%），合计 0.023309，相当于 point loss 0.26440 的 **8.8%**；同一切换下 fixed 与 bar 的 point loss 分别是 0.26442 与 0.26440，`log_ioi_ratio` 目标 MAE 都是 0.1371，即**每个 chord 只被监督一次这条不变量在两种布局下成立**，换布局没有偷走或重复监督。窗口数由 88 个 core（平均窗宽 148.6、最大 160）降到 28 个（平均 398.8、最大 511）。各 horizon 的候选与可监督跨度：h=1 920/1065、h=4 651/982、h=16 273/667——h=16 大量落在 native mask 不合格上，而不是被 core 边界切掉；`local_rows_excluded` 与 `clipped_rows` 均为 0。据此第一个 pilot 已确定把三个权重都取 1.0（用户 2026-10-11 决定；bar 约占目标函数 9%，无需再缩放）；按"占比 5%/10%/25%"反解出的权重点（例如 3.24/2.99/4.19）只是可选刻度，会让 bar 占到约 30%，不作为默认。这个 1.0 是起点不是结论，最终取值仍要由小 pilot 的验证损失与作品级指标决定，本探针不构成任何接入依据。

首个 matched GPU pilot（2026-10-11，用户授权可训练后执行；产物 `E:\auto_music_player_ml\data\training_runs\v4\bar_matched_20261011\{fixed,bar}_seed2026100{3,4,5}`，对比汇总 `E:\auto_music_player_ml\data\evaluations\20261003_capability\bar_matched_pilot_20261011`，summary sha256 `0fa80c43a007773cf4858db61ba4d8e8306393427c1a5d32d1f6017fd9758d99`）：均衡集 250 train / 50 validation，GRU hidden 64、dropout 0.1、lr 1e-3、wd 1e-4、batch 32、5 epoch、3 个 seed，两臂只差 `window_policy` 与三个 bar 权重（均 1.0，bar 臂 max window 512）。

结论是**负面的，而且必须按三 seed 读**：

- 第 5 epoch 的 point loss 配对差值（bar − fixed）为 -0.00389 / +0.00067 / -0.00816，均值 -0.00379，**跨度 0.00884 大于均值本身**，其中 seed 20261005 的"改善"其实来自 fixed 基线在第 5 epoch 退化（0.33034 比它自己第 3 epoch 的 0.32335 更差），不是 bar 臂变好。所以这条不能报成收益。
- `log_ioi_ratio` MAE 均值 +0.00066（bar 略差）、`log_key_duration_ratio` +0.00016（基本持平）、`micro_offset_ms` -0.007 ms（可忽略）。
- 唯一 3/3 同号的稳定效应是反向的：`onset_delta_ms` 变差 +1.87 / +6.11 / +6.80 ms（均值 +4.93 ms）。早期 epoch 也更差（第 1 epoch point loss 均值 +0.0098），即辅助项在训练前期抢走了共享 chord head 的容量。
- 收敛时三路 bar 分量的验证值是 local 0.0186、elapsed 0.0078、tempo 0.0076：**local 一项就占总辅助量的一半以上**，而 chord head 同时输出 onset 与 log IOI，这正好是 onset 退化的嫌疑路径。

因此**不要上调权重**。下一轮的可比做法是：把 local 单独降档（例如 elapsed/tempo 1.0、local 0.25）或给 onset 与 IOI 拆分输出头，并把 epoch 数拉长后按同一 best-epoch 规则对齐两臂（当前 fixed 会在第 5 epoch 反弹，直接比 final 会得出方向相反的假结论）。本轮不构成任何听感、作品级 bootstrap 或播放器接入证据。

对照臂与更正（同日追加，推翻上一节两处归因）：把 `bar_local_weight` 从 1.0 降到 0.25（产物 `bar_local025_pilot_20261011`，summary sha256 `79b07d0a6a4356a2e68ca6b150e5cbcee4418b2638d901920e2e78aeda44217d`）几乎不改变任何指标——第 5 epoch 的 point loss 配对差值 -0.00370 对 -0.00379，onset +5.35 对 +4.93 ms。于是再加一条对照臂：`window_policy=bar` 但三个权重全为 0（`bar_layoutonly_pilot_20261011`，summary sha256 `ee3746b8295a0309ad5ba78812113ddd3f5fdfdd1298bc4acb0a50cde69ce0c4`）：

| 臂 | e1 point Δ | e5 point Δ | e5 ioi Δ | e5 onset Δ | onset 同号 |
|---|---|---|---|---|---|
| bar 1.0/1.0/1.0 | +0.00975 | -0.00379 | +0.00066 | +4.93 ms | 3/3 |
| bar 1.0/1.0/0.25 | +0.00987 | -0.00370 | +0.00139 | +5.35 ms | 3/3 |
| bar 布局，权重全 0 | +0.00860 | **-0.00441** | +0.00014 | +3.43 ms | 3/3 |

**把辅助 loss 完全关掉，point loss 的"改善"不但没消失反而更大**。所以上一节里"bar 臂后期 point loss 更低"不能算辅助监督的收益，那是布局效应：core 从 88 个降到 28 个、窗宽 160→511，改变了 padding 比例与 bidirectional GRU 的上下文长度。onset 退化同样主要是布局带来的（没有辅助项就已有 +3.43 ms）；辅助项是否再额外贡献约 1.5 ms，在 3 seed、seed 间跨度 2.2–6.8 ms 的方差下无法定方向。

由此确立的方法规则：**这条管线上任何 bar 实验都必须带 layout-only 对照臂**，否则测到的是窗口尺寸而不是监督信号。当前站得住的结论只有两条：250/50、5 epoch 预算下 tempo/local 辅助监督没有可证明的增益；bar 对齐布局自身带来小的 point loss 效应和稳定的 onset 退化。下一步应该拉长 epoch 并统一 best-epoch 协议，或者去查 onset 通道本身，而不是继续调 bar 权重。

15 epoch 三臂与交叉评估（2026-10-11，用户选定"拉长 epoch + 统一 best-epoch 协议"后执行）：`epochs 15 / patience 15`（关早停，保证两臂在每个 epoch 号可比），3 seed × 3 臂（fixed、bar 1.0/1.0/1.0、bar layout-only），其余超参与 5-epoch pilot 逐字相同。按 seed 配对差值（bar − fixed）：`bar_long15_bar_pilot`（sha256 `3bbb3ca6ccad05a9fafa81715e4d658ab2ed59cef0d87a02cbc8c136acbb3dd0`）在第 5/10/15 epoch 的 point loss 差值为 -0.00379 / +0.00247 / +0.00205；`bar_long15_layoutonly_pilot`（sha256 `8f2b887daf2049d762ae693c3116104b880387a04914122d84176cab933fa2fc`）为 -0.00441 / +0.00254 / +0.00286。两条 bar 臂在第 10/15 epoch 都比 fixed 差，且彼此几乎重合；best-epoch point loss 三个 seed 全部是 fixed 最低（fixed 0.31910/0.31781/0.31789，bar 0.31980/0.32161/0.31935，layout-only 0.32124/0.32063/0.32098）。第 5 epoch 的数值与之前 5-epoch run 完全一致，说明这轮比较本身可复现。

决定性的一笔来自 `tools/evaluate_bar_objective_across_runs.py`（产物 `bar_objective_cross_eval_20261011`，sha256 `291583cdf66a28cd998d420ff7d8065b966f8e0623f03ac67030ab9b15520eb0`）：用同一批 50 个锁定验证样本、以权重 1e-9 的探针（只测量不参与训练）评估三个 checkpoint 的 bar 目标值。结果是 elapsed/tempo/local 三项上，**训练过 bar 目标的臂并不比从没见过这个目标的 fixed 臂更低**（elapsed 0.007643 对 0.007455，tempo 0.007610 对 0.007468，local 0.018461 对 0.018345），而 layout-only 臂的 elapsed 反而是三者最低（0.007045）。也就是说：辅助目标没有被拟合，权重 1.0 在梯度上基本等于关闭。

机制解释与代价：elapsed/tempo/local 都是 point loss 已经在逐行监督的同一批 log IOI 的函数，因此它与主监督高度冗余——per-row 残差定下来后，聚合量就被决定了，把它再作为 loss 相加几乎不增加信息，只会稀释梯度。要让这条路有意义，必须给辅助项提供 point loss 看不到的信息（例如真人小节级 tempo 曲线作为目标、或对 within-bar 漂移做非对称/方向性惩罚），或者把权重量级提高若干数量级并始终配 layout-only 对照，用交叉评估确认辅助目标真的被学到之后再看指标。另注：fixed 臂 15 epoch 的 train/val 为 0.2858/0.3198，过拟合方向与历史一致，所以"更长训练"本身也不是收益来源。

据此的建议是把 tempo/local 辅助监督这条线**归档为当前形态下无收益**，把预算转向 onset 与 key-duration 通道（onset MAE 长期 ~585 ms、`log_key_duration_ratio` ~0.60 几乎不动）与标签质量线。本轮同样不构成任何接入证据。

目标通道审计（`tools/audit_target_channels.py`，只读；产物 `target_channel_audit_validation_20261011` sha256 `52d9e478070df62d65ae9d87e2082c7b2901aa569509be3df205844231d23016`、`target_channel_audit_train_20261011` sha256 `6c06e95b46b4263d8067b5571d178d686d775e904d5a7206715a863fa0be23fa`）：把每个被监督的目标通道跟两条模型够不着的平凡基线对照——全部预测 0，以及减去该演奏自身中位数（衡量"每曲常数偏移"）。均衡集 validation（train 同向）结果：

| 通道 | 监督行数 | 预测 0 的 MAE | 去每曲中位数后 | 中位数占误差 | 模型实测 MAE |
|---|---|---|---|---|---|
| `onset_delta_ms` | 38829 | 578.68 ms | 543.22 ms | 6.1% | 584.23 ms（**比预测 0 更差**） |
| `log_ioi_ratio` | 61246 | 0.1843 | 0.1757 | 4.7% | 0.1771（好 3.9%） |
| `log_key_duration_ratio` | 112618 | 0.6668 | 0.5386 | 19.2% | 0.6222（好 6.7%） |
| `micro_offset_ms` | 112618 | 5.84 ms | 5.84 ms | 0%（构造上已居中） | 6.12 ms（**比预测 0 更差**） |
| `velocity_robust_z` | 112618 | 0.5737 | 0.5737 | 0% | 未评估 |

结论改变了优先级排序：**当前序列模型在四个可评估通道里有三个打不过或几乎打不过"全部预测 0"**。而且 `onset_delta_ms` 的 6.1% 居中收益说明那 ~400 ms 不是每曲常数偏移，而是逐事件散点（|值| p50 407 ms、p99 1.92 s，紧贴 2.0 s 监督上限）——长线漂移与对齐失配已经由 `log_ioi_ratio` 承担，这个绝对残差通道基本是噪声。`log_key_duration_ratio` 确有 19% 的每曲持续策略成分。

由此立两条规则：一，任何训练或评估报告都必须同时给出"预测 0"与"去每曲中位数"两条基线，否则所谓改善不可判定；二，下一步预算投向目标定义本身（`onset_delta_ms` 撤下或改为去掉本曲慢漂移后的高通残差、duration 做每曲持续策略居中或分源处理），而不是窗口布局或辅助 loss 尺度——本轮全部 bar 相关证据也说明那条线不是瓶颈。

onset 通道的容量检验（2026-10-11，`onset_weight` 0.1→0.0，15 epoch × 3 seed，其余超参与 fixed 臂逐字相同；产物 `onset_zero_pilot_20261011`，summary sha256 `38962bdabfa052af...`）：**撤掉噪声通道没有释放任何容量**——第 15 epoch 的 `log_ioi_ratio` 差值均值为 +0.00004、`log_key_duration_ratio` +0.00060，逐 seed 正负都有（ioi +0.00003/+0.00034/-0.00023），完全在噪声内。但它给出了三段对照：训 onset 得到 581.2–585.5 ms，完全不训 onset 退化到 605.2–730.0 ms，而"全部输出 0"的平凡基线是 578.68 ms。也就是这个通道**有可学成分，可学到的部分仍然不敌一个常数**，因此它作为训练目标目前净贡献为负（占梯度、不产出可用预测），作为诊断量则有意义。

由此补一条比较规则：**不同 loss 权重的两臂之间，`validation_loss` 与 `point_loss` 不可比**（它们是加权均值，权重一变口径就变），跨臂只能比未加权的逐通道 MAE。`tools/compare_matched_sequence_runs.py` 现在支持 `--pair-by/--baseline-value/--candidate-value`，可以按任意 `run_config` 键（如 `onset_weight`）配对，不再只能比 fixed/bar。

跨演奏重现性天花板（2026-10-11，`tools/audit_cross_performance_reproducibility.py`，只读；产物 `cross_performance_reproducibility_20261011/{train,validation}`，summary sha256 分别为 `84f724fe3e90b7adf66abd4ce16f74a9480681590684a2d906941483b2518c99` 与 `28c762b78e34a8fed573c2b2106ee5eba83bec9022de9308cd6bc66927d61518`）：同一份谱面在 PianoCoRe 里有多次演奏，把**两个不同演奏者**在同一和弦位置上的标签相减，差值就是任何只看谱面的模型永远拿不到的部分。口径：train 侧 932 组符合条件（≥2 次演奏且拍数完全一致）、实际配对 708 组 3588 对，另有 224 组因为全部录音都落在同一个 `performer` 值上而被丢弃（这种组说不出"重现性"）；validation 侧 92 组 / 67 组 / 337 对。每组最多取 4 次演奏、每位 performer 只取 1 次（`performer` 为空自成一桶，属于保守处理），位置按 `score_onset_beats` 逐位相等对齐（容差 1e-6 拍），只保留 `qc_status` 为 accepted/ok/空 的行，不等长的组整组丢弃（实测丢弃 0 组）。`implied_per_performance_rms = 跨演奏 RMS / √2` 是**每次演奏标签的不可约噪声 σ**（只用 E[(X1−X2)²]=2σ²，不依赖分布假设），ρ = 2r/(1+r) 是 Spearman–Brown 可靠度，√ρ 则是完美模型与单个观测标签的相关上限。

| 通道（validation / train） | 两次演奏的中位绝对差 | 不可约 σ | 跨演奏相关 r | 可靠度 ρ | 相关上限 √ρ | 高斯读法下的 MAE 底 | 模型 MAE | 预测 0 的 MAE |
|---|---|---|---|---|---|---|---|---|
| `onset_delta_ms` | 403.50 / 439.19 ms | 593.65 / 629.40 ms | 0.270 / 0.337 | 0.425 / 0.505 | 0.652 / 0.710 | 473.67 / 502.20 ms | 586.04 ms | 578.68 ms |
| `log_ioi_ratio` | 0.1183 / 0.1309 | 0.1978 / 0.2142 | 0.542 / 0.524 | 0.703 / 0.688 | 0.838 / 0.830 | 0.1578 / 0.1710 | 0.1796 | 0.1843 |
| `log_key_duration_ratio` | 0.3576 / 0.3952 | 0.5811 / 0.6256 | 0.415 / 0.400 | 0.586 / 0.572 | 0.766 / 0.756 | 0.4637 / 0.4992 | 0.6106 | 0.6668 |

模型 MAE 取 15 epoch × 3 seed 的 fixed 臂 best-epoch 均值，因此与上一节的"预测 0"基线同源可比。读法：**onset 已经封顶**——完美模型最多也只能与观测标签相关 0.65、解释 42.5% 方差，而模型现在 586.04 ms 相对高斯读法下限 473.67 ms 是 1.237 倍，同时还比"预测 0"差 1.3%，即这个通道的剩余空间既小又不可用于任何接入主张；`log_ioi_ratio` 是 1.138 倍、比预测 0 好 2.6%；只有 `log_key_duration_ratio` 还留着 1.317 倍（相对下限 32% 的空间）、比预测 0 好 8.4%。注意 MAE 与 σ 不同量纲，最后一列按高斯换算（MAE = √(2/π)σ ≈ 0.798σ），而实测分布是重尾的（onset 的 p90 绝对差 1402.8 ms 已达 RMS 839.5 ms 的 1.67 倍），所以"下限"只会更低、模型相对下限的差距只会更大——这不影响"onset 无空间、duration 空间最大"的排序。

由此给这条线定两条硬规则：一，以后任何"模型还差多少"的讨论必须对着这张表说，**onset 与 log IOI 已经贴近噪声底，继续调窗口布局、辅助 loss 或权重不会带来可用收益**；二，若还要在这三个通道里投入，只有 `log_key_duration_ratio` 值得（既有 19% 的每曲策略成分，又离下限最远），而它的下一步是目标定义而不是训练技巧。

内存红线（同一天的事故复盘）：这个探针最初写成"组内全部演奏两两配对 + 每位置 Python 元组累积"，在 2026-10-11 14:23 把 `python.exe` 推到提交内存 56.5 GB（本机 commit 上限 48.15 GB），整机僵死后于 14:28:56 非正常断电。实测校正：爆炸项**不是**分片加载——`full_20261003_w8` 有 1238 个 npz，单片解压后仅 29.8 MiB——而是 O(n²) 配对（单个 `piece_group` 最多 1778 次演奏，948 个组全配对是 18,098,229 对）再乘上千个和弦位置。现在有每组上限、每位演奏者只取一次、float32 分块累积，以及"加载前先算配对位置数、超过 `--max-pair-positions`（默认 4000 万）直接拒绝"的前置检查；本轮 train 估算 3,451,886 个位置，实测全程 python 提交内存 1.53 GB。代价是每组仍会调用一次 `load_selected_samples`，其内部会重扫 130,238 行的 manifest，因此 train 侧一轮约 35 分钟（validation 约 2 分钟），这是解析与 IO 开销而非计算量，批量载入可优化但不改变语义。测试见 `tests/test_training_cross_performance.py`（8）。

### duration 专项：每曲水平到底可不可学，居中之后形状有没有变好（2026-10-11）

上表把预算指向 `log_key_duration_ratio`，本轮先把"居中能不能救"这两件事分别量清楚，工具是 `tools/audit_piece_offset_reproducibility.py`（把每次录音在某通道的标签中位数当作它的"水平"，用单向 ICC(1) = (MSB−MSW)/(MSB+(k̄−1)MSW) 检验同一作品不同演奏者的水平是否一致；ICC≈1 说明水平是谱面属性、模型有可能学到，ICC≈0 说明它是演奏者习惯，居中只是把误差换个地方放）与 `tools/analyze_target_error_decomposition.py`（用既有 checkpoint 走真实推理路径 `predict_sample`，把误差按录音拆成 `offset_mae`=|中位误差| 与 `scatter_mae`=去掉每录音一个自由常数后的误差，并给出两侧各自居中后的对称口径 `shape_mae` 与其基线 `label_scatter_mae`="每录音预测一个常数"）。

水平可学性（train 708 曲 / 3588 对，summary sha256 `ae7e245765c169ed160a1bd4d169996dda6ef7c4bda4e12fbe5f312d4e4f5b76`；validation 67 曲 / 337 对，sha256 `8dae9a08f8b64d6477afa694653f36d3634881e74fcf513b60a25b6a635d150b`）：

| 通道 | ICC(1) train / validation | 跨演奏水平差的中位数 | 曲间 sd | 曲内（演奏者）sd |
|---|---|---|---|---|
| `onset_delta_ms` | 0.185 / 0.266 | 149.1 ms / 147.7 ms | 394 ms | 292 ms |
| `log_ioi_ratio` | 0.641 / 0.674 | 0.0245 / 0.0222 | 0.103 | 0.038 |
| `log_key_duration_ratio` | 0.331 / 0.306 | 0.229 / 0.169 | 0.664 | 0.397 |

也就是说 duration 的每曲水平**约三分之一确实是谱面属性**（并非"完全不可学"，这推翻了我上一轮"居中大概没用"的猜测），而剩下三分之二是演奏者自己的全局快慢/松紧选择。注意上面那张表的 σ 来自**按和弦聚合**的 duration，而模型的 duration 目标是**逐音符**的，两者不能直接混用（见下一节的粒度修正）。

每曲居中臂（`--duration-centering`：训练与验证两侧都用"该录音自身监督行中位数"把 duration 目标居中，其余超参与 fixed 臂逐字相同——已用 run_config 差异核对，除中心标志与 `implementation_sha256` 外无任何不同；15 epoch × 3 seed，产物 `v4/duration_centered_20261011/centered_seed*`，`_SUCCESS` 分别 `4f25e101…`/`a67a2c8a…`/`bebb2ca0…`；分解报告 `target_error_decomposition_20261011`，sha256 `fafbc7836abf8abe1680744eea9753e6c4c2d219e20198583767fed00929f796`）：

| 臂 | duration `shape_mae` | 同臂 `raw_mae` | duration `offset_mae` | 基线 `label_scatter_mae` | ioi `shape_mae` | onset `shape_mae` |
|---|---|---|---|---|---|---|
| fixed（3 seed） | 0.5109 / 0.5054 / 0.5044 | 0.6222 / 0.6062 / 0.6034 | 0.334 / 0.322 / 0.311 | 0.5386 | 0.1721 / 0.1734 / 0.1751 | 548.0 / 548.9 / 547.4 ms |
| 居中（3 seed） | 0.5071 / 0.5083 / 0.5071 | 0.5107 / 0.5117 / 0.5103 | 0.054 / 0.054 / 0.052 | 0.5386 | 0.1735 / 0.1731 / 0.1751 | 548.1 / 549.7 / 545.6 ms |

结论是**负向且干净**：对称口径下居中臂形状均值 0.5075 对 fixed 臂 0.5069，差 +0.0006，而两臂各自的 seed 内跨度就有 0.0065——效应量只有种子噪声的十分之一，方向还是反的；居中臂那个漂亮的 `offset_mae≈0.05` 是构造出来的假象（它的目标里根本没有水平），所以跨臂的 `raw_mae`、`validation_loss` 一律不可比，只能比 `shape_mae`。同时**没有释放任何容量**：ioi 与 onset 的形状误差在两臂之间逐 seed 都在原地。

粒度修正（重要，先前这里写过"两臂仍比形状噪声底高约 38%"，那个 0.366 是把**逐和弦均值**的噪声底套在**逐音符**误差上得到的，属于口径混用，已作废）。`tools/audit_cross_performance_reproducibility.py` 现在支持 `--granularity note`：manifest 里 932 个合格组的 `note_count` 全部相等（还有 `score_semantic_sha256` 佐证同一谱面），所以逐音符可以按索引严格对齐；chord 模式在改动前后逐字段数字相同，只多了 `granularity`/`channels_compared`/`label_transform` 三个描述键。validation 侧 67 曲 / 337 对 / 671,306 个音符配对（`cross_performance_note_20261011/validation_note_{raw,centred}`）：

| duration 口径 | 不可约 σ | 高斯读法 MAE 底 | 模型误差 | 相对底 | 剩余空间 | 模型已吃到的信号 |
|---|---|---|---|---|---|---|
| 逐音符原始标签 | 0.6376 | 0.5087 | 0.6106 | 1.200 倍 | 0.102 | 35.6% |
| 逐音符去水平（形状） | 0.5328 | 0.4251 | 0.5069 | 1.192 倍 | 0.0818 | 27.9% |

粒度切换隐含的"演奏者水平 sd"= √(0.6376²−0.5328²) = 0.3502，与独立用 ICC 测到的 0.357/0.348 吻合。**修正后的结论：duration 也已经离墙不远，剩余约 0.08–0.10 MAE（占它当前误差的 13–17%），而不是先前说的三分之一**。同时 duration 是**欠拟合而非过拟合**：fixed 臂 train 0.633–0.638 对 val 0.603–0.622、居中臂 train 0.552–0.555 对 val 0.510–0.512，训练集内一点都不比验证集好，说明连见过的演奏都没拟合上——所以若还要投这 0.08，杠杆在容量/优化与特征覆盖，不在数据量或正则化，更不在目标重参数化。

由此给 duration 定方向：**问题不在目标怎么重新参数化，而在模型根本没学到曲内形状**——要继续投入，就得改形状目标本身或输入特征（键/声部分组、pedal 与延音导致的实际发声长度、把每音符 duration 聚到和弦的口径），而不是居中、改权重或换窗口。这条与 onset 一样，本轮不构成任何接入证据。

容量探针（2026-10-11，`--hidden-dim` 64→256，参数量 52,872→752,136 即 14.2 倍，其余超参与 fixed 臂逐字相同——run_config 差异只有 `hidden_dim` 一个键；15 epoch × 3 seed，产物 `v4/duration_capacity_20261011/hidden256_seed*`，分解报告 `duration_capacity_probe_20261011`，sha256 `86e8d1e915901e22a0c09c714087d59b2df9732f3da0cd162f3a873ebcfcad15`）：**容量也被排除**。逐音符 duration `shape_mae` 三 seed 均值 0.5134（hidden 256）对 0.5069（hidden 64），`raw_mae` 均值 0.6206 对 0.6106，ioi 形状 0.1739 对 0.1735，onset 形状 548.24 对 548.09 ms——差值全部落在各自 seed 跨度内且方向略偏坏；更关键的是**训练集内的 duration 误差同样没下降**（train 0.6218/0.6268/0.6410 对 0.6326/0.6357/0.6377），即加了 14 倍参数连"记住见过的演奏"都做不到。至此四个训练侧杠杆被逐个证伪：窗口布局与 bar 辅助 loss（必须带 layout-only 对照）、loss 权重（`onset_weight=0`）、目标重参数化（每曲居中）、模型容量（14 倍参数）。剩下唯一站得住的候选是**输入里根本没有那个信息**：`NOTE_FEATURE_NAMES` 15 列里有 `score_duration_log1p`/`chord_size_log1p`/`rank_in_chord`/前后和弦 IOI，却没有同声部下一音符的距离，而 pedal 只出现在目标侧（`pedal_at_onset`/`pedal_coverage`）——补它需要重建数据集分片，属于独立立项，不是再跑一轮训练。

因此这条线的正确汇报方式是：**duration 剩余可谈空间约 0.08 MAE（占其误差 13%），且已证明无法靠训练侧旋钮取得**；在没有新输入特征之前继续训练只是烧 GPU。下一步该补的是序列管线缺的量具——作品级 paired bootstrap（已有的 `compare_quality_multiseed.py` 属旧的 flat-row 管线），用来判断已登记门槛"相对最强简单基线改善 ≥5% 且作品 95% 区间排除 0"到底有没有任一通道通过；若没有，真人 A/B 不应启动。

### 作品级 paired bootstrap：序列管线的数值门槛判定（2026-10-11）

`tools/bootstrap_sequence_vs_baselines.py` 补上了序列管线缺的量具（此前只有旧 flat-row 管线的 `compare_quality_multiseed.py`）。协议写死在工具里：重采样单元是 `performance_dataset::piece_group`，组内先平均录音再平均 seed，组间等权，2000 次抽样、种子 20261003，取 2.5/97.5 百分位；**不足 5 组不报区间**；`--split test` 直接拒绝。两条基线是 `predict_zero`（照原谱弹、完全不增强，可交付）与 `predict_recording_median`（每条录音预测它自己的中位数，用了推理时拿不到的信息，作为"一个常数能解释多少"的上界）。

fixed 臂 3 seed 在均衡集 50 条验证演奏（= 50 个作品组，onset 38,829 / ioi 61,246 / duration 112,618 行）上的结果，产物 `sequence_piece_bootstrap_validation_20261011`，sha256 `54bf37dfcb09a7a9c242c253dd38b71c61d7dc4a4fb10d0779ea8d7180374ecc`：

| 通道 | 模型（作品等权 MAE） | vs 每录音常数：配对差 [95% 区间] | vs 预测 0：配对差 [95% 区间] | 相对最强基线 | 门槛 |
|---|---|---|---|---|---|
| `onset_delta_ms` | 548.84 ms | **+34.22 ms** [+22.53, +49.39] | **+8.03 ms** [+0.15, +15.74] | −6.6% | 未过 |
| `log_ioi_ratio` | 0.15761 | +0.00518 [−0.00121, +0.01224] | +0.00033 [−0.00703, +0.00817] | −3.4% | 未过 |
| `log_key_duration_ratio` | 0.55450 | **+0.08234** [+0.04790, +0.11813] | −0.02178 [−0.06869, +0.02551] | −17.4% | 未过 |

三个通道**没有一个通过已登记门槛**（"相对最强简单基线改善 ≥5% 且作品级区间支持改善"）。两条被区间锁死的显著更差必须直说：duration 比"每条录音预测一个常数"显著更差（P(model worse)=1.000），onset 比"照原谱弹"显著更差（P=0.976）。同时**行加权读数会高估成果**：同一批预测在行加权下是"duration 比预测 0 好 8.4%"，作品等权后缩到 3.8% 且区间跨零，所以那句不能再用。

结论与行动边界：模型当前没有可辩护的数值优势，**真人 A/B 不应启动**（那等于让受试者验证一个数值上还不成立的模型）；四个训练侧杠杆已全部证伪，剩下的解释是输入缺少形状证据（同声部下一音符的距离、延音导致的实际发声长度），而补它需要重建数据集分片——那是需要用户先立项的独立工程，不要在它之前再排训练。另有一条与门槛直接相关的观察：`predict_recording_median` 之所以强，是因为"整条录音整体偏快/偏慢、偏长/偏短"本身就解释了很大一部分误差，而模型连这个常数级偏置都没学到——这正好和 ICC 那轮"duration 水平的 1/3 是谱面属性"呼应，是下一个可检验的建模问题（显式预测作品级偏移，而不是让逐点回归去猜）。

### A-lite：不重建分片的声部连续性特征（2026-10-11，中性偏负）

因为 `midi_features.py` 写的是 `pitch_centered = (pitch − 60) / 24`、`score_onset_beats` 原样存、第 4 列是 `log1p(duration_beats)`，而 MIDI 音高是整数、网格间距 1/24 远大于 float32 精度，所以**音高与时值可以从已建好的分片无损反解**——补特征不必然要重建 1238 个 npz。`training/shape_features.py` 据此在加载期派生三列纯谱面几何特征：同音高上一/下一次击键的距离（以本曲拍跨度归一、log1p）、以及"本音符记谱结束瞬间还有多少个其它声部仍在响"。它们只读 `note_features`，不读任何目标列，因此不构成标签泄漏。

一个必须记住的实现细节：`expm1(float32(log1p(2)))` 的往返误差是 ~4e-9 拍，会把"恰好在下一个音符开始时结束"（键盘音乐里的绝大多数写法）判成重叠。修法是 `searchsorted(..., side="right")` 定出"≤"语义，再用 `BEAT_TOLERANCE = 1e-6` 拍解并列。这个容差按实测校准过：50 条验证录音里**最小的不同起点间隔是 0.0208 拍，是容差的约 2.08 万倍**，所以它只吃掉存储噪声，不会合并真实事件。

第二个必须记住的接线不变式：`checkpoint_preparation_flags()`（在 `sequence_expression.py`）把"该 checkpoint 用了哪些输入/目标变换"从 run_config 里读回来，`audit_sequence_expression.py`、`tools/analyze_target_error_decomposition.py`、`tools/bootstrap_sequence_vs_baselines.py` 全部按它分组加载。之前两个评估工具用默认 17 列加载样本去评 20 列的形状臂，直接撞上 `operands could not be broadcast together with shapes (235,17) (20,)` —— 这是真跑出来的错。

臂本身（`--shape-features`，15 epoch × 3 seed，其余超参与 fixed 臂逐字相同，输入宽度 17→20、参数 52,872→53,064；产物 `v4/shape_features_20261011/shape_seed*`，分解报告 `shape_feature_probe_20261011/decomposition` sha256 `290e52f3d397d3e7dab95c9aac3f5fe722857a1cc6e0560cedd3835f391a3b32`，bootstrap `shape_feature_probe_20261011/bootstrap` sha256 `abfe35baa47ee650b18dad5720d4aef24457c77556cc11923974358ee9b4f026`）：

| 指标 | fixed（无新特征） | 形状特征臂 | 差 |
|---|---|---|---|
| duration `shape_mae`（3 seed 均值） | 0.5069（跨度 0.0065） | 0.5047（跨度 0.0032） | −0.0022 |
| duration 作品等权 MAE | 0.55450 | 0.55379 | −0.0007 |
| duration `raw_mae` | 0.6106 | 0.6039 | −0.0067 |
| ioi / onset `shape_mae` | 0.17353 / 548.09 ms | 0.17402 / 547.56 ms | +0.0005 / −0.53 ms |

形状特征把 duration 推低了约 1.1% 的原始 MAE，但**对称口径只有 −0.0022，落在种子跨度之内**，作品级 bootstrap 仍判未过门槛（duration vs predict-zero 的配对差 −0.0225，CI [−0.0731, +0.0299] 跨零；onset 仍显著差于原谱，+9.51 ms，CI [+2.66, +17.50]）。结论：**"形状缺口来自输入缺少声部连续性"这个假设没有被这三列支持**。由于 ρ_shape=0.657 说明缺口原则上可由谱面解释，要继续追就得换解释变量（力度/踏板的记谱标记、短语边界、局部速度上下文）或转向两部分建模，而不是再加同类几何列——每加一列都要付"旧 checkpoint 全部作废 + 重训基线 + 重跑 bootstrap"的代价。

测试见 `tests/test_training_bar_windows.py`（17）、`tests/test_training_bar_objective.py`（15）、`tests/test_training_bar_policy.py`（18，覆盖布局唯一划分、fixed 逐字兼容、collate 局部索引与 native 过滤、`_run_epoch` 计数与梯度、危险配置拒绝、audit/benchmark 侧车、`max_window_chords` 为 `None` 时回落默认 512）、`tests/test_training_matched_compare.py`（3，锁定跨 seed 配对差值与跨度的算法、拒绝覆盖未完成的 run，以及同一 seed 传入两个同名 policy 时直接报错而不是静默覆盖）、`tests/test_training_duration_centering.py`（7：默认路径逐字不变、居中语义只动 duration 列、未监督行不参与取中位但仍平移、无监督行时不产生 NaN、CLI 标志落到 trainer）、`tests/test_training_piece_offsets.py`（6：ICC(1) 手算核对、组大小不等时的加权正确性、每位演奏者只取一条录音、产物封装）、`tests/test_training_error_decomposition.py`（8：偏移/形状/对称口径与"每录音预测常数"基线、中位而非均值定义水平、空通道不崩、两臂标签单位换算）、`tests/test_training_sequence_bootstrap.py`（9：作品等权而非行加权的算法验证、组内先平均录音再平均 seed、缺 seed 时如实跳过、配对差符号约定、区间跨零不算支持、同 seed 可复现、少于 5 组不报区间、拒绝 test split 与过小 draws、封装产物与"退化基线"标记）、`tests/test_training_shape_features.py`（9：音高无损反解、同音高距离按拍跨度归一、并列不吞掉真实间隔、**存储往返不得把相邻音符判成重叠**、真重叠仍要看见、退化谱面有限、非有限时值与空输入报错）、`tests/test_training_shape_wiring.py`（7：默认宽度逐字不变、加列只追加在尾部、派生必须在相对位置缩放之前、CLI 标志落到 trainer、`predict_sample` 从 checkpoint config 读回两个变换开关、缺键时回落 legacy 宽度、`checkpoint_preparation_flags` 默认值）。

## 训练前门禁与备用方案

2026-10-04 已续验的下一轮前置工具：`build_development_selection.py` 确定性抽取五来源各 50 train/10 validation、共 300 个 piece/hash 去重作品组；`audit_game_expressibility.py` 覆盖真实 MIDI 导入、60ms 播放预处理及独立 legacy/Delta 路径；`bar_supervision.py` / `audit_bar_supervision.py` 提供真实小节几何、128-core/512 容量覆盖与 tempo/local 标签分解诊断。新集合属于开发数据；原标签、历史门禁与模型保持原样。原始来源副本、参考和完整审计输出均留在 E 盘。

新验证集 16 小节可信窗口在现有 128 core 中只覆盖 17.74%；因此下一阶段需要按小节组织长跨度监督。原前置审计见 [均衡开发域与可表达性记录](../docs/2026-10-03_均衡开发域_游戏可表达性与小节监督记录.md)。

2026-10-04 已修复短片段合并超过 16 拍的拒绝边界，并新增 `source_game_mapping.py` / `audit_source_game_mapping.py`：同 300 样本、600 案例重新审计为 598 complete、2 超过 30 分钟的导入拒绝；原 native masks 不变。来源旁文件只提供严格身份、单音时值及连续完整原生和弦 IOI 的诊断候选，不能视为调度可行性或模型接入许可。下一步转向真实小节窗口与 tempo/local 监督，尚未启动新训练。命令、覆盖量、指纹和边界见 [播放边界与来源映射记录](../docs/2026-10-04_技术-播放边界与来源映射-report.md)。

正式训练前依次执行：

1. `smoke` 与重复构建 hash 对比；
2. 8-16 个作品的 tiny-overfit，确认标签、mask 和 loss 能工作；
3. target shuffle 负对照，排除输入泄漏；
4. heuristic / identity / 简单统计基线；
5. full build 后按来源分别报告 timing、duration、velocity 和 pedal 指标；
6. 最后才接入 timing/hold 候选计划并做 A/B 偏好测试。

若过拟合，优先保持 piece/performer 分组切分、early stopping、weight decay、dropout、减小模型和数据增强；若欠拟合，先检查目标尺度、mask 与基线，再增加上下文、模型容量或分来源条件，不用测试集反复调参。PianoCoRe 主训练、本地真人样本域适配、A/B 偏好优化应保持三个阶段可独立回滚。
