# Auto Music Player v1.4.1

v1.4.1 在 v1.4.0 功能基础上恢复启动安全契约，并修复 Windows 打包时可能误收集第三方 ICU DLL 的问题。该版本同时包含 MIDI/JSON 导入、播放控制、练习与录制编辑能力。

## 主要更新

### 乐谱导入与统一播放链路

- 支持本项目 JSON、绝对时间 JSON、口风琴模拟器单曲 `events` 与整库 `library` 信封。
- 支持标准 MIDI Type 0/1、全局速度图、多轨选择或合并、旋律整理、移调和八度折回。
- 黑键、打击乐、超范围音符、复调清理和其他有损转换会写入导入或播放降级报告。
- 外部来源先统一为 `SourceSong → canonical score`；默认档位继续使用传统 `Player`，三角洲档位继续经过 `Profile → IR → Compiler → EventPlayer`，没有合并两条输入路径。

### 播放控制与 Windows 可靠性

- 每首乐谱、每个 Profile 可保存 BPM、移调、片段和练习设置，SQLite 原有 `scores` 数据结构保持兼容。
- 增加片段播放、进度定位、暂停/断点续播、焦点丢失保护和目标权限预检。
- 按键松开与 `panic_release` 不受前台窗口保护回调阻断；停止、异常和进程退出继续执行紧急释放。
- PyInstaller 包继续携带 `mido`、`pynput`、默认配置、Profile 与第三方许可声明。

### 练习、双轨提示与录制编辑

- 练习模式只监听实际输入，不调用 `KeyboardDriver`、传统 `Player` 或 `EventPlayer`。
- 双轨区域分别显示期望输入与用户输入，并统计命中、错误和进度；三角洲档位可选严格校验鼠标修饰键。
- 上传识别页支持步进录制和物理键实时录制，可量化时值并识别同时按下的和弦。
- 乐谱库支持编辑现有 SQLite 记录；校对和编辑流程支持添加、删除、移动、撤销与重做。

## v1.4.1 修复

- 恢复风险警示的 3 秒阅读锁定、简单教程、免费开源防诈骗声明和联系方式。
- “下次启动不再弹出”只在倒计时结束、用户确认继续后持久化；退出、Esc 或关闭窗口不保存该选择。
- 对话框接受、退出和关闭时都会停止倒计时定时器。
- 主窗口与 `version.txt` 统一为 v1.4.1。
- 恢复所有路径形式下的 `icu*.dll` 打包过滤，并增加防回退测试，避免构建机中的 Poppler/Conda ICU 遮蔽 Windows 系统 ICU、造成 QtCore 启动失败。

## 验证结果

- 旧缺陷定向回归：`228 passed, 2 skipped, 6 subtests passed`。
- 完整 unittest：`422 tests, OK (skipped=2)`。
- 完整 pytest：`489 passed, 2 skipped, 14 subtests passed`。
- 离屏 GUI 冒烟：`GUI smoke OK`。
- Windows 单文件候选包连续运行 5 秒，随后按精确进程路径清理，未留下残余进程。
- 包内检查：`THIRD_PARTY_NOTICES.md`、`config.yaml`、2 个 Profile、`mido` 与 `pynput` 均存在；`icu*.dll` 为 0 个。

## Windows 候选产物

- 文件：`AutoMusicPlayer.exe`
- 大小：`28,674,528` 字节（约 27.35 MiB）
- SHA-256：`C306F2C6CA6357DC2715FAF5BB7B7508C10DAAF4E19B46B014906C194678DC9A`
- Authenticode：`NotSigned`
- 构建环境：Windows 11、Python 3.14.7、PyInstaller 6.22.2、PyQt6/Qt 6.11.0、mido 1.3.3

## 升级与验收边界

- 升级前建议备份 `data/scores.db`；启动时会按需创建兼容的 `score_preferences` 表，不修改既有乐谱记录格式。
- 复杂 MIDI 转为有限音域的游戏谱属于有意改编，不保证无损还原全部声部、踏板、力度和演奏法。
- 当前产物未做代码签名。上述结论覆盖源码自动化测试、包内容检查和普通权限启动冒烟，不等同于真实游戏、管理员权限/UIPI、不同反作弊环境或长期演奏验收。
