# Auto Music Player v1.5.0

## 发布摘要

- 完善离线 PDF/DOCX 文本导入、MusicXML/MXL 导入和人工确认门禁。
- 增加 Audiveris 五线谱与 jpeditor/Jianpu OMR 的按需离线组件协议、打包脚本和来源校验记录。
- MIDI 导入保留多轨、力度、拍号、速度、踏板和原始来源 sidecar，同时维持现有游戏投影。
- 演奏人性化计划支持可追溯 seed、乐句相关抖动和两条播放路径共用的时序塑形。
- 增加 SQLite 数据迁移、便携/安装双模式、GitHub/Gitee 双镜像签名 manifest 的第一阶段检查提示。

## 安装与更新

- GitHub/Gitee 发行版提供基础 Windows EXE；Audiveris/Jianpu OMR 作为按需离线组件包单独提供。
- 第一阶段更新器只校验签名 manifest 并提示，不会自动下载或替换程序。
- 本版直接下载 EXE 尚未配置可信 Authenticode 证书；请核对发行页 SHA-256，并优先使用 Microsoft Store MSIX 版本（后续发布）。

## 已验证范围

- Windows 11 本地：`516 passed, 2 skipped, 14 subtests`。
- PyQt6 offscreen GUI smoke 通过。
- 组件包展开体积低于 500 MiB；识别结果仍必须人工确认后入库。

## 已知限制

- 手写简谱识别仍为实验性入口，不能视为准确率验收通过。
- 当前便携 EXE 未完成可信 Authenticode 签名；正式自动替换更新暂不启用。
