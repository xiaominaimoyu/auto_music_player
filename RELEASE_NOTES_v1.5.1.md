# Auto Music Player v1.5.1

## 发布摘要

- 修复 Windows 高 DPI 清单，改为 `PerMonitorV2` 并保持普通用户权限运行。
- 完成基础 Windows 单文件 EXE 构建；下载后可直接运行，不需要 Python 环境。
- 补齐 Audiveris 5.11.0 的 AGPL 许可证、来源声明和第三方通知材料。
- 重新验证 jpeditor/Jianpu OMR 离线组件；识别结果仍必须人工确认后入库。
- 保留 GitHub/Gitee 第一阶段更新检查：只校验并提示，不自动替换程序。

## 发布资产

- `AutoMusicPlayer.exe`：完整便携版 Windows x64 程序。
- `Audiveris-5.11.0-windows-x86_64-portable.zip`：印刷五线谱离线组件。
- `jpeditor-omr-0.7.6-autoplayer-win32-x64.zip`：印刷/实验性手写简谱离线组件。
- `omr-components.catalog.json`：组件版本、SHA-256 和许可证元数据。

## 验证范围

- `526 passed, 2 skipped, 14 subtests passed`。
- Jianpu 离线图片 smoke test 识别 56 个音符，人工确认标记为 true。
- Audiveris 5.11.0 CLI 启动和帮助命令通过。
- 基础 MSIX AppCert 总体结果为 PASS；OMR 组件按需提供，不内置到基础 MSIX。

## 已知限制

- 当前便携 EXE 未配置可信 Authenticode 证书，首次运行可能出现 Windows 信任提示；请以发行页 SHA-256 为准。
- 第二阶段自动下载/替换更新暂未启用。
- 尚未用真实印刷五线谱样本完成 Audiveris 识别准确率验收；手写简谱仍为实验性能力。
