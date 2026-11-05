# 本次变更的数据流

桌面界面由 PyQt6 实现。启动入口解析数据目录，ScoreDB 管理 SQLite 曲库和相对路径保存的来源附件。更新检查沿用 Ed25519 签名清单，下载文件先暂存，完成校验后交给用户。试听沿用 WinMM MIDI 输出，并由演奏档位提供音高和节奏规则。

## 背景与外观

main.py 将现有解析后的 data_dir 注入主窗口。core/appearance.py 负责 appearance.json 的规范化、相对素材路径约束及原子保存；素材放在 backgrounds/。gui/background.py 负责限量解码、图片优化和根容器缓存绘制；gui/appearance_dialog.py 管理预览草稿、保存及取消。gui/theme.py 生成限定于主窗口根容器的半透明表面样式。图片与配置成功提交后才回收旧素材；预览不写盘。JPEG 插件加入现有 PyInstaller 白名单。
