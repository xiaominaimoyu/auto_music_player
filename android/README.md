# 游戏自动演奏器 · Android 端

桌面版(auto_music_player)的安卓适配:识别乐谱 → 触摸注入演奏。
与桌面版"模拟物理键盘扫描码"不同,安卓端把音符映射为**屏幕琴键坐标**,通过
**系统无障碍手势接口(dispatchGesture)** 注入多点触摸事件实现演奏 ——
**免 root、免 adb、不依赖任何外部工具,安装即用**(仅需开启无障碍服务)。

> 仅供学习交流与个人娱乐,请遵守游戏用户协议,使用产生的风险自行承担。

## 架构映射(桌面版 → 安卓版)

| 桌面版(core/) | 安卓版(android/app/src/main/java/com/automusic/player/) | 说明 |
|---|---|---|
| parser.py | core/JianpuParser.kt | 简谱解析,协议一致 |
| recognizer.py | core/recognizer/ | OkHttp SSE 流式,prompt 复用 |
| database.py | core/db/ | Room(SQLite) |
| settings_store.py | core/settings/SettingsStore.kt | DataStore |
| keymap.py | core/KeyPointMap.kt | 音符 → **屏幕坐标**(归一化 0..1) |
| player.py | core/PlayerEngine.kt | 协程 + 绝对时钟调度 |
| keyboard_driver.py(SendInput) | input/TouchInjector.kt + AmpAccessibilityService | 无障碍手势注入 |
| gui/(PyQt6) | ui/(Jetpack Compose) | 暗色琥珀金主题对齐 |

## 环境要求

- Android Studio(Koala 及以上),内置 JDK 需 17(JBR 25 及以上会因 Gradle 8.7 不支持而失败,可安装 Temurin 17 并在 Gradle JVM 中选择)
- 真机 Android 8+(建议 11+;已实测华为 nova 11 / HarmonyOS 4.2)

## 构建

```bash
# 方式一:Android Studio 直接打开 android/ 目录,等待 Sync 后 Run
# 方式二:命令行(需 JDK 17)
cd android
./gradlew :app:assembleDebug     # 产物: app/build/outputs/apk/debug/app-debug.apk
```

## 首次使用(仅需一次)

1. 安装 APK 并打开
2. 「演奏」页 → 触摸注入卡 → **"去开启"**(直达服务详情页)→ 打开主开关并允许
3. 回 App 状态点变绿即可使用

## 使用流程

1. **标定**:导入游戏演奏界面截图(鸣潮/原神内置按真机实测的默认布局),
   微调 21 个琴键点位(或"四角推算"自动插值),保存;可用"测试点击"验证落点
2. **模型设置**(可选):添加 OpenAI 兼容多模态供应商 → 测试 → 设为激活;不配置则用内置样例
3. **识别**(两种途径任选):
   - 在线识别:导入乐谱图 → 开始识别 → 校对 → 保存入库
   - 外部 AI 识别(免配置):点「复制提示词」→ 粘贴到任意外部 AI 工具并附上乐谱图片 → 把返回的简谱粘贴到结果框 → 校对 → 保存入库
4. **演奏**:选谱 → 选布局 → 调 BPM → 开始演奏 → 3 秒倒计时内切到游戏

## 技术要点

- **注入通道**:无障碍 `dispatchGesture`,多 stroke 同一手势 = 多指和弦
  (`ACTION_DOWN`→按住→自动抬起由 stroke duration 表达),按屏幕绝对坐标路由到前台游戏窗口
- **坐标**:全部按整块显示器归一化存储(`maximumWindowMetrics`),演奏时换算真实像素;
  App 以悬浮小窗/分屏运行时坐标依然正确
- **时序**:协程 + 绝对时钟调度(`delayUntil`),每音符周期严格 = 时值 + gap,
  修正了桌面版"周期 = dur×hold + gap"的节奏偏快问题
- **多布局**:鸣潮/原神预设按真机截图圆点检测生成,支持自建布局与四角双线性插值
- **保活**:演奏期间启动前台服务(`specialUse`),防止进程被杀中断演奏

## 关于注入通道的历史

曾实现 Shizuku + `InputManager.injectInputEvent`(反射 hidden API + shell uid 转发)路径,
在华为 HarmonyOS 4.2 上被系统对抗策略拒绝(注入返回 false,adb 会话进程除外),
且需要外部激活;无障碍通道在同等场景下稳定工作,故最终仅保留无障碍通道,
App 完全自主运行。输入层接口(`dispatchChord`)保持中立,未来可插拔其他通道(Root/adb 等)。

## 已知边界

- 游戏版本更新可能改变琴键布局 → 到标定页重新标定即可
- 触摸屏多指上限一般 5~10 点,3 音和弦无压力;特大和弦暂不支持
- 无障碍手势由系统派发,极快节奏(BPM>200 连十六分)时可能引入轻微抖动
- 注入式触摸理论上可能被游戏检测,与桌面版同风险等级
