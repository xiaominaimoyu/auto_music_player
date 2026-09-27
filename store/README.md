# Microsoft Store MSIX 构建说明

Store 版本与 GitHub/Gitee 便携版是两个独立分发通道：

- 便携版继续使用 `updates/stable.json` 的 Ed25519 签名检查；
- MSIX 版本由 Microsoft Store 管理主程序更新；
- 首版仅打包 x64 主程序，Audiveris/Jianpu OMR 通过应用内按需组件安装；
- 用户数据使用包身份对应的 `LocalState`，首次启动从旧安装版数据目录复制且不删除来源。

`AppxManifest.xml.template` 中的身份字段不能自行猜测，必须替换为 Partner Center
分配的 `Name`、`Publisher` 和四段式版本号。Store 提交阶段由 Microsoft 重新签名；
本地 sideload 测试仍需要开发证书。

## 构建前提

1. 使用 Release PyInstaller EXE，不能使用 Debug 产物。
2. 安装 Windows SDK，确保 `makeappx.exe` 可用。
3. 从 Partner Center 复制精确的包身份字段。
4. 使用 `tools/build_msix.ps1` 生成包；组件和构建输出位于被 `.gitignore` 忽略的 `build/`。

示例（PowerShell）：

```powershell
pwsh -File .\tools\build_msix.ps1 `
  -PyInstallerExe .\build\release-v1.5.0\AutoMusicPlayer.exe `
  -IdentityName "<Partner Center Identity Name>" `
  -Publisher "CN=<Partner Center Publisher>" `
  -PublisherDisplayName "<Publisher Display Name>" `
  -PackageVersion "1.5.1.0"
```

默认不签名，适用于提交 Store 前的包生成。若要本地安装，需要在受控测试机上
使用开发证书签名；不要把 `.pfx` 或私钥放入仓库。

## 验证门槛

- 包身份、版本、架构和图标通过 manifest 校验；
- 安装目录没有任何写入依赖；
- 首次启动完成旧数据迁移并保留来源；
- 普通权限、管理员目标游戏、UAC 拒绝、离线状态均可安全退出；
- OMR 组件下载后 SHA-256、版本和许可证检查通过；
- Private/Flight 测试通过后再公开提交。
