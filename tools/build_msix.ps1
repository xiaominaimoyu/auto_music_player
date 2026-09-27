param(
    [Parameter(Mandatory = $true)]
    [string]$PyInstallerExe,
    [Parameter(Mandatory = $true)]
    [string]$IdentityName,
    [Parameter(Mandatory = $true)]
    [string]$Publisher,
    [Parameter(Mandatory = $true)]
    [string]$PublisherDisplayName,
    [string]$PackageVersion,
    [string]$OutputDirectory = "build/msix",
    [string]$MakeAppxPath,
    [string]$SignToolPath,
    [string]$CertificatePath,
    [string]$CertificatePassword
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$templatePath = Join-Path $repoRoot "store\AppxManifest.xml.template"
$assetScript = Join-Path $repoRoot "tools\prepare_msix_assets.ps1"
function Resolve-RepoPath([string]$value) {
    if ([System.IO.Path]::IsPathRooted($value)) {
        return (Resolve-Path -LiteralPath $value).Path
    }
    return (Resolve-Path (Join-Path $repoRoot $value)).Path
}

$exePath = Resolve-RepoPath $PyInstallerExe
$outputPath = if ([System.IO.Path]::IsPathRooted($OutputDirectory)) {
    $OutputDirectory
} else {
    Join-Path $repoRoot $OutputDirectory
}
$outputRoot = [System.IO.Path]::GetFullPath($outputPath)
$stage = Join-Path $outputRoot "stage"
$packagePath = Join-Path $outputRoot "AutoMusicPlayer.msix"

if (-not (Test-Path -LiteralPath $templatePath -PathType Leaf)) {
    throw "找不到 MSIX manifest 模板: $templatePath"
}
if (-not (Test-Path -LiteralPath $exePath -PathType Leaf)) {
    throw "找不到 Release EXE: $exePath"
}
if ([System.IO.Path]::GetExtension($exePath).ToLowerInvariant() -ne ".exe") {
    throw "-PyInstallerExe 必须指向 .exe"
}

if ([string]::IsNullOrWhiteSpace($PackageVersion)) {
    $versionText = (Get-Content -LiteralPath (Join-Path $repoRoot "version.txt") -Raw).Trim()
    $parts = $versionText.Split('.')
    if ($parts.Count -gt 4) { throw "版本号不能超过四段: $versionText" }
    while ($parts.Count -lt 4) { $parts += "0" }
    $PackageVersion = ($parts -join ".")
}
if ($PackageVersion -notmatch '^\d+\.\d+\.\d+\.\d+$') {
    throw "MSIX 包版本必须是四段数字，例如 1.5.0.0: $PackageVersion"
}

function Resolve-Tool([string]$requested, [string]$name) {
    if (-not [string]::IsNullOrWhiteSpace($requested)) {
        $resolved = (Resolve-Path -LiteralPath $requested).Path
        if (-not (Test-Path -LiteralPath $resolved -PathType Leaf)) {
            throw "找不到工具 $name`: $resolved"
        }
        return $resolved
    }
    $command = Get-Command $name -ErrorAction SilentlyContinue
    if ($command) { return $command.Source }
    $kitRoot = Join-Path ${env:ProgramFiles(x86)} "Windows Kits\10\bin"
    if (Test-Path -LiteralPath $kitRoot) {
        $candidate = Get-ChildItem -LiteralPath $kitRoot -Recurse -Filter $name -File -ErrorAction SilentlyContinue |
            Where-Object { $_.FullName -match '\\x64\\' } |
            Sort-Object FullName -Descending |
            Select-Object -First 1
        if ($candidate) { return $candidate.FullName }
    }
    throw "未找到 $name。请安装 Windows SDK，或通过参数显式指定路径。"
}

$makeappx = Resolve-Tool $MakeAppxPath "makeappx.exe"

if (Test-Path -LiteralPath $stage) {
    $stageFull = [System.IO.Path]::GetFullPath($stage)
    if (-not $stageFull.StartsWith($outputRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "拒绝清理输出目录之外的路径: $stageFull"
    }
    Remove-Item -LiteralPath $stage -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $stage | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $stage "Assets") | Out-Null

Copy-Item -LiteralPath $exePath -Destination (Join-Path $stage "AutoMusicPlayer.exe")
foreach ($file in @("THIRD_PARTY_NOTICES.md", "LICENSE")) {
    $source = Join-Path $repoRoot $file
    if (Test-Path -LiteralPath $source -PathType Leaf) {
        Copy-Item -LiteralPath $source -Destination (Join-Path $stage $file)
    }
}

& powershell.exe -NoProfile -ExecutionPolicy Bypass -File $assetScript `
    -SourceIcon (Join-Path $repoRoot "app.ico") `
    -OutputDirectory (Join-Path $stage "Assets")
if ($LASTEXITCODE -ne 0) { throw "生成 MSIX 图标失败，退出码 $LASTEXITCODE" }

$manifest = Get-Content -LiteralPath $templatePath -Raw
$manifest = $manifest.Replace("__STORE_IDENTITY_NAME__", $IdentityName)
$manifest = $manifest.Replace("__STORE_PUBLISHER__", $Publisher)
$manifest = $manifest.Replace("__STORE_PUBLISHER_DISPLAY_NAME__", $PublisherDisplayName)
$manifest = $manifest.Replace("__STORE_VERSION__", $PackageVersion)
if ($manifest.Contains("__STORE_")) { throw "manifest 仍包含未替换的身份占位符" }
$manifestPath = Join-Path $stage "AppxManifest.xml"
Set-Content -LiteralPath $manifestPath -Value $manifest -Encoding UTF8

$bytes = (Get-ChildItem -LiteralPath $stage -Recurse -File | Measure-Object -Property Length -Sum).Sum
if ($bytes -gt 500MB) {
    throw "MSIX 基础包超过项目约定的 500 MB 上限: $bytes bytes"
}

New-Item -ItemType Directory -Force -Path $outputRoot | Out-Null
if (Test-Path -LiteralPath $packagePath) { Remove-Item -LiteralPath $packagePath -Force }
& $makeappx pack /d $stage /p $packagePath /o
if ($LASTEXITCODE -ne 0) { throw "makeappx pack 失败，退出码 $LASTEXITCODE" }

if (-not [string]::IsNullOrWhiteSpace($CertificatePath)) {
    $signtool = Resolve-Tool $SignToolPath "signtool.exe"
    $signArgs = @("sign", "/fd", "SHA256", "/a", "/f", $CertificatePath)
    if (-not [string]::IsNullOrWhiteSpace($CertificatePassword)) {
        $signArgs += @("/p", $CertificatePassword)
    }
    $signArgs += $packagePath
    & $signtool @signArgs
    if ($LASTEXITCODE -ne 0) { throw "signtool 签名失败，退出码 $LASTEXITCODE" }
} else {
    Write-Warning "未签名输出：Store 提交由 Microsoft 重签；本地 sideload 前必须使用开发证书签名。"
}

Write-Output "MSIX package: $packagePath"
Write-Output (Get-FileHash -LiteralPath $packagePath -Algorithm SHA256 | Format-Table -HideTableHeaders | Out-String).Trim()
