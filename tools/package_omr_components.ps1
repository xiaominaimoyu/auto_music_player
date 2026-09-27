[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$JianpuPackageDirectory,

    [Parameter(Mandatory = $true)]
    [string]$AudiverisDirectory,

    [string]$NodeExecutable = (Get-Command node.exe -ErrorAction Stop).Source,
    [string]$OutputDirectory = (Join-Path (Resolve-Path 'dist') 'omr-components'),
    [string]$JianpuLicensePath,
    [string]$SmokeTestInput,
    [switch]$RunSmokeTest
)

$ErrorActionPreference = 'Stop'
$expectedNodeHash = '58e74bf02fc5bbacc41dcb8bef089961cd5bddd37830b87784e4fc624d145d1f'
$expectedJianpuPackageHash = '1025e8757c8a77362f84c5ecc26750719f180817360fb8b91640c13f68e9c2c8'

function Require-File([string]$Path, [string]$Label) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "$Label does not exist: $Path"
    }
    return (Resolve-Path -LiteralPath $Path).Path
}

function Require-Directory([string]$Path, [string]$Label) {
    if (-not (Test-Path -LiteralPath $Path -PathType Container)) {
        throw "$Label does not exist: $Path"
    }
    return (Resolve-Path -LiteralPath $Path).Path
}

$sourceJianpu = Require-Directory $JianpuPackageDirectory 'jpeditor package directory'
$sourceAudiveris = Require-Directory $AudiverisDirectory 'Audiveris administrative image'
$nodePath = Require-File $NodeExecutable 'Node runtime'
if (-not (Test-Path -LiteralPath (Join-Path $sourceJianpu 'omr.js') -PathType Leaf)) {
    throw 'The jpeditor package must contain omr.js'
}
if (-not (Test-Path -LiteralPath (Join-Path $sourceAudiveris 'Audiveris.exe') -PathType Leaf)) {
    throw 'The Audiveris directory must contain Audiveris.exe'
}

$nodeHash = (Get-FileHash -LiteralPath $nodePath -Algorithm SHA256).Hash.ToLowerInvariant()
if ($nodeHash -ne $expectedNodeHash) {
    throw "Node runtime hash mismatch: expected $expectedNodeHash, got $nodeHash"
}

$packageMarker = Get-ChildItem -LiteralPath (Split-Path $sourceJianpu) -Filter 'jpeditor-omr-*.zip' -File -ErrorAction SilentlyContinue |
    Where-Object { (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant() -eq $expectedJianpuPackageHash } |
    Select-Object -First 1
if (-not $packageMarker) {
    Write-Warning 'The upstream ZIP marker was not found beside the package directory; record the exact package hash in the release manifest.'
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$templateRoot = Join-Path $repoRoot 'omr-components\jianpu_omr'
$workRoot = Join-Path $repoRoot ('build\omr-assembly-' + [DateTime]::UtcNow.ToString('yyyyMMdd-HHmmss'))
$stageAudiveris = Join-Path $workRoot 'audiveris'
$stageJianpu = Join-Path $workRoot 'jianpu_omr'
New-Item -ItemType Directory -Force -Path $stageAudiveris, $stageJianpu, $OutputDirectory | Out-Null

Copy-Item -Path (Join-Path $sourceAudiveris '*') -Destination $stageAudiveris -Recurse -Force
Copy-Item -Path (Join-Path $sourceJianpu '*') -Destination $stageJianpu -Recurse -Force
Copy-Item -LiteralPath (Join-Path $templateRoot 'amp_adapter.mjs'), (Join-Path $templateRoot 'amp_worker.mjs'), (Join-Path $templateRoot 'jianpu_omr.cmd'), (Join-Path $templateRoot 'README.md') -Destination $stageJianpu -Force
Copy-Item -LiteralPath $nodePath -Destination (Join-Path $stageJianpu 'node.exe') -Force
$componentManifest = [ordered]@{
    schema_version = 1
    managed_by_app = $true
    component = [ordered]@{ id = 'audiveris-staff'; version = '5.11.0'; upstream = 'https://github.com/Audiveris/audiveris/releases/tag/5.11.0' }
}
$componentManifest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $stageAudiveris 'component.json') -Encoding utf8
$componentManifest = [ordered]@{
    schema_version = 1
    managed_by_app = $true
    component = [ordered]@{ id = 'jpeditor-jianpu'; version = '0.7.6'; upstream = 'https://github.com/lodebar2026/jpeditor/tree/v0.7.6' }
}
$componentManifest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $stageJianpu 'component.json') -Encoding utf8

$licenseDir = Join-Path $stageJianpu 'licenses'
New-Item -ItemType Directory -Force -Path $licenseDir | Out-Null
if ($JianpuLicensePath) {
    Copy-Item -LiteralPath (Require-File $JianpuLicensePath 'jpeditor license') -Destination (Join-Path $licenseDir 'jpeditor-LICENSE') -Force
} else {
    Invoke-WebRequest -Uri 'https://raw.githubusercontent.com/lodebar2026/jpeditor/v0.7.6/LICENSE' -OutFile (Join-Path $licenseDir 'jpeditor-LICENSE')
}
Invoke-WebRequest -Uri 'https://raw.githubusercontent.com/nodejs/node/v24.14.1/LICENSE' -OutFile (Join-Path $licenseDir 'nodejs-LICENSE')
Copy-Item -LiteralPath (Join-Path $repoRoot 'THIRD_PARTY_NOTICES.md') -Destination (Join-Path $licenseDir 'AutoMusicPlayer-THIRD_PARTY_NOTICES.md') -Force

if ($RunSmokeTest) {
    if (-not $SmokeTestInput) {
        throw 'RunSmokeTest requires -SmokeTestInput pointing to a known score bitmap'
    }
    $sample = Require-File $SmokeTestInput 'OMR smoke-test input'
    $smokeOutput = Join-Path $workRoot 'smoke-result.json'
    $workerPath = Join-Path $stageJianpu 'jianpu_omr.cmd'
    $workerArgs = @('--input', [string]$sample, '--output', [string]$smokeOutput, '--mode', 'printed')
    & $workerPath @workerArgs
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $smokeOutput)) {
        throw "Jianpu worker smoke test failed with exit code $LASTEXITCODE"
    }
    $smoke = Get-Content -LiteralPath $smokeOutput -Raw | ConvertFrom-Json
    if ($smoke.format -ne 'auto-music-player-source' -or @($smoke.notes).Count -eq 0) {
        throw 'Jianpu worker smoke output is not a non-empty rich source JSON'
    }
}

$audiverisZip = Join-Path $OutputDirectory 'Audiveris-5.11.0-windows-x86_64-portable.zip'
$jianpuZip = Join-Path $OutputDirectory 'jpeditor-omr-0.7.6-autoplayer-win32-x64.zip'
Compress-Archive -Path $stageAudiveris -DestinationPath $audiverisZip -CompressionLevel Optimal -Force
Compress-Archive -Path $stageJianpu -DestinationPath $jianpuZip -CompressionLevel Optimal -Force

$expandedBytes = (Get-ChildItem -LiteralPath $workRoot -Recurse -File | Measure-Object -Property Length -Sum).Sum
if ($expandedBytes -gt 500MB) { throw "OMR expanded bundle exceeds 500 MiB: $expandedBytes" }
$catalog = [ordered]@{
    schema_version = 1
    generated_at = [DateTime]::UtcNow.ToString('o')
    expanded_bytes = [int64]$expandedBytes
    max_bytes = 500MB
    components = @(
        [ordered]@{
            id = 'audiveris-staff'
            version = '5.11.0'
            archive = (Split-Path $audiverisZip -Leaf)
            sha256 = (Get-FileHash -LiteralPath $audiverisZip -Algorithm SHA256).Hash.ToLowerInvariant()
            license = 'AGPL-3.0'
            upstream = 'https://github.com/Audiveris/audiveris/releases/tag/5.11.0'
            supports = @('printed-staff')
        },
        [ordered]@{
            id = 'jpeditor-jianpu'
            version = '0.7.6'
            archive = (Split-Path $jianpuZip -Leaf)
            sha256 = (Get-FileHash -LiteralPath $jianpuZip -Algorithm SHA256).Hash.ToLowerInvariant()
            license = 'MIT plus dependency notices'
            upstream = 'https://github.com/lodebar2026/jpeditor/tree/v0.7.6'
            supports = @('printed-jianpu', 'handwritten-experimental')
            node_sha256 = $nodeHash
        }
    )
}
$catalogPath = Join-Path $OutputDirectory 'omr-components.catalog.json'
$catalog | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $catalogPath -Encoding utf8
Get-ChildItem -LiteralPath $OutputDirectory -File | Select-Object Name,Length
