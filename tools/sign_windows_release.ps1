[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$FilePath,

    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[0-9A-Fa-f]{40}$')]
    [string]$CertificateThumbprint,

    [Parameter(Mandatory = $true)]
    [ValidatePattern('^https?://')]
    [string]$TimestampUrl,

    [string]$Description = 'AutoMusicPlayer'
)

$ErrorActionPreference = 'Stop'

$resolvedFile = (Resolve-Path -LiteralPath $FilePath).Path
if (-not [System.IO.File]::Exists($resolvedFile)) {
    throw "Release file does not exist: $resolvedFile"
}

$thumbprint = $CertificateThumbprint.Replace(' ', '').ToUpperInvariant()
$certificate = @(
    Get-ChildItem -Path Cert:\CurrentUser\My, Cert:\LocalMachine\My -CodeSigningCert |
        Where-Object { $_.Thumbprint -eq $thumbprint -and $_.HasPrivateKey }
) | Select-Object -First 1
if ($null -eq $certificate) {
    throw "A trusted code-signing certificate with a private key was not found: $thumbprint"
}
if ($certificate.NotAfter -le (Get-Date)) {
    throw "The code-signing certificate has expired: $($certificate.NotAfter.ToString('u'))"
}

$signTool = Get-Command signtool.exe -ErrorAction SilentlyContinue |
    Select-Object -ExpandProperty Source -First 1
if (-not $signTool) {
    $kitsRoot = Join-Path ${env:ProgramFiles(x86)} 'Windows Kits\10\bin'
    $signTool = Get-ChildItem -LiteralPath $kitsRoot -Filter signtool.exe -File -Recurse |
        Where-Object { $_.FullName -match '\\x64\\signtool\.exe$' } |
        Sort-Object FullName -Descending |
        Select-Object -ExpandProperty FullName -First 1
}
if (-not $signTool) {
    throw 'signtool.exe was not found. Install the Windows SDK signing tools.'
}

& $signTool sign /fd SHA256 /sha1 $thumbprint /tr $TimestampUrl /td SHA256 /d $Description $resolvedFile
if ($LASTEXITCODE -ne 0) {
    throw "SignTool sign failed with exit code $LASTEXITCODE"
}

& $signTool verify /pa /all /v $resolvedFile
if ($LASTEXITCODE -ne 0) {
    throw "SignTool verification failed with exit code $LASTEXITCODE"
}

$signature = Get-AuthenticodeSignature -LiteralPath $resolvedFile
if ($signature.Status -ne [System.Management.Automation.SignatureStatus]::Valid) {
    throw "Authenticode verification is not Valid: $($signature.Status) $($signature.StatusMessage)"
}

$hash = Get-FileHash -LiteralPath $resolvedFile -Algorithm SHA256
[pscustomobject]@{
    Path = $resolvedFile
    Sha256 = $hash.Hash.ToLowerInvariant()
    Signer = $signature.SignerCertificate.Subject
    Thumbprint = $signature.SignerCertificate.Thumbprint
    TimestampCertificate = $signature.TimeStamperCertificate.Subject
}
