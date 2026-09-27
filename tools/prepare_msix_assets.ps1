param(
    [Parameter(Mandatory = $true)]
    [string]$SourceIcon,
    [Parameter(Mandatory = $true)]
    [string]$OutputDirectory
)

$ErrorActionPreference = "Stop"
$source = (Resolve-Path -LiteralPath $SourceIcon).Path
$output = [System.IO.Path]::GetFullPath($OutputDirectory)
New-Item -ItemType Directory -Force -Path $output | Out-Null

Add-Type -AssemblyName System.Drawing
$icon = New-Object System.Drawing.Icon($source)

function Write-IconPng([string]$name, [int]$size) {
    $bitmap = New-Object System.Drawing.Bitmap($size, $size)
    $graphics = [System.Drawing.Graphics]::FromImage($bitmap)
    try {
        $graphics.Clear([System.Drawing.Color]::Transparent)
        $graphics.InterpolationMode = [System.Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic
        $graphics.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::HighQuality
        $graphics.DrawIcon($icon, 0, 0)
        $bitmap.Save((Join-Path $output $name), [System.Drawing.Imaging.ImageFormat]::Png)
    }
    finally {
        $graphics.Dispose()
        $bitmap.Dispose()
    }
}

try {
    Write-IconPng "StoreLogo.png" 50
    Write-IconPng "Square44x44Logo.png" 44
    Write-IconPng "Square150x150Logo.png" 150
}
finally {
    $icon.Dispose()
}

Write-Output "MSIX assets prepared in $output"
