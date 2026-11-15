param(
    [Parameter(Mandatory = $true)][string]$ImagePath,
    [string]$Language = 'zh-Hans-CN',
    [string]$OutputPath
)

$ErrorActionPreference = 'Stop'
& (Join-Path $PSScriptRoot '../assets/windows_ocr.ps1') @PSBoundParameters
