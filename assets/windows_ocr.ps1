[CmdletBinding(DefaultParameterSetName = 'Image')]
param(
    [Parameter(Mandatory = $true, ParameterSetName = 'Image', Position = 0)][string]$ImagePath,
    [Parameter(Position = 1)][string]$Language = 'zh-Hans-CN',
    [Parameter(ParameterSetName = 'Image', Position = 2)][string]$OutputPath,
    [Parameter(Mandatory = $true, ParameterSetName = 'Stream')][switch]$Stream
)

# Run with Windows PowerShell 5.1; PowerShell 7 does not project these WinRT types.
$ErrorActionPreference = 'Stop'
$OutputEncoding = [Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
$totalWatch = [System.Diagnostics.Stopwatch]::StartNew()
Add-Type -AssemblyName System.Runtime.WindowsRuntime
[Windows.Media.Ocr.OcrEngine, Windows.Foundation, ContentType = WindowsRuntime] | Out-Null
[Windows.Globalization.Language, Windows.Globalization, ContentType = WindowsRuntime] | Out-Null
[Windows.Storage.StorageFile, Windows.Storage, ContentType = WindowsRuntime] | Out-Null
[Windows.Storage.FileAccessMode, Windows.Storage, ContentType = WindowsRuntime] | Out-Null
[Windows.Storage.Streams.IRandomAccessStream, Windows.Storage.Streams, ContentType = WindowsRuntime] | Out-Null
[Windows.Graphics.Imaging.BitmapDecoder, Windows.Graphics.Imaging, ContentType = WindowsRuntime] | Out-Null
[Windows.Graphics.Imaging.SoftwareBitmap, Windows.Graphics.Imaging, ContentType = WindowsRuntime] | Out-Null
[Windows.Media.Ocr.OcrResult, Windows.Foundation, ContentType = WindowsRuntime] | Out-Null

$asTask = [System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object {
    $_.Name -eq 'AsTask' -and $_.IsGenericMethod -and $_.GetParameters().Count -eq 1 -and
    $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1'
} | Select-Object -First 1

function Await-Operation($Operation, [Type]$ResultType) {
    $task = $asTask.MakeGenericMethod($ResultType).Invoke($null, @($Operation))
    $task.GetAwaiter().GetResult()
}

function Read-OcrImage([string]$Image, [System.Diagnostics.Stopwatch]$Watch) {
    $path = (Resolve-Path -LiteralPath $Image).ProviderPath
    $imageStream = $null
    $bitmap = $null
    try {
        $loadWatch = [System.Diagnostics.Stopwatch]::StartNew()
        $file = Await-Operation ([Windows.Storage.StorageFile]::GetFileFromPathAsync($path)) ([Windows.Storage.StorageFile])
        $imageStream = Await-Operation ($file.OpenAsync([Windows.Storage.FileAccessMode]::Read)) ([Windows.Storage.Streams.IRandomAccessStream])
        $decoder = Await-Operation ([Windows.Graphics.Imaging.BitmapDecoder]::CreateAsync($imageStream)) ([Windows.Graphics.Imaging.BitmapDecoder])
        $bitmap = Await-Operation ($decoder.GetSoftwareBitmapAsync()) ([Windows.Graphics.Imaging.SoftwareBitmap])
        $loadWatch.Stop()
        if ($bitmap.PixelWidth -gt [Windows.Media.Ocr.OcrEngine]::MaxImageDimension -or
            $bitmap.PixelHeight -gt [Windows.Media.Ocr.OcrEngine]::MaxImageDimension) {
            throw "Image exceeds Windows OCR maximum dimension $([Windows.Media.Ocr.OcrEngine]::MaxImageDimension)."
        }
        $recognizeWatch = [System.Diagnostics.Stopwatch]::StartNew()
        $result = Await-Operation ($engine.RecognizeAsync($bitmap)) ([Windows.Media.Ocr.OcrResult])
        $recognizeWatch.Stop()
        $words = [System.Collections.Generic.List[object]]::new()
        $lines = [System.Collections.Generic.List[object]]::new()
        foreach ($line in $result.Lines) {
            $lineIndex = $lines.Count
            $lines.Add([ordered]@{ text = $line.Text; word_start = $words.Count; word_count = @($line.Words).Count })
            foreach ($word in $line.Words) {
                $box = $word.BoundingRect
                $words.Add([ordered]@{
                    text = $word.Text
                    bbox = @($box.X, $box.Y, $box.Width, $box.Height)
                    line = $lineIndex
                })
            }
        }
        $Watch.Stop()
        [ordered]@{
            schema = 'windows-ocr-v1'
            path = $path
            language = $engine.RecognizerLanguage.LanguageTag
            width = $bitmap.PixelWidth
            height = $bitmap.PixelHeight
            text = $result.Text
            text_angle = $result.TextAngle
            elapsed_ms = [ordered]@{
                load = [Math]::Round($loadWatch.Elapsed.TotalMilliseconds, 2)
                recognize = [Math]::Round($recognizeWatch.Elapsed.TotalMilliseconds, 2)
                total = [Math]::Round($Watch.Elapsed.TotalMilliseconds, 2)
            }
            lines = $lines.ToArray()
            words = $words.ToArray()
        }
    }
    finally {
        if ($null -ne $bitmap) { $bitmap.Dispose() }
        if ($null -ne $imageStream) { $imageStream.Dispose() }
    }
}

$engine = [Windows.Media.Ocr.OcrEngine]::TryCreateFromLanguage([Windows.Globalization.Language]::new($Language))
if ($Stream) {
    [Console]::InputEncoding = [System.Text.UTF8Encoding]::new($false)
    while ($null -ne ($request = [Console]::ReadLine())) {
        if ($null -eq $engine) {
            [Console]::WriteLine('{"error":"language_unavailable"}')
            break
        }
        try {
            $image = (ConvertFrom-Json -InputObject $request).path
            $result = Read-OcrImage $image ([System.Diagnostics.Stopwatch]::StartNew())
            $json = @{ result = $result } | ConvertTo-Json -Depth 8 -Compress
        }
        catch {
            $json = '{"error":"recognition_failed"}'
        }
        [Console]::WriteLine($json)
    }
    return
}
if ($null -eq $engine) {
    $available = [Windows.Media.Ocr.OcrEngine]::AvailableRecognizerLanguages.LanguageTag -join ', '
    throw "OCR language '$Language' is unavailable. Installed OCR languages: $available"
}
if ($OutputPath -and (Test-Path -LiteralPath $OutputPath)) {
    throw "Output already exists: $OutputPath"
}
$json = Read-OcrImage $ImagePath $totalWatch | ConvertTo-Json -Depth 8 -Compress
if ($OutputPath) {
    [System.IO.File]::WriteAllText([System.IO.Path]::GetFullPath($OutputPath), $json, [System.Text.UTF8Encoding]::new($false))
}
[Console]::WriteLine($json)
