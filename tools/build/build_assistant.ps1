param([string]$Python = 'python')
$ErrorActionPreference = 'Stop'
$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$releaseRoot = Join-Path $projectRoot 'dist/Ninja3Assistant'
$legacyRoot = Join-Path $projectRoot 'data/realtime-v2/publish/Ninja3Assistant'
$running = Get-CimInstance Win32_Process | Where-Object {
    $_.ExecutablePath -and ($_.ExecutablePath.StartsWith($releaseRoot, [StringComparison]::OrdinalIgnoreCase) -or
                           $_.ExecutablePath.StartsWith($legacyRoot, [StringComparison]::OrdinalIgnoreCase))
}
if ($running) { throw 'Exit Ninja3Assistant and its worker before building.' }
Push-Location $projectRoot
try {
    & $Python -m tools.build.check_assets
    if ($LASTEXITCODE -ne 0) { throw 'Prepare local assets first; see assets/README.md.' }
    & $Python -m tools.build.build_intelligence_portrait_batch
    if ($LASTEXITCODE -ne 0) { throw 'Intelligence portrait graph generation failed.' }
    & $Python -m PyInstaller --noconfirm --distpath build/worker-dist --workpath build/pyinstaller packaging/Ninja3Worker.spec
    if ($LASTEXITCODE -ne 0) { throw 'Worker build failed.' }
    & cmd /c native_touch\build.cmd
    if ($LASTEXITCODE -ne 0) { throw 'Native input build failed.' }
    & dotnet publish winui/Ninja3Assistant.csproj -c Release -p:Platform=x64 -r win-x64 --self-contained true `
        -p:PublishReadyToRun=false "-p:Ninja3WorkerPath=$projectRoot/build/worker-dist/Ninja3Worker/Ninja3Worker.exe" `
        "-p:NinjaTouchPath=$projectRoot/build/ninja-touch/NinjaTouch.exe" -o $releaseRoot
    if ($LASTEXITCODE -ne 0) { throw 'UI publish failed.' }
    foreach ($name in @('intelligence-simulation','weekend-race','hayabusa-trials','boss-trial-recordings','daily-cleanup','xumi')) {
        $runtimePath = Join-Path $projectRoot "data/runtime/$name"
        $linkPath = Join-Path $releaseRoot $name
        if (!(Test-Path -LiteralPath $runtimePath)) { New-Item -ItemType Directory -Path $runtimePath -Force | Out-Null }
        if (!(Test-Path -LiteralPath $linkPath)) { New-Item -ItemType Junction -Path $linkPath -Target $runtimePath | Out-Null }
    }
    @'
from pathlib import Path
import hashlib,json,sys
root=Path(sys.argv[1]); release=root/'dist/Ninja3Assistant'
names=('Ninja3Assistant.exe','Ninja3Assistant.dll','Ninja3Worker.exe','NinjaTouch.exe','MainWindow.xbf','Ninja3Assistant.pri','Microsoft.Windows.AI.MachineLearning.dll')
components=[*(release/n for n in names), *sorted(p for p in (release/'_internal').rglob('*') if p.is_file())]
sources=[*root.joinpath('src').rglob('*.py'),*root.joinpath('packaging').glob('*.spec'),root/'pyproject.toml',root/'requirements.txt',root/'native_touch/main.cpp',root/'winui/MainWindow.xaml',*root.joinpath('winui').glob('*.cs'),root/'winui/Ninja3Assistant.csproj',*root.joinpath('winui/Assets').glob('Ninja3Icon.*')]
manifest={'release':'dist/Ninja3Assistant','components':{p.relative_to(release).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in components},
          'sources':{p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}
(release/'build-manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
'@ | & $Python - $projectRoot
    if ($LASTEXITCODE -ne 0) { throw 'Manifest creation failed.' }
    Write-Host "Ready: $projectRoot/启动助手.cmd"
} finally { Pop-Location }
