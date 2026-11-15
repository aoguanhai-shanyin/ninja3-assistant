# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all
from pathlib import Path

ROOT = Path(SPECPATH).resolve().parent

datas = [(str(ROOT/'assets'), 'assets')]
binaries = []
hiddenimports = []
for package in ('onnxruntime', 'cv2'):
    package_data, package_binaries, package_imports = collect_all(package)
    datas += package_data
    binaries += package_binaries
    hiddenimports += package_imports

a = Analysis([str(ROOT/'src/ninja3_assistant/entrypoints/worker.py')], pathex=[str(ROOT/'src')], binaries=binaries, datas=datas, hiddenimports=hiddenimports)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, a.binaries, a.datas, [], name='Ninja3Worker',
          console=True, upx=False, disable_windowed_traceback=False)
