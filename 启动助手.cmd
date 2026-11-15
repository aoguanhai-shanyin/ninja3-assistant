@echo off
setlocal
set "APP=%~dp0dist\Ninja3Assistant\Ninja3Assistant.exe"
if not exist "%APP%" (
  echo Ninja3Assistant is missing. See README.md for build instructions.
  pause
  exit /b 1
)
start "" /D "%~dp0dist\Ninja3Assistant" "%APP%"
