@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "PS1=%SCRIPT_DIR%run_app.ps1"

if not exist "%PS1%" (
  echo Missing launcher script: "%PS1%"
  exit /b 1
)

powershell -NoProfile -ExecutionPolicy Bypass -File "%PS1%" %*
set "EXIT_CODE=%ERRORLEVEL%"
exit /b %EXIT_CODE%
