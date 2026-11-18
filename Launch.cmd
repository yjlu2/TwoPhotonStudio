@echo off
setlocal
set PYTHONNOUSERSITE=1
set PYTHONUTF8=1
set PYTHONPATH=
cd /d "%~dp0"
if not exist ".local\launch.cmd" (
  echo Installation is not complete. See README_zh.md.
  pause
  exit /b 1
)
call ".local\launch.cmd"
if errorlevel 1 pause

