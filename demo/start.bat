@echo off
rem CDU demo launcher (Windows). English only - console codepage varies.
rem No absolute paths: everything is relative to this file's folder.
cd /d "%~dp0"

set PY=python
where py >nul 2>nul && set PY=py -3

%PY% -c "import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)" 2>nul
if errorlevel 1 (
  echo Python 3.12 or newer is required. Install it, then run this file again.
  pause
  exit /b 1
)

echo Serving this folder on http://localhost:8000/ - press Ctrl+C to stop.
start "" http://localhost:8000/pfd.html
%PY% -m http.server 8000
