@echo off
chcp 65001 >nul
set PYTHONUTF8=1
cd /d "%~dp0"
set PY=%LOCALAPPDATA%\stock-venv\Scripts\python.exe
set URL=http://127.0.0.1:8000

rem --- already running? just open the browser ---
powershell -NoProfile -Command "try{Invoke-WebRequest -UseBasicParsing %URL% -TimeoutSec 2 | Out-Null; exit 0}catch{exit 1}"
if %errorlevel%==0 (
  echo Server is already running. Opening browser...
  start msedge %URL%
  exit /b
)

rem --- first run: create Python environment on C: ---
if not exist "%PY%" (
  echo Creating Python environment, please wait...
  py -3.12 -m venv "%LOCALAPPDATA%\stock-venv"
  "%PY%" -m pip install pandas numpy matplotlib yfinance fastapi uvicorn requests playwright
)

rem --- open browser only after the server is ready (up to 90 s) ---
start "" /b powershell -NoProfile -WindowStyle Hidden -Command "for($i=0;$i -lt 90;$i++){try{Invoke-WebRequest -UseBasicParsing %URL% -TimeoutSec 2 | Out-Null; Start-Process msedge '%URL%'; break}catch{Start-Sleep 1}}"

echo Starting server... (keep this window open; closing it stops the system)
"%PY%" app.py
echo.
echo Server stopped. If you see an error above, send a screenshot to Claude.
pause
