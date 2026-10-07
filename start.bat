@echo off
chcp 65001 >nul
set PYTHONUTF8=1
cd /d "%~dp0"
set PY=%LOCALAPPDATA%\stock-venv\Scripts\python.exe
if not exist "%PY%" (
  echo 建立 Python 環境中 ^(放在 C 槽^)...
  py -3.12 -m venv "%LOCALAPPDATA%\stock-venv"
  "%PY%" -m pip install pandas numpy matplotlib yfinance fastapi uvicorn requests playwright
)
start "" http://127.0.0.1:8000
"%PY%" app.py
pause
