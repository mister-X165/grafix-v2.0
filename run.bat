@echo off
rem Запуск медицинского фактчекера на Windows (ТЗ §94).
rem Требует Python 3.10+ в PATH. GUI открывается окном PySide6.
cd /d "%~dp0"
where py >nul 2>nul
if %errorlevel%==0 (
    py -3 run.py
) else (
    python run.py
)
if errorlevel 1 pause
