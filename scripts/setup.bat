@echo off
rem =====================================================
rem  video-enhance setup (Windows)
rem  Creates venv, installs requirements, downloads models
rem =====================================================
chcp 65001 >nul
setlocal
cd /d "%~dp0.."

echo [1/4] Checking Python...
where python >nul 2>nul || (echo ERROR: Python nahi mila - https://python.org se install karo & pause & exit /b 1)

echo [2/4] Creating venv...
if not exist ".venv" python -m venv .venv
call ".venv\Scripts\activate.bat"

echo [3/4] Installing requirements...
python -m pip install --upgrade pip
pip install -r requirements.txt

echo [4/4] Downloading AI upscale models...
python scripts\download_models.py

echo.
echo =====================================================
echo DONE! Ab chalao:
echo   python scripts\enhance.py video.mp4 fastai 25
echo   python scripts\enhance_gui.py
echo =====================================================
pause
