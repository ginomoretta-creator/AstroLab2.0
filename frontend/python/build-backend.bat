@echo off
REM Build script for bundling Python backend with PyInstaller
REM Run this before building the Electron app for distribution

echo === ASL-Sandbox Python Backend Bundler ===
echo.

REM Check if PyInstaller is installed
pip show pyinstaller >nul 2>&1
if errorlevel 1 (
    echo Installing PyInstaller...
    pip install pyinstaller
)

REM Navigate to the desktop-app/python directory
cd /d "%~dp0"

REM Run PyInstaller
echo Building Python backend...
pyinstaller backend.spec --noconfirm --clean

REM Copy the output to the Electron app resources
echo Copying to Electron resources...
if not exist "..\resources\backend" mkdir "..\resources\backend"
xcopy /E /Y "dist\asl-sandbox-backend*" "..\resources\backend\"

echo.
echo === Build Complete ===
echo Backend executable is in: resources\backend\
pause
