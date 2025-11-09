@echo off
REM THRML MNIST Diffusion Demo - Quick Setup Script for Windows

echo ==========================================
echo THRML MNIST Diffusion Demo Setup
echo ==========================================
echo.

REM Check Python
python --version
if errorlevel 1 (
    echo Python is not installed or not in PATH!
    exit /b 1
)

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo.
    echo Creating virtual environment...
    python -m venv venv
) else (
    echo.
    echo Virtual environment already exists.
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo.
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install THRML
echo.
echo Installing THRML...
pip install -e .

REM Install demo requirements
echo.
echo Installing demo requirements...
pip install -r demo_requirements.txt

echo.
echo ==========================================
echo Setup Complete!
echo ==========================================
echo.
echo To run the demo:
echo   1. Activate the virtual environment:
echo        venv\Scripts\activate.bat
echo.
echo   2. Run the demo:
echo        python demo_mnist_diffusion.py --digit 1 --epochs 3
echo.
echo   Or for a quick test:
echo        python demo_mnist_diffusion.py --digit 1 --epochs 1
echo.
echo ==========================================

pause
