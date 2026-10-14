@echo off
title Omok++ AI Arena - AlphaZero 2x Champion
:: Switch working drive and directory to the exact directory where this batch file lives
cd /d "%~dp0"

echo ======================================================================
echo   Omok++ AI Arena (AlphaZero 2x 10-ResBlock Champion)
echo   Project Location: %~dp0
echo   Working Directory: %CD%
echo ======================================================================
echo.
echo Launching Omok++ Arena GUI...

python "%~dp0play.py"

if errorlevel 1 (
    echo.
    echo [ERROR] Failed to run GUI. Please make sure Python and PyTorch are installed in PATH.
    pause
)
