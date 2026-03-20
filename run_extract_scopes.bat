@echo off
title PDF Scope Extractor
echo ================================================
echo   PDF Scope Description Extractor
echo ================================================
echo.

REM ── Set your Anthropic API key here if not already in your environment ──
REM SET ANTHROPIC_API_KEY=sk-ant-...

REM Check if files were dragged onto this script
IF "%~1"=="" (
    echo No PDF files dragged onto this script.
    echo Searching for all PDFs in this folder instead...
    echo.
    python "%~dp0extract_scopes.py" --dir "%~dp0" --output "%~dp0scope_descriptions.html"
) ELSE (
    echo Processing dragged files...
    echo.
    python "%~dp0extract_scopes.py" %* --output "%~dp0scope_descriptions.html"
)

echo.
IF %ERRORLEVEL%==0 (
    echo Done! Output saved to: scope_descriptions.html
    echo Opening result in browser...
    start "" "%~dp0scope_descriptions.html"
) ELSE (
    echo Something went wrong. See error above.
)

echo.
pause
