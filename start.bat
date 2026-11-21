@echo off
setlocal EnableExtensions
chcp 65001 >nul 2>nul
set "PYTHONUTF8=1"
set "APP_DIR=%~dp0"

set "PYTHON_CMD="

python -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)" >nul 2>nul
if not errorlevel 1 (
    set "PYTHON_CMD=python"
    goto :found_python
)

py -3 -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)" >nul 2>nul
if not errorlevel 1 (
    set "PYTHON_CMD=py -3"
    goto :found_python
)

echo [ERROR] Python 3.11 or newer was not found on PATH. 1>&2
echo Please install Python 3.11+ and ensure 'python' or 'py' is available in PATH. 1>&2
exit /b 1

:found_python
%PYTHON_CMD% "%APP_DIR%app.py" %*
set "EXIT_CODE=%ERRORLEVEL%"
exit /b %EXIT_CODE%
