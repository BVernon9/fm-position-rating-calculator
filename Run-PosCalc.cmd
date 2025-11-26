@echo off
setlocal
set "SCRIPT=%~dp0PosCalc.py"

REM Use venv Python if present; else fall back to system "python"
if exist "%~dp0venv\Scripts\python.exe" (
  set "PY=%~dp0venv\Scripts\python.exe"
) else (
  set "PY=python"
)

REM No files dropped? Open file picker (handled by PosCalc.py)
if "%~1"=="" (
  "%PY%" "%SCRIPT%"
  goto :eof
)

REM One or more files dropped: run each
for %%F in (%*) do (
  "%PY%" "%SCRIPT%" "%%~fF"
)
pause
