@echo off
setlocal
pushd "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo JeePeeTee virtual environment was not found.
    echo Expected: %~dp0.venv\Scripts\python.exe
    popd
    exit /b 1
)

".venv\Scripts\python.exe" -m mini_llm.interactive %*
set "JEEPEETEE_EXIT=%ERRORLEVEL%"
popd
exit /b %JEEPEETEE_EXIT%
