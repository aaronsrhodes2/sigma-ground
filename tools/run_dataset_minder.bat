@echo off
REM Dataset Minder daily job — wrapper for Windows Task Scheduler.
REM Logs to misc\dataset_minder.log. Mirrors sigma_ground\mcp\benchmark\run_daily.bat.
REM Edit PYTHON_EXE below if your install path differs.

set PYTHON_EXE=C:\Users\aaron\AppData\Local\Programs\Python\Python313\python.exe
set REPO=D:\Aaron\development\sigma-ground
set LOG=%REPO%\misc\dataset_minder.log

cd /d "%REPO%"
echo. >> "%LOG%"
echo ============================================================ >> "%LOG%"
echo Dataset Minder started %DATE% %TIME% >> "%LOG%"
echo ============================================================ >> "%LOG%"

"%PYTHON_EXE%" tools\dataset_minder.py --run >> "%LOG%" 2>&1

echo. >> "%LOG%"
echo Dataset Minder finished %DATE% %TIME% >> "%LOG%"
