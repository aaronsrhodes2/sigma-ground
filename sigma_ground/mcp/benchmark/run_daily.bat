@echo off
REM Daily benchmark job — wrapper for Windows Task Scheduler.
REM Logs to misc\daily_job.log so failures are visible in the morning.
REM Edit PYTHON_EXE or LOG_DIR below if your install path differs.

set PYTHON_EXE=C:\Users\aaron\AppData\Local\Programs\Python\Python313\python.exe
set REPO=D:\Aaron\development\sigma-ground
set LOG=%REPO%\misc\daily_job.log

cd /d "%REPO%"
echo. >> "%LOG%"
echo ============================================================ >> "%LOG%"
echo Daily benchmark job started %DATE% %TIME% >> "%LOG%"
echo ============================================================ >> "%LOG%"

"%PYTHON_EXE%" -m sigma_ground.mcp.benchmark.daily_job >> "%LOG%" 2>&1

echo. >> "%LOG%"
echo Daily benchmark job finished %DATE% %TIME% >> "%LOG%"
