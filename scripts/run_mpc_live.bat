@echo off
REM Daily MPC live scoreboard (misc/CANDIDATE_LEDGER.md C27): wrapper for the Windows task 'sigma-ground mpc live daily'.
REM Same pattern as sigma_ground\mcp\benchmark\run_daily.bat. Logs to misc\mpc_live_job.log; the per-day logs are under D:\datasets\mpc_live\<date>\.

set PYTHON_EXE=C:\Users\aaron\AppData\Local\Programs\Python\Python313\python.exe
set REPO=D:\Aaron\development\sigma-ground
set LOG=%REPO%\misc\mpc_live_job.log
set PYTHONUTF8=1

cd /d "%REPO%"
echo. >> "%LOG%"
echo ============================================================ >> "%LOG%"
echo MPC live scoreboard started %DATE% %TIME% >> "%LOG%"
echo ============================================================ >> "%LOG%"

"%PYTHON_EXE%" -X utf8 scripts\mpc_live_driver.py >> "%LOG%" 2>&1

echo MPC live scoreboard finished %DATE% %TIME% exit %ERRORLEVEL% >> "%LOG%"
