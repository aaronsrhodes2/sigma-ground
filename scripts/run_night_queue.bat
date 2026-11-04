@echo off
REM The night window (scripts/night_queue.py): wrapper for the Windows task 'sigma-ground night window' (daily 01:00).
REM Drains the queue at D:\datasets\sigma_runs\night_queue one job at a time until 07:00; logs to misc\night_queue_job.log, per-job logs under the queue's logs\.

set PYTHON_EXE=C:\Users\aaron\AppData\Local\Programs\Python\Python313\python.exe
set REPO=D:\Aaron\development\sigma-ground
set LOG=%REPO%\misc\night_queue_job.log
set PYTHONUTF8=1

cd /d "%REPO%"
echo. >> "%LOG%"
echo ============================================================ >> "%LOG%"
echo night window started %DATE% %TIME% >> "%LOG%"
echo ============================================================ >> "%LOG%"

"%PYTHON_EXE%" -X utf8 scripts\night_queue.py run >> "%LOG%" 2>&1

echo night window finished %DATE% %TIME% exit %ERRORLEVEL% >> "%LOG%"
