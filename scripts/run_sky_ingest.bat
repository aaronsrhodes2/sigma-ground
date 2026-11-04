@echo off
REM Whole-sky daily ingest (scripts/mpc_sky_ingest.py): wrapper for the Windows task 'sigma-ground sky ingest' (daily 06:00, after MPCORB.DAT lands at 13:27 UT).
REM Orbits to parquet, starts frozen, then the priority set's observation files fetched one request per 5 s. Logs to misc\sky_ingest_job.log.

set PYTHON_EXE=C:\Users\aaron\AppData\Local\Programs\Python\Python313\python.exe
set REPO=D:\Aaron\development\sigma-ground
set LOG=%REPO%\misc\sky_ingest_job.log
set PYTHONUTF8=1

cd /d "%REPO%"
echo. >> "%LOG%"
echo ============================================================ >> "%LOG%"
echo sky ingest started %DATE% %TIME% >> "%LOG%"
echo ============================================================ >> "%LOG%"

"%PYTHON_EXE%" -X utf8 scripts\mpc_sky_ingest.py --queue-night --fetch-obs >> "%LOG%" 2>&1

echo sky ingest finished %DATE% %TIME% exit %ERRORLEVEL% >> "%LOG%"
