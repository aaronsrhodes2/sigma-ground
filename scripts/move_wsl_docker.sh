#!/bin/bash
# PLATINUM 22: move Ubuntu-24.04 and the Docker Desktop disk images from C: to D:. Logs to D:/tmp/logs/move_wsl_docker.log
LOG=/d/tmp/logs/move_wsl_docker.log
exec > >(tee -a "$LOG") 2>&1
say() { echo "[$(date +%H:%M:%S)] $*"; }

say "stopping Docker Desktop"
for p in "Docker Desktop.exe" com.docker.backend.exe com.docker.build.exe docker-agent.exe docker-mcp.exe docker-sandbox.exe com.docker.proxy.exe; do
  taskkill //F //IM "$p" >/dev/null 2>&1
done
sleep 5
say "wsl --shutdown"
wsl.exe --shutdown
sleep 8
wsl.exe -l -v | tr -d '\0'

say "moving Ubuntu-24.04 to D:\\wsl\\Ubuntu-24.04"
mkdir -p /d/wsl/Ubuntu-24.04
wsl.exe --manage Ubuntu-24.04 --move 'D:\wsl\Ubuntu-24.04' 2>&1 | tr -d '\0'
echo "ubuntu move exit ${PIPESTATUS[0]}"
ls -l /d/wsl/Ubuntu-24.04

say "copying Docker disk images to D:\\docker\\wsl"
mkdir -p /d/docker/wsl
MSYS_NO_PATHCONV=1 robocopy 'C:\Users\aaron\AppData\Local\Docker\wsl' 'D:\docker\wsl' /E /COPY:DAT /R:2 /W:2 /NFL /NDL /NP
echo "docker copy exit $?"
say "done: C: Docker copy is still in place; settings not yet changed"
