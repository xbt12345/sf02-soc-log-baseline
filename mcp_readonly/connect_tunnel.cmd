@echo off
powershell.exe -NoProfile -File "%~dp0connect_tunnel.ps1"
if errorlevel 1 pause
