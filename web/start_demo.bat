@echo off
rem Starts NEXA with its local database: no internet needed, nothing leaves
rem this machine. The browser opens by itself once the server is up.
rem Keep this window open while presenting; close it (or Ctrl+C) to stop.
rem
rem Analyses, uploaded raw IQ and the audit trail are kept in data\local\.
cd /d "%~dp0.."
set "PY=python"
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"
"%PY%" scripts\serve_local.py %*
if errorlevel 1 pause
