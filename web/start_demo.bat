@echo off
rem Starts the OMNI demo from this folder with no internet needed, then opens it.
rem Keep this window open while presenting; close it to stop the demo.
cd /d "%~dp0"
start "" http://localhost:8080/index.html
python -m http.server 8080
