@echo off
rem Double-click to start severity rating. Answers save automatically; close this window when done.
cd /d "%~dp0"
start "" http://localhost:8766
"..\..\.venv\Scripts\python.exe" rating_server.py
