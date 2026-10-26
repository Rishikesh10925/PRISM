@echo off
rem Double-click to start the mask review. Ratings save automatically; close this window when done.
cd /d "%~dp0"
start "" http://localhost:8765
"..\..\.venv\Scripts\python.exe" mask_review_server.py
