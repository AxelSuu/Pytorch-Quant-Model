@echo off
REM PyStock CLI wrapper for Windows Command Prompt
REM This allows you to run: pystock.bat train --symbol AAPL

"%~dp0.venv\Scripts\python.exe" "%~dp0pystock.py" %*
