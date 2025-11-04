# PyStock CLI wrapper for PowerShell
# This allows you to run: .\pystock.ps1 train --symbol AAPL

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
& "$scriptDir\.venv\Scripts\python.exe" "$scriptDir\pystock.py" $args
