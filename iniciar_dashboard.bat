@echo off
rem Abre LABOR INSIGHTS en el navegador: http://127.0.0.1:8050
rem Requiere el entorno .venv de esta carpeta (ver README.md).
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Falta el entorno .venv. Siga los pasos de README.md.
  pause
  exit /b 1
)
start "" http://127.0.0.1:8050
".venv\Scripts\python.exe" app.py
