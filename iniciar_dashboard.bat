@echo off
rem Abre el dashboard en el navegador: http://127.0.0.1:8050
rem Requiere el entorno .venv de esta carpeta (ver README.md).
cd /d "%~dp0"
start "" http://127.0.0.1:8050
".venv\Scripts\python.exe" -c "import app; app.app.run(debug=False, port=8050)"
