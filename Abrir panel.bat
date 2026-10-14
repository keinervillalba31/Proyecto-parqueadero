@echo off
rem Doble clic: enciende el servicio (si no lo estaba) y abre el panel en el navegador.
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (
    echo No se encontro el entorno .venv. Sigue la seccion "Instalacion" del README.
    pause
    exit /b 1
)
start "" ".venv\Scripts\pythonw.exe" run_service.py --panel
