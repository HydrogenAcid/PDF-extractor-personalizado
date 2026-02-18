Extractor de PDF version 1.0
Instrucciones para ejecutar el extractor de PDF
=============================================

Requisitos previos
- Tener instalado Python 3.8 o superior.

Pasos (desde la terminal)
1. Abrir una terminal y situarse en la carpeta del proyecto:

   cd "PDF extractor personalizado"

2. Crear un entorno virtual (recomendado):

   python -m venv .venv

3. Activar el entorno virtual:

   - PowerShell:

     .venv\Scripts\Activate.ps1

   - CMD (símbolo del sistema):

     .venv\Scripts\activate.bat

   - Git Bash / WSL / bash:

     source .venv/bin/activate

4. Actualizar pip (opcional pero recomendable):

   python -m pip install --upgrade pip

5. Instalar dependencias desde requirements.txt:

   pip install -r requirements.txt

6. Ejecutar el programa:

   python PDF-CustomE.py

#

7. Una vez ejecutado, te aparecera un servidor local para visualizarlo en el navegador
para entrar pulsa:
ctrl+click

8. Para salir pulsar:
ctrl+c
Solución de problemas comunes
- Si PowerShell bloquea la activación: ejecutar PowerShell como administrador y, si es necesario, permitir scripts:

  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

- Si falta algún paquete después de la instalación: comprobar la ruta y volver a ejecutar pip install -r requirements.txt con el entorno activado.

Notas
- Si prefieres no usar un entorno virtual, puedes instalar las dependencias globalmente, pero no lo recomiendo.
-Todavia se le pueden hacer varias mejoras, el OCR NO sirve aun, espera a la nueva versión
-Se añade un pdf de ejemplo c:
Fin.
By: HydrogenAcid

