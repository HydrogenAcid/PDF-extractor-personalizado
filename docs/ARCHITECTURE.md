# Arquitectura del proyecto

El proyecto es una aplicacion Flask.

- `main.py`: punto de entrada principal y registro de rutas.
- `templates/`: paginas HTML de cada modulo.
- `templates/partials/`: fragmentos reutilizables de interfaz.
- `static/`: estilos CSS compartidos.
- `assets/`: recursos visuales servidos por `/assets/<archivo>`.
- `src/ui/`: utilidades de interfaz, como la ruta de assets de bienvenida.
- `src/utils/`: helpers compartidos.
- `src/pdf/`: espacio preparado para utilidades de reportes o salidas PDF.
- `output/`: resultados generados localmente.

Los modulos de analisis (`hurst.py`, `small_world.py`, `fractals.py`, etc.) se mantienen en la raiz para no romper imports ni rutas existentes.
