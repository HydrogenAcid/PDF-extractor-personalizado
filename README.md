# PDF Extractor Personalizado

Aplicacion web en Flask para analizar PDFs desde varias perspectivas:

- ley de Zipf y entropia de Shannon
- distribucion de pares de vocales
- grafos de coocurrencia de texto
- redes small-world de Watts-Strogatz

La interfaz usa un menu lateral para cambiar entre ventanas, pero todo corre dentro del mismo servidor Flask.

## Cambios de hoy

Hoy se agrego un nuevo modulo de redes small-world basado en el paper de Watts y Strogatz (1998).

Se incorporaron estos cambios:

- nuevo archivo `small_world.py`
- nueva vista `templates/small_world.html`
- integracion de la ruta `/small_world` en `PDF-CustomE.py`
- enlace nuevo en el menu lateral de todas las ventanas
- estilos adicionales en `static/style.css`
- tarjeta informativa en la ventana principal para entrar al modulo small-world

El nuevo modulo permite:

- demostrar las relaciones teoricas de `Average Shortest Path` y `Average Clustering Coefficient`
- comparar red regular `p = 0` contra red aleatoria `p = 1`
- reproducir la figura 2 del paper con parametros por defecto `n = 1000`, `k = 10`, `20` realizaciones
- mostrar tablas con teoria, simulacion y error relativo

## Resumen de la aplicacion

La app recibe un PDF, intenta extraer texto con PyMuPDF y, si el texto es insuficiente, usa OCR con Tesseract.

Con ese texto base, cada modulo aplica un analisis distinto:

- `Zipf + Shannon`: frecuencia de palabras y distribucion de longitudes
- `Vocales`: distancias entre pares de vocales y su CDF
- `Grafos`: red de coocurrencia por pagina con metricas estructurales
- `Small-World`: simulacion de redes Watts-Strogatz

## Estructura del proyecto

```text
PDF-CustomE.py
graph_text.py
small_world.py
vowels.py
requirements.txt
.gitignore
static/
    style.css
templates/
    index.html
    vocales.html
    grafo_texto.html
    small_world.html
```

## Script principal

### `PDF-CustomE.py`

Es el punto de entrada de la aplicacion.

Responsabilidades:

- crea la app Flask
- registra las rutas de `vowels.py`
- registra las rutas de `graph_text.py`
- registra las rutas de `small_world.py`
- implementa la ventana principal `Zipf + Shannon`
- hace la extraccion automatica de texto para ese modulo
- calcula metricas basicas del corpus

Flujo del modulo principal:

1. Se recibe un PDF en `/process`.
2. Se intenta extraer texto nativo con PyMuPDF.
3. Si el texto es insuficiente, se usa OCR con Tesseract.
4. Se tokeniza el texto.
5. Se construye la distribucion rango-frecuencia.
6. Se estima el mejor intervalo para el ajuste de Zipf.
7. Se calcula la distribucion de longitudes y la entropia de Shannon.
8. Se devuelve JSON a la interfaz para dibujar graficas y tablas.

## Modulos auxiliares

### `vowels.py`

Implementa el analisis de pares de vocales.

Responsabilidades:

- extraer texto con PyMuPDF u OCR
- normalizar texto segun idioma
- convertir chino a pinyin cuando aplica
- localizar posiciones de vocales
- calcular distancias entre pares de vocales
- construir una CDF por cada par
- exponer las rutas `/vocales` y `/process_vowels`

Idiomas contemplados:

- espanol
- ingles
- frances
- aleman
- mandarin en pinyin

Salida principal:

- una serie CDF por cada par de vocales configurado
- metadatos del archivo y de la configuracion elegida

### `graph_text.py`

Implementa el analisis de grafos de coocurrencia de palabras.

Responsabilidades:

- extraer texto con PyMuPDF u OCR
- tokenizar por idioma
- eliminar stopwords
- construir un grafo no dirigido por pagina
- calcular metricas de red
- generar una distribucion de grados
- devolver un subgrafo compacto para visualizacion
- exponer las rutas `/grafo_texto` y `/process_graph_text`

Metricas calculadas:

- numero de nodos
- numero de aristas
- grado medio
- densidad
- numero de componentes conexas
- tamano de la componente gigante
- assortativity
- clustering promedio

### `small_world.py`

Implementa el modulo de redes small-world.

Responsabilidades:

- validar los parametros `n`, `k`, `realizations` y `log_points`
- generar grafos Watts-Strogatz conectados
- calcular `Average Shortest Path`
- calcular `Average Clustering Coefficient`
- comparar teoria contra simulacion
- construir los datos necesarios para la figura 2
- exponer las rutas `/small_world` y `/process_small_world`

Conceptos clave:

- `n`: numero de nodos de la red
- `k`: numero de vecinos iniciales por nodo en la red regular
- `p`: probabilidad de reconexion de una arista

Relaciones teoricas implementadas:

- red regular `p = 0`
  - `L(0) ~ n / (2k)`
  - `C(0) = 3(k-2) / (4(k-1)) ~ 3/4`
- red aleatoria `p = 1`
  - `L(1) ~ ln(n) / ln(k)`
  - `C(1) ~ k / n`

Nota de implementacion:

Para mantener la simulacion interactiva, el `Average Shortest Path` se estima por muestreo BFS cuando la red es grande. Esto reduce el tiempo de espera sin perder la forma general de la figura 2.

## Ventanas de la aplicacion

### 1. Ventana `Zipf + Shannon`

Archivo asociado:

- `templates/index.html`

Que muestra:

- formulario para subir un PDF
- grafica log-log de rango vs frecuencia
- tabla con paginas, tokens, vocabulario, `R^2` y pendiente Zipf
- grafica de longitud de palabra vs frecuencia
- tabla con entropia de Shannon
- acceso rapido al modulo small-world

Uso esperado:

- subir varios PDFs uno por uno
- comparar sus curvas y tablas en la misma interfaz

### 2. Ventana `Vocales`

Archivo asociado:

- `templates/vocales.html`

Que muestra:

- selector de idioma
- carga de PDF
- una o varias graficas CDF de distancias entre pares de vocales
- una tabla con idioma, paginas, `max_chars` y `max_dist`

Uso esperado:

- comparar estructura vocalica entre idiomas o entre distintos textos

### 3. Ventana `Grafos`

Archivo asociado:

- `templates/grafo_texto.html`

Que muestra:

- carga de PDF
- parametros `span`, `minFreq` y `maxVocab`
- grafica de distribucion de grado
- tabla con metricas del grafo
- visualizacion interactiva del subgrafo con vis-network

Uso esperado:

- estudiar relaciones locales entre palabras frecuentes
- observar densidad, clustering y estructura de conexiones

### 4. Ventana `Small-World`

Archivo asociado:

- `templates/small_world.html`

Que muestra:

- parametros `n`, `k`, numero de realizaciones y numero de puntos de la figura
- formulas teoricas visibles en la cabecera
- tarjetas resumen con `L(0)`, `C(0)` y zona small-world
- grafica de la figura 2 con `L(p)/L(0)` y `C(p)/C(0)`
- tabla comparativa entre teoria y simulacion
- tabla con datos numericos para cada valor de `p`

Uso esperado:

- exponer el modelo de Watts-Strogatz
- demostrar el efecto small-world
- justificar matematicamente los extremos `p = 0` y `p = 1`

## Plantillas y frontend

### `templates/index.html`

Vista principal del modulo Zipf + Shannon. Usa Chart.js para mostrar:

- grafica de Zipf
- grafica de longitudes
- tablas comparativas

### `templates/vocales.html`

Renderiza varias graficas CDF, una por cada par de vocales. Crea dinamicamente los charts necesarios.

### `templates/grafo_texto.html`

Combina Chart.js para la distribucion de grado y vis-network para el subgrafo interactivo.

### `templates/small_world.html`

Renderiza la simulacion de Watts-Strogatz con:

- formulario de parametros
- grafica principal de la figura 2
- tabla de relaciones teoricas
- tabla de valores simulados

### `static/style.css`

Hoja de estilos compartida por todas las ventanas.

Incluye:

- layout general con sidebar
- tarjetas y tablas reutilizables
- estilos de formularios
- estilos nuevos para la vista small-world, como `heroCard`, `statsGrid` y `featureCallout`

## Dependencias

El archivo `requirements.txt` declara:

- `flask`
- `pdfplumber`
- `pymupdf`
- `pytesseract`
- `pillow`
- `networkx`
- `matplotlib`
- `pypinyin`

Observacion:

- `pdfplumber` y `matplotlib` no son el centro del flujo actual, pero siguen en el entorno declarado
- `networkx` aparece repetido en `requirements.txt`; no rompe la instalacion, pero puede limpiarse despues

## Flujo general de uso

1. Levantar el servidor Flask.
2. Abrir `http://127.0.0.1:5000/`.
3. Elegir la ventana desde el menu lateral.
4. Cargar un PDF o ejecutar una simulacion.
5. Revisar graficas, tablas y metricas.
6. Resetear si se desea iniciar una comparacion limpia.

## Instalacion

### 1. Clonar el repositorio

```bash
git clone https://github.com/HydrogenAcid/PDF-extractor-personalizado.git
cd PDF-extractor-personalizado
```

### 2. Crear un entorno virtual

```bash
python -m venv .venv
```

### 3. Activar el entorno virtual

PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Git Bash:

```bash
source .venv/Scripts/activate
```

### 4. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 5. Ejecutar el servidor

```bash
python PDF-CustomE.py
```

### 6. Abrir en el navegador

```text
http://127.0.0.1:5000/
```

## Archivos ignorados por git

`.gitignore` excluye:

- entorno virtual `.venv/`
- caches de Python
- archivos temporales `.pyc`
- logs
- archivos PDF

Esto evita subir archivos grandes, resultados temporales y dependencias locales.

## Estado actual

La aplicacion ya integra cuatro modulos en una sola interfaz:

- analisis estadistico de texto
- analisis vocalico
- analisis de grafos de texto
- simulacion small-world

Es un entorno de trabajo util para cursos de:

- teoria de redes
- analisis de lenguaje
- estadistica de texto
- complejidad y sistemas dinamicos
