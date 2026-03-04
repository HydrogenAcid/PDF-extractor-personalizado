# Extractor de PDFs versión 2.0 - Análisis estadístico de texto: Ley de Zipf y Entropía de Shannon

## Descripción

Extractor de PDFs enfocado en análisis cuantitativo de texto. Permite subir múltiples libros en formato PDF y obtener, para cada uno:

-Distribución Rango vs Frecuencia (Ley de Zipf)
-Pendiente en plano log-log
-Distribución Longitud de palabra vs Frecuencia
-Entropía de Shannon sobre la distribución de longitudes
-Métricas básicas del corpus

Cada PDF se agrega dinámicamente a las gráficas con un color distinto y se muestran tablas comparativas.

## Funcionalidades

### 1. Extracción automática de texto

-Intenta primero extraer texto nativo usando PyMuPDF (ordenando bloques para mejorar lectura en columnas).
-Si el texto es insuficiente, realiza OCR automático con Tesseract.
-No requiere selección manual de columnas ni activación de OCR.

### 2. Ley de Zipf

Para cada libro:
1.Se tokeniza el texto.
2.Se cuentan frecuencias.
3.Se ordenan por frecuencia descendente.
4.Se asigna rango.
5.Se grafica en plano log-log.
6.Se calcula la pendiente mediante regresión lineal en log(rank) vs log(freq).

**Ajuste por intervalo óptimo:**
-La pendiente se calcula en el intervalo con mejor ajuste (máximo $$R^2$$)
-Se evita la cola de frecuencia 1.
-Se ignoran los primeros ranks dominados por stopwords.
-Se descartan ventanas con baja variación.

### 3. Distribución de longitudes

Para cada libro:
-Se agrupan palabras por longitud (1–23 caracteres).
-Se suma la frecuencia total por longitud.
-Se grafica longitud vs frecuencia.
-Se calcula la entropía de Shannon.

$$H = -\sum P(x_i) \log(P(x_i))$$

$$P(x_i) = \frac{f(x_i)}{\sum f(x_i)}$$

La entropía se reporta en nats.

### 4. Comparación múltiple

-Se pueden subir múltiples PDFs consecutivamente.
-Cada uno aparece como un nuevo dataset en ambas gráficas.
-Se puede resetear el análisis sin reiniciar el servidor.

### 5. Tooltips enriquecidos

Al pasar el cursor sobre un punto de Zipf se muestra:
-palabra
-rank
-frecuencia

## Métricas calculadas

-Tokens: total de palabras (con repetición)
-Vocab: número de palabras únicas
-Plot max rank
-Zipf slope
-Zipf R²
-Entropía de Shannon

## Limitaciones actuales

1.OCR en despliegue Render no incluye Tesseract por defecto.
2.Tokenización simple: usa regex básica.
3.Zipf heurístico: el intervalo se selecciona con heurísticas.
4.Rendimiento: no se grafica todo el vocabulario si es muy grande.
5.Memoria: el texto completo se procesa en memoria.

## Estructura del proyecto

```
PDF-CustomE.py
vowels.py
requirements.txt
Procfile
static/
    style.css
templates/
    index.html
    vocales.html
README.md
```

## Instalación

1.Clonar repositorio

```bash
git clone <repositorio>
cd <directorio>
```

2.Crear entorno virtual

```bash
python -m venv .venv
```

3.Activar entorno

**Git Bash:**
```bash
source .venv/Scripts/activate
```

**PowerShell:**
```powershell
..venv\Scripts\Activate.ps1
```

4.Instalar dependencias

```bash
pip install -r requirements.txt
```

5.Ejecutar servidor python PDF-CustomE.py

Abrir navegador http://127.0.0.1:5000

## 6. Análisis de pares de vocales (Vowel n-grams)

Se incorporó una nueva sección para analizar **pares y combinaciones de vocales dentro de palabras** con el objetivo de comparar patrones fonotácticos entre distintos idiomas.

Esta funcionalidad permite estudiar la **estructura vocálica de textos** y comparar distribuciones entre lenguas.

### Características

El sistema analiza automáticamente:

- Frecuencia de pares de vocales
- Frecuencia de combinaciones vocálicas relevantes (diptongos y algunos triptongos)
- Distribución de combinaciones vocálicas en textos largos

El resultado se grafica como:

- **Eje X:** índice del n-grama  
- **Eje Y:** frecuencia del par (o n-grama) de vocales

Cada libro cargado aparece como una nueva serie en la gráfica para facilitar comparaciones entre textos o idiomas.

### Idiomas soportados

El análisis incluye configuraciones específicas por idioma.

#### Español

Pares analizados:

- ae, ai, ao, au  
- ea, ei, eo, eu  
- ia, ie, io, iu  
- oa, oe, oi, ou  
- ua, ue, ui, uo

#### Inglés

Incluye combinaciones donde **y puede funcionar como vocal**:

- ae, ai, ao, au  
- ea, ei, eo, eu  
- ia, ie, io, iu  
- oa, oe, oi, ou  
- ua, ue, ui, uo  
- ay, ey, iy, oy, uy  
- ya, ye, yi, yo, yu

#### Francés

Similar al inglés, considerando **y como vocal** en ciertas posiciones.

#### Alemán

Se incluyen **umlauts** y diptongos característicos del idioma:

- ie, ei, eu, äu, au, öu, üe, üa

#### Mandarín (pinyin)

Se analizan combinaciones propias del sistema fonológico del pinyin:

- ai, ei, ao, ou  
- ia, ie, iao, iu  
- ua, uo, ui  
- üe, üa

### Conversión automática a pinyin

Si el usuario selecciona **mandarín** y el texto contiene **caracteres chinos**, el sistema convierte automáticamente el texto a **pinyin** antes de realizar el análisis.

Esto permite subir directamente:

- Texto en chino (汉字): soportado (se convierte a pinyin)
- Texto en pinyin: soportado
- PDF escaneado: soportado (mediante OCR)

La conversión se realiza utilizando la librería `pypinyin`.

### Métricas adicionales

El módulo calcula métricas adicionales sobre las vocales del corpus:

- **vowel_chars:** número total de vocales individuales encontradas en el texto.
- **ngrams_total:** número total de combinaciones de vocales detectadas.

Estas métricas permiten comparar la **densidad de combinaciones vocálicas** entre idiomas o textos.

### Interfaz

Se agregó una nueva sección accesible desde el menú lateral de la aplicación:

- Zipf + Shannon
- Vocales

Esto permite navegar entre los distintos tipos de análisis disponibles del corpus.

Próximas mejoras

-Exportación CSV
-Eliminación de stopwords
-Lematización
-Selección dinámica del intervalo Zipf
-Entropía en bits
-Docker para OCR

Versión Extractor de PDFs 2.1
