# Extractor de PDFs versión 2.0  
Análisis estadístico de texto: Ley de Zipf y Entropía de Shannon

---

## Descripción

Extractor de PDFs enfocado en análisis cuantitativo de texto.  
Permite subir múltiples libros en formato PDF y obtener, para cada uno:

- Distribución Rango vs Frecuencia (Ley de Zipf)
- Pendiente en plano log-log
- Distribución Longitud de palabra vs Frecuencia
- Entropía de Shannon sobre la distribución de longitudes
- Métricas básicas del corpus

Cada PDF se agrega dinámicamente a las gráficas con un color distinto y se muestran tablas comparativas.

---

## Funcionalidades

### 1. Extracción automática de texto

- Intenta primero extraer texto nativo usando PyMuPDF.
- Si el texto es insuficiente, realiza OCR automático con Tesseract.
- No requiere selección manual de columnas ni activación de OCR.

---

### 2. Ley de Zipf

Para cada libro:

1. Se tokeniza el texto.
2. Se cuentan frecuencias.
3. Se ordenan por frecuencia descendente.
4. Se asigna rango.
5. Se grafica en plano log-log.
6. Se calcula la pendiente usando regresión lineal sobre:

\[
\log(\text{rank}) \quad \text{vs} \quad \log(\text{frequency})
\]

Rango de ajuste actual: 1 a 300.

Se muestra en tabla:

- Páginas procesadas
- Tokens
- Vocabulario
- Pendiente Zipf

---

### 3. Distribución de longitudes

Para cada libro:

- Se agrupan palabras por longitud (1 a 23 caracteres).
- Se suma la frecuencia total por longitud.
- Se grafica longitud vs frecuencia.
- Se calcula la entropía de Shannon:

$\[
H = - \sum_i P(x_i)\log P(x_i)
\]
$
donde:

\[
P(x_i) = \frac{f(x_i)}{\sum_j f(x_j)}
\]

La entropía se reporta en nats.

---

### 4. Comparación múltiple

- Se pueden subir múltiples PDFs consecutivamente.
- Cada uno aparece como un nuevo dataset en ambas gráficas.
- Se puede resetear el análisis sin reiniciar el servidor.

---

## Métricas calculadas

- Tokens: total de palabras (con repetición).
- Vocab: número de palabras únicas.
- Zipf slope.
- Entropía de Shannon (longitudes).

---

## Limitaciones actuales

1. OCR en despliegue
   - Render no incluye Tesseract por defecto.
   - OCR requiere instalación del binario o Docker.
   - En Render puede funcionar solo extracción de texto nativo.

2. Tokenización simple
   - Usa regex básica.
   - No hay lematización.
   - No se eliminan stopwords.
   - No se separan contracciones complejas.

3. Longitud máxima fija
   - Solo se consideran palabras de 1 a 23 caracteres.
   - Palabras más largas se descartan.

4. Ajuste Zipf fijo
   - Pendiente calculada sobre ranks 1–300.
   - No se calcula R² aún.

5. Rendimiento
   - PDFs muy grandes pueden tardar.
   - OCR incrementa significativamente el tiempo.

6. Memoria
   - Todo el texto se procesa en memoria.
   - No hay procesamiento por streaming.

---

## Estructura del proyecto

```
.
├── PDF-CustomE.py
├── requirements.txt
├── Procfile
├── static/
│   └── style.css
├── templates/
│   └── index.html
└── README.md
```

---

## Instalación y ejecución local

### 1. Clonar repositorio

```bash
git clone <URL_DEL_REPO>
cd <NOMBRE_DEL_PROYECTO>
```

### 2. Crear entorno virtual

```bash
python -m venv .venv
```

Activar:

Git Bash:
```bash
source .venv/Scripts/activate
```

PowerShell:
```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. (Opcional) Instalar Tesseract para OCR

Descargar e instalar:
https://github.com/tesseract-ocr/tesseract

Asegurarse de que esté en el PATH.

### 5. Ejecutar servidor

```bash
python PDF-CustomE.py
```

Abrir en navegador:

```
http://127.0.0.1:5000
```




---

## Próximas mejoras posibles

- Cálculo de R² para Zipf.
- Exportación CSV de métricas.
- Eliminación de stopwords.
- Lematización.
- Selección dinámica de rango de ajuste.
- Conversión de entropía a bits.
- Dockerización para OCR en producción.

---

## Versión

Extractor de PDFs 2.0  
Análisis estadístico comparativo multi-documento.

