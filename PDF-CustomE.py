# PDF-CustomE.py
from flask import Flask, render_template, request, jsonify
import os, re, math
from collections import Counter

import fitz  # PyMuPDF
import pytesseract
from PIL import Image

app = Flask(__name__)

WORD_RE = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+")

# -------------------- Extracción (texto -> OCR fallback) --------------------

def extract_text_pymupdf_blocks(pdf_path: str):
    pages = []
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            blocks = page.get_text("blocks") or []
            blocks = sorted(blocks, key=lambda b: (b[1], b[0]))  # y luego x
            text = "\n".join((b[4] or "").strip() for b in blocks if (b[4] or "").strip())
            pages.append(text)
        doc.close()
    except Exception as e:
        print(f"Error en PyMuPDF: {e}")
    return pages

def extract_text_ocr_tesseract(pdf_path: str):
    pages = []
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            pix = page.get_pixmap(dpi=200)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            pages.append(pytesseract.image_to_string(img, lang="spa") or "")
        doc.close()
    except Exception as e:
        print(f"Error en OCR Tesseract: {e}")
    return pages

def is_text_usable(pages, min_chars_total=500):
    return len(("".join(pages)).strip()) >= min_chars_total

def extract_text_auto(pdf_path: str):
    pages = extract_text_pymupdf_blocks(pdf_path)
    if not is_text_usable(pages):
        pages = extract_text_ocr_tesseract(pdf_path)
    return pages

# -------------------- Métricas --------------------

def tokenize(text: str):
    return WORD_RE.findall(text.lower())

def rank_frequency(counter: Counter, max_rank: int):
    items = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
    items = items[:max_rank]
    ranks = list(range(1, len(items) + 1))
    words = [w for w, _ in items]
    freqs = [f for _, f in items]
    return ranks, words, freqs

def length_frequency_from_wordfreq(word_freq: Counter, max_len: int = 15):
    lf = Counter()
    for w, f in word_freq.items():
        L = len(w)
        if 1 <= L <= max_len:
            lf[L] += f
    xs = list(range(1, max_len + 1))
    ys = [lf[L] for L in xs]  # rellena 0 si no existe
    return xs, ys

def shannon_entropy_from_counts(counts):
    total = sum(counts)
    if total <= 0:
        return None
    h = 0.0
    for c in counts:
        if c <= 0:
            continue
        p = c / total
        h -= p * math.log(p)  # nats
    return h

def downsample_zipf(ranks, words, freqs, max_points: int):
    n = len(ranks)
    if n <= max_points:
        return ranks, words, freqs
    step = max(1, n // max_points)
    return ranks[::step], words[::step], freqs[::step]

def downsample_xy(xs, ys, max_points: int):
    n = len(xs)
    if n <= max_points:
        return xs, ys
    step = max(1, n // max_points)
    return xs[::step], ys[::step]

# -------------------- Ajuste Zipf: mejor intervalo (robusto) --------------------

def linreg_loglog_stats(ranks, freqs, start_idx, end_idx):
    xs, ys = [], []
    for r, f in zip(ranks[start_idx:end_idx], freqs[start_idx:end_idx]):
        if r <= 0 or f <= 0:
            return None
        xs.append(math.log(r))
        ys.append(math.log(f))

    m = len(xs)
    if m < 2:
        return None

    mean_x = sum(xs) / m
    mean_y = sum(ys) / m

    sxx = sum((x - mean_x) ** 2 for x in xs)
    if sxx == 0:
        return None

    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = mean_y - slope * mean_x

    # R^2; si sst==0 => y constante => NO queremos esa ventana (evita bug cola freq=1)
    sst = sum((y - mean_y) ** 2 for y in ys)
    if sst == 0:
        return None

    sse = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
    r2 = 1.0 - (sse / sst)
    return slope, intercept, r2

def last_rank_with_min_freq(freqs, min_freq: int):
    # freqs ordenadas desc; devuelve rank (1-based) del último con f>=min_freq
    last = 0
    for i, f in enumerate(freqs, start=1):
        if f >= min_freq:
            last = i
        else:
            break
    return last

def best_zipf_interval(
    ranks,
    freqs,
    *,
    min_rank=20,
    max_rank_cap=20000,
    window=200,
    min_fit_freq=2,
    min_distinct_freqs=3
):
    """
    Busca ventana [a..b] que maximiza R^2 en log-log, evitando:
    - cola con freq constante (sst==0)
    - zonas donde ya entró freq=1 (min_fit_freq)
    - ventanas con muy poca variación (pocas frecuencias distintas)
    """
    n = len(ranks)
    if n < window:
        return None

    # No buscamos más allá de donde f>=min_fit_freq (para evitar slope=0 en hapax)
    last_ok = last_rank_with_min_freq(freqs, min_fit_freq)
    if last_ok < (min_rank + window - 1):
        return None

    end_max_rank = min(max_rank_cap, last_ok)
    if end_max_rank < window:
        return None

    start_min_idx = max(0, min_rank - 1)
    end_max_idx = end_max_rank  # exclusivo (rank)

    best = None  # (r2, slope, a_rank, b_rank)
    range_limit = max(start_min_idx + 1, end_max_idx - window + 1)
    
    if range_limit <= start_min_idx:
        return None
    
    for start_idx in range(start_min_idx, range_limit):
        end_idx = start_idx + window
        
        if end_idx > len(ranks):
            break

        # variación mínima de frecuencias (evita “ventana plana”)
        wfreqs = freqs[start_idx:end_idx]
        if len(set(wfreqs)) < min_distinct_freqs:
            continue

        stats = linreg_loglog_stats(ranks, freqs, start_idx, end_idx)
        if not stats:
            continue

        slope, _, r2 = stats
        a_rank = ranks[start_idx]
        b_rank = ranks[end_idx - 1]

        # Zipf típico slope negativo; si sale positivo raro, no necesariamente malo,
        # pero puedes filtrarlo. Por ahora no filtramos por signo.
        if (best is None) or (r2 > best[0]):
            best = (r2, slope, a_rank, b_rank)

    return best

# -------------------- Flask --------------------

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/process", methods=["POST"])
def process():
    file = request.files.get("pdf")
    if not file:
        return jsonify({"error": "No se recibió PDF"}), 400

    # ---- parámetros (sin UI) ----
    # Para UI: no intentes graficar todo el vocabulario si es enorme.
    # Ajusta 15000/25000 según tu PC.
    max_rank_plot_cap = 27000
    max_points_zipf = 4300

    # Selección automática intervalo
    auto_min_rank = 20
    auto_window = 200
    auto_min_fit_freq = 2        # <- clave para evitar slope=0 en cola
    auto_min_distinct = 3
    auto_max_rank_cap = 20000

    # Longitudes
    max_len = 15
    max_points_len = 60

    temp = "temp.pdf"
    file.save(temp)

    try:
        pages = extract_text_auto(temp)
        text = " ".join(pages)

        tokens = tokenize(text)
        wf = Counter(tokens)

        vocab = len(wf)
        max_rank_plot = min(vocab, max_rank_plot_cap)
        if max_rank_plot < 2:
            return jsonify({"error": "Texto insuficiente para análisis"}), 400

        ranks, words, freqs = rank_frequency(wf, max_rank=max_rank_plot)
        
        if not ranks:
            return jsonify({"error": "No se pudo extraer palabras significativas del PDF"}), 400

        best = best_zipf_interval(
            ranks, freqs,
            min_rank=auto_min_rank,
            max_rank_cap=min(auto_max_rank_cap, len(ranks)),
            window=min(auto_window, len(ranks)),
            min_fit_freq=auto_min_fit_freq,
            min_distinct_freqs=auto_min_distinct
        )
        if best:
            best_r2, best_slope, best_a, best_b = best
        else:
            best_r2, best_slope, best_a, best_b = None, None, None, None

        lens_x, lens_y = length_frequency_from_wordfreq(wf, max_len=max_len)
        shannon_h = shannon_entropy_from_counts(lens_y)

        # downsample para el navegador (solo afecta el PLOT, no el fit)
        ranks_ds, words_ds, freqs_ds = downsample_zipf(ranks, words, freqs, max_points_zipf)
        lens_x_ds, lens_y_ds = downsample_xy(lens_x, lens_y, max_points_len)

        return jsonify({
            "name": file.filename or "PDF",
            "meta": {
                "pages": len(pages),
                "tokens": sum(wf.values()),
                "vocab": vocab,
                "max_rank_plot": max_rank_plot
            },
            "zipf": {
                "ranks": ranks_ds,
                "words": words_ds,
                "freqs": freqs_ds,
                "slope": best_slope,
                "r2": best_r2,
                "fit_min_rank": best_a,
                "fit_max_rank": best_b,
                "fit_window": auto_window,
                "min_fit_freq": auto_min_fit_freq
            },
            "lengths": {
                "x": lens_x_ds,
                "freqs": lens_y_ds,
                "shannon_entropy_nats": shannon_h,
                "max_len": max_len
            }
        })
    finally:
        if os.path.exists(temp):
            os.remove(temp)

if __name__ == "__main__":
    app.run(debug=True)