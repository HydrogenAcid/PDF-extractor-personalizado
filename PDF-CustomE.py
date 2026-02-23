from flask import Flask, render_template, request, jsonify
import os, re, math
from collections import Counter

import fitz  # PyMuPDF
import pytesseract
from PIL import Image

app = Flask(__name__)

WORD_RE = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+")

# ---------- Extracción robusta (texto -> fallback OCR) ----------

def extract_text_pymupdf_blocks(pdf_path: str):
    """
    Usa blocks para ordenar lectura y mejorar PDFs con columnas.
    """
    pages = []
    doc = fitz.open(pdf_path)
    for page in doc:
        blocks = page.get_text("blocks") or []
        # block: (x0, y0, x1, y1, text, block_no, block_type)
        blocks = sorted(blocks, key=lambda b: (b[1], b[0]))  # y luego x
        text = "\n".join((b[4] or "").strip() for b in blocks if (b[4] or "").strip())
        pages.append(text)
    doc.close()
    return pages

def extract_text_ocr_tesseract(pdf_path: str):
    pages = []
    doc = fitz.open(pdf_path)
    for page in doc:
        pix = page.get_pixmap(dpi=200)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        pages.append(pytesseract.image_to_string(img, lang="spa") or "")
    doc.close()
    return pages

def is_text_usable(pages, min_chars_total=500):
    return len(("".join(pages)).strip()) >= min_chars_total

def extract_text_auto(pdf_path: str):
    pages = extract_text_pymupdf_blocks(pdf_path)
    if not is_text_usable(pages):
        pages = extract_text_ocr_tesseract(pdf_path)
    return pages

# ---------- Métricas ----------

def tokenize(text: str):
    return WORD_RE.findall(text.lower())

def rank_frequency(counter: Counter, max_rank: int):
    items = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
    items = items[:max_rank]
    ranks = list(range(1, len(items) + 1))
    freqs = [f for _, f in items]
    return ranks, freqs

def linreg_slope_loglog(ranks, freqs, fit_min_rank: int, fit_max_rank: int):
    xs, ys = [], []
    for r, f in zip(ranks, freqs):
        if r < fit_min_rank or r > fit_max_rank:
            continue
        if f <= 0:
            continue
        xs.append(math.log(r))
        ys.append(math.log(f))
    m = len(xs)
    if m < 2:
        return None

    mean_x = sum(xs) / m
    mean_y = sum(ys) / m
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den = sum((x - mean_x) ** 2 for x in xs)
    if den == 0:
        return None
    return num / den

def length_frequency_from_wordfreq(word_freq: Counter, max_len: int = 23):
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

def downsample_xy(xs, ys, max_points: int):
    n = len(xs)
    if n <= max_points:
        return xs, ys
    step = max(1, n // max_points)
    xs2 = xs[::step]
    ys2 = ys[::step]
    return xs2, ys2

# ---------- Rutas ----------

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/process", methods=["POST"])
def process():
    file = request.files.get("pdf")
    if not file:
        return jsonify({"error": "No se recibió PDF"}), 400

    # parámetros (fijos, sin UI)
    max_rank_plot = 4000
    fit_min_rank = 1
    fit_max_rank = 300
    max_points_zipf = 2500
    max_points_len = 60  # longitudes no suelen ser tantas

    temp = "temp.pdf"
    file.save(temp)

    try:
        pages = extract_text_auto(temp)
        text = " ".join(pages)

        tokens = tokenize(text)
        wf = Counter(tokens)

        ranks, freqs = rank_frequency(wf, max_rank=max_rank_plot)
        slope = linreg_slope_loglog(ranks, freqs, fit_min_rank, fit_max_rank)

        lens_x, lens_y = length_frequency_from_wordfreq(wf, max_len=23)
        shannon_h = shannon_entropy_from_counts(lens_y)

        # downsample para no matar el navegador
        ranks_ds, freqs_ds = downsample_xy(ranks, freqs, max_points_zipf)
        lens_x_ds, lens_y_ds = downsample_xy(lens_x, lens_y, max_points_len)

        return jsonify({
            "name": file.filename or "PDF",
            "meta": {
                "pages": len(pages),
                "tokens": sum(wf.values()),
                "vocab": len(wf),
                "fit_min_rank": fit_min_rank,
                "fit_max_rank": fit_max_rank,
            },
            "zipf": {
                "ranks": ranks_ds,
                "freqs": freqs_ds,
                "slope": slope
            },
            "lengths": {
                "x": lens_x_ds,
                "freqs": lens_y_ds,
                "shannon_entropy_nats": shannon_h
            }
        })
    finally:
        if os.path.exists(temp):
            os.remove(temp)

if __name__ == "__main__":
    app.run(debug=True)