# vowels.py
from __future__ import annotations

import os
import re
import unicodedata
from collections import Counter
from typing import Dict, List, Tuple

import fitz  # PyMuPDF
import pytesseract
from PIL import Image
from pypinyin import pinyin, Style

from flask import render_template, request, jsonify

# ---------- Extracción (mismo enfoque: PyMuPDF -> OCR fallback) ----------

def extract_text_pymupdf_blocks(pdf_path: str) -> List[str]:
    pages: List[str] = []
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            blocks = page.get_text("blocks") or []
            blocks = sorted(blocks, key=lambda b: (b[1], b[0]))
            text = "\n".join((b[4] or "").strip() for b in blocks if (b[4] or "").strip())
            pages.append(text)
        doc.close()
    except Exception as e:
        print(f"[vocales] Error PyMuPDF: {e}")
    return pages

def extract_text_ocr_tesseract(pdf_path: str, lang: str = "spa") -> List[str]:
    pages: List[str] = []
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            pix = page.get_pixmap(dpi=200)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            pages.append(pytesseract.image_to_string(img, lang=lang) or "")
        doc.close()
    except Exception as e:
        print(f"[vocales] Error OCR: {e}")
    return pages

def is_text_usable(pages: List[str], min_chars_total: int = 500) -> bool:
    return len(("".join(pages)).strip()) >= min_chars_total

def extract_text_auto(pdf_path: str, ocr_lang: str = "spa") -> List[str]:
    pages = extract_text_pymupdf_blocks(pdf_path)
    if not is_text_usable(pages):
        pages = extract_text_ocr_tesseract(pdf_path, lang=ocr_lang)
    return pages

# ---------- Normalización / tokenización ----------

WORD_RE_LATIN = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñÄÖÜäöüŸÿÇç]+", re.UNICODE)

PINYIN_U_MAP = str.maketrans({
    "ǖ": "ü", "ǘ": "ü", "ǚ": "ü", "ǜ": "ü",
    "ü": "ü",
})
def contains_chinese(text: str) -> bool:
    # CJK Unified Ideographs
    return any('\u4e00' <= ch <= '\u9fff' for ch in text)

def chinese_to_pinyin(text: str) -> str:
    # NORMAL = sin tonos (xiao wangzi ...)
    # si quisieras tonos: Style.TONE3 y luego limpiar números
    res = pinyin(text, style=Style.NORMAL, errors="ignore")
    # res es lista de listas: [[xiao],[wang],[zi],...]
    return " ".join(s[0] for s in res if s and s[0])

def strip_diacritics_keep_umlaut(s: str) -> str:
    # Quita tildes/acentos comunes (á->a, é->e, ñ->n, ç->c, etc.)
    # PERO en alemán queremos poder conservar äöü como vocales distintas, así que esto NO se usa para "de".
    nfkd = unicodedata.normalize("NFD", s)
    out = []
    for ch in nfkd:
        if unicodedata.category(ch) == "Mn":
            continue
        out.append(ch)
    return "".join(out)

def normalize_text_for_lang(text: str, lang: str) -> str:
    t = text.lower()

    if lang == "zh_pinyin":
        # 1) colapsa tonos de ü -> ü
        t = t.translate(PINYIN_U_MAP)
        # 2) quita diacríticos en vocales con tono (āáǎà -> a, etc.)
        t = strip_diacritics_keep_umlaut(t)
        return t

    if lang == "de":
        # alemán: conservar umlauts como símbolos propios (ä ö ü)
        # pero puedes normalizar ß si quieres:
        t = t.replace("ß", "ss")
        return t

    # es/en/fr: quitar diacríticos para comparar "limpio"
    t = strip_diacritics_keep_umlaut(t)
    return t

def words_from_text(text: str, lang: str) -> List[str]:
    # En pinyin, sigue siendo latin; para de/fr puede venir con diacríticos.
    return WORD_RE_LATIN.findall(text)

MAX_DIST_DEFAULT = 40_000

def pairs_for_lang(lang: str) -> List[Tuple[str, str]]:
    # devuelve lista de pares (v1, v2)
    if lang == "es":
        return [("a", "e"), ("a", "i"), ("a", "o"), ("a", "u")]
    if lang == "en":
        # mismos que español (según tu requerimiento)
        return [("a", "e"), ("a", "i"), ("a", "o"), ("a", "u")]
    if lang == "de":
        # umlauts (5 gráficas)
        return [("a", "ä"), ("o", "ö"), ("u", "ü"), ("a", "ü"), ("o", "ü")]
    if lang == "fr":
        # y como vocal en pares con a,e,o,u
        return [("y", "a"), ("y", "e"), ("y", "o"), ("y", "u")]
    if lang == "zh_pinyin":
        # 5 pares (de tu set previo, escogidos como pares de vocales)
        return [("a", "i"), ("e", "i"), ("a", "o"), ("o", "u"), ("u", "ü")]
    return [("a", "e")]

def _positions_of_chars(text: str, targets: set[str], max_chars: int) -> Dict[str, List[int]]:
    # guarda posiciones (índices) para cada target, limitado a max_chars
    pos: Dict[str, List[int]] = {t: [] for t in targets}
    upto = min(len(text), max_chars)
    for i in range(upto):
        ch = text[i]
        if ch in pos:
            pos[ch].append(i)
    return pos

def _nearest_distances(A: List[int], B: List[int], max_dist: int) -> List[int]:
    # para cada posición en A, distancia al B más cercano (O(n))
    if not A or not B:
        return []
    dists: List[int] = []
    j = 0
    m = len(B)
    for a in A:
        while j + 1 < m and abs(B[j + 1] - a) <= abs(B[j] - a):
            j += 1
        d = abs(B[j] - a)
        if d > max_dist:
            d = max_dist
        dists.append(d)
    return dists

def _cdf_from_distances(dists: List[int], max_dist: int) -> Tuple[List[int], List[float]]:
    # devuelve puntos (x,y) de CDF; y llega a 1 en max_dist (por capping)
    if not dists:
        return [0, max_dist], [0.0, 0.0]

    dists.sort()
    n = len(dists)

    xs: List[int] = []
    ys: List[float] = []

    # puntos solo en cambios (compacto)
    prev = None
    for idx, d in enumerate(dists, start=1):
        if prev is None or d != prev:
            xs.append(d)
            ys.append(idx / n)
            prev = d
        else:
            ys[-1] = idx / n

    # fuerza el último punto en max_dist con 1.0 (porque cap)
    if xs[-1] != max_dist:
        xs.append(max_dist)
        ys.append(1.0)
    else:
        ys[-1] = 1.0

    return xs, ys

def analyze_vowel_pairs_cdf(text: str, lang: str, max_chars: int = MAX_DIST_DEFAULT, max_dist: int = MAX_DIST_DEFAULT) -> Dict:
    t = normalize_text_for_lang(text, lang)
    pairs = pairs_for_lang(lang)

    # targets únicos
    targets = set([v for p in pairs for v in p])
    pos = _positions_of_chars(t, targets, max_chars=max_chars)

    series = []
    for v1, v2 in pairs:
        A = pos.get(v1, [])
        B = pos.get(v2, [])
        d1 = _nearest_distances(A, B, max_dist=max_dist)     # v1 -> nearest v2
        d2 = _nearest_distances(B, A, max_dist=max_dist)     # v2 -> nearest v1
        dists = d1 + d2                                      # simétrico

        xs, ys = _cdf_from_distances(dists, max_dist=max_dist)

        series.append({
            "pair": f"{v1}/{v2}",
            "x": xs,
            "y": ys,
            "counts": {
                "n_v1": len(A),
                "n_v2": len(B),
                "n_dists": len(dists),
            }
        })

    return {
        "lang": lang,
        "max_chars": max_chars,
        "max_dist": max_dist,
        "series": series,
    }
# ---------- Rutas Flask (se registran desde el main) ----------

def register_vowels(app):
    @app.route("/vocales")
    def vocales_page():
        return render_template("vocales.html")

    @app.route("/process_vowels", methods=["POST"])
    def process_vowels():
        file = request.files.get("pdf")
        lang = (request.form.get("lang") or "es").strip()

        # OCR lang por si subes escaneados (tesseract codes):
        # spa, eng, fra, deu; pinyin normalmente eng
        ocr_map = {"es": "spa", "en": "eng", "fr": "fra", "de": "deu", "zh_pinyin": "eng"}
        ocr_lang = ocr_map.get(lang, "eng")

        if not file:
            return jsonify({"error": "No se recibió PDF"}), 400

        temp = "temp_vowels.pdf"
        file.save(temp)
        try:
            pages = extract_text_auto(temp, ocr_lang=ocr_lang)
            text = " ".join(pages)
            # Si el usuario eligió zh_pinyin y el PDF trae 汉字, convierte antes del análisis
            if lang == "zh_pinyin" and contains_chinese(text):
                text = chinese_to_pinyin(text)
            MAX_CHARS = 40_000
            MAX_DIST = 40_000
            out = analyze_vowel_pairs_cdf(text, lang, max_chars=MAX_CHARS, max_dist=MAX_DIST)
            return jsonify({
                "name": file.filename or "PDF",
                "pages": len(pages),
                **out,
            })
        finally:
            if os.path.exists(temp):
                os.remove(temp)