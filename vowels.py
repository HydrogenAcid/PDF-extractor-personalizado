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

# ---------- N-gramas de vocales ----------

def vowel_inventory(lang: str) -> str:
    if lang == "de":
        # consideramos umlauts explícitos
        return "aeiouyäöü"
    if lang == "fr":
        # francés: considerar y como vocal
        return "aeiouy"
    if lang == "en":
        return "aeiouy"  # puedes quitar y si no la quieres
    if lang == "es":
        return "aeiou"
    if lang == "zh_pinyin":
        # pinyin: ü existe; y en pinyin no es vocal suelta típica
        return "aeiouü"
    return "aeiou"

def ngrams_for_lang(lang: str) -> List[str]:
    if lang == "zh_pinyin":
        return ["ai","ei","ao","ou","ia","ie","iao","iu","ua","uo","ui","üe","üa"]


    base = [
        "ae","ai","ao","au",
        "ea","ei","eo","eu",
        "ia","ie","io","iu",
        "oa","oe","oi","ou",
        "ua","ue","ui","uo",
    ]

    if lang in ("fr","en"):
        base += ["ay","ey","iy","oy","uy","ya","ye","yi","yo","yu"]

    if lang == "de":
        # diptongos alemanes con umlauts
        base += ["äu","eu","ie","ei","au","öu","üa","üe"]

    seen = set()
    out = []
    for g in base:
        if g not in seen:
            seen.add(g)
            out.append(g)
    return out

def count_ngrams_in_word(word: str, grams: List[str]) -> Counter:
    c = Counter()
    if not word:
        return c
    for g in grams:
        L = len(g)
        if L == 0 or L > len(word):
            continue
        # conteo con traslape
        start = 0
        while True:
            j = word.find(g, start)
            if j < 0:
                break
            c[g] += 1
            start = j + 1
    return c

def analyze_vowels(text: str, lang: str) -> Dict:
    t = normalize_text_for_lang(text, lang)
    words = words_from_text(t, lang)

    vowels = set(vowel_inventory(lang))
    grams = ngrams_for_lang(lang)

    total_words = 0
    total_vowel_chars = 0
    total_ngrams = 0
    agg = Counter()

    for w in words:
        total_words += 1
        # opcional: filtrar solo letras “relevantes” (vocales+consonantes)
        # aquí solo contamos ngramas dentro de la palabra ya normalizada.
        # pero calculamos total vocal chars para métricas.
        total_vowel_chars += sum(1 for ch in w if ch in vowels)

        # Para que no cuente cosas raras, si el gram contiene ü, el word debe tener ü literal (ya normalizado)
        # Conteo directo de substrings
        cw = count_ngrams_in_word(w, grams)
        if cw:
            agg.update(cw)
            total_ngrams += sum(cw.values())

    freqs = [int(agg.get(g, 0)) for g in grams]
    xs = list(range(1, len(grams) + 1))

    return {
        "lang": lang,
        "grams": grams,
        "x": xs,
        "freqs": freqs,
        "meta": {
            "words": total_words,
            "vowel_chars": total_vowel_chars,
            "ngrams_total": total_ngrams,
        }
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
            out = analyze_vowels(text, lang)
            return jsonify({
                "name": file.filename or "PDF",
                "pages": len(pages),
                **out,
            })
        finally:
            if os.path.exists(temp):
                os.remove(temp)