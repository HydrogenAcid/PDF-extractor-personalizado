from flask import Flask, render_template, request, jsonify
import pdfplumber, fitz, pytesseract
from PIL import Image
import re
from collections import Counter
import networkx as nx
import os

app = Flask(__name__)

# -------- TEXTO --------
def extract_text(pdf_path, ocr=False, two_columns=False):
    pages = []

    if not ocr:
        with pdfplumber.open(pdf_path) as pdf:
            for p in pdf.pages:
                if two_columns:
                    w = p.width/2
                    left = p.crop((0,0,w,p.height)).extract_text() or ""
                    right = p.crop((w,0,p.width,p.height)).extract_text() or ""
                    pages.append(left + " " + right)
                else:
                    pages.append(p.extract_text() or "")
    else:
        doc = fitz.open(pdf_path)
        for page in doc:
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            pages.append(pytesseract.image_to_string(img, lang="spa"))

    return pages


# -------- FRECUENCIA --------
def top_words(text, top=15):
    words = re.findall(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+", text.lower())
    freq = Counter(words)
    return freq.most_common(top)


# -------- SERIE --------
def serie(pages):
    return [len(re.findall(r"\w+", p)) for p in pages]


# -------- GRAFO FRASES --------
def graph_phrases(text):
    words = re.findall(r"\w+", text)
    edges = []
    for i in range(len(words)-1):
        edges.append((words[i], words[i+1]))
    return edges[:40]


# -------- PAGINAS CON LONGITUDES DE PALABRAS --------
def pages_format_with_lengths(pages):
    result = []
    for i, text in enumerate(pages):
        lengths_list = [len(w) for w in re.findall(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+", text)]
        result.append(f"{text} {{{','.join(map(str,lengths_list))}}} hoja {i+1}")
    return result


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/process", methods=["POST"])
def process():
    file = request.files["pdf"]

    # Manejo de campos vacíos
    start_str = request.form.get("start", "1").strip()
    end_str = request.form.get("end", "9999").strip()
    start = int(start_str) if start_str else 1
    end = int(end_str) if end_str else 9999

    ocr = bool(request.form.get("ocr"))
    two = bool(request.form.get("two"))

    temp = "temp.pdf"
    file.save(temp)

    pages = extract_text(temp, ocr, two)
    pages = pages[start-1:end]

    text = " ".join(pages)

    data = {
        "serie": serie(pages),
        "top": top_words(text),
        "edges": graph_phrases(text),
        "pages": pages_format_with_lengths(pages)  # <-- Cambiado aquí
    }

    os.remove(temp)
    return jsonify(data)


if __name__ == "__main__":
    app.run(debug=True)