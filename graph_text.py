# graph_text.py
from __future__ import annotations

import os
import re
from collections import Counter
from typing import Dict, List, Tuple

import fitz
import pytesseract
from PIL import Image
import networkx as nx

from flask import render_template, request, jsonify

WORD_RE = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñÀÂÆÇÈÉÊËÎÏÔŒÙÛÜŸàâæçèéêëîïôœùûüÿ']+")

# -------------------- Extracción --------------------

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
        print(f"[graph] Error PyMuPDF: {e}")
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
        print(f"[graph] Error OCR: {e}")
    return pages

def is_text_usable(pages: List[str], min_chars_total: int = 500) -> bool:
    return len(("".join(pages)).strip()) >= min_chars_total

def extract_text_auto(pdf_path: str, ocr_lang: str = "spa") -> List[str]:
    pages = extract_text_pymupdf_blocks(pdf_path)
    if not is_text_usable(pages):
        pages = extract_text_ocr_tesseract(pdf_path, lang=ocr_lang)
    return pages

# -------------------- Stopwords --------------------

STOPWORDS_ES = {
    "el","la","los","las","de","del","y","o","u","a","ante","bajo","cabe","con","contra",
    "desde","durante","en","entre","hacia","hasta","para","por","segun","sin","sobre","tras",
    "un","una","unos","unas","al","que","se","su","sus","lo","le","les","me","te","nos","os",
    "mi","mis","tu","tus","si","ya","no","como","más","mas","muy","pero","porque","pues",
    "es","son","era","eran","fue","fueron","ser","estar","esta","este","estos","estas","ese",
    "esa","esos","esas","aqui","aquí","alli","allí","hay"
}

STOPWORDS_EN = {
    "the","a","an","and","or","but","of","to","in","on","at","for","from","by","with","as",
    "is","are","was","were","be","been","being","it","its","that","this","these","those",
    "he","she","they","them","his","her","their","you","your","i","we","our","not","no","yes",
    "do","does","did","have","has","had","will","would","can","could","shall","should","may",
    "might","if","then","than","so","because","about","into","over","under","out","up","down"
}

STOPWORDS_FR = {
    "le","la","les","un","une","des","du","de","d","et","ou","à","a","au","aux","en","dans",
    "sur","sous","par","pour","avec","sans","vers","chez","entre","mais","donc","or","ni","car",
    "que","qui","quoi","dont","où","ou","ce","cet","cette","ces","il","ils","elle","elles",
    "je","tu","nous","vous","me","te","se","mon","ton","son","ma","ta","sa","mes","tes","ses",
    "est","sont","était","etaient","être","etre","avoir","ont","pas","plus","moins","oui","non"
}

def get_stopwords(lang: str) -> set[str]:
    if lang == "es":
        return STOPWORDS_ES
    if lang == "en":
        return STOPWORDS_EN
    if lang == "fr":
        return STOPWORDS_FR
    return set()

# -------------------- Tokenización --------------------

def tokenize(text: str) -> List[str]:
    text = text.lower()
    tokens = WORD_RE.findall(text)
    tokens = [t.strip("'") for t in tokens]
    tokens = [t for t in tokens if len(t) >= 2]
    return tokens

def tokenize_without_stopwords(text: str, lang: str) -> List[str]:
    sw = get_stopwords(lang)
    toks = tokenize(text)
    return [t for t in toks if t not in sw]

def tokenize_pages_without_stopwords(pages: List[str], lang: str) -> List[List[str]]:
    return [tokenize_without_stopwords(page, lang) for page in pages]

# -------------------- Grafo por página --------------------

def build_page_cooccurrence_graph(
    token_pages: List[List[str]],
    span: int = 1,
    min_freq: int = 2,
    max_vocab: int = 400
) -> Tuple[nx.Graph, Counter]:
    """
    span=1 -> conecta cada palabra con la siguiente dentro de la misma página
    span=2 -> conecta con las siguientes dos dentro de la misma página
    """
    freq = Counter()
    for toks in token_pages:
        freq.update(toks)

    vocab_items = [w for w, c in freq.items() if c >= min_freq]
    vocab_items.sort(key=lambda w: (-freq[w], w))
    vocab = set(vocab_items[:max_vocab])

    G = nx.Graph()
    for w in vocab:
        G.add_node(w, freq=freq[w])

    for toks in token_pages:
        filtered = [t for t in toks if t in vocab]
        n = len(filtered)

        for i in range(n):
            wi = filtered[i]
            upper = min(i + span + 1, n)
            for j in range(i + 1, upper):
                wj = filtered[j]
                if wi == wj:
                    continue
                if G.has_edge(wi, wj):
                    G[wi][wj]["weight"] += 1
                else:
                    G.add_edge(wi, wj, weight=1)

    return G, freq

# -------------------- Métricas --------------------

def degree_distribution(G: nx.Graph) -> Dict[str, List[int]]:
    degs = [d for _, d in G.degree()]
    if not degs:
        return {"k": [], "count": []}

    c = Counter(degs)
    ks = sorted(c.keys())
    counts = [c[k] for k in ks]
    return {"k": ks, "count": counts}

def safe_assortativity(G: nx.Graph):
    try:
        val = nx.degree_assortativity_coefficient(G)
        if val != val:
            return None
        return float(val)
    except Exception:
        return None

def safe_clustering(G: nx.Graph):
    try:
        if G.number_of_nodes() == 0:
            return None
        return float(nx.average_clustering(G))
    except Exception:
        return None

def giant_component_size(G: nx.Graph) -> int:
    if G.number_of_nodes() == 0:
        return 0
    comps = list(nx.connected_components(G))
    if not comps:
        return 0
    return max(len(c) for c in comps)

def compute_graph_metrics(G: nx.Graph) -> Dict:
    n = G.number_of_nodes()
    m = G.number_of_edges()

    if n == 0:
        return {
            "nodes": 0,
            "edges": 0,
            "avg_degree": None,
            "density": None,
            "connected_components": 0,
            "giant_component_size": 0,
            "assortativity": None,
            "avg_clustering": None,
        }

    return {
        "nodes": n,
        "edges": m,
        "avg_degree": (2.0 * m / n),
        "density": nx.density(G) if n > 1 else None,
        "connected_components": nx.number_connected_components(G),
        "giant_component_size": giant_component_size(G),
        "assortativity": safe_assortativity(G),
        "avg_clustering": safe_clustering(G),
    }

# -------------------- Subgrafo de visualización --------------------

def top_subgraph_for_display(
    G: nx.Graph,
    max_nodes: int = 35,
    max_edges: int = 80
) -> Dict:
    if G.number_of_nodes() == 0:
        return {"nodes": [], "edges": []}

    ranked_nodes = sorted(
        G.nodes(data=True),
        key=lambda x: (-x[1].get("freq", 0), x[0])
    )[:max_nodes]

    keep = {node for node, _ in ranked_nodes}

    edges = []
    for u, v, data in G.edges(data=True):
        if u in keep and v in keep:
            edges.append((u, v, data.get("weight", 1)))

    edges.sort(key=lambda x: (-x[2], x[0], x[1]))
    edges = edges[:max_edges]

    used_nodes = set()
    for u, v, _ in edges:
        used_nodes.add(u)
        used_nodes.add(v)

    node_payload = []
    for node, data in ranked_nodes:
        if node in used_nodes:
            node_payload.append({
                "id": node,
                "label": node,
                "value": int(data.get("freq", 1)),
                "title": f"{node} (freq={data.get('freq', 1)})"
            })

    edge_payload = []
    for u, v, w in edges:
        edge_payload.append({
            "from": u,
            "to": v,
            "value": int(w),
            "title": f"{u} — {v} (w={w})"
        })

    return {"nodes": node_payload, "edges": edge_payload}

# -------------------- Flask --------------------

def register_graph_text(app):
    @app.route("/grafo_texto")
    def grafo_texto_page():
        return render_template("grafo_texto.html")

    @app.route("/process_graph_text", methods=["POST"])
    def process_graph_text():
        file = request.files.get("pdf")
        lang = (request.form.get("lang") or "es").strip()
        span = int(request.form.get("span") or 1)
        min_freq = int(request.form.get("min_freq") or 2)
        max_vocab = int(request.form.get("max_vocab") or 400)

        if not file:
            return jsonify({"error": "No se recibió PDF"}), 400

        ocr_map = {
            "es": "spa",
            "en": "eng",
            "fr": "fra",
            "de": "deu"
        }
        ocr_lang = ocr_map.get(lang, "eng")

        temp = "temp_graph.pdf"
        file.save(temp)

        try:
            pages = extract_text_auto(temp, ocr_lang=ocr_lang)
            token_pages = tokenize_pages_without_stopwords(pages, lang)

            tokens_raw = sum(len(tokenize(p)) for p in pages)
            tokens_filtered = sum(len(tp) for tp in token_pages)

            if tokens_filtered < 10:
                return jsonify({"error": "Texto insuficiente tras quitar stopwords"}), 400

            G, freq = build_page_cooccurrence_graph(
                token_pages=token_pages,
                span=span,
                min_freq=min_freq,
                max_vocab=max_vocab
            )

            return jsonify({
                "name": file.filename or "PDF",
                "meta": {
                    "pages": len(pages),
                    "tokens_raw": tokens_raw,
                    "tokens_filtered": tokens_filtered,
                    "span": span,
                    "min_freq": min_freq,
                    "max_vocab": max_vocab,
                    "lang": lang,
                    "scope": "por_pagina"
                },
                "metrics": compute_graph_metrics(G),
                "degree_distribution": degree_distribution(G),
                "display_graph": top_subgraph_for_display(G, max_nodes=35, max_edges=80)
            })
        finally:
            if os.path.exists(temp):
                os.remove(temp)