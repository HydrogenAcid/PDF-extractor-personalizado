from __future__ import annotations

import base64
import io
import math
import random
from collections import Counter
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
from flask import jsonify, render_template, request

MAX_NODES = 250
MAX_EDGES = 350
DEFAULT_NODES = 120
DEFAULT_EDGES = 220
DEFAULT_DISTRIBUTION = "montecarlo"

DISTRIBUTION_LABELS = {
    "montecarlo": "Monte Carlo",
    "poisson": "Poisson",
    "gauss": "Gauss",
    "sbm": "SBM",
    "power_law_truncated": "Power Law Truncada",
}

DISTRIBUTION_DEFAULTS = {
    "montecarlo": {"nodes": 150, "edges": 320, "layout": "Nube uniforme"},
    "poisson": {"nodes": 165, "edges": 330, "layout": "Shell por fitness"},
    "gauss": {"nodes": 150, "edges": 300, "layout": "Eje latente"},
    "sbm": {"nodes": 180, "edges": 340, "layout": "Multipartite por bloque"},
    "power_law_truncated": {"nodes": 125, "edges": 240, "layout": "Shell por grado"},
}

SBM_COLORS = ["#2563eb", "#dc2626", "#059669", "#7c3aed"]


def validate_params(n: int, m: int, distribution: str) -> str | None:
    if distribution not in DISTRIBUTION_LABELS:
        return "Distribucion no soportada"
    if n < 20 or n > MAX_NODES:
        return f"El numero de nodos debe estar entre 20 y {MAX_NODES}"
    max_possible_edges = min(MAX_EDGES, (n * (n - 1)) // 2)
    if m < 10 or m > max_possible_edges:
        return f"El numero de aristas debe estar entre 10 y {max_possible_edges}"
    return None


def sample_poisson(lam: float, rng: random.Random) -> int:
    limit = math.exp(-lam)
    k = 0
    prod = 1.0
    while prod > limit:
        k += 1
        prod *= rng.random()
    return max(0, k - 1)


def sample_truncated_power_law(
    rng: random.Random,
    alpha: float = 2.4,
    xmin: float = 1.0,
    xmax: float = 18.0,
) -> float:
    u = rng.random()
    one_minus_alpha = 1.0 - alpha
    lower = xmin ** one_minus_alpha
    upper = xmax ** one_minus_alpha
    return (lower + u * (upper - lower)) ** (1.0 / one_minus_alpha)


def normalize(values: List[float]) -> List[float]:
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if math.isclose(lo, hi):
        return [0.5 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def histogram_payload(
    values: List[float],
    *,
    title: str,
    x_label: str,
    bins: int = 12,
) -> Dict:
    if not values:
        return {
            "title": title,
            "x_label": x_label,
            "y_label": "Frecuencia",
            "labels": [],
            "values": [],
        }

    lo = min(values)
    hi = max(values)
    if math.isclose(lo, hi):
        return {
            "title": title,
            "x_label": x_label,
            "y_label": "Frecuencia",
            "labels": [f"{lo:.2f}"],
            "values": [len(values)],
        }

    width = (hi - lo) / bins
    counts = [0] * bins
    for value in values:
        idx = min(bins - 1, int((value - lo) / width))
        counts[idx] += 1

    labels = []
    for idx in range(bins):
        start = lo + idx * width
        end = start + width
        labels.append(f"{start:.2f} a {end:.2f}")

    return {
        "title": title,
        "x_label": x_label,
        "y_label": "Frecuencia",
        "labels": labels,
        "values": counts,
    }


def block_payload(blocks: List[int]) -> Dict:
    counts = Counter(blocks)
    labels = [f"Bloque {idx + 1}" for idx in sorted(counts)]
    values = [counts[idx] for idx in sorted(counts)]
    return {
        "title": "Tamano de bloques (SBM)",
        "x_label": "Bloque",
        "y_label": "Nodos",
        "labels": labels,
        "values": values,
    }


def montecarlo_profile(n: int, rng: random.Random) -> Dict:
    node_values = [rng.random() for _ in range(n)]
    pair_scores = []
    for u in range(n):
        for v in range(u + 1, n):
            pair_scores.append((u, v, rng.random()))

    return {
        "node_values": node_values,
        "blocks": [0] * n,
        "pair_scores": pair_scores,
        "distribution_plot": histogram_payload(
            [score for _, _, score in pair_scores],
            title="Histograma Monte Carlo uniforme",
            x_label="Score U(0, 1) por arista candidata",
        ),
        "description": (
            "Cada arista candidata recibe un score uniforme aleatorio. "
            "Seleccionar las m mayores produce un baseline aleatorio tipo G(n, m)."
        ),
    }


def poisson_profile(n: int, rng: random.Random) -> Dict:
    node_values = [float(sample_poisson(3.4, rng)) for _ in range(n)]
    pair_scores = []
    for u in range(n):
        for v in range(u + 1, n):
            base = (node_values[u] + 1.0) * (node_values[v] + 1.0)
            score = base * (0.88 + 0.24 * rng.random())
            pair_scores.append((u, v, score))

    return {
        "node_values": node_values,
        "blocks": [0] * n,
        "pair_scores": pair_scores,
        "distribution_plot": histogram_payload(
            node_values,
            title="Distribucion Poisson de fitness por nodo",
            x_label="Fitness discreto del nodo",
            bins=10,
        ),
        "description": (
            "Cada nodo recibe un fitness Poisson. Nodos con fitness alto "
            "tienden a concentrar mas conexiones."
        ),
    }


def gauss_profile(n: int, rng: random.Random) -> Dict:
    node_values = [rng.gauss(0.0, 1.0) for _ in range(n)]
    sigma = 0.9
    pair_scores = []
    for u in range(n):
        for v in range(u + 1, n):
            proximity = math.exp(-((node_values[u] - node_values[v]) ** 2) / (2.0 * sigma**2))
            score = proximity * (0.9 + 0.2 * rng.random())
            pair_scores.append((u, v, score))

    return {
        "node_values": node_values,
        "blocks": [0] * n,
        "pair_scores": pair_scores,
        "distribution_plot": histogram_payload(
            node_values,
            title="Distribucion Gaussiana de atributos latentes",
            x_label="Atributo N(0, 1) por nodo",
        ),
        "description": (
            "Cada nodo recibe un atributo gaussiano. Los nodos con atributos "
            "parecidos reciben mayor score de conexion."
        ),
    }


def sbm_profile(n: int, rng: random.Random) -> Dict:
    blocks = []
    for _ in range(n):
        roll = rng.random()
        if roll < 0.42:
            blocks.append(0)
        elif roll < 0.74:
            blocks.append(1)
        else:
            blocks.append(2)

    node_values = [float(block + 1) for block in blocks]
    pair_scores = []
    for u in range(n):
        for v in range(u + 1, n):
            same_block = blocks[u] == blocks[v]
            base = 1.0 if same_block else 0.18
            block_bonus = 0.08 * (blocks[u] + blocks[v])
            score = (base + block_bonus) * (0.85 + 0.3 * rng.random())
            pair_scores.append((u, v, score))

    return {
        "node_values": node_values,
        "blocks": blocks,
        "pair_scores": pair_scores,
        "distribution_plot": block_payload(blocks),
        "description": (
            "Los nodos se reparten en bloques. Las aristas dentro del mismo bloque "
            "tienen mayor score que las aristas entre bloques."
        ),
    }


def truncated_power_law_profile(n: int, rng: random.Random) -> Dict:
    node_values = [sample_truncated_power_law(rng) for _ in range(n)]
    pair_scores = []
    for u in range(n):
        for v in range(u + 1, n):
            base = math.sqrt(node_values[u] * node_values[v])
            score = base * (0.86 + 0.28 * rng.random())
            pair_scores.append((u, v, score))

    return {
        "node_values": node_values,
        "blocks": [0] * n,
        "pair_scores": pair_scores,
        "distribution_plot": histogram_payload(
            node_values,
            title="Power Law truncada de fitness por nodo",
            x_label="Fitness truncado por nodo",
        ),
        "description": (
            "Cada nodo recibe un fitness heavy-tail acotado. Esto produce hubs, "
            "pero evitando valores extremos inestables."
        ),
    }


def distribution_profile(distribution: str, n: int, rng: random.Random) -> Dict:
    builders = {
        "montecarlo": montecarlo_profile,
        "poisson": poisson_profile,
        "gauss": gauss_profile,
        "sbm": sbm_profile,
        "power_law_truncated": truncated_power_law_profile,
    }
    return builders[distribution](n, rng)


def poisson_shells(G: nx.Graph) -> List[List[int]]:
    grouped: Dict[int, List[int]] = {}
    for node in G.nodes:
        bucket = int(G.nodes[node]["value"])
        grouped.setdefault(bucket, []).append(node)

    shells = []
    for bucket in sorted(grouped.keys(), reverse=True):
        shells.append(sorted(grouped[bucket]))
    return [shell for shell in shells if shell]


def degree_shells(G: nx.Graph) -> List[List[int]]:
    ranked = sorted(G.degree(), key=lambda item: item[1], reverse=True)
    nodes = [node for node, _ in ranked]
    n = len(nodes)
    if n <= 10:
        return [nodes]

    inner = max(4, round(0.08 * n))
    middle = max(10, round(0.22 * n))
    shells = [
        nodes[:inner],
        nodes[inner : inner + middle],
        nodes[inner + middle :],
    ]
    return [shell for shell in shells if shell]


def gauss_positions(G: nx.Graph) -> Dict[int, Tuple[float, float]]:
    values = {node: float(G.nodes[node]["value"]) for node in G.nodes}
    normalized_map = {
        node: norm
        for node, norm in zip(values.keys(), normalize(list(values.values())))
    }

    sorted_nodes = sorted(G.nodes, key=lambda node: values[node])
    positions: Dict[int, Tuple[float, float]] = {}
    rng = random.Random(31415)

    rows = 8
    row_counts = [0] * rows
    for idx, node in enumerate(sorted_nodes):
        row = idx % rows
        row_counts[row] += 1
        x = normalized_map[node] * 2.6 - 1.3
        y_base = 1.0 - (2.0 * row / max(1, rows - 1))
        y = y_base + rng.uniform(-0.07, 0.07)
        positions[node] = (x, y)

    return positions


def choose_layout(distribution: str, G: nx.Graph) -> Dict[int, Tuple[float, float]]:
    if distribution == "montecarlo":
        return nx.random_layout(G, seed=2026)

    if distribution == "poisson":
        return nx.shell_layout(G, nlist=poisson_shells(G))

    if distribution == "gauss":
        return gauss_positions(G)

    if distribution == "sbm":
        return nx.multipartite_layout(G, subset_key="block", align="vertical")

    if distribution == "power_law_truncated":
        return nx.shell_layout(G, nlist=degree_shells(G))

    return nx.spring_layout(G, seed=2026)


def graph_from_ranked_scores(
    profile: Dict, n: int, m: int, rng: random.Random
) -> nx.Graph:
    weighted = []
    for u, v, score in profile["pair_scores"]:
        weight = max(float(score), 1e-9)
        key = rng.random() ** (1.0 / weight)
        weighted.append((key, u, v, score))

    weighted.sort(reverse=True)
    chosen_edges = [(u, v, {"score": score}) for _, u, v, score in weighted[:m]]

    G = nx.Graph()
    for node_id in range(n):
        G.add_node(
            node_id,
            value=float(profile["node_values"][node_id]),
            block=int(profile["blocks"][node_id]),
        )
    G.add_edges_from(chosen_edges)
    return G


def repair_isolated_nodes(G: nx.Graph, profile: Dict) -> None:
    candidate_by_node: Dict[int, List[Tuple[int, float]]] = {node: [] for node in G.nodes}
    for u, v, score in sorted(profile["pair_scores"], key=lambda item: item[2], reverse=True):
        candidate_by_node[u].append((v, float(score)))
        candidate_by_node[v].append((u, float(score)))

    locked_edges = set()
    for node in list(nx.isolates(G)):
        if G.degree(node) > 0:
            continue

        chosen_neighbor = None
        chosen_score = 0.0
        for neighbor, score in candidate_by_node[node]:
            if not G.has_edge(node, neighbor):
                chosen_neighbor = neighbor
                chosen_score = score
                break

        if chosen_neighbor is None:
            continue

        new_edge = tuple(sorted((node, chosen_neighbor)))
        G.add_edge(node, chosen_neighbor, score=chosen_score)
        locked_edges.add(new_edge)

        removable = sorted(
            (
                (data.get("score", 0.0), u, v)
                for u, v, data in G.edges(data=True)
                if tuple(sorted((u, v))) not in locked_edges
                and G.degree(u) > 1
                and G.degree(v) > 1
            ),
            key=lambda item: item[0],
        )
        if removable:
            _, u, v = removable[0]
            G.remove_edge(u, v)


def giant_component_size(G: nx.Graph) -> int:
    if G.number_of_nodes() == 0:
        return 0
    return max((len(component) for component in nx.connected_components(G)), default=0)


def graph_metrics(G: nx.Graph) -> Dict:
    n = G.number_of_nodes()
    m = G.number_of_edges()
    degrees = [degree for _, degree in G.degree()]

    return {
        "nodes": n,
        "edges": m,
        "avg_degree": (2.0 * m / n) if n else 0.0,
        "max_degree": max(degrees) if degrees else 0,
        "density": float(nx.density(G)) if n > 1 else 0.0,
        "connected_components": nx.number_connected_components(G) if n else 0,
        "giant_component_size": giant_component_size(G),
        "avg_clustering": float(nx.average_clustering(G)) if n else 0.0,
        "equivalent_p": (2.0 * m / (n * (n - 1))) if n > 1 else 0.0,
    }


def node_colors(G: nx.Graph) -> List[str]:
    blocks = [G.nodes[node]["block"] for node in G.nodes]
    if any(blocks):
        return [SBM_COLORS[block % len(SBM_COLORS)] for block in blocks]

    values = [G.nodes[node]["value"] for node in G.nodes]
    normalized = normalize(values)
    cmap = plt.get_cmap("viridis")
    return [
        matplotlib.colors.to_hex(cmap(value), keep_alpha=False)  # type: ignore[attr-defined]
        for value in normalized
    ]


def render_network_figure(G: nx.Graph, label: str, distribution: str) -> str:
    pos = choose_layout(distribution, G)
    colors = node_colors(G)
    sizes = [48 + 14 * G.degree(node) for node in G.nodes]

    fig, ax = plt.subplots(figsize=(6.2, 6.2), dpi=150)
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#ffffff")

    nx.draw_networkx_edges(
        G,
        pos,
        ax=ax,
        edge_color="#94a3b8",
        alpha=0.45,
        width=0.8,
    )
    nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        node_color=colors,
        node_size=sizes,
        edgecolors="#ffffff",
        linewidths=0.35,
    )

    ax.set_title(f"Red simulada: {label}", fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    buffer = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def simulate_erdos_variant(distribution: str, n: int, m: int) -> Dict:
    rng = random.Random(2026 + n * 11 + m * 7 + len(distribution))
    profile = distribution_profile(distribution, n, rng)
    G = graph_from_ranked_scores(profile, n, m, rng)
    repair_isolated_nodes(G, profile)
    metrics = graph_metrics(G)

    return {
        "distribution": distribution,
        "distribution_label": DISTRIBUTION_LABELS[distribution],
        "description": profile["description"],
        "layout_label": DISTRIBUTION_DEFAULTS[distribution]["layout"],
        "recommended_defaults": DISTRIBUTION_DEFAULTS[distribution],
        "distribution_plot": profile["distribution_plot"],
        "network_image_b64": render_network_figure(
            G, DISTRIBUTION_LABELS[distribution], distribution
        ),
        "metrics": metrics,
    }


def register_erdos_renyi(app):
    @app.route("/erdos_renyi")
    def erdos_renyi_page():
        return render_template(
            "erdos_renyi.html",
            distribution_defaults=DISTRIBUTION_DEFAULTS,
            default_distribution=DEFAULT_DISTRIBUTION,
        )

    @app.route("/process_erdos_renyi", methods=["POST"])
    def process_erdos_renyi():
        distribution = (request.form.get("distribution") or DEFAULT_DISTRIBUTION).strip()
        defaults = DISTRIBUTION_DEFAULTS.get(distribution, {})
        n = int(request.form.get("nodes") or defaults.get("nodes") or DEFAULT_NODES)
        m = int(request.form.get("edges") or defaults.get("edges") or DEFAULT_EDGES)

        error = validate_params(n, m, distribution)
        if error:
            return jsonify({"error": error}), 400

        return jsonify(simulate_erdos_variant(distribution, n, m))
