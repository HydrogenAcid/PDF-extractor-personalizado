from __future__ import annotations

import math
import random
from copy import deepcopy
from functools import lru_cache
from statistics import fmean
from typing import Dict, Iterable, List, Tuple

import networkx as nx
from flask import jsonify, render_template, request

DEFAULT_N = 1000
DEFAULT_K = 10
DEFAULT_REALIZATIONS = 20
DEFAULT_LOG_POINTS = 13
DEFAULT_MIN_EXP = -4.0
DEFAULT_MAX_EXP = 0.0


def logspace_p_values(
    start_exp: float = DEFAULT_MIN_EXP,
    stop_exp: float = DEFAULT_MAX_EXP,
    num: int = DEFAULT_LOG_POINTS,
) -> List[float]:
    if num <= 1:
        return [10.0 ** stop_exp]

    step = (stop_exp - start_exp) / (num - 1)
    return [10.0 ** (start_exp + i * step) for i in range(num)]


def validate_params(n: int, k: int, realizations: int, log_points: int) -> str | None:
    if n < 20:
        return "n debe ser al menos 20"
    if k < 2:
        return "k debe ser al menos 2"
    if k >= n:
        return "k debe ser menor que n"
    if k % 2 != 0:
        return "k debe ser par para construir la red anillo de Watts-Strogatz"
    if realizations < 1 or realizations > 50:
        return "El numero de realizaciones debe estar entre 1 y 50"
    if log_points < 5 or log_points > 25:
        return "El numero de puntos para la figura debe estar entre 5 y 25"
    return None


def build_connected_ws_graph(n: int, k: int, p: float, seed: int) -> nx.Graph:
    if p <= 0.0:
        return nx.watts_strogatz_graph(n, k, 0.0, seed=seed)
    return nx.connected_watts_strogatz_graph(n, k, p, tries=200, seed=seed)


def graph_metrics(G: nx.Graph) -> Dict[str, float]:
    return {
        "nodes": int(G.number_of_nodes()),
        "edges": int(G.number_of_edges()),
        "avg_shortest_path": float(estimate_average_shortest_path(G)),
        "avg_clustering": float(nx.average_clustering(G)),
    }


def estimate_average_shortest_path(
    G: nx.Graph,
    max_sources: int = 140,
    seed: int = 2026,
) -> float:
    n = G.number_of_nodes()
    if n <= max_sources:
        return float(nx.average_shortest_path_length(G))

    rng = random.Random(seed)
    nodes = list(G.nodes())
    sources = rng.sample(nodes, k=max_sources)

    total_distance = 0
    total_pairs = 0
    for source in sources:
        lengths = nx.single_source_shortest_path_length(G, source)
        total_distance += sum(lengths.values())
        total_pairs += len(lengths) - 1

    return total_distance / total_pairs


def average_metrics_for_probability(
    n: int, k: int, p: float, realizations: int
) -> Dict[str, float]:
    metrics = []
    for run_idx in range(realizations):
        seed = 2026 + run_idx + int(round(p * 1_000_000))
        G = build_connected_ws_graph(n, k, p, seed)
        metrics.append(graph_metrics(G))

    return {
        "p": float(p),
        "avg_shortest_path": fmean(m["avg_shortest_path"] for m in metrics),
        "avg_clustering": fmean(m["avg_clustering"] for m in metrics),
        "nodes": int(metrics[0]["nodes"]),
        "edges": int(metrics[0]["edges"]),
    }


def regular_lattice_theory(n: int, k: int) -> Dict[str, float]:
    return {
        "avg_shortest_path_asymptotic": n / (2.0 * k),
        "avg_clustering_exact": (3.0 * (k - 2.0)) / (4.0 * (k - 1.0)),
        "avg_clustering_limit": 0.75,
    }


def random_graph_theory(n: int, k: int) -> Dict[str, float]:
    return {
        "avg_shortest_path_asymptotic": math.log(n) / math.log(k),
        "avg_clustering_asymptotic": k / n,
        "avg_clustering_er": k / (n - 1.0),
    }


def safe_relative_error(empirical: float, theoretical: float) -> float | None:
    if theoretical == 0:
        return None
    return (empirical - theoretical) / theoretical


def serialize_float_list(values: Iterable[float], digits: int = 8) -> List[float]:
    return [round(float(v), digits) for v in values]


@lru_cache(maxsize=16)
def cached_small_world_experiment(
    n: int,
    k: int,
    realizations: int,
    log_points: int,
) -> Dict:
    p_values = logspace_p_values(num=log_points)

    regular_empirical = average_metrics_for_probability(n, k, 0.0, 1)
    random_empirical = average_metrics_for_probability(n, k, 1.0, realizations)

    baseline_L = regular_empirical["avg_shortest_path"]
    baseline_C = regular_empirical["avg_clustering"]

    curves = [average_metrics_for_probability(n, k, p, realizations) for p in p_values]

    return {
        "params": {
            "n": n,
            "k": k,
            "realizations": realizations,
            "log_points": log_points,
            "path_estimator": "sampled_bfs",
        },
        "relations": {
            "regular_p0": {
                "p": 0.0,
                "formula_path": "L(0) ~ n / (2k)",
                "formula_clustering": "C(0) = 3(k-2) / (4(k-1)) ~ 3/4",
                "empirical": regular_empirical,
                "theoretical": regular_lattice_theory(n, k),
            },
            "random_p1": {
                "p": 1.0,
                "formula_path": "L(1) ~ ln(n) / ln(k)",
                "formula_clustering": "C(1) ~ k / n",
                "empirical": random_empirical,
                "theoretical": random_graph_theory(n, k),
            },
        },
        "figure2": {
            "p_values": serialize_float_list(p_values, digits=10),
            "l_values": serialize_float_list([item["avg_shortest_path"] for item in curves]),
            "c_values": serialize_float_list([item["avg_clustering"] for item in curves]),
            "l_ratio": serialize_float_list(
                [item["avg_shortest_path"] / baseline_L for item in curves]
            ),
            "c_ratio": serialize_float_list(
                [item["avg_clustering"] / baseline_C for item in curves]
            ),
            "baseline_l0": round(baseline_L, 8),
            "baseline_c0": round(baseline_C, 8),
        },
    }


def enrich_with_errors(payload: Dict) -> Dict:
    result = deepcopy(payload)
    regular = result["relations"]["regular_p0"]
    random_case = result["relations"]["random_p1"]

    reg_emp = regular["empirical"]
    reg_theory = regular["theoretical"]
    rnd_emp = random_case["empirical"]
    rnd_theory = random_case["theoretical"]

    regular["comparison"] = {
        "path_relative_error": safe_relative_error(
            reg_emp["avg_shortest_path"], reg_theory["avg_shortest_path_asymptotic"]
        ),
        "clustering_relative_error": safe_relative_error(
            reg_emp["avg_clustering"], reg_theory["avg_clustering_exact"]
        ),
    }
    random_case["comparison"] = {
        "path_relative_error": safe_relative_error(
            rnd_emp["avg_shortest_path"], rnd_theory["avg_shortest_path_asymptotic"]
        ),
        "clustering_relative_error": safe_relative_error(
            rnd_emp["avg_clustering"], rnd_theory["avg_clustering_asymptotic"]
        ),
    }

    return result


def register_small_world(app):
    @app.route("/small_world")
    def small_world_page():
        return render_template("small_world.html")

    @app.route("/process_small_world", methods=["POST"])
    def process_small_world():
        n = int(request.form.get("n") or DEFAULT_N)
        k = int(request.form.get("k") or DEFAULT_K)
        realizations = int(request.form.get("realizations") or DEFAULT_REALIZATIONS)
        log_points = int(request.form.get("log_points") or DEFAULT_LOG_POINTS)

        error = validate_params(n, k, realizations, log_points)
        if error:
            return jsonify({"error": error}), 400

        payload = cached_small_world_experiment(n, k, realizations, log_points)
        return jsonify(enrich_with_errors(payload))
