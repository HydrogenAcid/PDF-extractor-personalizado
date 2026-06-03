from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import networkx as nx
import numpy as np
from flask import jsonify, render_template, request, send_from_directory, url_for

TOTAL_NODES = 300
COMPONENT_SIZE = 150
WS_NEIGHBORS = 4
WS_REWIRE_P = 0.00001
BRIDGE_EDGE = (0, COMPONENT_SIZE)
FRAME_TIMES = list(range(20, 35000, 250))
MAX_TIME = FRAME_TIMES[-1]
MASTER_SEED = 20260414
RENDER_VERSION = 6
PLAYBACK_INTERVAL_MS = 375
FRAME_FIGSIZE = (10.8, 6.2)
FRAME_DPI = 100
LEFT_CENTER = (-2.2, 0.0)
RIGHT_CENTER = (2.2, 0.0)

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "output" / "frames" / "animacion_charles"
MANIFEST_PATH = OUTPUT_DIR / "manifest.json"
GIF_PATH = PROJECT_ROOT / "output" / "gifs" / "animacion_charles" / "animacion_charles.gif"


def _normalize_cluster(
    positions: Dict[int, np.ndarray],
    center: Tuple[float, float],
    width: float,
    height: float,
    bridge_node: int,
    bridge_direction: str,
) -> Dict[int, Tuple[float, float]]:
    nodes = list(positions.keys())
    coords = np.asarray([positions[node] for node in nodes], dtype=float)
    coords -= coords.mean(axis=0, keepdims=True)

    bridge_index = nodes.index(bridge_node)
    bridge_vector = coords[bridge_index]
    bridge_angle = float(np.arctan2(bridge_vector[1], bridge_vector[0]))
    target_angle = 0.0 if bridge_direction == "right" else np.pi
    rotation = target_angle - bridge_angle
    rotation_matrix = np.array(
        [
            [np.cos(rotation), -np.sin(rotation)],
            [np.sin(rotation), np.cos(rotation)],
        ]
    )
    coords = coords @ rotation_matrix.T

    radial_scale = max(float(np.percentile(np.linalg.norm(coords, axis=1), 95)), 1e-6)
    coords = coords / radial_scale
    coords[:, 0] *= width
    coords[:, 1] *= height
    coords[:, 0] += center[0]
    coords[:, 1] += center[1]

    return {
        node: (float(coord[0]), float(coord[1])) for node, coord in zip(nodes, coords)
    }


def compact_component_layout(
    graph: nx.Graph,
    nodes: List[int],
    seed: int,
    center: Tuple[float, float],
    bridge_node: int,
    bridge_direction: str,
) -> Dict[int, Tuple[float, float]]:
    subgraph = graph.subgraph(nodes).copy()
    layout_graph = subgraph.copy()
    rng = np.random.default_rng(seed)

    helper_edges_added = 0
    attempts = 0
    while helper_edges_added < 220 and attempts < 12000:
        node_u, node_v = rng.choice(nodes, size=2, replace=False)
        if not layout_graph.has_edge(int(node_u), int(node_v)):
            layout_graph.add_edge(int(node_u), int(node_v), weight=0.22)
            helper_edges_added += 1
        attempts += 1

    initial_positions = {
        node: rng.normal(loc=0.0, scale=0.45, size=2) for node in nodes
    }

    layout = nx.spring_layout(
        layout_graph,
        pos=initial_positions,
        seed=seed,
        weight="weight",
        iterations=320,
        k=0.18,
        threshold=1e-5,
        scale=1.0,
    )
    return _normalize_cluster(
        layout,
        center=center,
        width=0.92,
        height=0.84,
        bridge_node=bridge_node,
        bridge_direction=bridge_direction,
    )


def generar_red(seed: int = MASTER_SEED):
    left_seed = seed
    right_seed = seed + 1
    left_graph = nx.watts_strogatz_graph(
        COMPONENT_SIZE, WS_NEIGHBORS, WS_REWIRE_P, seed=left_seed
    )
    right_graph = nx.watts_strogatz_graph(
        COMPONENT_SIZE, WS_NEIGHBORS, WS_REWIRE_P, seed=right_seed
    )
    right_graph = nx.relabel_nodes(
        right_graph, {node: node + COMPONENT_SIZE for node in right_graph.nodes()}
    )

    graph = nx.compose(left_graph, right_graph)
    graph.add_edge(*BRIDGE_EDGE)

    left_nodes = list(range(COMPONENT_SIZE))
    right_nodes = list(range(COMPONENT_SIZE, TOTAL_NODES))

    positions = {}
    positions.update(
        compact_component_layout(
            graph,
            left_nodes,
            seed + 41,
            LEFT_CENTER,
            bridge_node=BRIDGE_EDGE[0],
            bridge_direction="right",
        )
    )
    positions.update(
        compact_component_layout(
            graph,
            right_nodes,
            seed + 59,
            RIGHT_CENTER,
            bridge_node=BRIDGE_EDGE[1],
            bridge_direction="left",
        )
    )

    intra_left_edges = [
        edge
        for edge in graph.edges()
        if edge[0] in left_nodes and edge[1] in left_nodes
    ]
    intra_right_edges = [
        edge
        for edge in graph.edges()
        if edge[0] in right_nodes and edge[1] in right_nodes
    ]

    metadata = {
        "left_nodes": left_nodes,
        "right_nodes": right_nodes,
        "bridge_edge": BRIDGE_EDGE,
        "intra_left_edges": intra_left_edges,
        "intra_right_edges": intra_right_edges,
        "start_node": COMPONENT_SIZE // 2,
        "component_size": COMPONENT_SIZE,
        "total_nodes": TOTAL_NODES,
        "ws_neighbors": WS_NEIGHBORS,
        "ws_p": WS_REWIRE_P,
    }
    return graph, positions, metadata


def _simulate_path(
    graph: nx.Graph, start_node: int, max_time: int, seed: int
) -> List[int]:
    rng = np.random.default_rng(seed)
    path = [start_node]
    current = start_node
    for _ in range(max_time):
        neighbors = sorted(graph.neighbors(current))
        current = neighbors[int(rng.integers(0, len(neighbors)))]
        path.append(current)
    return path


def _first_cross_time(path: List[int], target_nodes: set[int]) -> int | None:
    for time_index, node in enumerate(path):
        if node in target_nodes:
            return time_index
    return None


def simular_random_walk(
    graph: nx.Graph,
    start_node: int,
    right_nodes: List[int],
    max_time: int = MAX_TIME,
    seed: int = MASTER_SEED,
):
    target_nodes = set(right_nodes)

    preferred = None
    latest_cross = None
    base_candidate = None
    best_midrange = None
    for offset in range(2400):
        candidate_seed = seed + offset
        path = _simulate_path(graph, start_node, max_time, candidate_seed)
        first_cross = _first_cross_time(path, target_nodes)
        candidate = {
            "path": path,
            "seed": candidate_seed,
            "first_cross_time": first_cross,
        }
        if base_candidate is None:
            base_candidate = candidate
        if first_cross is not None:
            if 240 <= first_cross <= max_time:
                if preferred is None or first_cross > preferred["first_cross_time"]:
                    preferred = candidate
            if 180 <= first_cross <= max_time:
                score = abs(first_cross - 300)
                if best_midrange is None or score < best_midrange["score"]:
                    best_midrange = {**candidate, "score": score}
            if latest_cross is None or first_cross > latest_cross["first_cross_time"]:
                latest_cross = candidate

    chosen = preferred or best_midrange or latest_cross or base_candidate
    return chosen


def render_frame(
    graph: nx.Graph,
    positions: Dict[int, Tuple[float, float]],
    metadata: Dict,
    walk_data: Dict,
    time_step: int,
    output_path: Path,
):
    left_nodes = metadata["left_nodes"]
    right_nodes = metadata["right_nodes"]
    bridge_edge = metadata["bridge_edge"]
    walker_node = walk_data["path"][time_step]

    fig, ax = plt.subplots(figsize=FRAME_FIGSIZE, dpi=FRAME_DPI)
    fig.patch.set_facecolor("#fbfcfe")
    ax.set_facecolor("#fbfcfe")

    # La analogia de "tunelaje" es solo visual: el puente unico actua como
    # cuello de botella estructural para un random walk en una red casi regular
    # de Watts-Strogatz. El layout es solo una vista de comunidades compactas.
    left_patch = Ellipse(
        LEFT_CENTER,
        width=2.9,
        height=2.2,
        facecolor="#dbeafe",
        edgecolor="none",
        alpha=0.17,
    )
    right_patch = Ellipse(
        RIGHT_CENTER,
        width=2.9,
        height=2.2,
        facecolor="#fde68a",
        edgecolor="none",
        alpha=0.15,
    )
    ax.add_patch(left_patch)
    ax.add_patch(right_patch)

    nx.draw_networkx_edges(
        graph,
        positions,
        edgelist=metadata["intra_left_edges"],
        ax=ax,
        edge_color="#5b8ff9",
        width=0.75,
        alpha=0.34,
    )
    nx.draw_networkx_edges(
        graph,
        positions,
        edgelist=metadata["intra_right_edges"],
        ax=ax,
        edge_color="#f5a623",
        width=0.75,
        alpha=0.34,
    )
    nx.draw_networkx_edges(
        graph,
        positions,
        edgelist=[bridge_edge],
        ax=ax,
        edge_color="#ef4444",
        width=3.2,
        alpha=0.95,
    )

    nx.draw_networkx_nodes(
        graph,
        positions,
        nodelist=left_nodes,
        ax=ax,
        node_color="#0f172a",
        node_size=26,
        linewidths=0.1,
        edgecolors="#0f172a",
        alpha=0.94,
    )
    nx.draw_networkx_nodes(
        graph,
        positions,
        nodelist=right_nodes,
        ax=ax,
        node_color="#0f172a",
        node_size=26,
        linewidths=0.1,
        edgecolors="#0f172a",
        alpha=0.94,
    )

    nx.draw_networkx_nodes(
        graph,
        positions,
        nodelist=[walker_node],
        ax=ax,
        node_color="#ef4444",
        node_size=150,
        linewidths=1.2,
        edgecolors="#ffffff",
    )

    ax.text(
        -3.03, 1.45, "Componente A", fontsize=12, fontweight="bold", color="#1d4ed8"
    )
    ax.text(1.92, 1.45, "Componente B", fontsize=12, fontweight="bold", color="#b45309")
    ax.text(
        0.0,
        1.46,
        f"t = {time_step}",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color="#111827",
        bbox={
            "boxstyle": "round,pad=0.28",
            "facecolor": "#ffffff",
            "edgecolor": "#d1d5db",
        },
    )

    ax.set_xlim(-3.55, 3.55)
    ax.set_ylim(-1.55, 1.68)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def _manifest_is_valid() -> bool:
    if not MANIFEST_PATH.exists():
        return False
    try:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except Exception:
        return False
    frame_files = manifest.get("frame_files") or []
    disk_frames = sorted(path.name for path in OUTPUT_DIR.glob("frame_*.png"))
    return (
        manifest.get("render_version") == RENDER_VERSION
        and len(frame_files) == len(FRAME_TIMES)
        and frame_files == disk_frames
        and all((OUTPUT_DIR / name).exists() for name in frame_files)
    )


def _purge_stale_frames() -> None:
    for path in OUTPUT_DIR.glob("frame_*.png"):
        path.unlink()


def _restore_frames_from_existing_gif() -> bool:
    if not GIF_PATH.exists():
        return False
    try:
        from PIL import Image
    except Exception:
        return False

    with Image.open(GIF_PATH) as gif:
        frame_count = getattr(gif, "n_frames", 1)
        if frame_count != len(FRAME_TIMES):
            return False

        _purge_stale_frames()
        for frame_index in range(frame_count):
            gif.seek(frame_index)
            frame = gif.convert("RGBA")
            frame.save(OUTPUT_DIR / f"frame_{frame_index + 1:03d}.png", format="PNG")
    return True


def exportar_frames(force: bool = False) -> Dict:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not force and _manifest_is_valid():
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    previous_manifest: Dict[str, object] = {}
    if MANIFEST_PATH.exists():
        try:
            previous_manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        except Exception:
            previous_manifest = {}

    if not force and _restore_frames_from_existing_gif():
        frame_files = [f"frame_{index:03d}.png" for index in range(1, len(FRAME_TIMES) + 1)]
        manifest = {
            "render_version": RENDER_VERSION,
            "frame_files": frame_files,
            "frame_times": FRAME_TIMES,
            "first_cross_time": previous_manifest.get("first_cross_time"),
            "walk_seed": previous_manifest.get("walk_seed", MASTER_SEED),
            "start_node": previous_manifest.get("start_node", COMPONENT_SIZE // 2),
            "bridge_edge": previous_manifest.get("bridge_edge", list(BRIDGE_EDGE)),
            "output_dir": str(OUTPUT_DIR.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "total_nodes": TOTAL_NODES,
            "component_size": COMPONENT_SIZE,
            "ws_neighbors": WS_NEIGHBORS,
            "ws_p": WS_REWIRE_P,
            "playback_interval_ms": PLAYBACK_INTERVAL_MS,
        }
        MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest

    _purge_stale_frames()
    graph, positions, metadata = generar_red()
    walk_data = simular_random_walk(
        graph, metadata["start_node"], metadata["right_nodes"]
    )

    frame_files: List[str] = []
    for index, time_step in enumerate(FRAME_TIMES, start=1):
        filename = f"frame_{index:03d}.png"
        output_path = OUTPUT_DIR / filename
        render_frame(graph, positions, metadata, walk_data, time_step, output_path)
        frame_files.append(filename)

    manifest = {
        "render_version": RENDER_VERSION,
        "frame_files": frame_files,
        "frame_times": FRAME_TIMES,
        "first_cross_time": walk_data["first_cross_time"],
        "walk_seed": walk_data["seed"],
        "start_node": metadata["start_node"],
        "bridge_edge": list(metadata["bridge_edge"]),
        "output_dir": str(OUTPUT_DIR.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "total_nodes": metadata["total_nodes"],
        "component_size": metadata["component_size"],
        "ws_neighbors": metadata["ws_neighbors"],
        "ws_p": metadata["ws_p"],
        "playback_interval_ms": PLAYBACK_INTERVAL_MS,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _payload_from_manifest() -> Dict:
    manifest = exportar_frames(force=False)
    frame_records = []
    for index, (filename, time_step) in enumerate(
        zip(manifest["frame_files"], manifest["frame_times"]), start=1
    ):
        frame_records.append(
            {
                "index": index,
                "time": time_step,
                "filename": filename,
                "url": url_for("animacion_charles_frame", filename=filename),
            }
        )

    first_cross = manifest["first_cross_time"]
    if first_cross is None:
        cross_label = "No observado hasta t = 400"
    else:
        cross_label = f"Primer cruce en t = {first_cross}"

    return {
        "frames": frame_records,
        "frame_count": len(frame_records),
        "first_cross_time": first_cross,
        "first_cross_label": cross_label,
        "walk_seed": manifest["walk_seed"],
        "start_node": manifest["start_node"],
        "bridge_edge": manifest["bridge_edge"],
        "output_dir": manifest["output_dir"],
        "network": {
            "total_nodes": manifest["total_nodes"],
            "component_size": manifest["component_size"],
            "ws_neighbors": manifest["ws_neighbors"],
            "ws_p": manifest["ws_p"],
        },
        "playback_interval_ms": manifest.get("playback_interval_ms", PLAYBACK_INTERVAL_MS),
        "cache_key": (
            f"{manifest.get('render_version', RENDER_VERSION)}-"
            f"{manifest.get('walk_seed', MASTER_SEED)}-"
            f"{len(frame_records)}"
        ),
    }


def register_animacion_charles(app):
    @app.route("/animacion_charles")
    def animacion_charles_page():
        return render_template("animacion_charles.html")

    @app.route("/process_animacion_charles", methods=["POST"])
    def process_animacion_charles():
        force = (request.form.get("force") or "").strip() == "1"
        exportar_frames(force=force)
        return jsonify(_payload_from_manifest())

    @app.route("/animacion_charles_frames/<path:filename>")
    def animacion_charles_frame(filename: str):
        return send_from_directory(OUTPUT_DIR, filename)
