from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import numpy as np
from flask import jsonify, render_template, request

DEFAULT_FRACTAL = "mandelbrot"

FRACTAL_LABELS = {
    "mandelbrot": "Conjunto de Mandelbrot",
    "julia": "Conjunto de Julia",
    "sierpinski_triangle": "Triangulo de Sierpinski",
    "sierpinski_carpet": "Carpeta de Sierpinski",
    "cantor": "Conjunto de Cantor",
    "koch": "Copo de nieve de Koch",
}

FRACTAL_CONFIG = {
    "mandelbrot": {
        "size": 560,
        "max_iter": 140,
        "formula": "Referencia teorica: dim_HB(Mandelbrot) = 2",
    },
    "julia": {
        "size": 560,
        "max_iter": 150,
        "parameter": complex(-0.8, 0.156),
        "formula": "Estimacion box-counting para c = -0.8 + 0.156i",
    },
    "sierpinski_triangle": {
        "points": 100000,
        "mask_size": 1024,
        "formula": "Referencia teorica: log(3) / log(2)",
    },
    "sierpinski_carpet": {
        "level": 5,
        "formula": "Referencia teorica: log(8) / log(3)",
    },
    "cantor": {
        "level": 7,
        "formula": "Referencia teorica: log(2) / log(3)",
    },
    "koch": {
        "level": 7,
        "mask_size": 1800,
        "formula": "Referencia teorica: log(4) / log(3)",
    },
}


def geometric_box_sizes(limit: int, base: int) -> List[int]:
    sizes: List[int] = []
    size = 1
    while size <= limit:
        sizes.append(size)
        size *= base
    return sizes


def box_count(mask: np.ndarray, box_size: int) -> int:
    height, width = mask.shape
    pad_h = (-height) % box_size
    pad_w = (-width) % box_size
    data = np.pad(mask.astype(bool), ((0, pad_h), (0, pad_w)), constant_values=False)
    reshaped = data.reshape(
        data.shape[0] // box_size,
        box_size,
        data.shape[1] // box_size,
        box_size,
    )
    occupied = reshaped.any(axis=(1, 3))
    return int(np.count_nonzero(occupied))


def box_count_1d(occupied: np.ndarray, box_size: int) -> int:
    pad = (-occupied.shape[0]) % box_size
    data = np.pad(occupied.astype(bool), (0, pad), constant_values=False)
    reshaped = data.reshape(data.shape[0] // box_size, box_size)
    return int(np.count_nonzero(reshaped.any(axis=1)))


def box_counting_dimension(mask: np.ndarray, box_sizes: Sequence[int] | None = None) -> float:
    data = np.asarray(mask, dtype=bool)
    if data.ndim == 1:
        return box_counting_dimension_1d(data, box_sizes)

    min_dim = min(data.shape)
    if box_sizes is None:
        box_sizes = []
        size = 1
        while size <= max(4, min_dim // 2):
            box_sizes.append(size)
            size *= 2

    sizes: List[int] = []
    counts: List[int] = []
    for box_size in box_sizes:
        if box_size <= 0 or box_size > min_dim:
            continue
        count = box_count(data, int(box_size))
        if count > 0:
            sizes.append(int(box_size))
            counts.append(count)

    if len(sizes) < 2:
        return 0.0

    sizes_arr = np.array(sizes, dtype=float)
    counts_arr = np.array(counts, dtype=float)
    xs = np.log(max(data.shape) / sizes_arr)
    ys = np.log(counts_arr)
    return float(np.polyfit(xs, ys, 1)[0])


def box_counting_dimension_1d(
    occupied: np.ndarray,
    box_sizes: Sequence[int] | None = None,
) -> float:
    data = np.asarray(occupied, dtype=bool).reshape(-1)
    length = data.shape[0]
    if box_sizes is None:
        box_sizes = []
        size = 1
        while size <= max(4, length // 2):
            box_sizes.append(size)
            size *= 2

    sizes: List[int] = []
    counts: List[int] = []
    for box_size in box_sizes:
        if box_size <= 0 or box_size > length:
            continue
        count = box_count_1d(data, int(box_size))
        if count > 0:
            sizes.append(int(box_size))
            counts.append(count)

    if len(sizes) < 2:
        return 0.0

    sizes_arr = np.array(sizes, dtype=float)
    counts_arr = np.array(counts, dtype=float)
    xs = np.log(length / sizes_arr)
    ys = np.log(counts_arr)
    return float(np.polyfit(xs, ys, 1)[0])


def boundary_mask(mask: np.ndarray) -> np.ndarray:
    data = np.asarray(mask, dtype=bool)
    return data & (
        (data != np.roll(data, 1, axis=0))
        | (data != np.roll(data, -1, axis=0))
        | (data != np.roll(data, 1, axis=1))
        | (data != np.roll(data, -1, axis=1))
    )


def mandelbrot_escape(size: int, max_iter: int) -> Tuple[np.ndarray, np.ndarray]:
    x = np.linspace(-2.25, 1.05, size)
    y = np.linspace(-1.55, 1.55, size)
    c = x[np.newaxis, :] + 1j * y[:, np.newaxis]
    z = np.zeros_like(c)
    escape = np.full(c.shape, float(max_iter), dtype=np.float32)
    inside = np.ones(c.shape, dtype=bool)

    for step in range(max_iter):
        z[inside] = z[inside] * z[inside] + c[inside]
        escaped = np.abs(z) > 2.0
        newly_escaped = escaped & inside
        if np.any(newly_escaped):
            magnitudes = np.maximum(np.abs(z[newly_escaped]), 2.000001)
            smooth = step + 1.0 - np.log(np.log(magnitudes)) / np.log(2.0)
            escape[newly_escaped] = smooth.astype(np.float32)
        inside &= ~escaped
        if not np.any(inside):
            break

    return escape, inside


def julia_escape(size: int, max_iter: int, parameter: complex) -> Tuple[np.ndarray, np.ndarray]:
    x = np.linspace(-1.7, 1.7, size)
    y = np.linspace(-1.7, 1.7, size)
    z = x[np.newaxis, :] + 1j * y[:, np.newaxis]
    escape = np.full(z.shape, float(max_iter), dtype=np.float32)
    inside = np.ones(z.shape, dtype=bool)

    for step in range(max_iter):
        z[inside] = z[inside] * z[inside] + parameter
        escaped = np.abs(z) > 2.0
        newly_escaped = escaped & inside
        if np.any(newly_escaped):
            magnitudes = np.maximum(np.abs(z[newly_escaped]), 2.000001)
            smooth = step + 1.0 - np.log(np.log(magnitudes)) / np.log(2.0)
            escape[newly_escaped] = smooth.astype(np.float32)
        inside &= ~escaped
        if not np.any(inside):
            break

    return escape, boundary_mask(inside)


def sierpinski_triangle_points(total_points: int) -> np.ndarray:
    rng = np.random.default_rng(2026)
    vertices = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, math.sqrt(3.0) / 2.0],
        ]
    )
    point = np.array([0.21, 0.13])
    points = np.zeros((total_points, 2), dtype=float)
    for idx in range(total_points):
        vertex = vertices[rng.integers(0, 3)]
        point = (point + vertex) / 2.0
        points[idx] = point
    return points


def points_mask(points: np.ndarray, size: int) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    xs = np.clip((points[:, 0] * (size - 1)).astype(int), 0, size - 1)
    ys = np.clip(((1.0 - points[:, 1]) * (size - 1)).astype(int), 0, size - 1)
    mask[ys, xs] = True
    return mask


def sierpinski_carpet_mask(level: int) -> np.ndarray:
    tile = np.ones((3, 3), dtype=bool)
    tile[1, 1] = False
    mask = np.ones((1, 1), dtype=bool)
    for _ in range(level):
        mask = np.kron(mask, tile)
    return mask


def cantor_occupancy(level: int) -> np.ndarray:
    length = 3**level
    occupied = np.ones(length, dtype=bool)
    for idx in range(length):
        value = idx
        keep = True
        for _ in range(level):
            if value % 3 == 1:
                keep = False
                break
            value //= 3
        occupied[idx] = keep
    return occupied


def koch_segment_points(
    start: Tuple[float, float],
    end: Tuple[float, float],
    order: int,
) -> List[Tuple[float, float]]:
    points: List[Tuple[float, float]] = [start, end]
    cos60 = 0.5
    sin60 = math.sqrt(3.0) / 2.0
    for _ in range(order):
        next_points: List[Tuple[float, float]] = [points[0]]
        for idx in range(len(points) - 1):
            x1, y1 = points[idx]
            x2, y2 = points[idx + 1]
            dx = (x2 - x1) / 3.0
            dy = (y2 - y1) / 3.0
            p1 = (x1 + dx, y1 + dy)
            p3 = (x1 + 2.0 * dx, y1 + 2.0 * dy)
            peak = (
                p1[0] + dx * cos60 + dy * sin60,
                p1[1] - dx * sin60 + dy * cos60,
            )
            next_points.extend([p1, peak, p3, (x2, y2)])
        points = next_points
    return points


def koch_snowflake_points(order: int) -> List[Tuple[float, float]]:
    left = (0.0, 0.0)
    right = (1.0, 0.0)
    top = (0.5, math.sqrt(3.0) / 2.0)
    edges = [
        koch_segment_points(left, right, order),
        koch_segment_points(right, top, order),
        koch_segment_points(top, left, order),
    ]
    path: List[Tuple[float, float]] = []
    for index, edge in enumerate(edges):
        if index == 0:
            path.extend(edge)
        else:
            path.extend(edge[1:])
    return path


def koch_mask(order: int, size: int) -> np.ndarray:
    points = np.array(koch_snowflake_points(order), dtype=float)
    min_x, min_y = points.min(axis=0)
    max_x, max_y = points.max(axis=0)
    padding = max(12.0, size * 0.05)
    usable_w = max(1.0, size - 2.0 * padding)
    usable_h = max(1.0, size - 2.0 * padding)
    span_x = max(max_x - min_x, 1e-9)
    span_y = max(max_y - min_y, 1e-9)
    scale = min(usable_w / span_x, usable_h / span_y)

    projected = np.zeros_like(points)
    projected[:, 0] = padding + (points[:, 0] - min_x) * scale
    projected[:, 1] = size - (padding + (points[:, 1] - min_y) * scale)

    mask = np.zeros((size, size), dtype=bool)
    for idx in range(len(projected) - 1):
        x1, y1 = projected[idx]
        x2, y2 = projected[idx + 1]
        steps = max(2, int(math.hypot(x2 - x1, y2 - y1) * 2.4))
        xs = np.clip(np.rint(np.linspace(x1, x2, steps)).astype(int), 0, size - 1)
        ys = np.clip(np.rint(np.linspace(y1, y2, steps)).astype(int), 0, size - 1)
        mask[ys, xs] = True
    x1, y1 = projected[-1]
    x2, y2 = projected[0]
    steps = max(2, int(math.hypot(x2 - x1, y2 - y1) * 2.4))
    xs = np.clip(np.rint(np.linspace(x1, x2, steps)).astype(int), 0, size - 1)
    ys = np.clip(np.rint(np.linspace(y1, y2, steps)).astype(int), 0, size - 1)
    mask[ys, xs] = True
    return mask


def comparison_note(estimated: float, expected: float) -> str:
    difference = abs(estimated - expected)
    return f"Esperado: {expected:.6f}. Diferencia absoluta: {difference:.6f}."


def mandelbrot_payload() -> Dict:
    size = FRACTAL_CONFIG["mandelbrot"]["size"]
    max_iter = FRACTAL_CONFIG["mandelbrot"]["max_iter"]
    escape, inside = mandelbrot_escape(size, max_iter)
    dimension = box_counting_dimension(inside, [1, 2, 4, 8, 16])
    expected = 2.0
    return {
        "label": FRACTAL_LABELS["mandelbrot"],
        "dimension": dimension,
        "expected_dimension": expected,
        "dimension_method": "Estimacion box-counting del conjunto discretizado",
        "formula": FRACTAL_CONFIG["mandelbrot"]["formula"],
        "notes": (
            "Se estima sobre la mascara del conjunto completo, no sobre una imagen externa. "
            f"{comparison_note(dimension, expected)} Al aumentar resolucion e iteraciones la "
            "estimacion deberia acercarse a 2."
        ),
        "render": {
            "kind": "escape_grid",
            "palette": "grayscale_escape",
            "width": size,
            "height": size,
            "max_iter": max_iter,
            "values": np.round(escape.reshape(-1), 4).tolist(),
        },
    }


def julia_payload() -> Dict:
    size = FRACTAL_CONFIG["julia"]["size"]
    max_iter = FRACTAL_CONFIG["julia"]["max_iter"]
    parameter = FRACTAL_CONFIG["julia"]["parameter"]
    escape, boundary = julia_escape(size, max_iter, parameter)
    dimension = box_counting_dimension(boundary, [1, 2, 4, 8, 16, 32])
    return {
        "label": FRACTAL_LABELS["julia"],
        "dimension": dimension,
        "expected_dimension": None,
        "dimension_method": "Estimacion box-counting de la frontera discretizada",
        "formula": FRACTAL_CONFIG["julia"]["formula"],
        "notes": (
            "Se estima la frontera visible del conjunto de Julia para c = -0.8 + 0.156i. "
            "Este valor no es universal: cambia con el parametro elegido, la resolucion y el "
            "numero de iteraciones."
        ),
        "render": {
            "kind": "escape_grid",
            "palette": "grayscale_escape",
            "width": size,
            "height": size,
            "max_iter": max_iter,
            "values": np.round(escape.reshape(-1), 4).tolist(),
        },
    }


def sierpinski_triangle_payload() -> Dict:
    total_points = FRACTAL_CONFIG["sierpinski_triangle"]["points"]
    mask_size = FRACTAL_CONFIG["sierpinski_triangle"]["mask_size"]
    points = sierpinski_triangle_points(total_points)
    mask = points_mask(points, mask_size)
    dimension = box_counting_dimension(mask, geometric_box_sizes(mask_size, 2))
    expected = math.log(3.0) / math.log(2.0)
    return {
        "label": FRACTAL_LABELS["sierpinski_triangle"],
        "dimension": dimension,
        "expected_dimension": expected,
        "dimension_method": "Estimacion box-counting",
        "formula": FRACTAL_CONFIG["sierpinski_triangle"]["formula"],
        "notes": (
            "Se estima a partir de puntos generados por chaos game. "
            f"{comparison_note(dimension, expected)}"
        ),
        "render": {
            "kind": "chaos_game",
            "steps": total_points,
            "seed": [0.21, 0.13],
            "vertices": [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, math.sqrt(3.0) / 2.0],
            ],
            "style": "grayscale",
        },
    }


def sierpinski_carpet_payload() -> Dict:
    level = FRACTAL_CONFIG["sierpinski_carpet"]["level"]
    mask = sierpinski_carpet_mask(level)
    dimension = box_counting_dimension(mask, geometric_box_sizes(mask.shape[0], 3))
    expected = math.log(8.0) / math.log(3.0)
    return {
        "label": FRACTAL_LABELS["sierpinski_carpet"],
        "dimension": dimension,
        "expected_dimension": expected,
        "dimension_method": "Estimacion box-counting",
        "formula": FRACTAL_CONFIG["sierpinski_carpet"]["formula"],
        "notes": (
            f"Nivel usado: {level}. {comparison_note(dimension, expected)}"
        ),
        "render": {
            "kind": "carpet",
            "level": level,
            "style": "grayscale",
        },
    }


def cantor_payload() -> Dict:
    level = FRACTAL_CONFIG["cantor"]["level"]
    occupied = cantor_occupancy(level)
    dimension = box_counting_dimension_1d(occupied, geometric_box_sizes(occupied.shape[0], 3))
    expected = math.log(2.0) / math.log(3.0)
    return {
        "label": FRACTAL_LABELS["cantor"],
        "dimension": dimension,
        "expected_dimension": expected,
        "dimension_method": "Estimacion box-counting en 1D discretizado",
        "formula": FRACTAL_CONFIG["cantor"]["formula"],
        "notes": (
            f"Nivel usado: {level}. {comparison_note(dimension, expected)}"
        ),
        "render": {
            "kind": "cantor",
            "level": level,
            "style": "grayscale",
        },
    }


def koch_payload() -> Dict:
    order = FRACTAL_CONFIG["koch"]["level"]
    mask_size = FRACTAL_CONFIG["koch"]["mask_size"]
    mask = koch_mask(order, mask_size)
    dimension = box_counting_dimension(mask, [1, 2, 4, 8, 16, 32, 64, 128, 256])
    expected = math.log(4.0) / math.log(3.0)
    return {
        "label": FRACTAL_LABELS["koch"],
        "dimension": dimension,
        "expected_dimension": expected,
        "dimension_method": "Estimacion box-counting",
        "formula": FRACTAL_CONFIG["koch"]["formula"],
        "notes": (
            f"Orden usado: {order}. {comparison_note(dimension, expected)}"
        ),
        "render": {
            "kind": "koch",
            "order": order,
            "style": "grayscale",
        },
    }


def fractal_payload(fractal: str) -> Dict:
    builders = {
        "mandelbrot": mandelbrot_payload,
        "julia": julia_payload,
        "sierpinski_triangle": sierpinski_triangle_payload,
        "sierpinski_carpet": sierpinski_carpet_payload,
        "cantor": cantor_payload,
        "koch": koch_payload,
    }
    return builders[fractal]()


def register_fractals(app):
    @app.route("/fractals")
    def fractals_page():
        return render_template(
            "fractals.html",
            default_fractal=DEFAULT_FRACTAL,
        )

    @app.route("/process_fractals", methods=["POST"])
    def process_fractals():
        fractal = (request.form.get("fractal") or DEFAULT_FRACTAL).strip()
        if fractal not in FRACTAL_LABELS:
            return jsonify({"error": "Fractal no soportado"}), 400

        payload = fractal_payload(fractal)
        payload["fractal"] = fractal
        return jsonify(payload)
