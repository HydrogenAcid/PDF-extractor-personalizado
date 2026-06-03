from __future__ import annotations

import csv
import math
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from flask import jsonify, render_template, request

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = BASE_DIR / "output" / "beta_series"

DEFAULT_BETA_MIN = -3.0
DEFAULT_BETA_MAX = 4.5
DEFAULT_BETA_STEP = 0.5
DEFAULT_SERIES_LENGTH = 5000
DEFAULT_REALIZATIONS = 10
DEFAULT_SEED = 20260526

ESTIMATOR_LABELS = {
    "dfa": "DFA alpha",
    "hurst_rs": "Hurst R/S",
    "higuchi": "Higuchi D",
    "box_counting": "Box-counting D de la grafica",
}


def relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(BASE_DIR).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def parse_float(name: str, default: float) -> float:
    raw = request.form.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    return float(raw)


def parse_int(name: str, default: int) -> int:
    raw = request.form.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    return int(float(raw))


def clamp(value: int, min_value: int, max_value: int) -> int:
    return max(min_value, min(max_value, value))


def build_run_dir(prefix: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUT_ROOT / f"{prefix}_{stamp}"
    suffix = 1
    while run_dir.exists():
        run_dir = OUTPUT_ROOT / f"{prefix}_{stamp}_{suffix}"
        suffix += 1
    (run_dir / "plots").mkdir(parents=True, exist_ok=True)
    return run_dir


def float_grid(start: float, stop: float, step: float, max_points: int) -> List[float]:
    if step <= 0:
        raise ValueError("El paso debe ser positivo")
    if stop < start:
        raise ValueError("El beta maximo debe ser mayor o igual al beta minimo")

    values: List[float] = []
    current = start
    epsilon = abs(step) * 1e-8
    while current <= stop + epsilon:
        values.append(round(current, 10))
        current += step

    if values and abs(values[-1] - stop) > epsilon and values[-1] < stop:
        values.append(round(stop, 10))

    if len(values) > max_points:
        raise ValueError(f"Demasiados puntos en el barrido; maximo permitido: {max_points}")
    return values


def beta_simulation(beta: float, n: int, rng: np.random.Generator) -> np.ndarray:
    x_values = rng.normal(0.0, 1.0, n)
    spectrum_source = np.fft.fft(x_values)
    filtered = np.zeros(n, dtype=complex)
    indices = np.arange(1, n // 2)
    if indices.size:
        weights = (indices.astype(float) / float(n)) ** (-beta / 2.0)
        values = spectrum_source[indices] * weights
        filtered[indices] = values
        filtered[n - indices - 1] = values
    series = np.fft.ifft(filtered).real
    return standardize(series)


def standardize(series: Sequence[float]) -> np.ndarray:
    values = np.asarray(series, dtype=float)
    centered = values - float(np.mean(values))
    scale = float(np.std(centered))
    if scale <= 0 or not math.isfinite(scale):
        return centered
    return centered / scale


def choose_scales(min_scale: int, max_scale: int, max_points: int) -> List[int]:
    if max_scale < min_scale:
        return []
    if max_scale - min_scale + 1 <= max_points:
        return list(range(min_scale, max_scale + 1))
    raw = np.unique(
        np.floor(np.logspace(np.log10(min_scale), np.log10(max_scale), max_points)).astype(int)
    )
    scales = [int(value) for value in raw if min_scale <= value <= max_scale]
    if scales and scales[-1] != max_scale:
        scales.append(max_scale)
    return sorted(set(scales))


def detrended_variances(segments: np.ndarray, poly_degree: int) -> np.ndarray:
    segment_count, scale = segments.shape
    if segment_count == 0:
        return np.array([], dtype=float)
    x_axis = np.arange(scale, dtype=float)
    coefficients = np.polyfit(x_axis, segments.T, poly_degree)
    trend = np.polyval(coefficients, x_axis[:, np.newaxis]).T
    return np.mean((segments - trend) ** 2, axis=1)


def estimate_dfa_alpha(series: Sequence[float], poly_degree: int = 1) -> float:
    values = standardize(series)
    n = values.size
    if n < 64:
        raise ValueError("La serie debe tener al menos 64 puntos para DFA")

    profile = np.cumsum(values - float(np.mean(values)))
    scales = choose_scales(16, max(24, n // 4), 11)
    usable_scales: List[int] = []
    fluctuations: List[float] = []

    for scale in scales:
        segment_count = n // scale
        if segment_count < 2:
            continue
        segments = profile[: segment_count * scale].reshape(segment_count, scale)
        variances = detrended_variances(segments, poly_degree)
        variances = variances[np.isfinite(variances) & (variances > 0)]
        if variances.size:
            usable_scales.append(scale)
            fluctuations.append(float(np.sqrt(np.mean(variances))))

    if len(usable_scales) < 2:
        raise ValueError("No hubo suficientes escalas para estimar DFA")

    slope, _ = np.polyfit(np.log(usable_scales), np.log(fluctuations), 1)
    return float(slope)


def estimate_hurst_rs(series: Sequence[float]) -> float:
    values = standardize(series)
    n = values.size
    if n < 64:
        raise ValueError("La serie debe tener al menos 64 puntos para Hurst R/S")

    scales = choose_scales(16, max(24, n // 4), 11)
    usable_scales: List[int] = []
    rs_values: List[float] = []

    for scale in scales:
        segment_count = n // scale
        if segment_count < 2:
            continue
        segments = values[: segment_count * scale].reshape(segment_count, scale)
        centered = segments - np.mean(segments, axis=1, keepdims=True)
        profiles = np.cumsum(centered, axis=1)
        ranges = np.max(profiles, axis=1) - np.min(profiles, axis=1)
        std_values = np.std(centered, axis=1)
        valid = std_values > 0
        if np.any(valid):
            usable_scales.append(scale)
            rs_values.append(float(np.mean(ranges[valid] / std_values[valid])))

    if len(usable_scales) < 2:
        raise ValueError("No hubo suficientes escalas para estimar Hurst R/S")

    slope, _ = np.polyfit(np.log(usable_scales), np.log(rs_values), 1)
    return float(slope)


def estimate_higuchi_dimension(series: Sequence[float]) -> float:
    values = standardize(series)
    n = values.size
    if n < 64:
        raise ValueError("La serie debe tener al menos 64 puntos para Higuchi")

    k_values = choose_scales(1, min(64, max(4, n // 4)), 16)
    usable_k: List[int] = []
    lengths: List[float] = []

    for k_value in k_values:
        segment_lengths: List[float] = []
        for offset in range(k_value):
            sampled = values[offset:n:k_value]
            max_i = sampled.size - 1
            if max_i < 1:
                continue
            total = float(np.sum(np.abs(np.diff(sampled))))
            length = ((n - 1) / (max_i * k_value)) * (total / k_value)
            if length > 0 and math.isfinite(length):
                segment_lengths.append(length)
        if segment_lengths:
            usable_k.append(k_value)
            lengths.append(float(np.mean(segment_lengths)))

    if len(usable_k) < 2:
        raise ValueError("No hubo suficientes escalas para estimar Higuchi")

    slope, _ = np.polyfit(np.log(usable_k), np.log(lengths), 1)
    return float(-slope)


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
    return int(np.count_nonzero(reshaped.any(axis=(1, 3))))


def estimate_graph_box_counting_dimension(series: Sequence[float], grid_size: int = 512) -> float:
    values = standardize(series)
    n = values.size
    value_min = float(np.min(values))
    value_max = float(np.max(values))
    if value_max == value_min:
        return 1.0

    x_coords = np.linspace(0, grid_size - 1, n)
    y_coords = (1.0 - (values - value_min) / (value_max - value_min)) * (grid_size - 1)
    mask = np.zeros((grid_size, grid_size), dtype=bool)

    for idx in range(n - 1):
        x1, y1 = x_coords[idx], y_coords[idx]
        x2, y2 = x_coords[idx + 1], y_coords[idx + 1]
        steps = max(2, int(max(abs(x2 - x1), abs(y2 - y1))) + 1)
        xs = np.clip(np.rint(np.linspace(x1, x2, steps)).astype(int), 0, grid_size - 1)
        ys = np.clip(np.rint(np.linspace(y1, y2, steps)).astype(int), 0, grid_size - 1)
        mask[ys, xs] = True

    sizes = [1, 2, 4, 8, 16, 32, 64, 128]
    usable_sizes: List[int] = []
    counts: List[int] = []
    for size in sizes:
        count = box_count(mask, size)
        if count > 0:
            usable_sizes.append(size)
            counts.append(count)

    if len(usable_sizes) < 2:
        return 0.0
    slope, _ = np.polyfit(np.log(grid_size / np.asarray(usable_sizes)), np.log(counts), 1)
    return float(slope)


ESTIMATORS: Dict[str, Callable[[Sequence[float]], float]] = {
    "dfa": estimate_dfa_alpha,
    "hurst_rs": estimate_hurst_rs,
    "higuchi": estimate_higuchi_dimension,
    "box_counting": estimate_graph_box_counting_dimension,
}


def beta_exponent_payload() -> Dict[str, object]:
    beta_min = parse_float("beta_min", DEFAULT_BETA_MIN)
    beta_max = parse_float("beta_max", DEFAULT_BETA_MAX)
    beta_step = parse_float("beta_step", DEFAULT_BETA_STEP)
    length_n = clamp(parse_int("length_n", DEFAULT_SERIES_LENGTH), 512, 20000)
    realizations = clamp(parse_int("realizations", DEFAULT_REALIZATIONS), 1, 50)
    seed = parse_int("seed", DEFAULT_SEED)
    estimator_key = (request.form.get("estimator") or "dfa").strip()

    if estimator_key not in ESTIMATORS:
        raise ValueError("Exponente no soportado")

    beta_values = float_grid(beta_min, beta_max, beta_step, 41)
    estimator = ESTIMATORS[estimator_key]
    rng = np.random.default_rng(seed)

    rows: List[Dict[str, object]] = []
    for beta in beta_values:
        estimates = []
        for _ in range(realizations):
            series = beta_simulation(beta, length_n, rng)
            estimates.append(float(estimator(series)))
        estimates_arr = np.asarray(estimates, dtype=float)
        rows.append(
            {
                "beta": float(beta),
                "mean": float(np.mean(estimates_arr)),
                "std": float(np.std(estimates_arr, ddof=1)) if realizations > 1 else 0.0,
                "min": float(np.min(estimates_arr)),
                "max": float(np.max(estimates_arr)),
            }
        )

    run_dir = build_run_dir("task1_beta_exponent")
    csv_path = run_dir / "beta_exponent_summary.csv"
    plot_path = run_dir / "plots" / f"beta_vs_{estimator_key}.png"
    report_path = run_dir / f"reporte_tarea1_{estimator_key}.tex"

    write_beta_csv(csv_path, rows, ESTIMATOR_LABELS[estimator_key])
    write_beta_plot(plot_path, rows, estimator_key)
    write_beta_report(
        report_path,
        rows,
        plot_path,
        estimator_key,
        length_n,
        realizations,
        seed,
        beta_min,
        beta_max,
        beta_step,
    )

    return {
        "params": {
            "beta_min": beta_min,
            "beta_max": beta_max,
            "beta_step": beta_step,
            "length_n": length_n,
            "realizations": realizations,
            "seed": seed,
            "estimator": estimator_key,
            "estimator_label": ESTIMATOR_LABELS[estimator_key],
        },
        "rows": rows,
        "files": {
            "run_dir": relative_path(run_dir),
            "csv": relative_path(csv_path),
            "plot": relative_path(plot_path),
            "latex": relative_path(report_path),
        },
        "notes": (
            "La simulacion conserva el filtrado espectral del notebook Beta_Series.ipynb. "
            "Cada punto es el promedio sobre las realizaciones indicadas."
        ),
    }


def write_beta_csv(path: Path, rows: Sequence[Dict[str, object]], estimator_label: str) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["beta", f"mean_{estimator_label}", "std", "min", "max"])
        for row in rows:
            writer.writerow([row["beta"], row["mean"], row["std"], row["min"], row["max"]])


def write_beta_plot(path: Path, rows: Sequence[Dict[str, object]], estimator_key: str) -> None:
    beta_values = np.asarray([row["beta"] for row in rows], dtype=float)
    means = np.asarray([row["mean"] for row in rows], dtype=float)
    stds = np.asarray([row["std"] for row in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.errorbar(beta_values, means, yerr=stds, marker="o", linewidth=2, capsize=3)
    if estimator_key == "dfa":
        reference_mask = (beta_values >= -1.0) & (beta_values <= 3.0)
        reference_beta = beta_values[reference_mask]
        theoretical = (reference_beta + 1.0) / 2.0
        ax.plot(
            reference_beta,
            theoretical,
            linestyle="--",
            linewidth=1.4,
            label="Referencia central (beta + 1) / 2",
        )
        ax.legend()
    ax.set_xlabel("beta")
    ax.set_ylabel(ESTIMATOR_LABELS[estimator_key])
    ax.set_title(f"{ESTIMATOR_LABELS[estimator_key]} promedio vs beta")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def latex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
        "{": r"\{",
        "}": r"\}",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return text


def write_beta_report(
    path: Path,
    rows: Sequence[Dict[str, object]],
    plot_path: Path,
    estimator_key: str,
    length_n: int,
    realizations: int,
    seed: int,
    beta_min: float,
    beta_max: float,
    beta_step: float,
) -> None:
    plot_ref = plot_path.relative_to(path.parent).as_posix()
    table_rows = "\n".join(
        f"{row['beta']:.4f} & {row['mean']:.6f} & {row['std']:.6f} & "
        f"{row['min']:.6f} & {row['max']:.6f} \\\\"
        for row in rows
    )
    label = latex_escape(ESTIMATOR_LABELS[estimator_key])
    content = rf"""\documentclass[11pt]{{article}}
\usepackage[margin=2.2cm]{{geometry}}
\usepackage{{graphicx}}

\title{{Tarea 1: barrido de beta y {label}}}
\author{{Generado por la aplicacion Flask}}
\date{{\today}}

\begin{{document}}
\maketitle

\section*{{Objetivo}}
Se genero una familia de series sinteticas mediante filtrado espectral controlado por beta,
siguiendo la estructura de \texttt{{Beta\_Series.ipynb}}. Para cada beta se estimo el
exponente {label} y se reporto el promedio sobre {realizations} realizaciones de longitud
{length_n}.

\section*{{Parametros}}
\begin{{itemize}}
\item Rango de beta: [{beta_min:.4f}, {beta_max:.4f}] con paso {beta_step:.4f}.
\item Realizaciones por beta: {realizations}.
\item Longitud por serie: {length_n}.
\item Semilla aleatoria: {seed}.
\end{{itemize}}

\section*{{Grafica}}
\begin{{figure}}[htbp]
\centering
\includegraphics[width=0.88\linewidth]{{{plot_ref}}}
\caption{{Promedio de {label} contra beta. Las barras indican una desviacion estandar.}}
\end{{figure}}

\section*{{Resultados}}
\begin{{center}}
\begin{{tabular}}{{rrrrr}}
beta & promedio & desv. est. & minimo & maximo \\
\hline
{table_rows}
\end{{tabular}}
\end{{center}}

\section*{{Nota de interpretacion}}
Para DFA, la comparacion lineal \(\alpha \approx (\beta+1)/2\) es una referencia util
en el regimen central. En los extremos del barrido, especialmente \(\beta < -1\) y
\(\beta > 3\), la estimacion con DFA de primer orden y series finitas puede saturarse:
los exponentes negativos no son recuperados por DFA estandar y los exponentes muy
altos quedan dominados por tendencias suaves de baja frecuencia.

\end{{document}}
"""
    path.write_text(content, encoding="utf-8")


def mfdfa(
    series: Sequence[float],
    q_values: Sequence[float],
    scales: Sequence[int],
    poly_degree: int,
) -> Dict[str, object]:
    values = standardize(series)
    n = values.size
    profile = np.cumsum(values - float(np.mean(values)))
    q_arr = np.asarray(q_values, dtype=float)
    scales_arr = np.asarray(scales, dtype=int)
    fluctuation = np.full((q_arr.size, scales_arr.size), np.nan, dtype=float)

    for scale_index, scale in enumerate(scales_arr):
        segment_count = n // int(scale)
        if segment_count < 2:
            continue
        forward = profile[: segment_count * scale].reshape(segment_count, scale)
        backward = profile[n - segment_count * scale :].reshape(segment_count, scale)
        segments = np.vstack([forward, backward])
        variances = detrended_variances(segments, poly_degree)
        variances = variances[np.isfinite(variances) & (variances > 1e-30)]
        if variances.size == 0:
            continue

        for q_index, q_value in enumerate(q_arr):
            if abs(q_value) < 1e-12:
                fluctuation[q_index, scale_index] = float(
                    np.exp(0.5 * np.mean(np.log(variances)))
                )
            else:
                fluctuation[q_index, scale_index] = float(
                    np.mean(variances ** (q_value / 2.0)) ** (1.0 / q_value)
                )

    hq: List[float] = []
    intercepts: List[float] = []
    for q_index in range(q_arr.size):
        values_q = fluctuation[q_index]
        valid = np.isfinite(values_q) & (values_q > 0)
        if np.count_nonzero(valid) < 2:
            hq.append(float("nan"))
            intercepts.append(float("nan"))
            continue
        slope, intercept = np.polyfit(np.log(scales_arr[valid]), np.log(values_q[valid]), 1)
        hq.append(float(slope))
        intercepts.append(float(intercept))

    hq_arr = np.asarray(hq, dtype=float)
    tau = q_arr * hq_arr - 1.0
    holder_alpha = np.gradient(tau, q_arr)

    return {
        "q_values": q_arr,
        "scales": scales_arr,
        "fluctuation": fluctuation,
        "hq": hq_arr,
        "intercepts": np.asarray(intercepts, dtype=float),
        "tau": tau,
        "holder_alpha": holder_alpha,
    }


def mfdfa_payload() -> Dict[str, object]:
    length_n = clamp(parse_int("mfdfa_length_n", DEFAULT_SERIES_LENGTH), 512, 50000)
    q_min = parse_float("q_min", -5.0)
    q_max = parse_float("q_max", 5.0)
    q_step = parse_float("q_step", 0.5)
    seed = parse_int("mfdfa_seed", DEFAULT_SEED + 1)
    poly_degree = clamp(parse_int("poly_degree", 1), 1, 3)

    q_values = float_grid(q_min, q_max, q_step, 41)
    if q_min <= 0 <= q_max and all(abs(q) > 1e-12 for q in q_values):
        q_values = sorted(q_values + [0.0])
    if len(q_values) < 3:
        raise ValueError("MF-DFA requiere al menos tres valores de q")

    rng = np.random.default_rng(seed)
    white_noise = rng.normal(0.0, 1.0, length_n)
    scales = choose_scales(16, max(24, length_n // 4), 18)
    result = mfdfa(white_noise, q_values, scales, poly_degree)

    q_arr = result["q_values"]
    hq = result["hq"]
    tau = result["tau"]
    holder_alpha = result["holder_alpha"]
    q0_index = int(np.argmin(np.abs(q_arr)))
    q2_index = int(np.argmin(np.abs(q_arr - 2.0)))

    rows = [
        {
            "q": float(q_arr[idx]),
            "h": float(hq[idx]),
            "tau": float(tau[idx]),
            "holder_alpha": float(holder_alpha[idx]),
        }
        for idx in range(q_arr.size)
    ]

    run_dir = build_run_dir("task2_mfdfa_holder")
    csv_path = run_dir / "mfdfa_holder_white_noise.csv"
    series_plot = run_dir / "plots" / "white_noise.png"
    hq_plot = run_dir / "plots" / "mfdfa_hq.png"
    alpha_plot = run_dir / "plots" / "mfdfa_holder_alpha.png"
    fluctuation_plot = run_dir / "plots" / "mfdfa_fluctuations.png"
    report_path = run_dir / "reporte_tarea2_mfdfa_holder.tex"

    write_mfdfa_csv(csv_path, rows)
    write_mfdfa_plots(
        white_noise,
        result,
        series_plot,
        hq_plot,
        alpha_plot,
        fluctuation_plot,
    )
    write_mfdfa_report(
        report_path,
        rows,
        series_plot,
        hq_plot,
        alpha_plot,
        fluctuation_plot,
        length_n,
        seed,
        poly_degree,
        float(holder_alpha[q0_index]),
        float(holder_alpha[q2_index]),
        float(hq[q0_index]),
        float(hq[q2_index]),
    )

    return {
        "params": {
            "length_n": length_n,
            "q_min": q_min,
            "q_max": q_max,
            "q_step": q_step,
            "seed": seed,
            "poly_degree": poly_degree,
            "scales": [int(scale) for scale in result["scales"]],
        },
        "rows": rows,
        "summary": {
            "holder_alpha_q0": float(holder_alpha[q0_index]),
            "holder_alpha_q2": float(holder_alpha[q2_index]),
            "hq_q0": float(hq[q0_index]),
            "hq_q2": float(hq[q2_index]),
            "expected_white_noise_h": 0.5,
        },
        "series": {
            "index": downsample_indices(length_n, 520),
            "values": downsample_values(white_noise, 520),
        },
        "fluctuations": build_fluctuation_payload(result),
        "files": {
            "run_dir": relative_path(run_dir),
            "csv": relative_path(csv_path),
            "series_plot": relative_path(series_plot),
            "hq_plot": relative_path(hq_plot),
            "alpha_plot": relative_path(alpha_plot),
            "fluctuation_plot": relative_path(fluctuation_plot),
            "latex": relative_path(report_path),
        },
    }


def downsample_indices(length: int, max_points: int) -> List[int]:
    if length <= max_points:
        return list(range(length))
    return np.linspace(0, length - 1, max_points, dtype=int).tolist()


def downsample_values(values: Sequence[float], max_points: int) -> List[float]:
    data = np.asarray(values, dtype=float)
    if data.size <= max_points:
        return np.round(data, 6).tolist()
    indices = np.linspace(0, data.size - 1, max_points, dtype=int)
    return np.round(data[indices], 6).tolist()


def build_fluctuation_payload(result: Dict[str, object]) -> List[Dict[str, object]]:
    q_arr = result["q_values"]
    scales = result["scales"]
    fluctuation = result["fluctuation"]
    hq = result["hq"]
    intercepts = result["intercepts"]

    targets = [float(q_arr[0]), 0.0, float(q_arr[-1])]
    selected_indices = sorted({int(np.argmin(np.abs(q_arr - target))) for target in targets})
    payload: List[Dict[str, object]] = []
    for idx in selected_indices:
        q_value = float(q_arr[idx])
        values_q = fluctuation[idx]
        points = [
            {"x": int(scale), "y": float(value)}
            for scale, value in zip(scales, values_q)
            if math.isfinite(float(value)) and float(value) > 0
        ]
        fit = [
            {
                "x": int(scale),
                "y": float(math.exp(float(intercepts[idx])) * (int(scale) ** float(hq[idx]))),
            }
            for scale in scales
            if math.isfinite(float(hq[idx])) and math.isfinite(float(intercepts[idx]))
        ]
        payload.append({"q": q_value, "points": points, "fit": fit})
    return payload


def write_mfdfa_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["q", "h_q", "tau_q", "holder_alpha"])
        for row in rows:
            writer.writerow([row["q"], row["h"], row["tau"], row["holder_alpha"]])


def write_mfdfa_plots(
    white_noise: Sequence[float],
    result: Dict[str, object],
    series_plot: Path,
    hq_plot: Path,
    alpha_plot: Path,
    fluctuation_plot: Path,
) -> None:
    q_arr = result["q_values"]
    scales = result["scales"]
    fluctuation = result["fluctuation"]
    hq = result["hq"]
    intercepts = result["intercepts"]
    holder_alpha = result["holder_alpha"]

    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    ax.plot(np.asarray(white_noise), linewidth=0.8)
    ax.set_title("Ruido blanco N(0,1)")
    ax.set_xlabel("Indice")
    ax.set_ylabel("Valor")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(series_plot, dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(q_arr, hq, marker="o", linewidth=2)
    ax.axhline(0.5, linestyle="--", color="#666666", linewidth=1.2, label="Referencia ruido blanco")
    ax.set_title("Exponente generalizado h(q)")
    ax.set_xlabel("q")
    ax.set_ylabel("h(q)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(hq_plot, dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(q_arr, holder_alpha, marker="o", linewidth=2)
    ax.axhline(0.5, linestyle="--", color="#666666", linewidth=1.2, label="Referencia ruido blanco")
    ax.set_title("Exponente de Holder alpha(q)")
    ax.set_xlabel("q")
    ax.set_ylabel("alpha(q)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(alpha_plot, dpi=170)
    plt.close(fig)

    targets = [float(q_arr[0]), 0.0, float(q_arr[-1])]
    selected_indices = sorted({int(np.argmin(np.abs(q_arr - target))) for target in targets})
    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    for idx in selected_indices:
        values_q = fluctuation[idx]
        valid = np.isfinite(values_q) & (values_q > 0)
        ax.loglog(scales[valid], values_q[valid], marker="o", linewidth=0, label=f"q={q_arr[idx]:.2f}")
        fit_values = np.exp(intercepts[idx]) * (scales.astype(float) ** hq[idx])
        ax.loglog(scales, fit_values, linewidth=1.4)
    ax.set_title("Escalamiento MF-DFA")
    ax.set_xlabel("s")
    ax.set_ylabel("F_q(s)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fluctuation_plot, dpi=170)
    plt.close(fig)


def write_mfdfa_report(
    path: Path,
    rows: Sequence[Dict[str, object]],
    series_plot: Path,
    hq_plot: Path,
    alpha_plot: Path,
    fluctuation_plot: Path,
    length_n: int,
    seed: int,
    poly_degree: int,
    holder_alpha_q0: float,
    holder_alpha_q2: float,
    hq_q0: float,
    hq_q2: float,
) -> None:
    refs = [series_plot, hq_plot, alpha_plot, fluctuation_plot]
    rel_refs = [item.relative_to(path.parent).as_posix() for item in refs]
    table_rows = "\n".join(
        f"{row['q']:.4f} & {row['h']:.6f} & {row['tau']:.6f} & {row['holder_alpha']:.6f} \\\\"
        for row in rows
    )
    content = rf"""\documentclass[11pt]{{article}}
\usepackage[margin=2.2cm]{{geometry}}
\usepackage{{graphicx}}

\title{{Tarea 2: exponente de Holder por MF-DFA}}
\author{{Generado por la aplicacion Flask}}
\date{{\today}}

\begin{{document}}
\maketitle

\section*{{Objetivo}}
Se aplico Multifractal Detrended Fluctuation Analysis (MF-DFA) a un ruido blanco
gaussiano para estimar el exponente de Holder. A partir de \(h(q)\), obtenido de
\(F_q(s) \sim s^{{h(q)}}\), se calculo \(\tau(q) = qh(q)-1\) y
\(\alpha(q) = d\tau(q)/dq\).

\section*{{Parametros}}
\begin{{itemize}}
\item Longitud de la serie: {length_n}.
\item Semilla aleatoria: {seed}.
\item Grado polinomial de detrending: {poly_degree}.
\item h(q=0) estimado: {hq_q0:.6f}.
\item h(q=2) estimado: {hq_q2:.6f}.
\item Holder alpha en q=0: {holder_alpha_q0:.6f}.
\item Holder alpha en q=2: {holder_alpha_q2:.6f}.
\end{{itemize}}

\section*{{Graficas}}
\begin{{figure}}[htbp]
\centering
\includegraphics[width=0.84\linewidth]{{{rel_refs[0]}}}
\caption{{Ruido blanco usado como entrada.}}
\end{{figure}}

\begin{{figure}}[htbp]
\centering
\includegraphics[width=0.78\linewidth]{{{rel_refs[1]}}}
\caption{{Exponente generalizado h(q).}}
\end{{figure}}

\begin{{figure}}[htbp]
\centering
\includegraphics[width=0.78\linewidth]{{{rel_refs[2]}}}
\caption{{Exponente de Holder alpha(q).}}
\end{{figure}}

\begin{{figure}}[htbp]
\centering
\includegraphics[width=0.78\linewidth]{{{rel_refs[3]}}}
\caption{{Regresiones log-log de MF-DFA para valores representativos de q.}}
\end{{figure}}

\section*{{Resultados}}
\begin{{center}}
\begin{{tabular}}{{rrrr}}
q & h(q) & tau(q) & alpha(q) \\
\hline
{table_rows}
\end{{tabular}}
\end{{center}}

\end{{document}}
"""
    path.write_text(content, encoding="utf-8")


def register_beta_series(app):
    @app.route("/beta_series")
    def beta_series_page():
        return render_template(
            "beta_series.html",
            estimators=ESTIMATOR_LABELS,
            defaults={
                "beta_min": DEFAULT_BETA_MIN,
                "beta_max": DEFAULT_BETA_MAX,
                "beta_step": DEFAULT_BETA_STEP,
                "length_n": DEFAULT_SERIES_LENGTH,
                "realizations": DEFAULT_REALIZATIONS,
                "seed": DEFAULT_SEED,
            },
        )

    @app.route("/process_beta_exponent", methods=["POST"])
    def process_beta_exponent():
        try:
            payload = beta_exponent_payload()
            return jsonify(payload)
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

    @app.route("/process_mfdfa", methods=["POST"])
    def process_mfdfa():
        try:
            payload = mfdfa_payload()
            return jsonify(payload)
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400
