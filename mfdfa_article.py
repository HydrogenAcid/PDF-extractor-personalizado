from __future__ import annotations

import math
from typing import Dict, List, Sequence

import numpy as np
from flask import jsonify, render_template, request

from beta_series import choose_scales, detrended_variances, standardize

DEFAULT_SERIES_LENGTH = 65536
DEFAULT_SEED = 20260601
DEFAULT_Q_MIN = -20.0
DEFAULT_Q_MAX = 20.0
DEFAULT_Q_STEP = 1.0
DEFAULT_FIG1_FIT_MIN = 200
DEFAULT_FIG1_FIT_MAX = 5000
DEFAULT_BINOMIAL_FIT_MIN = 50
DEFAULT_BINOMIAL_FIT_MAX = 500
SELECTED_Q_VALUES = [-10.0, -2.0, -0.2, 0.2, 2.0, 10.0]
MONOFRACTAL_CASES = [
    {"key": "h075", "hurst": 0.75, "label": "H = 0.75"},
    {"key": "h050", "hurst": 0.50, "label": "H = 0.50"},
    {"key": "h025", "hurst": 0.25, "label": "H = 0.25"},
]


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


def clamp(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, value))


def q_grid(q_min: float, q_max: float, q_step: float, extra: Sequence[float] = ()) -> np.ndarray:
    if q_step <= 0:
        raise ValueError("El paso de q debe ser positivo")
    if q_max < q_min:
        raise ValueError("q max debe ser mayor que q min")

    values: List[float] = []
    current = q_min
    epsilon = q_step * 1e-8
    while current <= q_max + epsilon:
        values.append(round(current, 10))
        current += q_step
    if q_min <= 0 <= q_max and all(abs(item) > 1e-12 for item in values):
        values.append(0.0)
    for item in extra:
        if q_min <= item <= q_max:
            values.append(float(item))
    return np.asarray(sorted(set(values)), dtype=float)


def spectral_mono_series(hurst: float, n: int, rng: np.random.Generator) -> np.ndarray:
    beta = 2.0 * hurst - 1.0
    values = rng.normal(0.0, 1.0, n)
    spectrum = np.fft.rfft(values)
    freqs = np.fft.rfftfreq(n)
    weights = np.ones_like(freqs)
    valid = freqs > 0
    weights[valid] = freqs[valid] ** (-beta / 2.0)
    weights[0] = 0.0
    return standardize(np.fft.irfft(spectrum * weights, n=n))


def mfdfa_fluctuations(
    series: Sequence[float],
    q_values: Sequence[float],
    scales: Sequence[int],
    poly_degree: int,
    fit_min: int,
    fit_max: int,
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

    fit_mask = (scales_arr >= fit_min) & (scales_arr <= fit_max)
    if np.count_nonzero(fit_mask) < 2:
        fit_mask = np.ones_like(scales_arr, dtype=bool)

    hq: List[float] = []
    intercepts: List[float] = []
    for q_index in range(q_arr.size):
        values_q = fluctuation[q_index]
        valid = fit_mask & np.isfinite(values_q) & (values_q > 0)
        if np.count_nonzero(valid) < 2:
            hq.append(float("nan"))
            intercepts.append(float("nan"))
            continue
        slope, intercept = np.polyfit(np.log(scales_arr[valid]), np.log(values_q[valid]), 1)
        hq.append(float(slope))
        intercepts.append(float(intercept))

    hq_arr = np.asarray(hq, dtype=float)
    return {
        "q_values": q_arr,
        "scales": scales_arr,
        "fluctuation": fluctuation,
        "hq": hq_arr,
        "intercepts": np.asarray(intercepts, dtype=float),
        "tau": q_arr * hq_arr - 1.0,
    }


def points_from_arrays(x_values: Sequence[float], y_values: Sequence[float]) -> List[Dict[str, float]]:
    points: List[Dict[str, float]] = []
    for x_value, y_value in zip(x_values, y_values):
        if math.isfinite(float(x_value)) and math.isfinite(float(y_value)):
            points.append({"x": float(x_value), "y": float(y_value)})
    return points


def sorted_points_from_arrays(x_values: Sequence[float], y_values: Sequence[float]) -> List[Dict[str, float]]:
    return sorted(points_from_arrays(x_values, y_values), key=lambda point: point["x"])


def physical_spectrum_points(
    alpha_values: Sequence[float],
    f_alpha_values: Sequence[float],
    alpha_min: float,
    alpha_max: float,
) -> List[Dict[str, float]]:
    points: List[Dict[str, float]] = []
    lower = alpha_min - 0.02
    upper = alpha_max + 0.02
    for alpha, f_alpha in zip(alpha_values, f_alpha_values):
        alpha_float = float(alpha)
        f_float = float(f_alpha)
        if (
            math.isfinite(alpha_float)
            and math.isfinite(f_float)
            and lower <= alpha_float <= upper
            and 0.0 <= f_float <= 1.05
        ):
            points.append({"x": alpha_float, "y": f_float})
    return sorted(points, key=lambda point: point["x"])


def fluctuation_curves(result: Dict[str, object], selected_q: Sequence[float]) -> List[Dict[str, object]]:
    q_arr = result["q_values"]
    scales = result["scales"]
    fluctuation = result["fluctuation"]
    hq = result["hq"]
    intercepts = result["intercepts"]
    curves: List[Dict[str, object]] = []

    for shift_index, q_value in enumerate(selected_q):
        q_index = int(np.argmin(np.abs(q_arr - q_value)))
        shift = 4.0 ** shift_index
        values_q = fluctuation[q_index]
        point_data = []
        fit_data = []
        for scale, value in zip(scales, values_q):
            if math.isfinite(float(value)) and float(value) > 0:
                point_data.append({"x": int(scale), "y": float(value * shift)})
        if math.isfinite(float(hq[q_index])) and math.isfinite(float(intercepts[q_index])):
            for scale in scales:
                fit_value = math.exp(float(intercepts[q_index])) * (float(scale) ** float(hq[q_index]))
                fit_data.append({"x": int(scale), "y": float(fit_value * shift)})
        curves.append(
            {
                "q": float(q_arr[q_index]),
                "shift": shift,
                "points": point_data,
                "fit": fit_data,
            }
        )
    return curves


def build_figure1_payload() -> Dict[str, object]:
    n = clamp(parse_int("fig1_n", DEFAULT_SERIES_LENGTH), 4096, 131072)
    seed = parse_int("fig1_seed", DEFAULT_SEED)
    poly_degree = clamp(parse_int("fig1_poly_degree", 2), 1, 4)
    fit_min = clamp(parse_int("fig1_fit_min", DEFAULT_FIG1_FIT_MIN), 8, n // 2)
    fit_max = clamp(parse_int("fig1_fit_max", DEFAULT_FIG1_FIT_MAX), fit_min + 1, n // 2)
    q_values = q_grid(DEFAULT_Q_MIN, DEFAULT_Q_MAX, DEFAULT_Q_STEP, SELECTED_Q_VALUES)
    scales = choose_scales(8, max(16, n // 4), 34)
    rng = np.random.default_rng(seed)

    cases = []
    for case in MONOFRACTAL_CASES:
        hurst = float(case["hurst"])
        series = spectral_mono_series(hurst, n, rng)
        result = mfdfa_fluctuations(series, q_values, scales, poly_degree, fit_min, fit_max)
        q_arr = result["q_values"]
        hq = result["hq"]
        tau = result["tau"]
        cases.append(
            {
                "key": case["key"],
                "label": case["label"],
                "hurst": hurst,
                "fluctuations": fluctuation_curves(result, SELECTED_Q_VALUES),
                "hq": points_from_arrays(q_arr, hq),
                "tau": points_from_arrays(q_arr, tau),
                "hq_theory": points_from_arrays(q_arr, np.full_like(q_arr, hurst)),
                "tau_theory": points_from_arrays(q_arr, q_arr * hurst - 1.0),
            }
        )

    return {
        "params": {
            "n": n,
            "seed": seed,
            "poly_degree": poly_degree,
            "fit_min": fit_min,
            "fit_max": fit_max,
            "selected_q": SELECTED_Q_VALUES,
        },
        "cases": cases,
        "source": {
            "article": "Kantelhardt et al., Physica A 316 (2002), Fig. 1, page 94",
            "doi": "https://doi.org/10.1016/S0378-4371(02)01383-3",
        },
    }


def binomial_cascade(a: float, nmax: int) -> np.ndarray:
    n = 2**nmax
    counts = np.fromiter((int(index).bit_count() for index in range(n)), dtype=float, count=n)
    values = (a**counts) * ((1.0 - a) ** (nmax - counts))
    return np.asarray(values, dtype=float)


def binomial_theory(a: float, q_values: Sequence[float]) -> Dict[str, np.ndarray]:
    q_arr = np.asarray(q_values, dtype=float)
    hq = np.zeros_like(q_arr)
    tau = np.zeros_like(q_arr)
    alpha = np.zeros_like(q_arr)
    f_alpha = np.zeros_like(q_arr)
    log2 = math.log(2.0)
    b = 1.0 - a

    for idx, q_value in enumerate(q_arr):
        mass = (a**q_value) + (b**q_value)
        alpha[idx] = -(
            (a**q_value) * math.log(a) + (b**q_value) * math.log(b)
        ) / (mass * log2)
        if abs(q_value) < 1e-12:
            tau[idx] = -1.0
            hq[idx] = -0.5 * (math.log(a) + math.log(b)) / log2
        else:
            tau[idx] = -math.log(mass) / log2
            hq[idx] = (tau[idx] + 1.0) / q_value
        f_alpha[idx] = q_value * alpha[idx] - tau[idx]
    return {"hq": hq, "tau": tau, "alpha": alpha, "f_alpha": f_alpha}
####
#def spectral():
 #   tau = np.arange(-20.0, 20.1, 1.0)
#######


def singularity_spectrum(q_values: Sequence[float], tau_values: Sequence[float]) -> Dict[str, np.ndarray]:
    q_arr = np.asarray(q_values, dtype=float)
    tau_arr = np.asarray(tau_values, dtype=float)
    valid = np.isfinite(q_arr) & np.isfinite(tau_arr)
    q_valid = q_arr[valid]
    tau_valid = tau_arr[valid]
    if q_valid.size < 2:
        return {"alpha": np.asarray([], dtype=float), "f_alpha": np.asarray([], dtype=float)}

    edge_order = 2 if q_valid.size >= 3 else 1
    alpha = np.gradient(tau_valid, q_valid, edge_order=edge_order)
    f_alpha = q_valid * alpha - tau_valid
    return {"alpha": alpha, "f_alpha": f_alpha}

def downsample_series(values: Sequence[float], max_points: int) -> Dict[str, List[float]]:
    data = np.asarray(values, dtype=float)
    if data.size <= max_points:
        indices = np.arange(data.size, dtype=int)
    else:
        indices = np.linspace(0, data.size - 1, max_points, dtype=int)
    return {
        "index": indices.tolist(),
        "values": np.round(data[indices], 12).tolist(),
    }


def build_binomial_payload() -> Dict[str, object]:
    nmax = clamp(parse_int("binomial_nmax", 16), 8, 17)
    a = min(0.95, max(0.55, parse_float("binomial_a", 0.75)))
    poly_degree = clamp(parse_int("binomial_poly_degree", 1), 1, 4)
    fit_min = clamp(parse_int("binomial_fit_min", DEFAULT_BINOMIAL_FIT_MIN), 8, 2**nmax // 2)
    fit_max = clamp(parse_int("binomial_fit_max", DEFAULT_BINOMIAL_FIT_MAX), fit_min + 1, 2**nmax // 2)
    q_values = q_grid(-20.0, 20.0, 1.0, SELECTED_Q_VALUES)
    values = binomial_cascade(a, nmax)
    scales = choose_scales(8, max(16, values.size // 4), 34)
    result = mfdfa_fluctuations(values, q_values, scales, poly_degree, fit_min, fit_max)
    theory = binomial_theory(a, result["q_values"])
    theory_smooth_q = np.linspace(-20.0, 20.0, 401)
    theory_smooth = binomial_theory(a, theory_smooth_q)
    log2 = math.log(2.0)
    alpha_min = -math.log(max(a, 1.0 - a)) / log2
    alpha_max = -math.log(min(a, 1.0 - a)) / log2

    q_arr = result["q_values"]
    hq = result["hq"]
    tau = result["tau"]
    row_spectrum = singularity_spectrum(q_arr, tau)
    spectrum_mask = (q_arr >= -10.0) & (q_arr <= 10.0)
    spectrum = singularity_spectrum(q_arr[spectrum_mask], tau[spectrum_mask])
    alpha_numeric = (
        row_spectrum["alpha"] if row_spectrum["alpha"].size == q_arr.size else np.full_like(q_arr, np.nan)
    )
    f_alpha_numeric = (
        row_spectrum["f_alpha"]
        if row_spectrum["f_alpha"].size == q_arr.size
        else np.full_like(q_arr, np.nan)
    )
    rows = []
    for idx, q_value in enumerate(q_arr):
        rows.append(
            {
                "q": float(q_value),
                "h": float(hq[idx]),
                "h_theory": float(theory["hq"][idx]),
                "tau": float(tau[idx]),
                "tau_theory": float(theory["tau"][idx]),
                "alpha": float(alpha_numeric[idx]),
                "f_alpha": float(f_alpha_numeric[idx]),
            }
        )

    return {
        "params": {
            "nmax": nmax,
            "n": int(values.size),
            "a": a,
            "poly_degree": poly_degree,
            "fit_min": fit_min,
            "fit_max": fit_max,
            "selected_q": SELECTED_Q_VALUES,
        },
        "series": downsample_series(values, 900),
        "fluctuations": fluctuation_curves(result, SELECTED_Q_VALUES),
        "hq": points_from_arrays(q_arr, hq),
        "hq_theory": points_from_arrays(q_arr, theory["hq"]),
        "tau": points_from_arrays(q_arr, tau),
        "tau_theory": points_from_arrays(q_arr, theory["tau"]),
        "spectrum": physical_spectrum_points(
            spectrum["alpha"], spectrum["f_alpha"], alpha_min, alpha_max
        ),
        "spectrum_theory": sorted_points_from_arrays(
            theory_smooth["alpha"], theory_smooth["f_alpha"]
        ),
        "rows": rows,
        "source": {
            "article": "Kantelhardt et al., Physica A 316 (2002), Eq. (18)-(20)",
            "doi": "https://doi.org/10.1016/S0378-4371(02)01383-3",
        },
    }


def register_mfdfa_article(app):
    @app.route("/mfdfa_articulo")
    def mfdfa_article_page():
        return render_template("mfdfa_articulo.html")

    @app.route("/process_mfdfa_article_fig1", methods=["POST"])
    def process_mfdfa_article_fig1():
        try:
            return jsonify(build_figure1_payload())
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

    @app.route("/process_binomial_cascade", methods=["POST"])
    def process_binomial_cascade():
        try:
            return jsonify(build_binomial_payload())
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400
