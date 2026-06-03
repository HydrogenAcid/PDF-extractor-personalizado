from __future__ import annotations

import json
import math
from datetime import date, datetime, timedelta, timezone
from typing import Dict, List, Sequence, Tuple
from urllib.parse import quote
from urllib.request import Request, urlopen

import numpy as np
from flask import jsonify, render_template

DEFAULT_WHITE_NOISE_POINTS = 2048
DEFAULT_WHITE_NOISE_PLOT_POINTS = 420
DEFAULT_MAX_SCALING_POINTS = 12

MONTH_NAMES_ES = [
    "enero",
    "febrero",
    "marzo",
    "abril",
    "mayo",
    "junio",
    "julio",
    "agosto",
    "septiembre",
    "octubre",
    "noviembre",
    "diciembre",
]

MARKET_SERIES = [
    {"symbol": "VOO", "label": "VOO"},
    {"symbol": "NVDA", "label": "Nvidia (NVDA)"},
    {"symbol": "GAP", "label": "Gap (GAP)"},
]


def previous_calendar_month(today: date | None = None) -> Dict[str, str]:
    today = today or date.today()
    first_current = date(today.year, today.month, 1)
    last_previous = first_current - timedelta(days=1)
    first_previous = date(last_previous.year, last_previous.month, 1)
    return {
        "start": first_previous.isoformat(),
        "end_exclusive": first_current.isoformat(),
        "end_inclusive": last_previous.isoformat(),
        "label": f"{MONTH_NAMES_ES[first_previous.month - 1].capitalize()} {first_previous.year}",
    }


def iso_to_epoch_utc(day_iso: str) -> int:
    year, month, day = map(int, day_iso.split("-"))
    return int(datetime(year, month, day, tzinfo=timezone.utc).timestamp())


def fetch_yahoo_daily_series(
    symbol: str,
    start_iso: str,
    end_exclusive_iso: str,
) -> Tuple[List[str], List[float]]:
    period1 = iso_to_epoch_utc(start_iso)
    period2 = iso_to_epoch_utc(end_exclusive_iso)
    encoded_symbol = quote(symbol, safe="")
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{encoded_symbol}"
        f"?period1={period1}&period2={period2}&interval=1d&includePrePost=false&events=div%2Csplits"
    )
    request = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json",
        },
    )
    with urlopen(request, timeout=15) as response:
        payload = json.load(response)

    chart = payload.get("chart") or {}
    if chart.get("error"):
        raise ValueError(chart["error"].get("description") or "Yahoo Finance devolvio un error")

    result_list = chart.get("result") or []
    if not result_list:
        raise ValueError("Sin datos disponibles para la serie solicitada")

    result = result_list[0]
    timestamps = result.get("timestamp") or []
    indicators = result.get("indicators") or {}
    quote_list = indicators.get("quote") or [{}]
    close_values = quote_list[0].get("close") or []
    adj_values = (indicators.get("adjclose") or [{}])[0].get("adjclose")
    values = adj_values if adj_values else close_values

    dates: List[str] = []
    closes: List[float] = []
    for timestamp, value in zip(timestamps, values):
        if value is None or not math.isfinite(value):
            continue
        dates.append(datetime.utcfromtimestamp(int(timestamp)).strftime("%Y-%m-%d"))
        closes.append(float(value))

    if len(closes) < 8:
        raise ValueError("La serie descargada es demasiado corta para estimar los exponentes")

    return dates, closes


def _choose_scales(min_scale: int, max_scale: int, max_points: int = DEFAULT_MAX_SCALING_POINTS) -> List[int]:
    if max_scale < min_scale:
        return []
    total_points = max_scale - min_scale + 1
    if total_points <= max_points:
        return list(range(min_scale, max_scale + 1))

    raw = np.unique(
        np.floor(
            np.logspace(np.log10(min_scale), np.log10(max_scale), num=max_points)
        ).astype(int)
    )
    scales = [int(value) for value in raw if min_scale <= value <= max_scale]
    if scales and scales[-1] != max_scale:
        scales.append(max_scale)
    return sorted(set(scales))


def _build_scaling_payload(
    x_values: Sequence[float],
    y_values: Sequence[float],
    slope: float,
    intercept: float,
) -> Dict[str, List[Dict[str, float]]]:
    points = [
        {"x": float(x_value), "y": float(y_value)}
        for x_value, y_value in zip(x_values, y_values)
    ]
    fit = [
        {"x": float(x_value), "y": float(math.exp(intercept) * (x_value ** slope))}
        for x_value in x_values
    ]
    return {"points": points, "fit": fit}


def estimate_hurst_rs(series: Sequence[float]) -> Dict[str, object]:
    values = np.asarray(series, dtype=float)
    n = values.size
    if n < 10:
        raise ValueError("La serie debe tener al menos 10 puntos para estimar H")

    min_window = max(4, min(32, n // 8))
    max_window = min(max(min_window + 1, n // 4), 512)
    window_sizes = _choose_scales(min_window, max_window)

    usable_windows: List[int] = []
    rs_values: List[float] = []

    for window_size in window_sizes:
        segment_count = n // window_size
        if segment_count < 2:
            continue

        segment_ratios: List[float] = []
        for segment_index in range(segment_count):
            start = segment_index * window_size
            end = start + window_size
            segment = values[start:end]
            mean_value = float(np.mean(segment))
            centered = segment - mean_value
            profile = np.cumsum(centered)
            range_value = float(np.max(profile) - np.min(profile))
            std_value = float(np.sqrt(np.mean(centered ** 2)))
            if std_value > 0 and math.isfinite(range_value) and math.isfinite(std_value):
                segment_ratios.append(range_value / std_value)

        if segment_ratios:
            usable_windows.append(int(window_size))
            rs_values.append(float(np.mean(segment_ratios)))

    if len(usable_windows) < 2:
        raise ValueError("No hubo suficientes escalas utilizables para la regresion log-log de H")

    xs = np.log(np.asarray(usable_windows, dtype=float))
    ys = np.log(np.asarray(rs_values, dtype=float))
    slope, intercept = np.polyfit(xs, ys, 1)
    return {
        "hurst": float(slope),
        "window_sizes": usable_windows,
        "rs_values": rs_values,
        "intercept": float(intercept),
    }


def estimate_hurst_increment_scaling(series: Sequence[float]) -> Dict[str, object]:
    values = np.asarray(series, dtype=float)
    n = values.size
    if n < 10:
        raise ValueError("La serie debe tener al menos 10 puntos para estimar H por incrementos")

    max_lag = max(3, min(n // 2, 32))
    lags = _choose_scales(1, max_lag, max_points=min(DEFAULT_MAX_SCALING_POINTS, max_lag))

    usable_lags: List[int] = []
    scale_values: List[float] = []

    for lag in lags:
        delta = values[lag:] - values[:-lag]
        if delta.size < 2:
            continue

        sigma = float(np.std(delta, ddof=1))
        if sigma > 0 and math.isfinite(sigma):
            usable_lags.append(int(lag))
            scale_values.append(sigma)

    if len(usable_lags) < 2:
        raise ValueError("No hubo suficientes escalas utilizables para la regresion log-log de H por incrementos")

    xs = np.log(np.asarray(usable_lags, dtype=float))
    ys = np.log(np.asarray(scale_values, dtype=float))
    slope, intercept = np.polyfit(xs, ys, 1)
    return {
        "hurst": float(slope),
        "lags": usable_lags,
        "scales": scale_values,
        "intercept": float(intercept),
    }


def estimate_dfa(series: Sequence[float], poly_degree: int = 1) -> Dict[str, object]:
    values = np.asarray(series, dtype=float)
    n = values.size
    if n < 12:
        raise ValueError("La serie debe tener al menos 12 puntos para calcular DFA")

    profile = np.cumsum(values - float(np.mean(values)))
    max_window = min(max(5, n // 4), max(5, n // 2))
    window_sizes = _choose_scales(4, max_window)

    usable_scales: List[int] = []
    fluctuation_values: List[float] = []
    for window_size in window_sizes:
        segment_count = n // window_size
        if segment_count < 2:
            continue

        x_axis = np.arange(window_size, dtype=float)
        fluctuation_by_segment: List[float] = []
        for segment_index in range(segment_count):
            start = segment_index * window_size
            end = start + window_size
            segment = profile[start:end]
            fit_coefficients = np.polyfit(x_axis, segment, poly_degree)
            fit_values = np.polyval(fit_coefficients, x_axis)
            fs2 = float(np.mean((segment - fit_values) ** 2))
            if fs2 > 0 and math.isfinite(fs2):
                fluctuation_by_segment.append(fs2)

        if fluctuation_by_segment:
            usable_scales.append(int(window_size))
            fluctuation_values.append(float(math.sqrt(np.mean(fluctuation_by_segment))))

    if len(usable_scales) < 2:
        raise ValueError("No hubo suficientes ventanas utilizables para estimar DFA")

    xs = np.log(np.asarray(usable_scales, dtype=float))
    ys = np.log(np.asarray(fluctuation_values, dtype=float))
    slope, intercept = np.polyfit(xs, ys, 1)
    return {
        "alpha": float(slope),
        "window_sizes": usable_scales,
        "fluctuations": fluctuation_values,
        "intercept": float(intercept),
    }


def estimate_higuchi_dimension(series: Sequence[float]) -> Dict[str, object]:
    values = np.asarray(series, dtype=float)
    n = values.size
    if n < 12:
        raise ValueError("La serie debe tener al menos 12 puntos para calcular Higuchi")

    max_k = min(max(4, n // 3), 32)
    k_values = _choose_scales(1, max_k, max_points=min(DEFAULT_MAX_SCALING_POINTS, max_k))

    usable_ks: List[int] = []
    average_lengths: List[float] = []
    for k in k_values:
        segment_lengths: List[float] = []
        for m in range(k):
            max_i = (n - 1 - m) // k
            if max_i < 1:
                continue

            total = 0.0
            for i in range(1, max_i + 1):
                current_index = m + i * k
                previous_index = m + (i - 1) * k
                total += abs(values[current_index] - values[previous_index])

            length_m = ((n - 1) / (max_i * k)) * (total / k)
            if length_m > 0 and math.isfinite(length_m):
                segment_lengths.append(length_m)

        if segment_lengths:
            usable_ks.append(int(k))
            average_lengths.append(float(np.mean(segment_lengths)))

    if len(usable_ks) < 2:
        raise ValueError("No hubo suficientes escalas utilizables para Higuchi")

    xs = np.log(np.asarray(usable_ks, dtype=float))
    ys = np.log(np.asarray(average_lengths, dtype=float))
    slope, intercept = np.polyfit(xs, ys, 1)
    return {
        "dimension": float(-slope),
        "slope": float(slope),
        "intercept": float(intercept),
        "k_values": usable_ks,
        "lengths": average_lengths,
    }


def normalize_base_100(series: Sequence[float]) -> List[float]:
    values = np.asarray(series, dtype=float)
    if values.size == 0 or values[0] == 0:
        return []
    return np.round((values / values[0]) * 100.0, 4).tolist()


def downsample_series(values: Sequence[float], max_points: int) -> List[float]:
    items = list(values)
    if len(items) <= max_points:
        return items
    indices = np.linspace(0, len(items) - 1, num=max_points, dtype=int)
    return [items[idx] for idx in indices]


def interpret_hurst_rs(value: float) -> str:
    if value < 0.45:
        return "Anti-persistente"
    if value <= 0.55:
        return "Sin memoria marcada"
    if value <= 0.9:
        return "Persistente"
    if value <= 1.1:
        return "Compatible con una trayectoria integrada"
    if value <= 1.35:
        return "Persistencia fuerte"
    return "Escalamiento alto"


def interpret_hurst_increment(value: float) -> str:
    if value < 0.1:
        return "Casi ruido blanco"
    if value < 0.45:
        return "Anti-persistente"
    if value <= 0.55:
        return "Aproximadamente browniano"
    if value <= 0.75:
        return "Persistente"
    return "Persistencia fuerte"


def interpret_dfa(value: float) -> str:
    if value < 0.45:
        return "Anti-correlacion"
    if value <= 0.55:
        return "Sin memoria marcada"
    return "Persistencia de largo alcance"


def interpret_higuchi(value: float) -> str:
    if value >= 1.8:
        return "Grafica muy rugosa"
    if value >= 1.45:
        return "Rugosidad intermedia"
    return "Trayectoria relativamente suave"


def analyze_series_metrics(series: Sequence[float]) -> Dict[str, object]:
    hurst_rs = estimate_hurst_rs(series)
    hurst_increment = estimate_hurst_increment_scaling(series)
    dfa = estimate_dfa(series)
    higuchi = estimate_higuchi_dimension(series)

    return {
        "hurst_rs": hurst_rs["hurst"],
        "hurst_increment": hurst_increment["hurst"],
        "dfa_alpha": dfa["alpha"],
        "higuchi_dimension": higuchi["dimension"],
        "hurst_rs_interpretation": interpret_hurst_rs(hurst_rs["hurst"]),
        "hurst_increment_interpretation": interpret_hurst_increment(hurst_increment["hurst"]),
        "dfa_interpretation": interpret_dfa(dfa["alpha"]),
        "higuchi_interpretation": interpret_higuchi(higuchi["dimension"]),
        "hurst_rs_scaling": _build_scaling_payload(
            hurst_rs["window_sizes"],
            hurst_rs["rs_values"],
            hurst_rs["hurst"],
            hurst_rs["intercept"],
        ),
        "dfa_scaling": _build_scaling_payload(
            dfa["window_sizes"],
            dfa["fluctuations"],
            dfa["alpha"],
            dfa["intercept"],
        ),
        "higuchi_scaling": _build_scaling_payload(
            higuchi["k_values"],
            higuchi["lengths"],
            higuchi["slope"],
            higuchi["intercept"],
        ),
    }


def build_white_noise_validation() -> Dict[str, object]:
    rng = np.random.default_rng(20260416)
    white_noise = rng.normal(0.0, 1.0, DEFAULT_WHITE_NOISE_POINTS)
    metrics = analyze_series_metrics(white_noise)
    plot_series = downsample_series(
        np.round(white_noise, 6).tolist(), DEFAULT_WHITE_NOISE_PLOT_POINTS
    )
    plot_index = list(range(len(plot_series)))
    return {
        "label": "Ruido blanco N(0,1)",
        "symbol": "np.random.normal(0,1)",
        "series": plot_series,
        "index": plot_index,
        "points": int(DEFAULT_WHITE_NOISE_POINTS),
        "hurst_rs": metrics["hurst_rs"],
        "hurst_increment": metrics["hurst_increment"],
        "dfa_alpha": metrics["dfa_alpha"],
        "higuchi_dimension": metrics["higuchi_dimension"],
        "expected_hurst_rs": 0.5,
        "expected_hurst_increment": 0.0,
        "expected_dfa_alpha": 0.5,
        "expected_higuchi_dimension": 2.0,
        "hurst_rs_interpretation": "Validacion local R/S: H debe acercarse a 0.5",
        "hurst_increment_interpretation": "Validacion local por incrementos: H debe acercarse a 0",
        "dfa_interpretation": "Validacion local: DFA debe acercarse a 0.5",
        "higuchi_interpretation": "Validacion local: Higuchi debe acercarse a 2",
        "hurst_rs_scaling": metrics["hurst_rs_scaling"],
        "dfa_scaling": metrics["dfa_scaling"],
        "higuchi_scaling": metrics["higuchi_scaling"],
    }


def build_brownian_validation() -> Dict[str, object]:
    rng = np.random.default_rng(20260417)
    increments = rng.normal(0.0, 1.0, DEFAULT_WHITE_NOISE_POINTS)
    brownian = np.cumsum(increments)
    metrics = analyze_series_metrics(brownian)
    plot_series = downsample_series(
        np.round(brownian, 6).tolist(), DEFAULT_WHITE_NOISE_PLOT_POINTS
    )
    plot_index = list(range(len(plot_series)))
    return {
        "label": "Movimiento browniano",
        "symbol": "cumsum(np.random.normal(0,1))",
        "series": plot_series,
        "index": plot_index,
        "points": int(DEFAULT_WHITE_NOISE_POINTS),
        "hurst_rs": metrics["hurst_rs"],
        "hurst_increment": metrics["hurst_increment"],
        "dfa_alpha": metrics["dfa_alpha"],
        "higuchi_dimension": metrics["higuchi_dimension"],
        "expected_hurst_rs": 1.0,
        "expected_hurst_increment": 0.5,
        "expected_dfa_alpha": 1.5,
        "expected_higuchi_dimension": 1.5,
        "hurst_rs_interpretation": "Validacion local R/S: H debe acercarse a 1",
        "hurst_increment_interpretation": "Validacion local por incrementos: H debe acercarse a 0.5",
        "dfa_interpretation": "Validacion local: DFA debe acercarse a 1.5",
        "higuchi_interpretation": "Validacion local: Higuchi debe acercarse a 1.5",
        "hurst_rs_scaling": metrics["hurst_rs_scaling"],
        "dfa_scaling": metrics["dfa_scaling"],
        "higuchi_scaling": metrics["higuchi_scaling"],
    }


def build_market_payload() -> Dict[str, object]:
    period = previous_calendar_month()
    market_series = []
    errors = []

    for item in MARKET_SERIES:
        symbol = item["symbol"]
        label = item["label"]
        try:
            dates, closes = fetch_yahoo_daily_series(
                symbol,
                period["start"],
                period["end_exclusive"],
            )
            metrics = analyze_series_metrics(closes)
            market_series.append(
                {
                    "symbol": symbol,
                    "label": label,
                    "dates": dates,
                    "close": np.round(closes, 4).tolist(),
                    "normalized": normalize_base_100(closes),
                    "points": len(closes),
                    "first_close": float(closes[0]),
                    "last_close": float(closes[-1]),
                    "hurst_rs": metrics["hurst_rs"],
                    "hurst_increment": metrics["hurst_increment"],
                    "dfa_alpha": metrics["dfa_alpha"],
                    "higuchi_dimension": metrics["higuchi_dimension"],
                    "hurst_rs_interpretation": metrics["hurst_rs_interpretation"],
                    "hurst_increment_interpretation": metrics["hurst_increment_interpretation"],
                    "dfa_interpretation": metrics["dfa_interpretation"],
                    "higuchi_interpretation": metrics["higuchi_interpretation"],
                }
            )
        except Exception as exc:  # pragma: no cover - red externa
            errors.append(f"{label}: {exc}")

    white_noise = build_white_noise_validation()
    brownian = build_brownian_validation()
    return {
        "period": period,
        "market_series": market_series,
        "validations": {
            "white_noise": white_noise,
            "brownian": brownian,
        },
        "errors": errors,
        "series_count": len(market_series),
        "source": {
            "name": "Yahoo Finance",
            "endpoint": "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}",
            "interval": "1d",
        },
        "references": {
            "hurst_rs_white": "Ruido blanco -> H aprox 0.5 con R/S",
            "hurst_rs_brownian": "Browniano -> H aprox 1 con R/S",
            "hurst_increment_white": "Ruido blanco -> H aprox 0 por incrementos",
            "hurst_increment_brownian": "Browniano -> H aprox 0.5 por incrementos",
            "dfa_white": "Ruido blanco -> alpha aprox 0.5",
            "dfa_brownian": "Browniano -> alpha aprox 1.5",
            "higuchi_white": "Ruido blanco -> D aprox 2",
            "higuchi_brownian": "Browniano -> D aprox 1.5",
        },
    }


def register_hurst(app):
    @app.route("/hurst")
    def hurst_page():
        return render_template("hurst.html")

    @app.route("/process_hurst", methods=["POST"])
    def process_hurst():
        payload = build_market_payload()
        return jsonify(payload)
