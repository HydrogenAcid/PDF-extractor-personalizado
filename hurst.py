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


def fetch_yahoo_daily_series(symbol: str, start_iso: str, end_exclusive_iso: str) -> Tuple[List[str], List[float]]:
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
        raise ValueError("La serie descargada es demasiado corta para estimar H")

    return dates, closes


def estimate_hurst_ec19(series: Sequence[float]) -> Dict[str, object]:
    values = np.asarray(series, dtype=float)
    n = values.size
    if n < 10:
        raise ValueError("La serie debe tener al menos 10 puntos para estimar H")

    max_lag = max(3, min(n // 2, 32))
    lags = np.unique(
        np.floor(np.logspace(np.log10(1), np.log10(max_lag), num=min(14, max_lag))).astype(int)
    )
    lags = lags[lags >= 1]

    scale_values: List[float] = []
    usable_lags: List[int] = []
    for lag in lags:
        delta = values[lag:] - values[:-lag]
        if delta.size < 2:
            continue
        sigma = float(np.std(delta, ddof=1))
        if sigma > 0 and math.isfinite(sigma):
            usable_lags.append(int(lag))
            scale_values.append(sigma)

    if len(usable_lags) < 2:
        raise ValueError("No hubo suficientes escalas utilizables para la regresion log-log")

    xs = np.log(np.asarray(usable_lags, dtype=float))
    ys = np.log(np.asarray(scale_values, dtype=float))
    slope, intercept = np.polyfit(xs, ys, 1)
    return {
        "hurst": float(slope),
        "lags": usable_lags,
        "scales": scale_values,
        "intercept": float(intercept),
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


def interpret_hurst(value: float) -> str:
    if value < 0.1:
        return "Casi ruido blanco"
    if value < 0.45:
        return "Anti-persistente"
    if value <= 0.55:
        return "Aproximadamente browniano"
    return "Persistente"


def build_white_noise_validation() -> Dict[str, object]:
    rng = np.random.default_rng(20260414)
    white_noise = rng.normal(0.0, 1.0, DEFAULT_WHITE_NOISE_POINTS)
    estimate = estimate_hurst_ec19(white_noise)
    plot_series = downsample_series(np.round(white_noise, 6).tolist(), DEFAULT_WHITE_NOISE_PLOT_POINTS)
    plot_index = list(range(len(plot_series)))
    return {
        "label": "Ruido blanco N(0,1)",
        "symbol": "np.random.normal(0,1)",
        "series": plot_series,
        "index": plot_index,
        "hurst": estimate["hurst"],
        "expected": 0.0,
        "interpretation": "Validacion: debe acercarse a 0",
        "lags": estimate["lags"],
        "scales": estimate["scales"],
        "points": int(DEFAULT_WHITE_NOISE_POINTS),
    }


def build_market_payload() -> Dict[str, object]:
    period = previous_calendar_month()
    market_series = []
    errors = []

    for item in MARKET_SERIES:
        symbol = item["symbol"]
        label = item["label"]
        try:
            dates, closes = fetch_yahoo_daily_series(symbol, period["start"], period["end_exclusive"])
            estimate = estimate_hurst_ec19(closes)
            market_series.append(
                {
                    "symbol": symbol,
                    "label": label,
                    "dates": dates,
                    "close": np.round(closes, 4).tolist(),
                    "normalized": normalize_base_100(closes),
                    "hurst": estimate["hurst"],
                    "expected": None,
                    "interpretation": interpret_hurst(estimate["hurst"]),
                    "points": len(closes),
                    "first_close": float(closes[0]),
                    "last_close": float(closes[-1]),
                    "lags": estimate["lags"],
                    "scales": estimate["scales"],
                }
            )
        except Exception as exc:  # pragma: no cover - red externa
            errors.append(f"{label}: {exc}")

    validation = build_white_noise_validation()
    return {
        "period": period,
        "market_series": market_series,
        "validation": validation,
        "errors": errors,
        "series_count": len(market_series),
        "source": {
            "name": "Yahoo Finance",
            "endpoint": "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}",
            "interval": "1d",
        },
        "notes": {
            "normalization": (
                "Base 100 significa reescalar cada serie para que su primer cierre del mes valga 100. "
                "Sirve para comparar trayectorias de instrumentos con precios muy distintos."
            ),
            "points": (
                "Observaciones significa numero de cierres diarios disponibles en el mes. "
                "Si aparecen 22 puntos, son 22 sesiones bursatiles del periodo analizado."
            ),
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
