"""Tests for the ``base_model_4d_shadow`` block of ``diagnose_solar`` (#954).

Path-B promotion metric: re-aggregates the last days_back days of base-bucket
EMA twice in parallel — once with ``solar_normalization_delta`` (3D, live) and
once with ``solar_normalization_delta_4d`` (4D shadow from 7b8cb0a) — and
reports per-cell drift plus a shoulder-aggregate RMS of per-step EMA jitter.

These tests target the report builder ``_compute_base_model_4d_shadow_report``
directly with a minimal mock coordinator.  An end-to-end pass through
``diagnose_solar`` would require wiring the full diagnose harness; the
unit-level tests on the report builder are sufficient because the method is
called once at a single call site.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import MagicMock

from custom_components.heating_analytics.diagnostics import DiagnosticsEngine


def _coord(hourly_log, *, correlation_data=None, learning_rate=0.05,
           energy_sensors=None):
    coord = MagicMock()
    coord._hourly_log = hourly_log
    coord._correlation_data = correlation_data or {}
    coord.learning_rate = learning_rate
    coord.energy_sensors = energy_sensors or ["sensor.heater1"]
    return coord


def _entry(ts, *, d3, d4=None, actual=1.5, sf=0.4, temp_key="10",
           wind_bucket="normal", aux=False, guest=0.0,
           mode="heating", sde=None):
    e = {
        "timestamp": ts,
        "actual_kwh": actual,
        "solar_normalization_delta": d3,
        "solar_factor": sf,
        "temp_key": temp_key,
        "wind_bucket": wind_bucket,
        "auxiliary_active": aux,
        "guest_impact_kwh": guest,
        "unit_modes": {"sensor.heater1": mode},
        "solar_dominant_entities": sde if sde is not None else [],
    }
    if d4 is not None:
        e["solar_normalization_delta_4d"] = d4
    return e


def _ts_range(n_hours, start=None):
    """Generate ISO timestamps starting near today (within days_back=30)."""
    if start is None:
        # Use yesterday so we're safely inside the cutoff window.
        start = datetime.now() - timedelta(days=1)
    return [(start + timedelta(hours=i)).isoformat() for i in range(n_hours)]


def test_shadow_reports_drift_when_4d_delta_differs():
    """4D delta consistently 0.1 kWh larger → cumulative bucket drift."""
    n = 30
    log = []
    for ts in _ts_range(n):
        # 3D delta = 0.2 kWh, 4D delta = 0.3 kWh (4D consistently larger).
        log.append(_entry(ts, d3=0.2, d4=0.3, actual=1.5, sf=0.4,
                          temp_key="10", wind_bucket="normal"))
    # Seed both sims at 1.0 via correlation_data so we observe a clean
    # convergence-to-target gap, not a from-zero ramp.
    coord = _coord(log, correlation_data={"10": {"normal": 1.0}})
    out = DiagnosticsEngine(coord)._compute_base_model_4d_shadow_report(30)

    assert out["available"] is True
    assert out["n_hours"] == n
    assert out["days_back"] == 30
    cell = out["per_cell"]["10/normal"]
    assert cell["n"] == n
    assert cell["bucket_3d_initial"] == 1.0
    # Targets: 3D → max(0, 1.5+0.2)=1.7; 4D → max(0, 1.5+0.3)=1.8.
    # After many EMA steps the 4D bucket should sit above the 3D bucket
    # in proportion to the delta gap, so drift_kwh > 0 and < 0.1.
    assert cell["drift_kwh"] > 0.0
    assert cell["drift_kwh"] <= 0.1
    assert cell["bucket_4d_final"] > cell["bucket_3d_final"]

    # Headline must surface a finite shoulder_ratio (temp_key=10 falls in
    # the [7..14] shoulder window with normal-wind).
    h = out["headline"]
    assert h["shoulder_cell_count"] == 1
    assert h["shoulder_ratio"] is not None
    assert h["bucket_drift_rms_3d_shoulder"] > 0.0
    assert h["bucket_drift_rms_4d_shoulder"] > 0.0


def test_shadow_unavailable_when_no_4d_tags():
    """No entries carry solar_normalization_delta_4d → available=False."""
    log = []
    for ts in _ts_range(20):
        # 3D delta only — 4D field missing entirely.
        log.append(_entry(ts, d3=0.2, d4=None))
    coord = _coord(log)
    out = DiagnosticsEngine(coord)._compute_base_model_4d_shadow_report(30)

    assert out["available"] is False
    assert out["n_hours"] == 0
    assert out["reason"] == "no_4d_tagged_hours"
    assert out["days_back"] == 30
    # All 20 entries had a 3D delta but no 4D — counter must reflect.
    assert out["n_hours_skipped_no_4d_delta"] == 20


def test_per_cell_excludes_aux_and_guest_hours():
    """Aux-active and guest-impact hours must not contribute to cell n."""
    timestamps = _ts_range(15)
    log = []
    # 5 clean qualifying hours.
    for ts in timestamps[:5]:
        log.append(_entry(ts, d3=0.2, d4=0.3))
    # 5 aux-active hours (must be skipped).
    for ts in timestamps[5:10]:
        log.append(_entry(ts, d3=0.2, d4=0.3, aux=True))
    # 5 guest-impact hours (must be skipped).
    for ts in timestamps[10:15]:
        log.append(_entry(ts, d3=0.2, d4=0.3, guest=0.5))

    coord = _coord(log)
    out = DiagnosticsEngine(coord)._compute_base_model_4d_shadow_report(30)

    assert out["available"] is True
    assert out["n_hours"] == 5
    assert out["per_cell"]["10/normal"]["n"] == 5


def test_per_cell_excludes_cooling_mode_hours():
    """Cooling-mode hours must not contribute (base learning is heating)."""
    timestamps = _ts_range(10)
    log = []
    for ts in timestamps[:6]:
        log.append(_entry(ts, d3=0.2, d4=0.3, mode="heating"))
    for ts in timestamps[6:]:
        log.append(_entry(ts, d3=0.2, d4=0.3, mode="cooling"))

    coord = _coord(log)
    out = DiagnosticsEngine(coord)._compute_base_model_4d_shadow_report(30)

    assert out["available"] is True
    assert out["n_hours"] == 6
    assert out["per_cell"]["10/normal"]["n"] == 6
