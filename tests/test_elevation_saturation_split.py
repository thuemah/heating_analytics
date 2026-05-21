"""Tier 1 elevation_diagnostics saturation split (#954 follow-up).

Verifies that ``elevation_diagnostics.instantaneous`` splits each
elevation bucket into ``unsaturated`` and ``saturated`` sub-blocks.
Pre-implementation step for the #954 4D shadow learner — needed to
distinguish HP-capacity censoring (saturated: ``actual < 0.05·base``,
equivalent to ``implied_solar ≥ BATCH_FIT_SATURATION_RATIO``) from
genuine Kasten elevation×airmass bias (unsaturated modulating hours).
Heating regime only.
"""
from datetime import datetime, timedelta
from unittest.mock import MagicMock

from custom_components.heating_analytics.diagnostics import DiagnosticsEngine

from tests.test_solar_diagnose import _hour_entry, _make_coord


def _coord_with_elevations(hourly_log, elevations):
    coord = _make_coord(hourly_log)
    elev_iter = iter(elevations)
    coord.solar.get_approx_sun_pos = MagicMock(
        side_effect=lambda _dt: (next(elev_iter), 180.0)
    )
    return coord


def test_elevation_buckets_split_by_saturation():
    """3 unsaturated + 3 saturated heating hours at elev 30° land in their
    respective sub-blocks of the ``15-30`` bucket (elev 30° is bucket-boundary
    half-open → falls in ``30-45``; use 22° to hit ``15-30``)."""
    base_dt = datetime(2026, 4, 1, 12, 0)
    entries: list[dict] = []
    elevations: list[float] = []
    # Unsaturated heating hours: actual = 1.0, base = 2.0 → actual/base = 0.5
    # (well above the 0.05 saturation gate).
    for i in range(3):
        ts = (base_dt + timedelta(hours=i)).isoformat()
        entries.append(_hour_entry(
            ts, solar_s=0.5, solar_e=0.0, solar_w=0.0,
            actual=1.0, base=2.0, mode="heating",
        ))
        elevations.append(22.0)
    # Saturated heating hours: actual = 0.005, base = 0.5 → actual/base = 0.01
    # < 0.05, base > 0.05 — triggers the saturation continue.
    for i in range(3, 6):
        ts = (base_dt + timedelta(hours=i)).isoformat()
        entries.append(_hour_entry(
            ts, solar_s=0.5, solar_e=0.0, solar_w=0.0,
            actual=0.005, base=0.5, mode="heating",
        ))
        elevations.append(22.0)

    coord = _coord_with_elevations(entries, elevations)
    result = DiagnosticsEngine(coord).diagnose_solar(days_back=30)
    ev = result["per_unit"]["sensor.heater1"]["elevation_diagnostics"]["instantaneous"]

    bucket = ev["15-30"]
    assert "unsaturated" in bucket
    assert "saturated" in bucket
    assert bucket["unsaturated"]["n"] == 3
    assert bucket["saturated"]["n"] == 3
    # With n=3 < min_samples=5 the summary collapses to {"n": ...} only,
    # which is the documented small-bucket behaviour — boost to 6 each so
    # the full block (and median_residual_normalised) is emitted.


def test_elevation_buckets_split_full_block_above_min_samples():
    """6+6 hours per sub-block emits the full median/MAD/normalised
    summary in both ``unsaturated`` and ``saturated`` sub-blocks."""
    base_dt = datetime(2026, 4, 1, 12, 0)
    entries: list[dict] = []
    elevations: list[float] = []
    for i in range(6):
        ts = (base_dt + timedelta(hours=i)).isoformat()
        entries.append(_hour_entry(
            ts, solar_s=0.5, solar_e=0.0, solar_w=0.0,
            actual=1.0, base=2.0, mode="heating",
        ))
        elevations.append(22.0)
    for i in range(6, 12):
        ts = (base_dt + timedelta(hours=i)).isoformat()
        entries.append(_hour_entry(
            ts, solar_s=0.5, solar_e=0.0, solar_w=0.0,
            actual=0.005, base=0.5, mode="heating",
        ))
        elevations.append(22.0)

    coord = _coord_with_elevations(entries, elevations)
    result = DiagnosticsEngine(coord).diagnose_solar(days_back=30)
    ev = result["per_unit"]["sensor.heater1"]["elevation_diagnostics"]["instantaneous"]

    unsat = ev["15-30"]["unsaturated"]
    sat = ev["15-30"]["saturated"]
    assert unsat["n"] == 6
    assert sat["n"] == 6
    # Both sub-blocks emit the full summary including a non-None normalised
    # residual (mean_potential > 1e-3 by construction).
    assert unsat["median_residual_normalised"] is not None
    assert sat["median_residual_normalised"] is not None
    for field in ("median_residual", "mad_residual", "mean_potential"):
        assert field in unsat
        assert field in sat
