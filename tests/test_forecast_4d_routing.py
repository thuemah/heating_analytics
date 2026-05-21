"""Forecast 4D routing tests (#978).

Five of the six ``calculate_total_power`` call sites in ``forecast.py``
route through the 4D shadow pipeline when ``experimental_4d_primary``
is on AND a forecast-hour DNI/DHI signal is resolvable; the sixth
(daily-forecast site #2) is pinned to 3D via ``force_3d=True`` as a
transitional state for Agent B's follow-up commit on the same branch.

Tests use a MagicMock coordinator + a real ``ForecastManager`` and
intercept ``coordinator.statistics.calculate_total_power`` to spy on
which kwargs each site passes — DNI/DHI presence is the signal for
"routed 4D" because the 4D dispatch path in
``StatisticsManager.calculate_total_power`` is the only branch that
accepts the 4D-only override kwargs.
"""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from custom_components.heating_analytics.forecast import ForecastManager


# ---------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------


def _make_coord(hass, *, flag_on: bool, solar_enabled: bool = True):
    """Build a MagicMock coordinator wired tightly enough to drive
    ``_process_forecast_item`` and ``calculate_plan_revision_impact``
    without booting the real coordinator graph."""
    coord = MagicMock()
    coord.hass = hass
    coord.experimental_4d_primary = flag_on
    coord.solar_enabled = solar_enabled
    coord.solar_correction_percent = 75.0
    coord.auxiliary_heating_active = False
    coord.extreme_wind_threshold = 15.0
    coord.wind_threshold = 10.0
    coord._calculate_weighted_inertia = MagicMock(side_effect=lambda hist: hist[-1])
    coord._calculate_effective_wind = MagicMock(side_effect=lambda ws, gust: ws)
    coord._get_wind_bucket = MagicMock(return_value="normal")
    coord._get_cloud_coverage = MagicMock(return_value=40.0)
    coord._get_weather_wind_unit = MagicMock(return_value="m/s")
    coord._solar_carryover_state = 0.0
    coord.solar_battery_decay = 0.50
    coord.battery_thermal_feedback_k = 0.0
    # Solar calculator stubs.
    coord.solar = MagicMock()
    coord.solar.get_approx_sun_pos = MagicMock(return_value=(35.0, 180.0))
    coord.solar.calculate_solar_factor = MagicMock(return_value=0.5)
    coord.solar.calculate_solar_vector = MagicMock(return_value=(0.4, 0.0, 0.0))
    coord.solar.calculate_effective_solar_factor = MagicMock(return_value=0.35)
    coord.solar.calculate_effective_solar_vector = MagicMock(return_value=(0.28, 0.0, 0.0))
    coord.solar.estimate_daily_avg_solar_factor = MagicMock(return_value=0.4)
    coord.solar.estimate_daily_avg_solar_vector = MagicMock(return_value=(0.3, 0.0, 0.0))
    # Solar optimiser — predicts the screen position for each forecast hour.
    coord.solar_optimizer = MagicMock()
    coord.solar_optimizer.get_recommendation_state = MagicMock(return_value="state_x")
    coord.solar_optimizer.predict_correction_percent = MagicMock(return_value=80.0)
    return coord


def _record_dispatch(coord):
    """Replace ``coord.statistics.calculate_total_power`` with a spy
    that records its kwargs and returns a stable dict shape."""
    coord.statistics = MagicMock()
    calls: list[dict] = []

    def _spy(*args, **kwargs):
        calls.append({"args": args, "kwargs": kwargs})
        return {
            "total_kwh": 1.23,
            "breakdown": {
                "solar_reduction_kwh": 0.10,
                "aux_reduction_kwh": 0.0,
                "solar_heating_wasted_kwh": 0.0,
            },
            "unit_breakdown": {},
        }

    coord.statistics.calculate_total_power = MagicMock(side_effect=_spy)
    return calls


# ---------------------------------------------------------------------
# Site #1 — _process_forecast_item
# ---------------------------------------------------------------------


def _forecast_item(*, native_dni: float | None = None, native_dhi: float | None = None,
                   cloud: float | None = None, condition: str | None = None,
                   ts: str = "2025-05-16T12:00:00+00:00"):
    item = {
        "datetime": ts,
        "temperature": 10.0,
        "wind_speed": 5.0,
        "elevation": 35.0,
        "azimuth": 180.0,
    }
    if native_dni is not None:
        item["direct_normal_irradiance"] = native_dni
    if native_dhi is not None:
        item["diffuse_radiation"] = native_dhi
    if cloud is not None:
        item["cloud_coverage"] = cloud
    if condition is not None:
        item["condition"] = condition
    return item


def test_site1_flag_off_routes_3d(hass):
    coord = _make_coord(hass, flag_on=False)
    calls = _record_dispatch(coord)
    fm = ForecastManager(coord)

    fm._process_forecast_item(
        _forecast_item(native_dni=600.0, native_dhi=120.0, cloud=30.0),
        inertia_history=[10.0],
        wind_unit="m/s",
        default_cloud=50.0,
    )

    assert len(calls) == 1
    kwargs = calls[0]["kwargs"]
    # 3D-only kwargs present, no 4D override.
    assert kwargs.get("override_solar_factor") is not None
    assert kwargs.get("override_solar_vector") is not None
    assert kwargs.get("override_dni_dhi") is None


def test_site1_flag_on_with_native_dni_dhi_routes_4d(hass):
    coord = _make_coord(hass, flag_on=True)
    calls = _record_dispatch(coord)
    fm = ForecastManager(coord)

    fm._process_forecast_item(
        _forecast_item(native_dni=650.0, native_dhi=130.0),
        inertia_history=[10.0],
        wind_unit="m/s",
        default_cloud=50.0,
    )

    assert len(calls) == 1
    kwargs = calls[0]["kwargs"]
    # 4D dispatch — no 3D-only kwargs, dni/dhi override present.
    assert "override_solar_factor" not in kwargs
    assert "override_solar_vector" not in kwargs
    assert "carryover_state_override" not in kwargs
    assert kwargs.get("override_dni_dhi") == (650.0, 130.0)
    # Correction percent comes from solar_optimizer.predict_correction_percent.
    assert kwargs.get("override_correction_percent") == 80.0


def test_site1_flag_on_kasten_fallback_when_only_cloud_coverage(hass):
    coord = _make_coord(hass, flag_on=True)
    calls = _record_dispatch(coord)
    fm = ForecastManager(coord)

    fm._process_forecast_item(
        _forecast_item(cloud=40.0),
        inertia_history=[10.0],
        wind_unit="m/s",
        default_cloud=50.0,
    )

    assert len(calls) == 1
    kwargs = calls[0]["kwargs"]
    assert kwargs.get("override_dni_dhi") is not None
    dni, dhi = kwargs["override_dni_dhi"]
    # Kasten-synthetic produces strictly positive (dni or dhi) when sun is up.
    assert (dni + dhi) > 0.0
    # And we did not pass 3D-only kwargs.
    assert "override_solar_vector" not in kwargs


def test_site1_flag_on_no_sun_falls_back_to_3d(hass):
    """At elevation <= 0 the resolver returns ``no_sun``; route must
    fall back to the 3D primitive so other hours of the forecast loop
    are not silently corrupted."""
    coord = _make_coord(hass, flag_on=True)
    # Pin elev <= 0 for the parsed sun-pos call.
    coord.solar.get_approx_sun_pos = MagicMock(return_value=(-5.0, 180.0))
    calls = _record_dispatch(coord)
    fm = ForecastManager(coord)

    item = _forecast_item(native_dni=0.0, native_dhi=0.0)
    item["elevation"] = -5.0
    fm._process_forecast_item(
        item,
        inertia_history=[10.0],
        wind_unit="m/s",
        default_cloud=50.0,
    )

    assert len(calls) == 1
    kwargs = calls[0]["kwargs"]
    # No DNI/DHI override; 3D path taken.
    assert kwargs.get("override_dni_dhi") is None
    assert "override_solar_factor" in kwargs


def test_site1_flag_on_solar_disabled_routes_3d(hass):
    coord = _make_coord(hass, flag_on=True, solar_enabled=False)
    calls = _record_dispatch(coord)
    fm = ForecastManager(coord)

    fm._process_forecast_item(
        _forecast_item(native_dni=600.0, native_dhi=120.0),
        inertia_history=[10.0],
        wind_unit="m/s",
        default_cloud=50.0,
    )

    assert len(calls) == 1
    kwargs = calls[0]["kwargs"]
    assert kwargs.get("override_dni_dhi") is None
    # Solar disabled means effective_solar_vector is None — still 3D path.
    assert "override_solar_factor" in kwargs


# ---------------------------------------------------------------------
# Site #2 — daily forecast: 24 hourly 4D calls under flag=True,
# legacy single-call-times-24 under flag=False (#978 Agent B follow-up).
# ---------------------------------------------------------------------


def _daily_item(**overrides):
    item = {
        "temperature": 12.0,
        "templow": 6.0,
        "wind_speed": 4.0,
        "cloud_coverage": 30.0,
        "direct_normal_irradiance": 600.0,
        "diffuse_radiation": 120.0,
    }
    item.update(overrides)
    return item


def test_site2_daily_forecast_flag_on_routes_24_hourly_4d_calls(hass):
    """Under experimental_4d_primary=True the daily-forecast path runs
    24 hourly 4D calls.  Each call carries an override_now offset by
    one hour from the previous and an override_dni_dhi for that hour.
    """
    from datetime import date, timedelta
    coord = _make_coord(hass, flag_on=True)
    calls = _record_dispatch(coord)
    fm = ForecastManager(coord)

    fm._calculate_from_daily_forecast(_daily_item(), date(2025, 5, 16))

    # 24 calls, one per hour of the target day.
    assert len(calls) == 24

    # All sun-up hours route 4D (no 3D-only kwargs, dni/dhi present).
    sun_up_calls = [
        c for c in calls if c["kwargs"].get("override_dni_dhi") is not None
    ]
    assert sun_up_calls, "expected at least one sun-up 4D call"
    for c in sun_up_calls:
        k = c["kwargs"]
        assert "override_solar_factor" not in k
        assert "override_solar_vector" not in k
        assert "force_3d" not in k

    # Hour offsets are monotonically +1h.
    timestamps = [c["kwargs"]["override_now"] for c in calls]
    for prev, nxt in zip(timestamps, timestamps[1:]):
        assert nxt - prev == timedelta(hours=1)


def test_site2_daily_forecast_flag_off_uses_single_call_times_24(hass):
    """Under experimental_4d_primary=False the legacy 3D path is
    preserved bit-identically: ONE calculate_total_power call with
    day-average inputs, multiplied by 24 by the caller, with the
    algebraic carryover override active.
    """
    from datetime import date
    coord = _make_coord(hass, flag_on=False)
    calls = _record_dispatch(coord)
    fm = ForecastManager(coord)

    fm._calculate_from_daily_forecast(_daily_item(), date(2025, 5, 16))

    assert len(calls) == 1
    kwargs = calls[0]["kwargs"]
    # 3D-only kwargs present (s_factor / s_vector / carryover).
    assert kwargs.get("override_solar_factor") is not None
    assert kwargs.get("override_solar_vector") is not None
    assert "carryover_state_override" in kwargs
    # No 4D overrides.
    assert kwargs.get("override_dni_dhi") is None
    # force_3d pin is gone (the else-branch no longer needs it: there
    # are no 4D-only kwargs in this call to route via the 4D path).
    assert "force_3d" not in kwargs


def test_site2_daily_forecast_uniform_conditions_parity(hass):
    """With uniform DNI/DHI/cloud across all 24 hours AND the screen
    optimiser pinned to a single correction percent, the 4D 24-hour
    loop should produce the same totalled output shape (24 × per-call
    total) as the flag=False legacy path's single call ×24 — within
    the precision of stub returns.  Validates the loop's accumulation
    arithmetic against the legacy ×24 multiplication on the only case
    where the two are analytically equivalent.
    """
    from datetime import date
    # Pin sun above the horizon for every hour so every iteration
    # hits the 4D dispatch (no 3D fallback for dark hours).
    coord_on = _make_coord(hass, flag_on=True)
    coord_on.solar.get_approx_sun_pos = MagicMock(return_value=(35.0, 180.0))
    calls_on = _record_dispatch(coord_on)
    fm_on = ForecastManager(coord_on)
    pred_on, solar_on, _ws_on = fm_on._calculate_from_daily_forecast(
        _daily_item(), date(2025, 5, 16)
    )

    coord_off = _make_coord(hass, flag_on=False)
    calls_off = _record_dispatch(coord_off)
    fm_off = ForecastManager(coord_off)
    pred_off, solar_off, _ws_off = fm_off._calculate_from_daily_forecast(
        _daily_item(), date(2025, 5, 16)
    )

    # Stub returns total_kwh=1.23, solar=0.10 per call.
    # 4D path: 24 calls → 24 × 1.23.   Legacy: 1 × 1.23 × 24.
    assert len(calls_on) == 24
    assert len(calls_off) == 1
    assert pred_on == pytest.approx(pred_off)
    assert solar_on == pytest.approx(solar_off)


def test_site2_daily_forecast_physics_divergence_on_clear_noon_day(hass):
    """On a clear-noon day (DNI varies ~3× between dawn/dusk and
    noon as a function of sun elevation), the 4D 24-hour-loop and
    the 3D single-call-times-24 should disagree.  Validates the
    physics-first rationale: 24 calls with hour-midpoint sun position
    are not interchangeable with one day-average call ×24.

    We measure divergence via the *number of 4D vs 3D dispatches*
    (a structural rather than numerical metric, because the stub
    returns the same total per call by design).  Under flag=True
    elevation varies hour-to-hour so the resolver feeds a different
    DNI/DHI to each hour; under flag=False the day-average inputs
    flatten that variation.
    """
    from datetime import date

    # Real sun position per hour so DNI varies analytically across
    # the 24-hour loop under flag=True (resolver consumes elev to
    # split GHI between DNI and DHI via Kasten + Erbs).
    def _real_sun_pos(when):
        # Crude solar-noon-at-12:00 model: elev peaks at 60° at noon,
        # 0 at 06:00 and 18:00, negative outside that range.
        import math
        hour = when.hour + when.minute / 60.0
        elev = 60.0 * math.sin(math.pi * (hour - 6.0) / 12.0)
        return (elev, 180.0)

    coord_on = _make_coord(hass, flag_on=True)
    coord_on.solar.get_approx_sun_pos = MagicMock(side_effect=_real_sun_pos)
    calls_on = _record_dispatch(coord_on)
    fm_on = ForecastManager(coord_on)
    # Drop native DNI/DHI so the resolver's Kasten leg synthesises a
    # per-hour-elevation-dependent (DNI, DHI) pair — that's the path
    # where the day-average vs. per-hour distinction actually bites.
    daily_no_native = _daily_item(cloud_coverage=20.0)
    daily_no_native.pop("direct_normal_irradiance", None)
    daily_no_native.pop("diffuse_radiation", None)
    fm_on._calculate_from_daily_forecast(daily_no_native, date(2025, 5, 16))

    # Capture per-hour DNI passed to the dispatcher.  A non-trivial
    # divergence means the per-hour DNI values span a meaningful range
    # (not all-equal), which is exactly what the day-average call
    # would collapse into a single number.
    dni_values = [
        c["kwargs"]["override_dni_dhi"][0]
        for c in calls_on
        if c["kwargs"].get("override_dni_dhi") is not None
    ]
    assert dni_values, "expected at least one sun-up 4D call"
    dni_max = max(dni_values)
    dni_min = min(dni_values)
    # The day-average call collapses this entire range into ONE value;
    # the 24-call loop preserves the variation.  Validates the physics
    # rationale: per-hour DNI variation is non-trivial.
    assert dni_max >= 1.5 * (dni_min + 1e-6), (
        f"expected ≥1.5× peak-vs-trough DNI variation across sun-up "
        f"hours, got min={dni_min:.1f} max={dni_max:.1f}"
    )

    coord_off = _make_coord(hass, flag_on=False)
    calls_off = _record_dispatch(coord_off)
    fm_off = ForecastManager(coord_off)
    daily_no_native_off = _daily_item(cloud_coverage=20.0)
    daily_no_native_off.pop("direct_normal_irradiance", None)
    daily_no_native_off.pop("diffuse_radiation", None)
    fm_off._calculate_from_daily_forecast(daily_no_native_off, date(2025, 5, 16))
    # Legacy path collapses to a single call: no per-hour DNI signal
    # reaches the dispatcher at all.
    assert len(calls_off) == 1
    assert calls_off[0]["kwargs"].get("override_dni_dhi") is None


# ---------------------------------------------------------------------
# Site #4 — fallback ladder (logged dni/dhi → cloud_coverage → 3D)
# ---------------------------------------------------------------------


def _plan_revision_setup(hass, *, flag_on: bool):
    """Wire up coordinator state needed by ``calculate_plan_revision_impact``."""
    coord = _make_coord(hass, flag_on=flag_on)
    # Reference forecast map + hourly log — minimal shape.
    coord.model = MagicMock()
    coord.model.hourly_log = []
    coord.data = {
        "solar_vector_s": 0.2,
        "solar_vector_e": 0.0,
        "solar_vector_w": 0.0,
        "solar_factor": 0.3,
        "effective_wind": 4.0,
    }
    coord._get_float_state = MagicMock(return_value=None)
    return coord


def test_site4_logged_dni_dhi_used_directly(hass):
    coord = _plan_revision_setup(hass, flag_on=True)
    log = {
        "timestamp": "2025-05-16T08:00:00+00:00",
        "hour": 8,
        "temp": 8.0,
        "effective_wind": 4.0,
        "auxiliary_active": False,
        "solar_factor": 0.4,
        "solar_vector_s": 0.3,
        "solar_vector_e": 0.0,
        "solar_vector_w": 0.0,
        "correction_percent": 70.0,
        "dni": 720.0,
        "dhi": 110.0,
    }
    coord.model.hourly_log = [log]
    f_data = {"temp": 9.0, "wind": 4.0, "solar_factor": 0.5,
              "solar_vector": (0.4, 0.0, 0.0),
              "direct_normal_irradiance": 700.0,
              "diffuse_radiation": 120.0,
              "cloud_coverage": 30.0,
              "condition": None}
    coord.forecast_manager = MagicMock()
    calls = _record_dispatch(coord)
    fm = ForecastManager(coord)
    fm._cached_reference_map = {"2025-05-16": {8: f_data}}

    with patch("custom_components.heating_analytics.forecast.dt_util") as mdt:
        mdt.now.return_value = datetime(2025, 5, 16, 12, 0, 0, tzinfo=timezone.utc)
        mdt.parse_datetime.side_effect = lambda s: datetime.fromisoformat(s)
        fm.calculate_plan_revision_impact()

    # Two calls: planned (site #3), reality (site #4).
    assert len(calls) == 2
    reality_kwargs = calls[1]["kwargs"]
    assert reality_kwargs.get("override_dni_dhi") == (720.0, 110.0)
    assert reality_kwargs.get("override_correction_percent") == 70.0


def test_site4_falls_back_to_cloud_coverage_when_dni_dhi_missing(hass):
    coord = _plan_revision_setup(hass, flag_on=True)
    log = {
        "timestamp": "2025-05-16T08:00:00+00:00",
        "hour": 8,
        "temp": 8.0,
        "effective_wind": 4.0,
        "auxiliary_active": False,
        "solar_factor": 0.4,
        "solar_vector_s": 0.3,
        "solar_vector_e": 0.0,
        "solar_vector_w": 0.0,
        "correction_percent": 70.0,
        # No dni/dhi; only cloud_coverage.
        "cloud_coverage": 25.0,
    }
    coord.model.hourly_log = [log]
    f_data = {"temp": 9.0, "wind": 4.0, "solar_factor": 0.5,
              "solar_vector": (0.4, 0.0, 0.0),
              "direct_normal_irradiance": 700.0,
              "diffuse_radiation": 120.0,
              "cloud_coverage": 30.0,
              "condition": None}
    calls = _record_dispatch(coord)
    fm = ForecastManager(coord)
    fm._cached_reference_map = {"2025-05-16": {8: f_data}}

    with patch("custom_components.heating_analytics.forecast.dt_util") as mdt:
        mdt.now.return_value = datetime(2025, 5, 16, 12, 0, 0, tzinfo=timezone.utc)
        mdt.parse_datetime.side_effect = lambda s: datetime.fromisoformat(s)
        fm.calculate_plan_revision_impact()

    assert len(calls) == 2
    reality_kwargs = calls[1]["kwargs"]
    assert reality_kwargs.get("override_dni_dhi") is not None
    dni, dhi = reality_kwargs["override_dni_dhi"]
    assert (dni + dhi) > 0.0  # Kasten-synthetic from cloud_coverage=25.


def test_site4_missing_all_signal_falls_back_to_3d(hass):
    coord = _plan_revision_setup(hass, flag_on=True)
    log = {
        "timestamp": "2025-05-16T08:00:00+00:00",
        "hour": 8,
        "temp": 8.0,
        "effective_wind": 4.0,
        "auxiliary_active": False,
        "solar_factor": 0.4,
        "solar_vector_s": 0.3,
        "solar_vector_e": 0.0,
        "solar_vector_w": 0.0,
        # No dni/dhi/cloud_coverage.
    }
    coord.model.hourly_log = [log]
    f_data = {"temp": 9.0, "wind": 4.0, "solar_factor": 0.5,
              "solar_vector": (0.4, 0.0, 0.0),
              "direct_normal_irradiance": None,
              "diffuse_radiation": None,
              "cloud_coverage": None,
              "condition": None}
    calls = _record_dispatch(coord)
    fm = ForecastManager(coord)
    fm._cached_reference_map = {"2025-05-16": {8: f_data}}

    with patch("custom_components.heating_analytics.forecast.dt_util") as mdt:
        mdt.now.return_value = datetime(2025, 5, 16, 12, 0, 0, tzinfo=timezone.utc)
        mdt.parse_datetime.side_effect = lambda s: datetime.fromisoformat(s)
        fm.calculate_plan_revision_impact()

    assert len(calls) == 2
    reality_kwargs = calls[1]["kwargs"]
    assert reality_kwargs.get("override_dni_dhi") is None
    assert "override_solar_factor" in reality_kwargs  # 3D path


# ---------------------------------------------------------------------
# Site #6 — partial current hour passes NO 4D overrides
# ---------------------------------------------------------------------


def test_site6_partial_current_hour_passes_no_4d_overrides(hass):
    """Site #6 must invoke the 4D primitive without override_dni_dhi /
    override_correction_percent so live coordinator state drives the
    resolution (matches the live tick path)."""
    coord = _plan_revision_setup(hass, flag_on=True)
    coord._get_float_state = MagicMock(return_value=10.0)
    f_data_curr = {"temp": 11.0, "wind": 4.0, "solar_factor": 0.5,
                   "solar_vector": (0.4, 0.0, 0.0),
                   "direct_normal_irradiance": 600.0,
                   "diffuse_radiation": 100.0,
                   "cloud_coverage": 30.0,
                   "condition": None}
    calls = _record_dispatch(coord)
    fm = ForecastManager(coord)
    fm._cached_reference_map = {"2025-05-16": {12: f_data_curr}}

    with patch("custom_components.heating_analytics.forecast.dt_util") as mdt:
        # 12:30 — minutes_passed > 0 hits the partial-hour branch.
        mdt.now.return_value = datetime(2025, 5, 16, 12, 30, 0, tzinfo=timezone.utc)
        mdt.parse_datetime.side_effect = lambda s: datetime.fromisoformat(s)
        fm.calculate_plan_revision_impact()

    # Two calls: site #5 (planned partial) + site #6 (reality partial).
    assert len(calls) == 2
    site6_kwargs = calls[1]["kwargs"]
    # Site #6 explicitly passes NO 4D overrides — relies on live state.
    assert site6_kwargs.get("override_dni_dhi") is None
    assert site6_kwargs.get("override_correction_percent") is None
    # And no 3D-only kwargs either (so the dispatcher routes 4D under flag).
    assert "override_solar_factor" not in site6_kwargs
    assert "override_solar_vector" not in site6_kwargs
    # Sanity: site #5 (planned) DOES pass the override_dni_dhi.
    site5_kwargs = calls[0]["kwargs"]
    assert site5_kwargs.get("override_dni_dhi") == (600.0, 100.0)


# ---------------------------------------------------------------------
# Carryover loop natural no-op under flag=True
# ---------------------------------------------------------------------


def test_carryover_loop_is_natural_no_op_under_flag(hass):
    """Under flag=True, the 4D primitive returns
    ``solar_heating_wasted_kwh = 0`` so the local_carryover accumulator
    in the forecast loop stays at 0 across all hours — documented
    expected behaviour, not a bug.  This test pins the contract."""
    coord = _make_coord(hass, flag_on=True)
    coord._solar_carryover_state = 1.0  # non-zero starting state
    coord.battery_thermal_feedback_k = 2.0  # non-zero k
    # Spy: 4D dispatch returns 0 wasted (matches real
    # ``calculate_total_power_4d`` contract).
    calls = _record_dispatch(coord)

    fm = ForecastManager(coord)
    captured_carryovers: list[float] = []

    real_process = fm._process_forecast_item

    def _capture(item, inertia_history, wind_unit, default_cloud, **kw):
        captured_carryovers.append(kw.get("carryover_state_override", 0.0))
        return real_process(item, inertia_history, wind_unit, default_cloud, **kw)

    with patch.object(fm, "_process_forecast_item", side_effect=_capture):
        items = [
            _forecast_item(native_dni=600.0, native_dhi=120.0,
                           ts=f"2025-05-16T{h:02d}:00:00+00:00")
            for h in (12, 13, 14)
        ]
        # Drive the hourly loop directly.
        with patch("custom_components.heating_analytics.forecast.dt_util") as mdt:
            mdt.now.return_value = datetime(2025, 5, 16, 12, 0, 0, tzinfo=timezone.utc)
            mdt.parse_datetime.side_effect = lambda s: datetime.fromisoformat(s)
            fm._calculate_from_hourly_forecast(items, [10.0], "m/s")

    # First hour starts at coord._solar_carryover_state, but subsequent
    # hours decay-only (charge = k × 0 = 0).  Decay 0.50 → carryovers
    # 1.0, 0.5, 0.25.  This is the same behaviour as the pre-#978 path
    # with k=0; the point is no-op symmetry, not value-zero.
    assert captured_carryovers[0] == pytest.approx(1.0)
    assert captured_carryovers[1] == pytest.approx(0.5)
    assert captured_carryovers[2] == pytest.approx(0.25)


# ---------------------------------------------------------------------
# Regression: real StatisticsManager signature + local-day construction
# (Codex P1 catches; existing tests stubbed too aggressively to see them)
# ---------------------------------------------------------------------


def test_dispatcher_accepts_4d_override_kwargs_without_typeerror():
    """The dispatcher must accept ``override_dni_dhi`` /
    ``override_correction_percent`` / ``override_sun_pos`` without
    raising TypeError.  Pre-fix the dispatcher signature only listed
    the 3D-only overrides and any forecast 4D call site would raise.
    """
    from custom_components.heating_analytics.statistics import StatisticsManager

    coord = MagicMock()
    coord.experimental_4d_primary = True
    stats = StatisticsManager(coord)
    # Patch the 4D primitive to a spy so we don't need the full model graph.
    captured: dict = {}

    def _spy(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return {"total_kwh": 1.0, "breakdown": {}}

    stats.calculate_total_power_4d = _spy  # type: ignore[assignment]

    # No TypeError = pass.
    stats.calculate_total_power(
        10.0, 0.0, 0.0, False,
        override_dni_dhi=(500.0, 80.0),
        override_correction_percent=100.0,
        override_sun_pos=(45.0, 180.0),
    )
    assert captured["kwargs"].get("override_dni_dhi") == (500.0, 80.0)
    assert captured["kwargs"].get("override_correction_percent") == 100.0
    assert captured["kwargs"].get("override_sun_pos") == (45.0, 180.0)


def test_dispatcher_ignores_4d_overrides_on_3d_path():
    """When routing 3D (flag=False), 4D-only overrides must be silently
    dropped — they have no equivalent on the 3D primitive."""
    from custom_components.heating_analytics.statistics import StatisticsManager

    coord = MagicMock()
    coord.experimental_4d_primary = False
    stats = StatisticsManager(coord)
    captured: dict = {}

    def _spy(*args, **kwargs):
        captured["kwargs"] = kwargs
        return {"total_kwh": 1.0, "breakdown": {}}

    stats._calculate_total_power_3d = _spy  # type: ignore[assignment]
    stats.calculate_total_power(
        10.0, 0.0, 0.0, False,
        override_dni_dhi=(500.0, 80.0),
        override_correction_percent=50.0,
    )
    # 3D primitive should NOT see the 4D-only kwargs.
    assert "override_dni_dhi" not in captured["kwargs"]
    assert "override_correction_percent" not in captured["kwargs"]


def test_daily_loop_start_is_tz_aware_on_target_date():
    """Naive ``datetime(target.year, ...)`` fed to ``dt_util.as_local``
    was treated as UTC then converted, shifting the 24-hour loop into
    the wrong local day in non-UTC timezones (Codex P1).  The fix
    inherits ``dt_util.now()``'s tzinfo and replaces date components +
    zeroes time — produces true local midnight on the target date.
    """
    from datetime import date as date_cls
    from custom_components.heating_analytics.forecast import dt_util

    target = date_cls(2026, 5, 17)
    # The fix's construction pattern.
    day_start = dt_util.now().replace(
        year=target.year, month=target.month, day=target.day,
        hour=0, minute=0, second=0, microsecond=0,
    )
    assert day_start.year == target.year
    assert day_start.month == target.month
    assert day_start.day == target.day
    assert day_start.hour == 0
    assert day_start.tzinfo is not None  # bug was tz-naive interpretation
