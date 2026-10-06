"""Tests for the calibrate_balance_point service (#1045).

Synthetic ``daily_history`` with a known model::

    E_hour = b + U · DH(cp − Δ · S / S_ref)

per hour of the day.  The fit must recover the dark-day change point
``cp`` and the solar shift ``Δ``, suggest ``cp`` only when the evidence
supports a change, report the proportional-TDD zero ``cp + b/U`` beside
it, and never write configuration.
"""
from __future__ import annotations

import math
import random
from datetime import date, timedelta
from unittest.mock import MagicMock

import pytest
from homeassistant.util import dt as dt_util

from custom_components.heating_analytics.balance_point import (
    calibrate_balance_point,
    collect_days,
    degree_hours,
    fit_change_point,
    sweep,
)
from custom_components.heating_analytics.const import (
    BP_CALIBRATION_MIN_DAYS,
    BP_CALIBRATION_MIN_DAYS_PER_SIDE,
    BP_CALIBRATION_SOLAR_REFERENCE_QUANTILE,
    BP_CALIBRATION_SOLAR_SHIFT_MAX_C,
)


def _hourly_temps(mean: float, amp: float = 4.0) -> list[float]:
    # Coldest ~05:00, warmest ~17:00.
    return [round(mean + amp * math.sin(2 * math.pi * (h - 11) / 24), 3) for h in range(24)]


def _dh(temps: list[float], base: float) -> float:
    return sum(max(0.0, base - t) for t in temps) / len(temps)


def _history(
    days: int = 365,
    *,
    cp: float = 17.0,
    shift: float = 0.0,
    b: float = 0.05,
    u: float = 0.12,
    noise: float = 0.02,
    t_mid: float = 7.0,
    t_amp: float = 11.0,
    solar_max: float = 0.3,
    carryover: float = 0.0,
    cop_slope: float = 0.0,
    relative_noise: float = 0.0,
    cooling_cp: float | None = None,
    seed: int = 1,
    end: date | None = None,
) -> dict:
    """``daily_history`` over ``days`` ending yesterday."""
    rng = random.Random(seed)
    end = end or dt_util.now().date()
    days_out = []
    for i in range(days):
        d = end - timedelta(days=days - i)
        # Coldest mid-January, warmest mid-July; plus weather noise.
        season = -math.cos(2 * math.pi * (d.timetuple().tm_yday - 15) / 365)
        mean = t_mid + t_amp * season + rng.gauss(0.0, 2.0)
        solar = rng.uniform(0.0, solar_max) * (0.4 + 0.6 * (season + 1) / 2)
        days_out.append((d, mean, solar))
    s_ref_values = sorted(s for _, _, s in days_out)
    s_ref = s_ref_values[int(BP_CALIBRATION_SOLAR_REFERENCE_QUANTILE * (len(s_ref_values) - 1))]

    history = {}
    prev_solar = None
    for d, mean, solar in days_out:
        temps = _hourly_temps(mean)
        s_eff = solar if prev_solar is None else (1 - carryover) * solar + carryover * prev_solar
        prev_solar = solar
        base = cp - (shift * s_eff / s_ref if s_ref > 0 else 0.0)
        # Electrical energy = heat demand / COP, COP = 3 · (1 + κ·T) / 3.
        demand = sum(max(0.0, base - t) / (1.0 + cop_slope * t) for t in temps) / 24
        heat_h = max(
            0.0,
            (b + u * demand) * (1.0 + rng.gauss(0.0, relative_noise))
            + rng.gauss(0.0, noise),
        )
        cool_h = 0.0
        if cooling_cp is not None:
            cool_h = max(0.0, 0.2 * sum(max(0.0, t - cooling_cp) for t in temps) / 24)
        history[d.isoformat()] = {
            "kwh": round(24 * (heat_h + cool_h), 3),
            "temp": round(mean, 1),
            "wind": 2.0,
            "solar_factor": round(solar, 4),
            "aux_impact_kwh": 0.0,
            "guest_impact_kwh": 0.0,
            "regime_heating_kwh": round(24 * heat_h, 4),
            "regime_cooling_kwh": round(24 * cool_h, 4),
            "unit_breakdown": {"sensor.a": round(24 * (heat_h + cool_h), 3)},
            "hourly_vectors": {"temp": temps},
        }
    return history


def _coord(history, *, bp: float = 17.0, correlation=None, hourly_log=None,
           inertia_weights=None, wind_threshold: float = 8.0):
    coord = MagicMock()
    coord.model.daily_history = history
    coord.model.correlation_data = correlation or {}
    coord.model.hourly_log = hourly_log or []
    coord.balance_point = bp
    coord.inertia_weights = inertia_weights
    coord.wind_threshold = wind_threshold
    return coord


def _days(history, **kw):
    coord = _coord(history, **kw)
    return collect_days(coord, history, dt_util.now().date())["days"]


# ---------------------------------------------------------------------
# Degree-hours and the fit
# ---------------------------------------------------------------------

def test_degree_hours_match_the_direct_sum():
    history = _history(days=5)
    days = _days(history)
    for day, entry in zip(days, history.values()):
        temps = entry["hourly_vectors"]["temp"]
        for base in (-5.0, 3.3, 12.0, 30.0):
            assert degree_hours(day, base) == pytest.approx(_dh(temps, base))
            cool = sum(max(0.0, t - base) for t in temps) / 24
            assert degree_hours(day, base, heating=False) == pytest.approx(cool)


def test_fit_recovers_floor_and_slope_on_noiseless_data():
    days = _days(_history(noise=0.0, solar_max=0.0))
    fit = fit_change_point(days, 17.0)
    assert fit["b"] == pytest.approx(0.05, abs=1e-3)
    assert fit["U"] == pytest.approx(0.12, abs=1e-3)
    assert fit["sse"] == pytest.approx(0.0, abs=1e-4)


def test_sweep_recovers_the_change_point_under_noise():
    result = sweep(_days(_history()), current_bp=12.0)
    assert result["status"] == "fitted"
    assert result["change_point"] == pytest.approx(17.0, abs=0.5)
    lo, hi = result["interval"]
    assert lo <= 17.0 <= hi
    assert result["current_within_interval"] is False


def test_solar_shift_is_separated_from_the_dark_day_change_point():
    """Sun lowers the day's balance point; ``cp`` must stay the dark-day value.

    This is the failure the daily model exists to fix: on a sunny shoulder
    day the building needs no heat well below its balance point, and a fit
    that cannot say *why* reads that as a lower balance point.
    """
    days = _days(_history(shift=6.0, noise=0.01))
    result = sweep(days, current_bp=17.0)
    assert result["status"] == "fitted"
    assert result["change_point"] == pytest.approx(17.0, abs=0.5)
    assert result["solar_shift_c"] == pytest.approx(6.0, abs=1.0)

    # The no-solar model on the same data is pulled down — the bias.
    no_shift = min(
        (f for f in (fit_change_point(days, cp / 2) for cp in range(20, 45)) if f),
        key=lambda f: f["sse"],
    )
    assert no_shift["cp"] < result["change_point"] - 1.0


def test_yesterdays_sun_is_separated_from_the_change_point():
    """Heat stored from yesterday's sun still cuts today's demand.

    Without the carry-over term that reads as a lower balance point —
    about 1 °C low at 30 % carry-over in simulation.
    """
    result = sweep(_days(_history(shift=6.0, carryover=0.5, noise=0.01)))
    assert result["change_point"] == pytest.approx(17.0, abs=0.5)
    assert result["solar_carryover"] == pytest.approx(0.5)


def test_yesterdays_sun_after_a_gap_defaults_to_todays():
    base = dt_util.now().date() - timedelta(days=10)
    history = {}
    for offset, solar in ((0, 0.1), (1, 0.3), (3, 0.2)):
        history[(base + timedelta(days=offset)).isoformat()] = {
            "regime_heating_kwh": 1.0,
            "regime_cooling_kwh": 0.0,
            "solar_factor": solar,
            "hourly_vectors": {"temp": [5.0] * 24},
        }
    days = _days(history)
    assert [d["solar_prev"] for d in days] == [0.1, 0.1, 0.2]


def test_heat_pump_efficiency_curve_is_not_read_as_a_lower_balance_point():
    """COP rises with outdoor temperature, so energy against degree-hours
    bends; a straight arm crossed the floor at ~8 °C for a true 17 °C."""
    days = _days(_history(cop_slope=0.04, solar_max=0.0, relative_noise=0.05, noise=0.0))
    result = sweep(days, current_bp=17.0)
    assert result["change_point"] == pytest.approx(17.0, abs=1.0)
    assert result["cop_slope_per_c"] == pytest.approx(0.04, abs=0.01)

    linear = min(
        (f for f in (fit_change_point(days, cp / 2) for cp in range(16, 49)) if f),
        key=lambda f: f["sse"],
    )
    assert linear["cp"] < 14.0


def test_direct_electric_heating_fits_no_cop_slope():
    result = sweep(_days(_history(solar_max=0.0)))
    assert result["cop_slope_per_c"] == 0.0


def test_dark_day_balance_point_above_the_warmest_means_is_reachable():
    """Cool summer: few days average above 17 °C, but sun keeps sunny days
    on the flat side.  Capping the sweep at the K-th warmest daily mean made
    an 18 °C balance point unreachable."""
    history = _history(cp=18.0, shift=6.0, t_mid=5.0, t_amp=9.0, noise=0.01)
    days = _days(history)
    means = sorted(d["mean_t"] for d in days)
    assert means[-BP_CALIBRATION_MIN_DAYS_PER_SIDE] < 18.0
    result = sweep(days)
    assert result["optimum_at_sweep_boundary"] is False
    assert result["change_point"] == pytest.approx(18.0, abs=0.5)


def test_outlier_pass_does_not_drop_the_coldest_legitimate_days():
    """Heating-arm residuals scale with demand; one MAD across both arms
    dropped 3–9 % of clean days, all cold ones."""
    history = _history(relative_noise=0.10, noise=0.005, solar_max=0.0)
    result = sweep(_days(history))
    assert result["outlier_days_dropped"] <= 4


def test_no_sunshine_means_no_shift_is_swept():
    result = sweep(_days(_history(solar_max=0.0)))
    assert result["status"] == "fitted"
    assert result["solar_shift_c"] == 0.0
    assert result["solar_carryover"] == 0.0
    assert result["solar_shift_at_sweep_boundary"] is False


def test_sweep_reports_when_data_cannot_reach_the_range():
    """A winter-only climate cannot locate a 17 °C balance point."""
    cold = _history(t_mid=-14.0, t_amp=2.0, solar_max=0.0)
    result = sweep(_days(cold))
    assert result["status"] == "data_does_not_bracket_sweep_range"


def test_cold_climate_never_yields_a_suggestion():
    report = calibrate_balance_point(_coord(_history(t_mid=-8.0, t_amp=2.0), bp=12.0))
    assert report["suggested_balance_point"] is None
    assert report["verdict"] in ("insufficient_data", "optimum_at_sweep_boundary")


def test_min_days_per_side_is_enforced_by_the_sweep_range():
    days = _days(_history())
    result = sweep(days)
    means = sorted(d["mean_t"] for d in days)
    k = BP_CALIBRATION_MIN_DAYS_PER_SIDE
    assert result["swept_range"][0] >= means[k - 1] - 1e-9
    # The upper end reaches past the K-th warmest mean by the largest solar
    # shift: sunny days below cp can sit on the flat side.
    assert result["swept_range"][1] <= means[-k] + BP_CALIBRATION_SOLAR_SHIFT_MAX_C + 1e-9


def test_too_few_days_is_insufficient():
    result = sweep(_days(_history(days=BP_CALIBRATION_MIN_DAYS - 1)))
    assert result["status"] == "insufficient_days"


def test_proportional_zero_is_reported_beside_the_change_point():
    result = sweep(_days(_history(b=0.3, u=0.1, noise=0.01, solar_max=0.0)))
    assert result["proportional_gap_c"] == pytest.approx(3.0, abs=0.3)
    assert result["proportional_zero"] == pytest.approx(
        result["change_point"] + result["proportional_gap_c"], abs=0.01
    )


def test_outlier_days_are_dropped_and_the_fit_repeated():
    """A cold week with the heating off must not be fitted as the floor."""
    history = _history(noise=0.01, solar_max=0.0)
    cold_keys = sorted(history, key=lambda k: history[k]["temp"])[:7]
    for key in cold_keys:
        history[key]["regime_heating_kwh"] = 0.0
    result = sweep(_days(history))
    assert result["outlier_days_dropped"] >= 7
    assert set(cold_keys) <= set(result["outlier_dates"])
    assert result["change_point"] == pytest.approx(17.0, abs=0.5)


# ---------------------------------------------------------------------
# Day selection
# ---------------------------------------------------------------------

@pytest.mark.parametrize(
    "mutate,reason",
    [
        (lambda e: e.pop("regime_heating_kwh"), "missing_regime_split"),
        (lambda e: e.update(hourly_vectors={"temp": [5.0] * 20 + [None] * 4}), "incomplete_day"),
        (lambda e: e.update(aux_impact_kwh=0.4), "auxiliary_heat"),
        (lambda e: e.update(guest_impact_kwh=1.2), "guest_mode"),
        (lambda e: e.update(unit_breakdown={}), "unattributed_energy"),
        (lambda e: e.pop("unit_breakdown"), "unattributed_energy"),
        (lambda e: e.update(wind=9.5), "high_wind"),
        (lambda e: e["hourly_vectors"].update(wind=[9.0] * 7 + [2.0] * 17), "high_wind"),
        (lambda e: e.update(regime_heating_kwh=-1.0), "invalid_energy"),
        (lambda e: e.update(regime_cooling_kwh="x"), "invalid_energy"),
    ],
)
def test_excluded_days_are_counted_by_reason(mutate, reason):
    history = _history(days=3)
    key = sorted(history)[1]
    mutate(history[key])
    collected = collect_days(_coord(history), history, dt_util.now().date())
    assert len(collected["days"]) == 2
    assert collected["discarded"][reason] == 1
    assert collected["discarded"]["total_discarded"] == 1


def test_a_few_windy_hours_do_not_discard_the_day():
    history = _history(days=3)
    key = sorted(history)[1]
    history[key]["wind"] = 9.0  # the daily mean is not consulted when hours exist
    history[key]["hourly_vectors"]["wind"] = [9.0] * 6 + [2.0] * 18
    assert len(_days(history)) == 3


def test_csv_imported_days_are_not_fitted_as_zero_heating():
    """Imported days: real ``kwh``, no per-unit breakdown, split 0 / 0.

    Admitting them moved the fit to the sweep boundary; they are skipped.
    """
    history = _history(solar_max=0.0)
    for key in sorted(history)[::2]:
        history[key].update(regime_heating_kwh=0.0, regime_cooling_kwh=0.0, unit_breakdown={})
    report = calibrate_balance_point(_coord(history, bp=12.0))
    assert report["discarded_days"]["unattributed_energy"] == 183
    assert report["fit"]["change_point"] == pytest.approx(17.0, abs=0.5)


def test_dst_day_with_23_hours_is_kept():
    """Spring forward: 24 slots, one of them empty, and 23 hours of energy."""
    from zoneinfo import ZoneInfo
    from unittest.mock import patch

    history = _history(days=300)
    key = "2022-03-27"  # last Sunday of March, Europe/Oslo (test clock: 2023-01-01)
    history[key]["hourly_vectors"]["temp"][2] = None
    with patch.object(dt_util, "DEFAULT_TIME_ZONE", ZoneInfo("Europe/Oslo"), create=True):
        days = {d["date"]: d for d in _days(history)}
    assert days[key]["n"] == 23
    assert days[key]["y_heat"] == pytest.approx(history[key]["regime_heating_kwh"] / 23)


def test_gappy_day_is_not_used():
    """After downtime the missed hours' meter delta lands in the next logged
    hour: a 22-hour day carries ~24 hours of energy.  Only complete days."""
    history = _history(days=3)
    key = sorted(history)[1]
    history[key]["hourly_vectors"]["temp"][5] = None
    collected = collect_days(_coord(history), history, dt_util.now().date())
    assert [d["date"] for d in collected["days"]] == [k for k in sorted(history) if k != key]
    assert collected["discarded"]["incomplete_day"] == 1


def test_fall_back_day_divides_energy_by_25_hours():
    """Fall back: 25 hours of energy, two logs merged into one slot."""
    from zoneinfo import ZoneInfo
    from unittest.mock import patch

    history = _history(days=200)
    key = "2022-10-30"  # last Sunday of October, Europe/Oslo (test clock: 2023-01-01)
    assert key in history
    with patch.object(dt_util, "DEFAULT_TIME_ZONE", ZoneInfo("Europe/Oslo"), create=True):
        days = {d["date"]: d for d in _days(history)}
    assert days[key]["y_heat"] == pytest.approx(history[key]["regime_heating_kwh"] / 25)


def test_energy_is_per_hour_of_the_day():
    history = _history(days=2)
    key = sorted(history)[0]
    days = _days(history)
    assert days[0]["y_heat"] == pytest.approx(history[key]["regime_heating_kwh"] / 24)


def test_window_is_the_last_year_and_excludes_today():
    today = dt_util.now().date()
    history = _history(days=500, end=today + timedelta(days=1))
    collected = collect_days(_coord(history), history, today)
    dates = [d["date"] for d in collected["days"]]
    assert max(dates) < today.isoformat()
    assert min(dates) >= (today - timedelta(days=365)).isoformat()
    assert collected["days_in_window"] == 365


def test_inertia_temperature_is_reconstructed_across_consecutive_days():
    """Equal weights over two hours → each hour averages itself and the
    previous one, continuing across midnight and restarting after a gap."""
    base = dt_util.now().date() - timedelta(days=10)
    history = {}
    for i, offset in enumerate((0, 1, 3)):  # day 2 missing: gap before the third day
        history[(base + timedelta(days=offset)).isoformat()] = {
            "regime_heating_kwh": 1.0,
            "regime_cooling_kwh": 0.0,
            "hourly_vectors": {"temp": [float(10 * i + h % 2) for h in range(24)]},
        }
    days = _days(history, inertia_weights=[1.0, 1.0])
    first, second, third = days
    # First hour of the series has no history: its own temperature.
    assert first["sorted_t"][0] == pytest.approx(0.0)
    # Second day's first hour averages 10.0 with the previous day's last (1.0).
    assert pytest.approx(5.5) in second["sorted_t"]
    # After the gap the series restarts: 20.0 stands alone.
    assert third["sorted_t"][0] == pytest.approx(20.0)


def test_inertia_matches_the_live_coordinator_methods():
    """Replay live: each hour closes just after HH:00, calls the real
    ``_get_recent_log_temps`` (from the closed hour's start, as
    ``hourly_processor`` does) + ``_calculate_weighted_inertia``, then logs
    the hour stamped at its start.  The reconstruction must agree."""
    import types
    from datetime import datetime, timezone

    from custom_components.heating_analytics.coordinator import HeatingDataCoordinator
    from custom_components.heating_analytics.helpers import generate_exponential_kernel

    tau = 4.0
    weights = list(generate_exponential_kernel(tau=tau, window_hours=20))
    live = types.SimpleNamespace(inertia_weights=weights, inertia_tau=tau, _hourly_log=[])
    for name in ("_get_recent_log_temps", "_calculate_weighted_inertia"):
        setattr(live, name, types.MethodType(getattr(HeatingDataCoordinator, name), live))

    first = dt_util.now().date() - timedelta(days=4)
    history = {}
    live_inertia = {}
    for i in range(2):
        d = first + timedelta(days=i)
        temps = [round(5.0 + 6.0 * math.sin(h / 3.0) + i, 3) for h in range(24)]
        history[d.isoformat()] = {
            "regime_heating_kwh": 1.0, "regime_cooling_kwh": 0.0,
            "hourly_vectors": {"temp": temps},
        }
        for h, t in enumerate(temps):
            start = datetime(d.year, d.month, d.day, h, tzinfo=timezone.utc)
            window = live._get_recent_log_temps(start) + [t]
            live_inertia[(d.isoformat(), h)] = live._calculate_weighted_inertia(window)
            live._hourly_log.append({"timestamp": start.isoformat(), "temp": t})

    coord = _coord(history, inertia_weights=weights)
    coord.inertia_tau = tau
    days = collect_days(coord, history, dt_util.now().date())["days"]
    second = sorted(history)[1]
    day = next(x for x in days if x["date"] == second)
    expected = sorted(live_inertia[(second, h)] for h in range(24))
    assert day["sorted_t"] == pytest.approx(expected)


def test_a_missing_hour_does_not_reset_the_series():
    """The gappy day itself is dropped, but the next day's first hours still
    average across the gap, as live does."""
    base = dt_util.now().date() - timedelta(days=5)
    day1 = [float(h) for h in range(24)]
    day1[23] = None
    history = {
        base.isoformat(): {"regime_heating_kwh": 1.0, "regime_cooling_kwh": 0.0,
                           "hourly_vectors": {"temp": day1}},
        (base + timedelta(days=1)).isoformat(): {
            "regime_heating_kwh": 1.0, "regime_cooling_kwh": 0.0,
            "hourly_vectors": {"temp": [30.0] * 24}},
    }
    coord = _coord(history, inertia_weights=[1.0, 1.0, 1.0])
    coord.inertia_tau = 4.0  # window of τ − 1 = 3 hours
    days = collect_days(coord, history, dt_util.now().date())["days"]
    assert len(days) == 1
    # Day 2, hour 0: hours 22 (22.0) and 0 (30.0) — hour 23 is missing.
    assert pytest.approx(26.0) in days[0]["sorted_t"]


def test_raw_temperature_is_used_without_a_usable_kernel():
    history = _history(days=2)
    key = sorted(history)[0]
    for weights in (None, MagicMock(), [], ["x"], [0.0, 0.0]):
        days = _days(history, inertia_weights=weights)
        assert days[0]["sorted_t"] == sorted(history[key]["hourly_vectors"]["temp"])


# ---------------------------------------------------------------------
# The service report
# ---------------------------------------------------------------------

def test_suggests_the_change_point_when_configured_value_is_off():
    report = calibrate_balance_point(_coord(_history(shift=4.0), bp=12.0))
    assert report["verdict"] == "suggest_change"
    assert report["suggested_balance_point"] == pytest.approx(17.0, abs=0.5)
    assert report["stability"]["stable"] is True
    assert "impact" in report


def test_no_suggestion_when_configured_value_is_consistent():
    report = calibrate_balance_point(_coord(_history(shift=4.0), bp=17.0))
    assert report["verdict"] == "current_is_consistent"
    assert report["suggested_balance_point"] is None


def test_fit_is_independent_of_the_configured_value():
    history = _history()
    a = calibrate_balance_point(_coord(history, bp=12.0))
    b = calibrate_balance_point(_coord(history, bp=20.0))
    assert a["best_change_point"] == b["best_change_point"]


def test_halves_that_disagree_withhold_the_suggestion():
    even = _history(cp=14.0, seed=2)
    odd = _history(cp=19.0, seed=3)
    history = {
        k: (even[k] if date.fromisoformat(k).isocalendar()[1] % 2 == 0 else odd[k])
        for k in even
    }
    report = calibrate_balance_point(_coord(history, bp=10.0))
    assert report["verdict"] == "unstable_across_halves"
    assert report["suggested_balance_point"] is None
    cps = report["stability"]["half_change_points"]
    assert abs(cps[0] - cps[1]) > 1.0


def test_cop_sensitivity_reports_the_change_point_per_slope():
    """cp and κ trade off; the table shows how far cp moves with κ."""
    report = calibrate_balance_point(_coord(
        _history(cop_slope=0.03, relative_noise=0.05, noise=0.0, solar_max=0.0), bp=12.0,
    ))
    cop = report["fit"]["cop_sensitivity"]
    rows = cop["change_point_by_cop_slope"]
    assert [r["cop_slope"] for r in rows] == [0.0, 0.01, 0.02, 0.03, 0.04, 0.05]
    cps = [r["change_point"] for r in rows]
    assert cps == sorted(cps)  # a steeper COP curve moves cp up
    assert cps[0] < 14.0  # the linear arm's bias, visible in the table
    assert 0.03 in cop["supported_cop_slopes"]
    lo, hi = cop["change_point_range_over_supported"]
    assert lo <= 17.0 <= hi + 0.5


def test_direct_electric_change_point_does_not_depend_on_the_cop_slope():
    report = calibrate_balance_point(_coord(_history(solar_max=0.0), bp=12.0))
    cop = report["fit"]["cop_sensitivity"]
    assert cop["best_cop_slope"] == 0.0
    assert cop["supported_cop_slopes"]  # never empty: the best slope's row is in it
    assert 0.0 in cop["supported_cop_slopes"]
    assert cop["stable"] is True
    assert report["verdict"] == "suggest_change"


def test_cp_that_moves_with_the_cop_slope_is_withheld(monkeypatch):
    """Halves agree, but only given one κ: cp is not identified."""
    import custom_components.heating_analytics.balance_point as bp_mod

    fitted = {
        "status": "fitted", "change_point": 20.0, "solar_shift_c": 4.0,
        "cop_slope_per_c": 0.04, "limits_hit": [], "optimum_at_sweep_boundary": False,
        "proportional_zero": 21.0, "proportional_gap_c": 1.0,
        "current_within_interval": False,
        "cop_sensitivity": {"stable": False, "change_point_range_over_supported": [17.5, 20.0]},
    }
    monkeypatch.setattr(bp_mod, "sweep", lambda days, **kw: dict(fitted))
    report = calibrate_balance_point(_coord(_history(days=70), bp=17.0))
    assert report["verdict"] == "cp_depends_on_cop_slope"
    assert report["suggested_balance_point"] is None


def test_cp_moving_with_the_cop_slope_withholds_end_to_end():
    """Noisy mild-climate heat pump: the data supports κ from 0.01 to 0.04
    and cp moves 2 °C across them."""
    report = calibrate_balance_point(_coord(_history(
        shift=5, cop_slope=0.03, relative_noise=0.25, noise=0.05, t_mid=8, t_amp=6, seed=1,
    ), bp=12.0))
    cop = report["fit"]["cop_sensitivity"]
    lo, hi = cop["change_point_range_over_supported"]
    assert hi - lo > 1.0
    assert report["verdict"] == "cp_depends_on_cop_slope"
    assert report["suggested_balance_point"] is None


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_heteroscedastic_noise_keeps_the_true_cop_slope_supported(seed):
    """Heating energy scatters more on cold days, which carry the κ signal.
    An equal-variance threshold rejected the true slope about half the
    time; the weighted fit keeps it."""
    report = calibrate_balance_point(_coord(_history(
        cop_slope=0.03, relative_noise=0.1, noise=0.0, solar_max=0.0, seed=seed,
    ), bp=17.0))
    assert report["fit"]["weighted"] is True
    assert 0.03 in report["fit"]["cop_sensitivity"]["supported_cop_slopes"]
    assert report["suggested_balance_point"] is None


def test_regression_true_cop_slope_rejected_gave_a_false_suggestion():
    """Review reproduction: supported slopes were [0.04] only and the fit
    suggested 19.5 °C for a true 17 °C."""
    report = calibrate_balance_point(_coord(_history(
        shift=5, carryover=0.2, cop_slope=0.03, relative_noise=0.1, noise=0.03, seed=100,
    ), bp=17.0))
    assert report["suggested_balance_point"] is None
    assert 0.03 in report["fit"]["cop_sensitivity"]["supported_cop_slopes"]


def test_regression_empty_supported_set_is_never_stable():
    """Review reproduction: best κ off the table grid, neither neighbouring
    row within the threshold → no supported slope, read as stable, and an
    18 °C suggestion for a true 17 °C."""
    report = calibrate_balance_point(_coord(_history(seed=7), bp=17.0))
    cop = report["fit"]["cop_sensitivity"]
    assert cop["supported_cop_slopes"]
    assert report["suggested_balance_point"] is None


def test_off_grid_cop_slope_is_computed_on_demand():
    day = _days(_history(days=2))[0]
    base = 17.0
    expected = sum(
        max(0.0, base - t) / (1.0 + 0.033 * t) for t in day["sorted_t"]
    ) / day["n"]
    assert degree_hours(day, base, cop_slope=0.033) == pytest.approx(expected)


def test_poor_fit_in_one_half_is_reported_as_poor_fit():
    """A half the model does not describe is not "too few days"."""
    import random as _random

    history = _history(solar_max=0.0, noise=0.01)
    rng = _random.Random(3)
    odd = [k for k in sorted(history) if date.fromisoformat(k).isocalendar()[1] % 2 == 1]
    for key in rng.sample(odd, 40):
        history[key]["regime_heating_kwh"] *= rng.choice((0.1, 4.0))
    report = calibrate_balance_point(_coord(history, bp=12.0))
    assert report["fit"]["status"] == "fitted"
    assert report["stability"]["half_statuses"]["odd_weeks"] == "poor_fit"
    assert report["verdict"] == "poor_fit"


def test_cop_slope_beyond_its_limit_withholds_the_suggestion():
    """κ = 0.08 exceeds any air-to-air heat pump; the limit is not raised."""
    report = calibrate_balance_point(_coord(
        _history(cop_slope=0.08, relative_noise=0.05, noise=0.0, solar_max=0.0), bp=12.0,
    ))
    assert "cop_slope" in report["fit"]["limits_hit"]
    assert report["suggested_balance_point"] is None


def test_cop_stability_is_not_assessable_with_the_slope_on_its_limit():
    """Only the limit row supported: the slopes that would move cp further
    lie outside the table, so "stable" would be a blind answer."""
    report = calibrate_balance_point(_coord(
        _history(cop_slope=0.08, relative_noise=0.05, noise=0.0, solar_max=0.0), bp=12.0,
    ))
    cop = report["fit"]["cop_sensitivity"]
    assert cop["best_cop_slope"] == 0.05
    assert cop["stable"] is None
    assert report["suggested_balance_point"] is None


def test_carryover_at_its_limit_withholds_the_suggestion():
    report = calibrate_balance_point(_coord(_history(shift=6.0, carryover=0.9, noise=0.01), bp=12.0))
    assert report["limits_hit"] == ["solar_carryover"]
    assert report["verdict"] == "nuisance_at_limit"
    assert report["suggested_balance_point"] is None


def test_carryover_limit_does_not_count_without_a_material_shift():
    """Carry-over acts only through the shift: with a small shift it moves
    each day's base by a fraction of a degree and is not identified."""
    from custom_components.heating_analytics.balance_point import carryover_at_limit
    from custom_components.heating_analytics.const import (
        BP_CALIBRATION_CARRYOVER_MIN_SHIFT_C,
        BP_CALIBRATION_SOLAR_CARRYOVER_GRID,
    )

    grid = list(BP_CALIBRATION_SOLAR_CARRYOVER_GRID)
    top = grid[-1]
    assert carryover_at_limit(top, BP_CALIBRATION_CARRYOVER_MIN_SHIFT_C, grid) is True
    assert carryover_at_limit(top, BP_CALIBRATION_CARRYOVER_MIN_SHIFT_C - 0.5, grid) is False
    assert carryover_at_limit(grid[-2], 10.0, grid) is False
    assert carryover_at_limit(0.0, 10.0, [0.0]) is False  # no sun: nothing swept


def test_a_fit_that_needs_many_outliers_dropped_is_a_poor_fit():
    """Sun-driven spring cooling, say: a third of the days fit nothing."""
    import random as _random

    history = _history(solar_max=0.0, noise=0.01)
    rng = _random.Random(5)
    for key in rng.sample(sorted(history), 90):
        history[key]["regime_heating_kwh"] *= rng.choice((0.1, 4.0))
    report = calibrate_balance_point(_coord(history, bp=12.0))
    assert report["fit"]["status"] == "poor_fit"
    assert report["fit"]["outlier_share"] > 0.15
    assert report["verdict"] == "poor_fit"
    assert report["suggested_balance_point"] is None


def test_too_few_days_to_split_is_not_reported_as_disagreement():
    report = calibrate_balance_point(_coord(_history(days=100), bp=12.0))
    assert report["fit"]["status"] == "fitted"
    assert report["verdict"] == "insufficient_data_for_stability"
    assert report["stability"]["halves_fitted"] is False
    assert report["suggested_balance_point"] is None


def test_report_is_json_serialisable():
    import json

    report = calibrate_balance_point(_coord(_history(shift=4.0, cooling_cp=22.0), bp=12.0))
    json.dumps(report, allow_nan=False)


def test_insufficient_data_on_an_empty_history():
    report = calibrate_balance_point(_coord({}, bp=17.0))
    assert report["verdict"] == "insufficient_data"
    assert report["suggested_balance_point"] is None
    assert report["fit"]["status"] == "insufficient_days"


def test_history_without_regime_split_is_reported_not_guessed():
    history = _history()
    for entry in history.values():
        entry.pop("regime_heating_kwh")
    report = calibrate_balance_point(_coord(history))
    assert report["verdict"] == "insufficient_data"
    assert report["discarded_days"]["missing_regime_split"] == 365


def test_large_proportional_gap_is_flagged():
    report = calibrate_balance_point(_coord(_history(b=0.3, u=0.1, noise=0.01)))
    assert report["proportional_gap_large"] is True


def test_cooling_change_point_is_reported_not_suggested():
    report = calibrate_balance_point(_coord(_history(cooling_cp=22.0, t_mid=10.0, t_amp=12.0)))
    cooling = report["cooling_change_point"]
    assert cooling["status"] == "fitted"
    assert cooling["change_point"] == pytest.approx(22.0, abs=1.0)
    assert report["best_change_point"] == pytest.approx(17.0, abs=0.5)


def test_no_cooling_block_on_a_heating_only_install():
    report = calibrate_balance_point(_coord(_history()))
    assert "cooling_change_point" not in report


def test_u_curve_cross_check_reports_a_flat_bottom():
    corr = {str(t): {"normal": max(0.1, 0.12 * (15 - t))} for t in range(-5, 22)}
    report = calibrate_balance_point(_coord(_history(), correlation=corr))
    cross = report["u_curve_cross_check"]
    assert cross["status"] == "ok"
    lo, hi = cross["flat_bottom_range"]
    assert lo <= 15 <= hi
    assert cross["has_warm_arm"] is False


def test_nothing_is_written():
    coord = _coord(_history(), bp=12.0)
    calibrate_balance_point(coord)
    assert coord.balance_point == 12.0
    coord.hass.config_entries.async_update_entry.assert_not_called()


# ---------------------------------------------------------------------
# Service handler
# ---------------------------------------------------------------------

@pytest.mark.asyncio
async def test_handler_runs_the_fit_off_the_event_loop_on_a_snapshot():
    """The fit takes about a second on a year of history: it must run in
    the executor, on copies the coordinator cannot mutate mid-fit."""
    from unittest.mock import AsyncMock, patch

    from custom_components.heating_analytics import (
        SERVICE_CALIBRATE_BALANCE_POINT,
        async_setup_entry,
    )
    from custom_components.heating_analytics.balance_point import (
        calibrate_balance_point as service_fn,
    )
    from custom_components.heating_analytics.const import DOMAIN

    hass = MagicMock()
    hass.data = {}
    hass.config_entries.async_forward_entry_setups = AsyncMock()
    captured = {}

    def _register(domain, service, callback, schema=None, **kwargs):
        if domain == DOMAIN and service == SERVICE_CALIBRATE_BALANCE_POINT:
            captured["handler"] = callback

    hass.services.async_register = MagicMock(side_effect=_register)
    jobs = []

    async def _executor(fn, *args):
        jobs.append((fn, args))
        return fn(*args)

    hass.async_add_executor_job = AsyncMock(side_effect=_executor)
    entry = MagicMock()
    entry.entry_id = "test_entry"
    entry.data = {"outdoor_temp_sensor": "sensor.temp", "energy_sensors": ["sensor.a"]}

    history = _history(shift=4.0)
    coord = _coord(history, bp=12.0)
    coord.async_config_entry_first_refresh = AsyncMock()
    coord.storage.async_load_data = AsyncMock()
    coord._async_save_data = AsyncMock()
    coord.entry = entry
    with patch("custom_components.heating_analytics.HeatingDataCoordinator", return_value=coord), \
         patch("custom_components.heating_analytics._get_target_coordinator", return_value=coord):
        await async_setup_entry(hass, entry)
        call = MagicMock()
        call.data = {}
        report = await captured["handler"](call)

    assert len(jobs) == 1
    fn, (job_coord, snapshot) = jobs[0]
    assert fn is service_fn and job_coord is coord
    assert snapshot["daily_history"] == history
    assert snapshot["daily_history"] is not history
    assert report["verdict"] == "suggest_change"
