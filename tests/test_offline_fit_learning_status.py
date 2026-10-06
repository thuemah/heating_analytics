"""Offline solar fits skip the hours live learning skipped (#1087).

Live learning skips 20–80 % aux hours (``skipped_mixed_mode``), solar-and-aux
hours (``skipped_dual_interference``), post-aux cooldown for aux-affected
units, and every hour while learning is off (``disabled``).  Those statuses
live only in ``learning_status``; the ``auxiliary_active`` flag the offline
paths already checked is set only at ≥ 80 % aux.  Aux lowers ``actual``
independently of the sun, so an admitted mixed-mode hour reads as solar gain
and inflates the coefficient.

Covers the shared predicate and each of the five offline paths:
``_collect_batch_fit_samples`` (``batch_fit_solar`` and, via
``match_diagnose``, ``apply_implied_coefficient``),
``_collect_batch_fit_samples_4d``, ``fit_solar_obstruction``,
``diagnose_solar``'s per-unit implied coefficient, and the per-unit
min-base calibration.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from custom_components.heating_analytics.diagnostics import DiagnosticsEngine
from custom_components.heating_analytics.helpers import (
    UNIT_LEARNING_SKIP_REASONS,
    hour_learning_skip_reason,
    aux_affected_entities_of,
    unit_learning_skip_reason,
)
from custom_components.heating_analytics.learning import LearningManager

from tests.test_batch_fit_solar_4d import (
    _entry as _entry_4d,
    _make_coord as _make_coord_4d,
)
from tests.test_per_unit_min_base import (
    MockCoord as _MinBaseCoord,
    _calibrate,
    _fill_dark_samples,
)
from tests.test_solar_diagnose import (
    _hour_entry as _diag_entry,
    _make_coord as _make_diag_coord,
)
from tests.test_solar_obstruction_gate import (
    SENSOR as _OBS_SENSOR,
    _build_log_explicit,
    _make_fit_coord,
    _south_dominant_positions,
    _stub_get_approx_sun_pos,
)


# ---------------------------------------------------------------------------
# The predicate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        ("disabled", "learning_disabled"),
        ("skipped_no_data", "learning_no_data"),
        ("skipped_mixed_mode", "mixed_mode_aux"),
        ("skipped_dual_interference", "dual_interference"),
    ],
)
def test_statuses_where_unit_learning_did_not_run(status, expected):
    entry = {"learning_status": status}
    assert unit_learning_skip_reason(entry, "sensor.a", {"sensor.a"}) == expected
    # Independent of aux scope: per-unit learning did not run for anyone.
    assert unit_learning_skip_reason(entry, "sensor.b", {"sensor.a"}) == expected


@pytest.mark.parametrize(
    "status",
    [
        "active",
        "disabled_global_only",  # Track B/C: per-unit learning still runs
        "skipped_global_saturation",  # only the global base write is skipped
        "active_aux_update (1.20kW)",
        "active_aux_no_change",
        "aux_skipped_guest_mode",
    ],
)
def test_statuses_where_unit_learning_ran_are_kept(status):
    assert unit_learning_skip_reason({"learning_status": status}, "sensor.a") is None


def test_skipped_global_saturation_is_kept():
    """Not a ``startswith("skipped_")`` prefix rule.

    Solar-saturated hours skip only the global base write; per-unit NLMS
    still learns from them, and they are the censored rows Tobit uses.
    """
    entry = {"learning_status": "skipped_global_saturation"}
    assert unit_learning_skip_reason(entry, "sensor.a", None) is None


def test_cooldown_skips_only_aux_affected_entities():
    """Mirrors ``_process_per_unit_learning``: cooldown freezes aux-affected
    units only; the rest keep learning."""
    entry = {"learning_status": "cooldown_post_aux"}
    assert unit_learning_skip_reason(entry, "sensor.a", {"sensor.a"}) == "post_aux_cooldown"
    assert unit_learning_skip_reason(entry, "sensor.b", {"sensor.a"}) is None
    # Configured, but empty: nothing is aux-affected.
    assert unit_learning_skip_reason(entry, "sensor.a", set()) is None
    # None = every entity affected.
    assert unit_learning_skip_reason(entry, "sensor.b", None) == "post_aux_cooldown"


def test_cooldown_snapshot_is_authoritative_under_daily_learning():
    """Daily learning logs ``disabled_global_only`` on a cooldown hour while
    per-unit learning still skips aux-affected units; the snapshot says so."""
    entry = {
        "learning_status": "disabled_global_only",
        "aux_cooldown_entities": ["sensor.a"],
    }
    assert unit_learning_skip_reason(entry, "sensor.a", None) == "post_aux_cooldown"
    assert unit_learning_skip_reason(entry, "sensor.b", None) is None


def test_cooldown_snapshot_wins_over_current_aux_scope():
    """A later reconfiguration of ``aux_affected_entities`` must not rewrite
    which units a historical cooldown froze."""
    entry = {
        "learning_status": "cooldown_post_aux",
        "aux_cooldown_entities": ["sensor.a"],
    }
    assert unit_learning_skip_reason(entry, "sensor.a", {"sensor.b"}) == "post_aux_cooldown"
    assert unit_learning_skip_reason(entry, "sensor.b", {"sensor.b"}) is None
    # Empty snapshot: cooldown was active but froze no unit.
    empty = {"learning_status": "cooldown_post_aux", "aux_cooldown_entities": []}
    assert unit_learning_skip_reason(empty, "sensor.a", None) is None


def test_status_skip_takes_precedence_over_cooldown_snapshot():
    entry = {"learning_status": "disabled", "aux_cooldown_entities": []}
    assert unit_learning_skip_reason(entry, "sensor.a", None) == "learning_disabled"


def test_malformed_cooldown_snapshot_falls_back_to_status():
    # A string is not a snapshot; it must not substring-match entity ids.
    entry = {"learning_status": "active", "aux_cooldown_entities": "sensor.abc"}
    assert unit_learning_skip_reason(entry, "sensor.a", None) is None


@pytest.mark.parametrize("entry", [{}, {"learning_status": None}, {"learning_status": 3}])
def test_missing_or_malformed_status_is_kept(entry):
    """Pre-field logs carry no status; kept, as live learning did."""
    assert unit_learning_skip_reason(entry, "sensor.a") is None


@pytest.mark.parametrize(
    "entry",
    [
        {"learning_status": "disabled"},
        {"learning_status": "skipped_no_data"},
        {"learning_status": "skipped_mixed_mode"},
        {"learning_status": "skipped_dual_interference"},
        {"learning_status": "cooldown_post_aux"},
        {"learning_status": "disabled_global_only", "aux_cooldown_entities": ["e"]},
        {"learning_status": "skipped_global_saturation"},
        {"learning_status": "active"},
        {},
    ],
)
def test_hour_reason_is_the_entity_independent_part(entry):
    """``hour_learning_skip_reason`` returns exactly what every unit shares.

    Cooldown is left to the per-entity check; for everything else the
    per-unit answer is the same for every entity and equals the hour's.
    """
    hour = hour_learning_skip_reason(entry)
    per_unit = unit_learning_skip_reason(entry, "e", None)
    if per_unit == "post_aux_cooldown":
        assert hour is None
    else:
        assert hour == per_unit


def test_reason_keys_cover_every_return_value():
    returned = {
        unit_learning_skip_reason({"learning_status": s}, "e", None)
        for s in (
            "disabled", "skipped_no_data", "skipped_mixed_mode",
            "skipped_dual_interference", "cooldown_post_aux",
        )
    }
    assert returned == set(UNIT_LEARNING_SKIP_REASONS)


def test_aux_affected_entities_of():
    coord = MagicMock()
    coord.aux_affected_entities = ["sensor.a"]
    assert aux_affected_entities_of(coord) == {"sensor.a"}
    coord.aux_affected_entities = []
    assert aux_affected_entities_of(coord) == set()
    # A stub attribute is not a collection → conservative "all affected".
    assert aux_affected_entities_of(MagicMock()) is None
    assert aux_affected_entities_of(object()) is None


# ---------------------------------------------------------------------------
# 3D batch-fit collector (batch_fit_solar / apply_implied_coefficient)
# ---------------------------------------------------------------------------

_E3 = "sensor.heater3d"


def _collect_3d(statuses, *, for_tobit, match_diagnose=False, aux_affected=None):
    coordinator = MagicMock()
    coordinator.model.correlation_data_per_unit = {_E3: {"10": {"normal": 2.0}}}
    if aux_affected is not None:
        coordinator.aux_affected_entities = aux_affected
    hourly_log = [
        {
            "unit_modes": {_E3: "heating"},
            "unit_breakdown": {_E3: 1.5},  # impact 0.5, unsaturated
            "unit_expected_breakdown": {_E3: 2.0},
            "temp_key": "10",
            "wind_bucket": "normal",
            "learning_status": status,
        }
        for status in statuses
    ]
    potentials = [(1.0, 0.0, 0.0, 1.0)] * len(hourly_log)
    return LearningManager()._collect_batch_fit_samples(
        entity_id=_E3,
        regime="heating",
        hourly_log=hourly_log,
        entry_potentials=potentials,
        coordinator=coordinator,
        unit_threshold=0.1,
        screen_affected_entities=None,
        for_tobit=for_tobit,
        match_diagnose=match_diagnose,
        solar_coefficients_per_unit={},
    )


_MIXED_LOG = (
    ["active"] * 6
    + ["skipped_mixed_mode"] * 3
    + ["skipped_dual_interference"] * 2
    + ["disabled"]
    + ["skipped_global_saturation"] * 2
)


@pytest.mark.parametrize(
    ("for_tobit", "match_diagnose"),
    [(True, False), (False, True)],
    ids=["batch_fit_tobit", "apply_implied"],
)
def test_3d_collector_drops_learning_skipped_hours(for_tobit, match_diagnose):
    samples, _, drops = _collect_3d(
        _MIXED_LOG, for_tobit=for_tobit, match_diagnose=match_diagnose,
    )
    # 6 active + 2 global-saturation kept.
    assert len(samples) == 8, drops
    assert drops["mixed_mode_aux"] == 3
    assert drops["dual_interference"] == 2
    assert drops["learning_disabled"] == 1
    assert drops["post_aux_cooldown"] == 0


def test_3d_collector_reason_keys_always_reported():
    _, _, drops = _collect_3d(["active"] * 3, for_tobit=True)
    for reason in UNIT_LEARNING_SKIP_REASONS:
        assert drops[reason] == 0


def test_3d_collector_cooldown_respects_aux_scope():
    statuses = ["active"] * 4 + ["cooldown_post_aux"] * 3
    samples, _, drops = _collect_3d(statuses, for_tobit=True, aux_affected=[_E3])
    assert len(samples) == 4
    assert drops["post_aux_cooldown"] == 3
    samples, _, drops = _collect_3d(statuses, for_tobit=True, aux_affected=["sensor.other"])
    assert len(samples) == 7
    assert drops["post_aux_cooldown"] == 0


# ---------------------------------------------------------------------------
# 4D batch-fit collector (batch_fit_solar_4d)
# ---------------------------------------------------------------------------


def test_4d_collector_drops_learning_skipped_hours():
    sid = "sensor.heater1"
    coord = _make_coord_4d(correlation_data_per_unit={sid: {"10": {"normal": 2.5}}})
    statuses = ["logged"] * 4 + ["skipped_mixed_mode"] * 2 + ["disabled"]
    log = []
    for i, status in enumerate(statuses):
        e = _entry_4d(f"2026-04-{i + 1:02d}T11:00:00", sensor_id=sid, actual_kwh=1.5)
        e["learning_status"] = status
        log.append(e)
    samples, _, drops = LearningManager()._collect_batch_fit_samples_4d(
        entity_id=sid,
        regime="heating",
        hourly_log=log,
        coordinator=coord,
        screen_affected_entities=None,
    )
    assert len(samples) == 4, drops
    assert drops["mixed_mode_aux"] == 2
    assert drops["learning_disabled"] == 1


# ---------------------------------------------------------------------------
# fit_solar_obstruction
# ---------------------------------------------------------------------------


def _obstruction_run(skipped_status: str | None):
    coord, solar = _make_fit_coord()
    positions = _south_dominant_positions([float(e) for e in range(10, 71)] * 2)
    entries, sun_pos_by_ts = _build_log_explicit(positions, true_crit_s=30.0)
    n_skipped = 0
    if skipped_status is not None:
        for e in entries[::3]:
            e["learning_status"] = skipped_status
            n_skipped += 1
    solar.get_approx_sun_pos = _stub_get_approx_sun_pos(sun_pos_by_ts)
    result = LearningManager().fit_solar_obstruction(
        hourly_log=entries, coordinator=coord, dry_run=True,
    )
    return result, len(entries), n_skipped


def test_obstruction_fit_drops_learning_skipped_hours():
    baseline, n_total, _ = _obstruction_run(None)
    assert baseline[_OBS_SENSOR]["s"]["n_samples"] == n_total
    assert baseline["n_skipped_learning_status"]["mixed_mode_aux"] == 0

    result, _, n_skipped = _obstruction_run("skipped_mixed_mode")
    assert n_skipped > 0
    assert result["n_skipped_learning_status"]["mixed_mode_aux"] == n_skipped
    assert result[_OBS_SENSOR]["s"]["n_samples"] == n_total - n_skipped


def test_obstruction_fit_skips_shutdown_constraint_on_skipped_hour():
    """A mixed-mode hour's shutdown flag is aux, not sun: no constraint."""
    coord, solar = _make_fit_coord()
    positions = _south_dominant_positions([float(e) for e in range(10, 71)])
    entries, sun_pos_by_ts = _build_log_explicit(positions)
    for e in entries:
        e["solar_dominant_entities"] = [_OBS_SENSOR]
        e["learning_status"] = "skipped_mixed_mode"
    solar.get_approx_sun_pos = _stub_get_approx_sun_pos(sun_pos_by_ts)
    result = LearningManager().fit_solar_obstruction(
        hourly_log=entries, coordinator=coord, dry_run=True,
    )
    assert result["n_skipped_learning_status"]["mixed_mode_aux"] == len(entries)
    assert result[_OBS_SENSOR]["s"]["n_samples"] == 0


# ---------------------------------------------------------------------------
# diagnose_solar implied coefficient
# ---------------------------------------------------------------------------


def test_diagnose_implied_skips_learning_skipped_hours():
    from datetime import datetime, timedelta

    base_dt = datetime(2026, 4, 1, 12, 0)
    entries = []
    for i in range(12):
        e = _diag_entry(
            (base_dt + timedelta(hours=i)).isoformat(),
            solar_s=0.5, solar_e=0.15 if i % 2 else 0.05,
            actual=1.5, base=2.0, temp=5.0,
        )
        entries.append(e)
    clean = DiagnosticsEngine(_make_diag_coord(list(entries))).diagnose_solar(days_back=30)

    # Four more hours at the same weather, logged as mixed-mode aux with
    # actual pulled down by aux — they would read as extra solar.
    for i in range(4):
        e = _diag_entry(
            (base_dt + timedelta(hours=12 + i)).isoformat(),
            solar_s=0.5, solar_e=0.15, actual=0.2, base=2.0, temp=5.0,
        )
        e["learning_status"] = "skipped_mixed_mode"
        entries.append(e)
    mixed = DiagnosticsEngine(_make_diag_coord(entries)).diagnose_solar(days_back=30)
    assert clean["per_unit"]["sensor.heater1"]["implied_coefficient_30d"] is not None

    assert mixed["global"]["excluded"]["mixed_mode_aux"] == 4
    assert mixed["global"]["qualifying_hours"] == clean["global"]["qualifying_hours"]
    assert (
        mixed["per_unit"]["sensor.heater1"]["implied_coefficient_30d"]
        == clean["per_unit"]["sensor.heater1"]["implied_coefficient_30d"]
    )


# ---------------------------------------------------------------------------
# Per-unit min-base calibration
# ---------------------------------------------------------------------------


def test_min_base_calibration_skips_learning_skipped_hours():
    """Aux-depressed dark hours must not set the noise floor low."""
    sid = "sensor.a"
    coord = _MinBaseCoord([sid])
    _fill_dark_samples(coord, {sid: [0.20 + 0.01 * i for i in range(60)]})
    clean = _calibrate(coord)
    p10_clean = clean["units"][sid]["p10_actual"]

    # Add aux-depressed dark hours logged as mixed-mode.
    n_mixed = 30
    ts = coord._hourly_log[-1]["timestamp"]
    for _ in range(n_mixed):
        coord._hourly_log.append({
            "timestamp": ts,
            "hour": int(ts[11:13]),
            "solar_factor": 0.0,
            "auxiliary_active": False,
            "unit_modes": {sid: "heating"},
            "unit_breakdown": {sid: 0.05},
            "learning_status": "skipped_mixed_mode",
        })
    coord._per_unit_min_base_thresholds = {}
    result = _calibrate(coord)
    assert result["learning_skipped"]["mixed_mode_aux"] == n_mixed
    assert result["units"][sid]["dark_samples"] == clean["units"][sid]["dark_samples"]
    assert result["units"][sid]["p10_actual"] == p10_clean
    assert result["units"][sid]["status"] == clean["units"][sid]["status"]
