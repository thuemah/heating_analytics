"""Building-level thermal regime classification (#1051).

The label decides whether one regime's sign conventions may be applied to the
whole building, so it is deliberately demand-weighted and deliberately
conservative about committing to a side.
"""
import pytest

from custom_components.heating_analytics.const import (
    MODE_COOLING,
    MODE_DHW,
    MODE_GUEST_COOLING,
    MODE_GUEST_HEATING,
    MODE_HEATING,
    MODE_OFF,
)
from custom_components.heating_analytics.learning import (
    classify_thermal_regime,
    dominant_thermal_regime,
)


def test_all_heating_is_heating():
    modes = {"a": MODE_HEATING, "b": MODE_HEATING}
    energy = {"a": 5.0, "b": 3.0}
    assert classify_thermal_regime(modes, energy) == "heating"


def test_all_cooling_is_cooling():
    modes = {"a": MODE_COOLING, "b": MODE_COOLING}
    energy = {"a": 5.0, "b": 3.0}
    assert classify_thermal_regime(modes, energy) == "cooling"


def test_no_demand_is_idle():
    modes = {"a": MODE_HEATING, "b": MODE_COOLING}
    assert classify_thermal_regime(modes, {"a": 0.0, "b": 0.0}) == "idle"
    assert classify_thermal_regime(modes, {}) == "idle"
    assert classify_thermal_regime({}, {}) == "idle"


def test_sub_guard_demand_is_idle():
    """Below the 10 Wh energy guard there is nothing to classify."""
    modes = {"a": MODE_HEATING}
    assert classify_thermal_regime(modes, {"a": 0.005}) == "idle"


def test_classification_is_demand_weighted_not_unit_counted():
    """Six heaters idling do not outvote one AC doing all the work.

    Counting units would call this "heating" 6-to-1; by demand it is 95 %
    cooling, which is what the sign conventions have to follow.
    """
    modes = {f"h{i}": MODE_HEATING for i in range(6)}
    modes["ac"] = MODE_COOLING
    energy = {f"h{i}": 0.05 for i in range(6)}
    energy["ac"] = 6.0

    assert classify_thermal_regime(modes, energy) == "cooling"


def test_near_even_split_is_mixed_not_a_winner():
    """A 60/40 split must not be described with the majority's semantics."""
    modes = {"h": MODE_HEATING, "c": MODE_COOLING}
    assert classify_thermal_regime(modes, {"h": 6.0, "c": 4.0}) == "mixed"


@pytest.mark.parametrize("heating,cooling,expected", [
    (8.0, 2.0, "heating"),   # exactly at the 0.8 dominance share
    (7.9, 2.1, "mixed"),     # just below it
    (2.0, 8.0, "cooling"),   # mirror
    (2.1, 7.9, "mixed"),
])
def test_dominance_share_boundary(heating, cooling, expected):
    modes = {"h": MODE_HEATING, "c": MODE_COOLING}
    energy = {"h": heating, "c": cooling}
    assert classify_thermal_regime(modes, energy) == expected


def test_dhw_is_excluded_from_both_sides():
    """DHW consumes but is not driven by outdoor conditions.

    Counting it would make a DHW-heavy summer day read as carrying heating
    demand, which is exactly the misclassification the regime label exists to
    prevent.
    """
    modes = {"dhw": MODE_DHW, "ac": MODE_COOLING}
    energy = {"dhw": 9.0, "ac": 3.0}

    assert classify_thermal_regime(modes, energy) == "cooling"

    # DHW alone is not a thermal regime at all.
    assert classify_thermal_regime({"dhw": MODE_DHW}, {"dhw": 9.0}) == "idle"


def test_off_units_are_excluded():
    modes = {"off": MODE_OFF, "h": MODE_HEATING}
    energy = {"off": 4.0, "h": 3.0}
    assert classify_thermal_regime(modes, energy) == "heating"


def test_guest_modes_count_toward_their_regime():
    """Guest units are excluded from global learning but not from reporting.

    They consume real energy and the user experiences it as part of the
    building's behaviour, so the reporting split includes them.
    """
    modes = {"g": MODE_GUEST_COOLING}
    assert classify_thermal_regime(modes, {"g": 5.0}) == "cooling"

    modes = {"g": MODE_GUEST_HEATING}
    assert classify_thermal_regime(modes, {"g": 5.0}) == "heating"


def test_unknown_modes_are_excluded():
    modes = {"x": "some_future_mode", "h": MODE_HEATING}
    energy = {"x": 9.0, "h": 3.0}
    assert classify_thermal_regime(modes, energy) == "heating"


def test_negative_and_missing_energy_is_ignored():
    """A meter glitch must not flip the regime."""
    modes = {"h": MODE_HEATING, "c": MODE_COOLING}
    assert classify_thermal_regime(modes, {"h": 5.0, "c": -3.0}) == "heating"
    assert classify_thermal_regime(modes, {"h": 5.0}) == "heating"
    assert classify_thermal_regime(modes, {"h": 5.0, "c": None}) == "heating"


# --- dominant_thermal_regime ---------------------------------------------

def test_dominant_collapses_mixed_to_its_side():
    modes = {"h": MODE_HEATING, "c": MODE_COOLING}

    assert classify_thermal_regime(modes, {"h": 6.0, "c": 4.0}) == "mixed"
    assert dominant_thermal_regime(modes, {"h": 6.0, "c": 4.0}) == "heating"

    assert classify_thermal_regime(modes, {"h": 4.0, "c": 6.0}) == "mixed"
    assert dominant_thermal_regime(modes, {"h": 4.0, "c": 6.0}) == "cooling"


def test_dominant_is_none_without_demand():
    assert dominant_thermal_regime({"h": MODE_HEATING}, {"h": 0.0}) is None
    assert dominant_thermal_regime({}, {}) is None


def test_dominant_agrees_with_label_when_label_commits():
    modes = {"h": MODE_HEATING, "c": MODE_COOLING}
    for energy in ({"h": 9.0, "c": 1.0}, {"h": 1.0, "c": 9.0}):
        label = classify_thermal_regime(modes, energy)
        assert label in ("heating", "cooling")
        assert dominant_thermal_regime(modes, energy) == label


# --- Sparse mode maps -----------------------------------------------------

def test_empty_mode_map_is_heating_not_idle():
    """The default-heating install must not read as idle.

    ``coordinator._unit_modes`` starts empty and is written only when the user
    explicitly sets a mode, and the hourly log filters MODE_HEATING out to
    reduce clutter.  Both maps are sparse, so a classifier that iterates the
    mode map instead of the energy map sees nothing on a plain heating install
    and reports "idle" forever.
    """
    assert classify_thermal_regime({}, {"a": 5.0, "b": 3.0}) == "heating"
    assert dominant_thermal_regime({}, {"a": 5.0}) == "heating"


def test_sparse_mode_map_defaults_only_the_missing_units():
    """Units absent from the mode map default to heating; present ones don't."""
    # Only the AC is recorded; the two radiators are implicit heating.
    modes = {"ac": MODE_COOLING}
    energy = {"ac": 1.0, "rad1": 5.0, "rad2": 4.0}
    assert classify_thermal_regime(modes, energy) == "heating"

    # Same sparse map, but now the AC dominates demand.
    energy = {"ac": 9.0, "rad1": 0.5, "rad2": 0.5}
    assert classify_thermal_regime(modes, energy) == "cooling"


def test_modes_without_energy_do_not_classify():
    """A recorded mode with no consumption carries no weight."""
    modes = {"ac": MODE_COOLING}
    assert classify_thermal_regime(modes, {"ac": 0.0}) == "idle"
    # ...and does not drag a heating day toward mixed.
    assert classify_thermal_regime(modes, {"ac": 0.0, "rad": 5.0}) == "heating"


# --- Live regime: today's split must be attributed as it was consumed ------

from unittest.mock import MagicMock

from custom_components.heating_analytics.coordinator import HeatingDataCoordinator


def _live_coord(heating_kwh, cooling_kwh, *, data=None):
    """Stub carrying only the per-mode accumulators the properties read.

    The real properties are evaluated against it via ``fget`` rather than
    being patched onto the mock's class, which would leak between tests.
    """
    coord = MagicMock(spec=HeatingDataCoordinator)
    coord.data = {
        "accumulated_heating_kwh": heating_kwh,
        "accumulated_cooling_kwh": cooling_kwh,
    } if data is None else data
    coord._today_regime_split = HeatingDataCoordinator._today_regime_split.fget(coord)
    return coord


def _label(coord):
    return HeatingDataCoordinator.thermal_regime.fget(coord)


def _dominant(coord):
    return HeatingDataCoordinator.dominant_thermal_regime.fget(coord)


def test_intraday_mode_switch_keeps_the_dominant_side():
    """20 kWh heating overnight, then 1 kWh cooling — still a heating day.

    Classifying today's whole accumulation by the mode active *now* would
    count all 21 kWh as cooling and flip both the label and the driver's
    regime branch to cooling physics on a heating-dominated day.  The
    accumulators carry the per-hour attribution instead, matching what the
    daily-history split records.
    """
    coord = _live_coord(20.0, 1.0)
    assert _label(coord) == "heating"
    assert _dominant(coord) == "heating"


def test_live_split_reads_the_per_mode_accumulators():
    coord = _live_coord(3.0, 12.0)
    assert _label(coord) == "cooling"
    assert _dominant(coord) == "cooling"


def test_live_mixed_day_reports_mixed():
    coord = _live_coord(6.0, 4.0)
    assert _label(coord) == "mixed"
    assert _dominant(coord) == "heating"


def test_live_no_demand_is_idle():
    coord = _live_coord(0.0, 0.0)
    assert _label(coord) == "idle"
    assert _dominant(coord) is None


def test_live_missing_accumulators_do_not_raise():
    """Before the first update cycle the keys may be absent."""
    coord = _live_coord(0.0, 0.0, data={})
    assert _label(coord) == "idle"
    assert _dominant(coord) is None


def test_deviation_sensor_labels_its_own_heating_cooling_pair():
    """thermal_regime sits with the two figures it is derived from.

    The Deviation sensor already exposes accumulated_heating_kwh and
    accumulated_cooling_kwh; the label saves a consumer from re-deriving the
    dominance rule, and stays checkable against its own basis.
    """
    from custom_components.heating_analytics.const import (
        ATTR_DEVIATION_BREAKDOWN,
        ATTR_ENERGY_TODAY,
        ATTR_EXPECTED_TODAY,
        ATTR_FORECAST_TODAY,
        ATTR_PREDICTED,
    )
    from custom_components.heating_analytics.sensor import HeatingDeviationSensor

    coordinator = MagicMock()
    coordinator.data = {
        ATTR_FORECAST_TODAY: 100.0,
        ATTR_PREDICTED: 90.0,
        ATTR_ENERGY_TODAY: 50.0,
        ATTR_EXPECTED_TODAY: 45.0,
        ATTR_DEVIATION_BREAKDOWN: [],
        "plan_revision_impact": {},
        "weather_adjusted_deviation": {},
        "accumulated_heating_kwh": 18.0,
        "accumulated_cooling_kwh": 2.0,
    }
    coordinator.auxiliary_heating_active = False
    coordinator._collector.sample_count = 0
    coordinator.thermal_regime = HeatingDataCoordinator.thermal_regime.fget(
        _live_coord(18.0, 2.0)
    )

    entry = MagicMock()
    entry.entry_id = "test"

    attrs = HeatingDeviationSensor(coordinator, entry).extra_state_attributes

    assert attrs["accumulated_heating_kwh"] == 18.0
    assert attrs["accumulated_cooling_kwh"] == 2.0
    # 90 % heating clears the 0.8 dominance share.
    assert attrs["thermal_regime"] == "heating"
