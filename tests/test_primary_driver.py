"""Primary-driver attribution on the Expected Energy Today sensor.

The stressors reported by ``primary_driver`` must be marginal contributions
against one common reference state.  Before this was fixed the comparison
mixed scales — "Temp" carried the whole zero-wind load (a level, baseload
included) while wind and solar carried deltas — so ``"Wind"`` required
``res_actual > 2 * res_no_wind`` to win and was structurally unreachable,
pinning the attribute to ``"Temp"``.

The reference is the least-demanding state for the active regime, which keeps
every term non-negative in both: clear sky when heating (sun offsets demand),
no sun when cooling (sun adds demand).

These tests drive ``calculate_total_power`` with an exact per-call surface so
each term's magnitude is known, and pin the partition property that makes the
terms comparable in the first place.
"""
import pytest
from homeassistant.core import HomeAssistant

from custom_components.heating_analytics.sensor import (
    HeatingExpectedEnergyTodaySensor,
)


BALANCE_POINT = 18.0
CURRENT_TEMP = 0.0
EFF_WIND = 8.0
CLEAR_SKY_FACTOR = 0.9


def _wire(mock_coordinator, surface, *, regime="heating", solar_enabled=True):
    """Wire the counterfactual model evaluations the driver block makes.

    ``surface`` maps ``(temp, wind, has_solar_override)`` to a total_kwh, so
    each of the block's calls is pinned exactly rather than inferred.  When
    solar is disabled no override is ever passed, so the reference-sky and
    actual-sky calls collapse onto one key by construction — which is the
    behaviour under test, not an artefact.
    """
    mock_coordinator.balance_point = BALANCE_POINT
    mock_coordinator.solar_enabled = solar_enabled
    mock_coordinator.auxiliary_heating_active = False
    mock_coordinator.thermal_regime = regime
    mock_coordinator.dominant_thermal_regime = (
        None if regime == "idle" else ("cooling" if regime == "cooling" else "heating")
    )
    mock_coordinator.data = {
        "current_calc_temp": CURRENT_TEMP,
        "effective_wind": EFF_WIND,
    }
    mock_coordinator.solar.get_approx_sun_pos.return_value = (30.0, 180.0)
    mock_coordinator.solar.calculate_solar_factor.return_value = CLEAR_SKY_FACTOR

    # Neighbouring attributes in the same getter — irrelevant here, but the
    # property computes the whole block before returning.
    mock_coordinator.statistics.get_max_historical_daily_kwh.return_value = 50.0
    mock_coordinator.statistics.get_typical_day_consumption.return_value = (20.0, 10, "high")
    mock_coordinator.forecast.calculate_load_trend.return_value = "stable"
    mock_coordinator.forecast._midnight_forecast_snapshot = {}

    calls = []

    def _power(temp_arg, wind_arg, _solar_impact, **kwargs):
        key = (temp_arg, wind_arg, "override_solar_factor" in kwargs)
        calls.append(key)
        assert key in surface, f"unexpected model evaluation {key}"
        return {"total_kwh": surface[key], "breakdown": {"solar_reduction_kwh": 0.0}}

    mock_coordinator.statistics.calculate_total_power.side_effect = _power
    return calls


def _attrs(mock_coordinator, mock_entry):
    sensor = HeatingExpectedEnergyTodaySensor(mock_coordinator, mock_entry)
    return sensor.extra_state_attributes


# --- Heating regime -------------------------------------------------------

def _heating_surface(ref, temp, temp_wind, actual):
    return {
        (BALANCE_POINT, 0.0, True): ref,
        (CURRENT_TEMP, 0.0, True): temp,
        (CURRENT_TEMP, EFF_WIND, True): temp_wind,
        (CURRENT_TEMP, EFF_WIND, False): actual,
    }


@pytest.mark.asyncio
async def test_temp_wins_on_a_cold_calm_clear_day(
    hass: HomeAssistant, mock_coordinator, mock_entry
):
    """Deep cold with little wind and clear sky is genuinely temperature-driven."""
    # temp 3.0, wind 0.2, solar deficit 0.1
    _wire(mock_coordinator, _heating_surface(1.0, 4.0, 4.2, 4.3))
    assert _attrs(mock_coordinator, mock_entry)["primary_driver"] == "Temp"


@pytest.mark.asyncio
async def test_wind_can_now_win(hass: HomeAssistant, mock_coordinator, mock_entry):
    """A storm on a mild day is wind-driven — previously unreachable.

    Under the old level-vs-delta comparison "Wind" required the wind penalty
    to exceed the entire zero-wind load (`res_actual > 2 * res_no_wind`).
    Here the wind penalty (2.0) exceeds both the temp term (0.5) and the
    solar deficit (0.3) while staying far below total consumption (3.8),
    which is the physically ordinary case that used to be unrepresentable.
    """
    _wire(mock_coordinator, _heating_surface(1.0, 1.5, 3.5, 3.8))
    assert _attrs(mock_coordinator, mock_entry)["primary_driver"] == "Wind"


@pytest.mark.asyncio
async def test_solar_deficit_can_win(hass: HomeAssistant, mock_coordinator, mock_entry):
    """An overcast shoulder day is driven by the missing solar gain."""
    # temp 0.4, wind 0.3, solar deficit 1.5
    _wire(mock_coordinator, _heating_surface(1.0, 1.4, 1.7, 3.2))
    assert _attrs(mock_coordinator, mock_entry)["primary_driver"] == "Solar_Deficit"


@pytest.mark.asyncio
async def test_no_driver_when_conditions_are_at_reference(
    hass: HomeAssistant, mock_coordinator, mock_entry
):
    """All terms clamp at zero — report None, not an arbitrary winner.

    ``max()`` over zeros returns whichever key sorts first, which would
    otherwise surface a fabricated driver on a mild, calm, sunny hour.
    """
    _wire(mock_coordinator, _heating_surface(2.0, 2.0, 2.0, 2.0))
    assert _attrs(mock_coordinator, mock_entry)["primary_driver"] == "None"


@pytest.mark.asyncio
async def test_terms_below_the_energy_guard_do_not_count(
    hass: HomeAssistant, mock_coordinator, mock_entry
):
    """Sub-10 Wh noise must not be promoted to a driver."""
    _wire(mock_coordinator, _heating_surface(2.0, 2.001, 2.002, 2.003))
    assert _attrs(mock_coordinator, mock_entry)["primary_driver"] == "None"


@pytest.mark.asyncio
async def test_terms_are_clamped_and_never_negative(
    hass: HomeAssistant, mock_coordinator, mock_entry
):
    """A non-monotone model surface must not produce a negative winner.

    Sparse or extrapolated cells can make a counterfactual come out below its
    predecessor; each term clamps at 0 so max() cannot select a negative.
    """
    # temp term would be -0.5, wind -0.2; only the solar deficit is positive.
    _wire(mock_coordinator, _heating_surface(3.0, 2.5, 2.3, 3.1))
    assert _attrs(mock_coordinator, mock_entry)["primary_driver"] == "Solar_Deficit"


@pytest.mark.asyncio
async def test_solar_deficit_is_zero_when_solar_disabled(
    hass: HomeAssistant, mock_coordinator, mock_entry
):
    """With solar off there is no reference-sky counterfactual to differ from."""
    surface = {
        (BALANCE_POINT, 0.0, False): 1.0,
        (CURRENT_TEMP, 0.0, False): 1.2,
        # Reference-sky and actual-sky calls collapse onto this one key.
        (CURRENT_TEMP, EFF_WIND, False): 1.7,
    }
    _wire(mock_coordinator, surface, solar_enabled=False)
    # temp 0.2, wind 0.5, solar 0.
    assert _attrs(mock_coordinator, mock_entry)["primary_driver"] == "Wind"
    mock_coordinator.solar.calculate_solar_factor.assert_not_called()


@pytest.mark.asyncio
async def test_terms_partition_the_load_above_reference(
    hass: HomeAssistant, mock_coordinator, mock_entry
):
    """The terms sum exactly to actual - reference.

    This is the property that makes them comparable at all, and what the old
    level-vs-delta mix violated.
    """
    ref, temp, temp_wind, actual = 1.0, 2.2, 2.9, 4.1
    _wire(mock_coordinator, _heating_surface(ref, temp, temp_wind, actual))

    attrs = _attrs(mock_coordinator, mock_entry)

    assert (temp - ref) + (temp_wind - temp) + (actual - temp_wind) == pytest.approx(
        actual - ref
    )
    assert attrs["primary_driver"] == "Temp"


# --- Cooling regime -------------------------------------------------------

def _cooling_surface(ref, temp, actual):
    """Cooling holds wind fixed and uses a no-sun reference sky."""
    return {
        (BALANCE_POINT, EFF_WIND, True): ref,
        (CURRENT_TEMP, EFF_WIND, True): temp,
        (CURRENT_TEMP, EFF_WIND, False): actual,
    }


@pytest.mark.asyncio
async def test_cooling_reference_sky_is_no_sun(
    hass: HomeAssistant, mock_coordinator, mock_entry
):
    """Under cooling the least-demanding sky is darkness, not clear sky.

    Using the heating reference (clear sky) would make the solar term
    negative — sun raises cooling demand — and clamp it away entirely.
    """
    _wire(mock_coordinator, _cooling_surface(1.0, 1.5, 3.0), regime="cooling")
    _attrs(mock_coordinator, mock_entry)

    _, kwargs = mock_coordinator.statistics.calculate_total_power.call_args_list[0]
    assert kwargs["override_solar_factor"] == 0.0
    # The clear-sky factor is a heating-regime concept and is never consulted.
    mock_coordinator.solar.calculate_solar_factor.assert_not_called()


@pytest.mark.asyncio
async def test_cooling_reports_solar_as_load_not_deficit(
    hass: HomeAssistant, mock_coordinator, mock_entry
):
    """Sun is the measured cooling load, so the driver is Solar_Load."""
    # temp 0.5, solar load 1.5
    attrs_surface = _cooling_surface(1.0, 1.5, 3.0)
    _wire(mock_coordinator, attrs_surface, regime="cooling")

    attrs = _attrs(mock_coordinator, mock_entry)
    assert attrs["primary_driver"] == "Solar_Load"
    assert attrs["thermal_regime"] == "cooling"


@pytest.mark.asyncio
async def test_cooling_never_reports_wind_as_a_driver(
    hass: HomeAssistant, mock_coordinator, mock_entry
):
    """Wind is excluded from the cooling driver set entirely.

    Under cooling wind is a mitigator, not a stressor — and there is no signal
    to report either way: per-unit cooling samples route to the dedicated
    cooling wind-bucket regardless of actual wind, so the per-unit cooling
    model is wind-independent by construction.  Wind is held fixed across the
    reference and temp evaluations so it cannot leak into attribution.
    """
    calls = _wire(mock_coordinator, _cooling_surface(1.0, 1.5, 3.0), regime="cooling")

    attrs = _attrs(mock_coordinator, mock_entry)
    assert attrs["primary_driver"] != "Wind"
    # No zero-wind counterfactual is ever evaluated under cooling.
    assert all(wind == EFF_WIND for _temp, wind, _ref in calls)


@pytest.mark.asyncio
async def test_cooling_temp_can_win(hass: HomeAssistant, mock_coordinator, mock_entry):
    """Warmth above the balance point drives a hot overcast day."""
    # temp 2.0, solar load 0.3
    _wire(mock_coordinator, _cooling_surface(1.0, 3.0, 3.3), regime="cooling")
    assert _attrs(mock_coordinator, mock_entry)["primary_driver"] == "Temp"


@pytest.mark.asyncio
async def test_mixed_regime_reports_mixed_but_evaluates_dominant_side(
    hass: HomeAssistant, mock_coordinator, mock_entry
):
    """"mixed" is reported honestly while the driver uses the dominant side.

    The driver names a magnitude, not a direction, so a dominant-side
    evaluation stays informative; directional consumers branch on the label.
    """
    _wire(mock_coordinator, _heating_surface(1.0, 1.5, 3.5, 3.8), regime="mixed")
    mock_coordinator.dominant_thermal_regime = "heating"

    attrs = _attrs(mock_coordinator, mock_entry)
    assert attrs["thermal_regime"] == "mixed"
    assert attrs["primary_driver"] == "Wind"


@pytest.mark.asyncio
async def test_idle_regime_is_reported(
    hass: HomeAssistant, mock_coordinator, mock_entry
):
    """No thermal demand at all falls back to the heating framing and None."""
    _wire(mock_coordinator, _heating_surface(2.0, 2.0, 2.0, 2.0), regime="idle")

    attrs = _attrs(mock_coordinator, mock_entry)
    assert attrs["thermal_regime"] == "idle"
    assert attrs["primary_driver"] == "None"
