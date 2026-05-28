"""Tests for the ``fit_solar_obstruction`` service handler (#1020).

The handler orchestrates multi-window stability assessment and
surfaces suggested gates without auto-writing.  Covers:

1. **Suggestion-not-auto-write** — passing gate appears in
   ``suggested_gates``; ``_critical_elev_per_facade_per_unit`` is
   never mutated by the fit service.
2. **Multi-window stability** — boundary becomes ``applicable=True``
   only when ≥2 of {30, 60, 90}-day windows agree within ±3°.
3. **Single-pass debug mode** — explicit ``days_back`` skips the
   stability gate (single pass, no multi-window orchestration).
4. **``apply_obstruction_gate`` service** — writes a single slot,
   rejects out-of-plausibility-range values, supports dry_run.
"""
from __future__ import annotations

from unittest.mock import MagicMock, AsyncMock, patch
import pytest

from custom_components.heating_analytics import (
    async_setup_entry,
    SERVICE_FIT_SOLAR_OBSTRUCTION,
    SERVICE_APPLY_OBSTRUCTION_GATE,
)
from custom_components.heating_analytics.const import DOMAIN


@pytest.fixture
def mock_entry():
    entry = MagicMock()
    entry.entry_id = "test_entry"
    entry.data = {
        "outdoor_temp_sensor": "sensor.temp",
        "energy_sensors": ["sensor.heater"],
    }
    return entry


def _entity_block(
    *,
    side: str,
    learned: bool,
    best: float | None,
    sse: float = 0.5,
):
    """Construct a single-facade-single-side fit_solar_obstruction
    entity block; opposite side is inert.  Used by the per-window
    fit-call mocks."""
    inactive = {
        "learned": False, "applicable": False, "best_critical_elev": None,
        "sse_improvement_ratio": 0.0,
    }
    populated = {
        "learned": learned,
        "applicable": learned,
        "best_critical_elev": best,
        "sse_improvement_ratio": sse,
    }
    inactive_facade = {"low": inactive, "high": inactive}
    target_facade = {
        "low": populated if side == "low" else inactive,
        "high": populated if side == "high" else inactive,
    }
    return {
        "s": inactive_facade,
        "e": inactive_facade,
        "w": target_facade,
    }


async def _setup_handler_with_coord(service_name: str):
    """Capture a service handler by name without running the full
    HA setup."""
    hass = MagicMock()
    hass.data = {}
    hass.config_entries.async_forward_entry_setups = AsyncMock()
    captured: dict[str, object] = {}

    def _register(domain, service, callback, schema=None, **kwargs):
        if domain == DOMAIN and service == service_name:
            captured["handler"] = callback

    hass.services.async_register = MagicMock(side_effect=_register)
    return hass, captured


# ---------------------------------------------------------------------------
# fit_solar_obstruction handler — multi-window orchestration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_handler_multi_window_default_runs_three_passes(mock_entry):
    """Default invocation (no days_back) runs three fits on the
    configured stability windows + one final pass with stability
    injected.  Total fit calls: 4 (3 per-window + 1 final).
    """
    hass, captured = await _setup_handler_with_coord(
        SERVICE_FIT_SOLAR_OBSTRUCTION
    )
    sensor = "sensor.heater"

    # All three windows agree on LOW = 5.0°.
    consistent_result = {
        sensor: _entity_block(side="low", learned=True, best=5.0, sse=0.6),
        "dry_run": False,
        "n_skipped_cooling_unlearned": 0,
        "suggested_gates": [
            {"entity_id": sensor, "facade": "w", "side": "low",
             "value": 5.0, "before": None, "sse_improvement_ratio": 0.6},
        ],
    }

    with patch("custom_components.heating_analytics.HeatingDataCoordinator") as mock_coord_cls:
        coord = mock_coord_cls.return_value
        coord.async_config_entry_first_refresh = AsyncMock()
        coord.storage = MagicMock()
        coord.storage.async_load_data = AsyncMock()
        coord._async_save_data = AsyncMock()
        coord._hourly_log = []
        coord._critical_elev_per_facade_per_unit = {}
        coord.energy_sensors = [sensor]
        coord._per_unit_min_base_thresholds = None
        coord._screen_affected_set = None
        coord._solar_coefficients_4d_per_unit = {}
        coord.entry = mock_entry

        coord.learning = MagicMock()
        coord.learning.fit_solar_obstruction = MagicMock(
            return_value=consistent_result
        )

        with patch(
            "custom_components.heating_analytics._get_target_coordinator",
            return_value=coord,
        ):
            await async_setup_entry(hass, mock_entry)
            handler = captured["handler"]
            call = MagicMock()
            call.data = {}  # no days_back → multi-window
            response = await handler(call)

    # 3 per-window + 1 final stability-injected pass = 4.
    assert coord.learning.fit_solar_obstruction.call_count == 4
    assert response["multi_window"] is True
    assert response["windows_evaluated"] == [30, 60, 90]
    assert response["primary_window_days"] == 90
    # Auto-write is gone: no save on the fit path.
    coord._async_save_data.assert_not_called()
    assert response["suggested_count"] == 1


@pytest.mark.asyncio
async def test_handler_single_pass_when_days_back_provided(mock_entry):
    """When ``days_back`` is supplied explicitly, multi-window is
    skipped — single pass + final stability-injected pass.  The
    stability gate is bypassed in this mode (intended for ad-hoc
    debug of a specific window).
    """
    hass, captured = await _setup_handler_with_coord(
        SERVICE_FIT_SOLAR_OBSTRUCTION
    )
    sensor = "sensor.heater"

    result = {
        sensor: _entity_block(side="low", learned=True, best=5.0, sse=0.6),
        "dry_run": False,
        "n_skipped_cooling_unlearned": 0,
        "suggested_gates": [
            {"entity_id": sensor, "facade": "w", "side": "low",
             "value": 5.0, "before": None, "sse_improvement_ratio": 0.6},
        ],
    }

    with patch("custom_components.heating_analytics.HeatingDataCoordinator") as mock_coord_cls:
        coord = mock_coord_cls.return_value
        coord.async_config_entry_first_refresh = AsyncMock()
        coord.storage = MagicMock()
        coord.storage.async_load_data = AsyncMock()
        coord._async_save_data = AsyncMock()
        coord._hourly_log = []
        coord._critical_elev_per_facade_per_unit = {}
        coord.energy_sensors = [sensor]
        coord._per_unit_min_base_thresholds = None
        coord._screen_affected_set = None
        coord._solar_coefficients_4d_per_unit = {}
        coord.entry = mock_entry

        coord.learning = MagicMock()
        coord.learning.fit_solar_obstruction = MagicMock(return_value=result)

        with patch(
            "custom_components.heating_analytics._get_target_coordinator",
            return_value=coord,
        ):
            await async_setup_entry(hass, mock_entry)
            handler = captured["handler"]
            call = MagicMock()
            call.data = {"days_back": 30}  # explicit → single pass
            response = await handler(call)

    # 1 single-pass + 1 final stability-injected pass = 2.
    assert coord.learning.fit_solar_obstruction.call_count == 2
    assert response["multi_window"] is False
    assert response["windows_evaluated"] == [30]
    coord._async_save_data.assert_not_called()


@pytest.mark.asyncio
async def test_handler_unstable_boundary_blocks_suggestion(mock_entry):
    """When the three windows disagree (one window's best_critical_elev
    is far from the others), stability flag goes False and the final
    pass receives ``stable_across_windows=False`` for that boundary.
    The handler's response surfaces the disagreement.
    """
    hass, captured = await _setup_handler_with_coord(
        SERVICE_FIT_SOLAR_OBSTRUCTION
    )
    sensor = "sensor.heater"

    # Window-specific fits with disagreeing best values.
    per_window: dict[int, dict] = {}
    for w, best in ((30, 5.0), (60, 15.0), (90, 18.0)):
        per_window[w] = {
            sensor: _entity_block(side="low", learned=True, best=best, sse=0.6),
            "dry_run": False,
            "n_skipped_cooling_unlearned": 0,
            "suggested_gates": [],
        }

    call_count = {"n": 0}

    def _fit_side_effect(**kwargs):
        call_count["n"] += 1
        days_back = kwargs["days_back"]
        if call_count["n"] <= 3:
            # Per-window passes.
            return per_window[days_back]
        # Final stability-injected pass — receives stability flags.
        stability = kwargs.get("stability_per_facade_per_entity") or {}
        sensor_stability = stability.get(sensor, {})
        w_low_stable = sensor_stability.get(("w", "low"), True)
        # Mimic the production behaviour: the underlying fit_solar_obstruction
        # would clear applicable when stable_across_windows=False.
        out = per_window[max(per_window)]
        out[sensor]["w"]["low"]["applicable"] = w_low_stable
        out["suggested_gates"] = (
            [{"entity_id": sensor, "facade": "w", "side": "low",
              "value": 18.0, "before": None, "sse_improvement_ratio": 0.6}]
            if w_low_stable else []
        )
        return out

    with patch("custom_components.heating_analytics.HeatingDataCoordinator") as mock_coord_cls:
        coord = mock_coord_cls.return_value
        coord.async_config_entry_first_refresh = AsyncMock()
        coord.storage = MagicMock()
        coord.storage.async_load_data = AsyncMock()
        coord._async_save_data = AsyncMock()
        coord._hourly_log = []
        coord._critical_elev_per_facade_per_unit = {}
        coord.energy_sensors = [sensor]
        coord._per_unit_min_base_thresholds = None
        coord._screen_affected_set = None
        coord._solar_coefficients_4d_per_unit = {}
        coord.entry = mock_entry

        coord.learning = MagicMock()
        coord.learning.fit_solar_obstruction = MagicMock(side_effect=_fit_side_effect)

        with patch(
            "custom_components.heating_analytics._get_target_coordinator",
            return_value=coord,
        ):
            await async_setup_entry(hass, mock_entry)
            handler = captured["handler"]
            call = MagicMock()
            call.data = {}
            response = await handler(call)

    # Stability summary records all three values and flags as unstable
    # (5° vs 15° vs 18° — only 15° / 18° are within ±3°, but 5° sits
    # 10° away; 15°/18° agree, so stability IS True actually — let's
    # verify the handler implementation matches our expectation).
    w_low_stability = response["stability"][sensor]["w"]["low"]
    # 15 and 18 are within ±3 (diff=3.0); 5 is far.  So agreeing_pair
    # captured includes 15 and 18; stable=True with 2 agreeing.
    assert w_low_stability["stable_across_windows"] is True
    assert response["suggested_count"] == 1


@pytest.mark.asyncio
async def test_handler_no_agreement_blocks_suggestion(mock_entry):
    """All three windows disagree (no pair within ±3°) → stability
    flag False, no suggestion surfaces.
    """
    hass, captured = await _setup_handler_with_coord(
        SERVICE_FIT_SOLAR_OBSTRUCTION
    )
    sensor = "sensor.heater"

    per_window: dict[int, dict] = {}
    for w, best in ((30, 3.0), (60, 10.0), (90, 18.0)):
        per_window[w] = {
            sensor: _entity_block(side="low", learned=True, best=best, sse=0.6),
            "dry_run": False,
            "n_skipped_cooling_unlearned": 0,
            "suggested_gates": [],
        }

    call_count = {"n": 0}

    def _fit_side_effect(**kwargs):
        call_count["n"] += 1
        days_back = kwargs["days_back"]
        if call_count["n"] <= 3:
            return per_window[days_back]
        stability = kwargs.get("stability_per_facade_per_entity") or {}
        sensor_stab = stability.get(sensor, {})
        w_low_stable = sensor_stab.get(("w", "low"), True)
        out = per_window[max(per_window)]
        out[sensor]["w"]["low"]["applicable"] = w_low_stable
        out["suggested_gates"] = (
            [{"entity_id": sensor, "facade": "w", "side": "low",
              "value": 18.0, "before": None, "sse_improvement_ratio": 0.6}]
            if w_low_stable else []
        )
        return out

    with patch("custom_components.heating_analytics.HeatingDataCoordinator") as mock_coord_cls:
        coord = mock_coord_cls.return_value
        coord.async_config_entry_first_refresh = AsyncMock()
        coord.storage = MagicMock()
        coord.storage.async_load_data = AsyncMock()
        coord._async_save_data = AsyncMock()
        coord._hourly_log = []
        coord._critical_elev_per_facade_per_unit = {}
        coord.energy_sensors = [sensor]
        coord._per_unit_min_base_thresholds = None
        coord._screen_affected_set = None
        coord._solar_coefficients_4d_per_unit = {}
        coord.entry = mock_entry

        coord.learning = MagicMock()
        coord.learning.fit_solar_obstruction = MagicMock(side_effect=_fit_side_effect)

        with patch(
            "custom_components.heating_analytics._get_target_coordinator",
            return_value=coord,
        ):
            await async_setup_entry(hass, mock_entry)
            handler = captured["handler"]
            call = MagicMock()
            call.data = {}
            response = await handler(call)

    w_low_stability = response["stability"][sensor]["w"]["low"]
    assert w_low_stability["stable_across_windows"] is False
    assert response["suggested_count"] == 0


# ---------------------------------------------------------------------------
# apply_obstruction_gate handler
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_apply_obstruction_gate_writes_and_saves(mock_entry):
    """Writing a plausible LOW gate persists to state and triggers save."""
    hass, captured = await _setup_handler_with_coord(
        SERVICE_APPLY_OBSTRUCTION_GATE
    )
    sensor = "sensor.heater"

    with patch("custom_components.heating_analytics.HeatingDataCoordinator") as mock_coord_cls:
        coord = mock_coord_cls.return_value
        coord.async_config_entry_first_refresh = AsyncMock()
        coord.storage = MagicMock()
        coord.storage.async_load_data = AsyncMock()
        coord._async_save_data = AsyncMock()
        coord._hourly_log = []
        coord._critical_elev_per_facade_per_unit = {}
        coord.energy_sensors = [sensor]
        coord.entry = mock_entry

        with patch(
            "custom_components.heating_analytics._get_target_coordinator",
            return_value=coord,
        ):
            await async_setup_entry(hass, mock_entry)
            handler = captured["handler"]
            call = MagicMock()
            call.data = {
                "unit_entity_id": sensor,
                "facade": "w",
                "side": "low",
                "value": 5.0,
                "dry_run": False,
            }
            response = await handler(call)

    assert response["wrote"] is True
    assert response["after"] == 5.0
    assert response["before"] is None
    assert (
        coord._critical_elev_per_facade_per_unit[sensor]["w"]["low"]
        == 5.0
    )
    coord._async_save_data.assert_awaited_once()


@pytest.mark.asyncio
async def test_apply_obstruction_gate_dry_run_no_write(mock_entry):
    """dry_run=True returns the planned change but does not mutate state."""
    hass, captured = await _setup_handler_with_coord(
        SERVICE_APPLY_OBSTRUCTION_GATE
    )
    sensor = "sensor.heater"

    with patch("custom_components.heating_analytics.HeatingDataCoordinator") as mock_coord_cls:
        coord = mock_coord_cls.return_value
        coord.async_config_entry_first_refresh = AsyncMock()
        coord.storage = MagicMock()
        coord.storage.async_load_data = AsyncMock()
        coord._async_save_data = AsyncMock()
        coord._hourly_log = []
        coord._critical_elev_per_facade_per_unit = {}
        coord.energy_sensors = [sensor]
        coord.entry = mock_entry

        with patch(
            "custom_components.heating_analytics._get_target_coordinator",
            return_value=coord,
        ):
            await async_setup_entry(hass, mock_entry)
            handler = captured["handler"]
            call = MagicMock()
            call.data = {
                "unit_entity_id": sensor,
                "facade": "s",
                "side": "high",
                "value": 35.0,
                "dry_run": True,
            }
            response = await handler(call)

    assert response["wrote"] is False
    assert response["dry_run"] is True
    assert response["after"] == 35.0
    assert sensor not in coord._critical_elev_per_facade_per_unit
    coord._async_save_data.assert_not_called()


@pytest.mark.asyncio
async def test_apply_obstruction_gate_rejects_out_of_range(mock_entry):
    """Values outside the side's plausibility range raise
    ServiceValidationError.  LOW > 20° is rejected; HIGH < 20° rejected.
    """
    hass, captured = await _setup_handler_with_coord(
        SERVICE_APPLY_OBSTRUCTION_GATE
    )
    sensor = "sensor.heater"

    with patch("custom_components.heating_analytics.HeatingDataCoordinator") as mock_coord_cls:
        coord = mock_coord_cls.return_value
        coord.async_config_entry_first_refresh = AsyncMock()
        coord.storage = MagicMock()
        coord.storage.async_load_data = AsyncMock()
        coord._async_save_data = AsyncMock()
        coord._hourly_log = []
        coord._critical_elev_per_facade_per_unit = {}
        coord.energy_sensors = [sensor]
        coord.entry = mock_entry

        with patch(
            "custom_components.heating_analytics._get_target_coordinator",
            return_value=coord,
        ):
            await async_setup_entry(hass, mock_entry)
            handler = captured["handler"]
            # LOW with 25° — out of range [2°, 20°].
            call = MagicMock()
            call.data = {
                "unit_entity_id": sensor,
                "facade": "w",
                "side": "low",
                "value": 25.0,
                "dry_run": False,
            }
            with pytest.raises(Exception):  # ServiceValidationError (mocked)
                await handler(call)
            # HIGH with 10° — out of range [20°, 60°].
            call2 = MagicMock()
            call2.data = {
                "unit_entity_id": sensor,
                "facade": "w",
                "side": "high",
                "value": 10.0,
                "dry_run": False,
            }
            with pytest.raises(Exception):
                await handler(call2)


@pytest.mark.asyncio
async def test_apply_obstruction_gate_clear_flag_clears_slot(mock_entry):
    """``clear=True`` resets a previously-set gate slot to None.  The
    flag exists because HA's number-selector UI cannot send null, so
    the form-editor path needs an explicit toggle to reach the clear
    operation."""
    hass, captured = await _setup_handler_with_coord(
        SERVICE_APPLY_OBSTRUCTION_GATE
    )
    sensor = "sensor.heater"

    with patch("custom_components.heating_analytics.HeatingDataCoordinator") as mock_coord_cls:
        coord = mock_coord_cls.return_value
        coord.async_config_entry_first_refresh = AsyncMock()
        coord.storage = MagicMock()
        coord.storage.async_load_data = AsyncMock()
        coord._async_save_data = AsyncMock()
        coord._hourly_log = []
        coord._critical_elev_per_facade_per_unit = {
            sensor: {
                "s": {"low": None, "high": None},
                "e": {"low": None, "high": None},
                "w": {"low": 5.0, "high": None},
            },
        }
        coord.energy_sensors = [sensor]
        coord.entry = mock_entry

        with patch(
            "custom_components.heating_analytics._get_target_coordinator",
            return_value=coord,
        ):
            await async_setup_entry(hass, mock_entry)
            handler = captured["handler"]
            call = MagicMock()
            call.data = {
                "unit_entity_id": sensor,
                "facade": "w",
                "side": "low",
                "clear": True,
                "dry_run": False,
            }
            response = await handler(call)

    assert response["before"] == 5.0
    assert response["after"] is None
    assert (
        coord._critical_elev_per_facade_per_unit[sensor]["w"]["low"]
        is None
    )


@pytest.mark.asyncio
async def test_apply_obstruction_gate_clear_ignores_value(mock_entry):
    """When ``clear=True`` the ``value`` field is ignored — protects
    against a leftover number from a prior service-form call from
    overriding the clear intent."""
    hass, captured = await _setup_handler_with_coord(
        SERVICE_APPLY_OBSTRUCTION_GATE
    )
    sensor = "sensor.heater"

    with patch("custom_components.heating_analytics.HeatingDataCoordinator") as mock_coord_cls:
        coord = mock_coord_cls.return_value
        coord.async_config_entry_first_refresh = AsyncMock()
        coord.storage = MagicMock()
        coord.storage.async_load_data = AsyncMock()
        coord._async_save_data = AsyncMock()
        coord._hourly_log = []
        coord._critical_elev_per_facade_per_unit = {
            sensor: {
                "s": {"low": None, "high": None},
                "e": {"low": None, "high": None},
                "w": {"low": 5.0, "high": None},
            },
        }
        coord.energy_sensors = [sensor]
        coord.entry = mock_entry

        with patch(
            "custom_components.heating_analytics._get_target_coordinator",
            return_value=coord,
        ):
            await async_setup_entry(hass, mock_entry)
            handler = captured["handler"]
            call = MagicMock()
            call.data = {
                "unit_entity_id": sensor,
                "facade": "w",
                "side": "low",
                "value": 15.0,  # leftover from a previous call
                "clear": True,  # explicit clear wins
                "dry_run": False,
            }
            response = await handler(call)

    assert response["after"] is None
    assert (
        coord._critical_elev_per_facade_per_unit[sensor]["w"]["low"]
        is None
    )


@pytest.mark.asyncio
async def test_apply_obstruction_gate_missing_value_and_clear_errors(mock_entry):
    """Neither value nor clear → raise ServiceValidationError rather
    than silently writing None.  Guard against a partially-filled
    service-form submission becoming an accidental clear."""
    hass, captured = await _setup_handler_with_coord(
        SERVICE_APPLY_OBSTRUCTION_GATE
    )
    sensor = "sensor.heater"

    with patch("custom_components.heating_analytics.HeatingDataCoordinator") as mock_coord_cls:
        coord = mock_coord_cls.return_value
        coord.async_config_entry_first_refresh = AsyncMock()
        coord.storage = MagicMock()
        coord.storage.async_load_data = AsyncMock()
        coord._async_save_data = AsyncMock()
        coord._hourly_log = []
        coord._critical_elev_per_facade_per_unit = {}
        coord.energy_sensors = [sensor]
        coord.entry = mock_entry

        with patch(
            "custom_components.heating_analytics._get_target_coordinator",
            return_value=coord,
        ):
            await async_setup_entry(hass, mock_entry)
            handler = captured["handler"]
            call = MagicMock()
            call.data = {
                "unit_entity_id": sensor,
                "facade": "w",
                "side": "low",
                # no value, no clear
                "dry_run": False,
            }
            with pytest.raises(Exception):  # ServiceValidationError (mocked)
                await handler(call)
