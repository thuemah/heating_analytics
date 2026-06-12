"""Hour-boundary 4D routing under ``experimental_4d_primary`` (#1024).

Pre-fix, the hour-boundary analysis call (``res_analysis`` in
``hourly_processor``) and the end-of-hour gap fill passed the 3D-only
overrides (``override_solar_factor`` / ``override_solar_vector``), which
auto-route the ``calculate_total_power`` dispatcher back to the 3D
primitive regardless of the flag.  Result: ``expected_kwh`` (accumulated
from the live 4D minute loop) and the boundary accounting
(``solar_heating/cooling_applied_kwh``, ``solar_normalization_delta``,
battery charge, ``thermodynamic_gross_kwh``) were computed from two
different coefficient sets in the same log row.  Observed symptom on the
reporting install: phantom ``solar_cooling_applied_kwh`` of 0.10-0.19
kWh/h from stale 3D cooling coefficients and negative
``thermodynamic_gross_kwh`` on sunny midday hours.

Covers:
- ``_resolve_4d_boundary_overrides`` bundle construction (flag gate,
  night hours, kasten fallback, solar-disabled, mock coordinators).
- res_analysis routing: flag on -> 4D primitive, flag off -> legacy 3D
  call with the 3D overrides (bit-identical path).
- end-to-end regression: cooling unit with divergent 3D/4D coefficient
  sets logs the 4D applied split under the flag (no phantom cooling
  solar, gross == actual) and the 3D one without it.
- ``_close_hour_gap`` honours the bundle.
"""
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import datetime, timezone

import pytest

from homeassistant.core import HomeAssistant

from custom_components.heating_analytics.coordinator import HeatingDataCoordinator
from custom_components.heating_analytics.const import MODE_COOLING


# Test site: southern Norway (59.6N, 11.2E).  Midsummer 11:00-12:00 UTC
# is close to local solar noon — sun elevation ~53 deg.
TEST_LAT = 59.6
TEST_LON = 11.2
SUNNY_START = datetime(2026, 6, 21, 11, 0, 0, tzinfo=timezone.utc)
SUNNY_END = datetime(2026, 6, 21, 12, 0, 0, tzinfo=timezone.utc)
# A winter night hour at the same location.
NIGHT_START = datetime(2026, 1, 15, 23, 0, 0, tzinfo=timezone.utc)
NIGHT_END = datetime(2026, 1, 16, 0, 0, 0, tzinfo=timezone.utc)


# Shaped result for spying on calculate_total_power primitives without
# breaking the downstream consumers in _process_hourly_data.
def _shaped_result():
    return {
        "total_kwh": 1.0,
        "global_base_kwh": 1.0,
        "global_aux_reduction_kwh": 0.0,
        "unit_breakdown": {},
        "breakdown": {
            "base_kwh": 1.0,
            "aux_reduction_kwh": 0.0,
            "solar_reduction_kwh": 0.0,
            "solar_wasted_kwh": 0.0,
            "solar_heating_wasted_kwh": 0.0,
            "solar_heating_applied_kwh": 0.0,
            "solar_cooling_applied_kwh": 0.0,
        },
    }


def _build_coordinator(hass, flag_4d: bool):
    entry = MagicMock()
    entry.data = {
        "balance_point": 17.0,
        "learning_rate": 0.1,
        "energy_sensors": ["sensor.heater"],
        "solar_enabled": True,
        "experimental_4d_primary": flag_4d,
    }
    # Real coordinates so get_approx_sun_pos (astral) works on the
    # MagicMock hass fixture.
    hass.config.latitude = TEST_LAT
    hass.config.longitude = TEST_LON
    with patch("custom_components.heating_analytics.storage.Store"):
        coordinator = HeatingDataCoordinator(hass, entry)
    coordinator._async_save_data = AsyncMock()
    coordinator.storage.append_hourly_log_csv = AsyncMock()
    coordinator.learning.process_learning = MagicMock(return_value={
        "model_base_before": 1.0,
        "model_base_after": 1.0,
        "model_updated": False,
    })
    return coordinator


def _fill_sunny_collector(coordinator, *, avg_temp=16.0):
    col = coordinator._collector
    col.start_time = SUNNY_START
    col.sample_count = 60
    col.temp_sum = avg_temp * 60
    col.wind_values = [2.0] * 60
    col.correction_sum = 100.0 * 60  # screens fully open all hour
    # 3D effective vector averages (south-heavy midday hour)
    col.solar_sum = 0.3 * 60
    col.solar_vector_s_sum = 0.4 * 60
    col.solar_vector_e_sum = 0.0
    col.solar_vector_w_sum = 0.1 * 60
    # Native irradiance averages for the 4D ladder
    col.dni_sum = 600.0 * 60
    col.dni_count = 60
    col.dhi_sum = 150.0 * 60
    col.dhi_count = 60
    col.energy_hour = 0.05
    col.expected_energy_hour = 0.3
    coordinator._hourly_delta_per_unit = {"sensor.heater": 0.05}


def _seed_models(coordinator):
    """Cooling unit with divergent 3D vs 4D coefficient sets.

    3D cooling carries substantial coefficients (the stale-set scenario);
    4D cooling is explicitly learned at zero (forced-off anticorrelation
    converged honestly to "no coupling").
    """
    buckets = {str(k): {"normal": 0.5, "cooling": 0.2} for k in range(5, 25)}
    coordinator._correlation_data = {str(k): {"normal": 0.5} for k in range(5, 25)}
    coordinator._correlation_data_per_unit = {"sensor.heater": buckets}
    coordinator._unit_modes = {"sensor.heater": MODE_COOLING}
    coordinator._solar_coefficients_per_unit = {
        "sensor.heater": {
            "heating": {"s": 0.0, "e": 0.0, "w": 0.0, "learned": True},
            "cooling": {"s": 0.3, "e": 0.2, "w": 0.25, "learned": True},
        }
    }
    coordinator._solar_coefficients_4d_per_unit = {
        "sensor.heater": {
            "heating": {"s": 0.0, "e": 0.0, "w": 0.0, "diffuse": 0.0, "learned": True},
            "cooling": {"s": 0.0, "e": 0.0, "w": 0.0, "diffuse": 0.0, "learned": True},
        }
    }


# =====================================================================
# _resolve_4d_boundary_overrides — bundle construction
# =====================================================================


def test_bundle_none_when_flag_off(hass: HomeAssistant):
    coordinator = _build_coordinator(hass, flag_4d=False)
    _fill_sunny_collector(coordinator)
    assert coordinator._hourly_processor._resolve_4d_boundary_overrides(SUNNY_END) is None


def test_bundle_none_on_mock_coordinator_flag():
    """MagicMock auto-attribute is truthy but not ``is True`` — must bail."""
    from custom_components.heating_analytics.hourly_processor import HourlyProcessor
    proc = HourlyProcessor(MagicMock())
    assert proc._resolve_4d_boundary_overrides(SUNNY_END) is None


def test_bundle_native_dni_dhi(hass: HomeAssistant):
    coordinator = _build_coordinator(hass, flag_4d=True)
    _fill_sunny_collector(coordinator)
    bundle = coordinator._hourly_processor._resolve_4d_boundary_overrides(SUNNY_END)
    assert bundle is not None
    assert bundle["override_now"] == SUNNY_START.replace(minute=30)
    assert bundle["override_sun_pos"][0] > 30.0  # midday summer sun
    dni, dhi = bundle["override_dni_dhi"]
    assert dni == pytest.approx(600.0)
    assert dhi == pytest.approx(150.0)
    assert bundle["override_correction_percent"] == pytest.approx(100.0)


def test_bundle_kasten_fallback_from_cloud(hass: HomeAssistant):
    """No native DNI/DHI but cloud averages present -> synthetic > 0."""
    coordinator = _build_coordinator(hass, flag_4d=True)
    _fill_sunny_collector(coordinator)
    col = coordinator._collector
    col.dni_sum = col.dhi_sum = 0.0
    col.dni_count = col.dhi_count = 0
    col.cloud_coverage_sum = 20.0 * 60
    col.cloud_coverage_count = 60
    bundle = coordinator._hourly_processor._resolve_4d_boundary_overrides(SUNNY_END)
    assert bundle is not None
    dni, dhi = bundle["override_dni_dhi"]
    assert dni > 0.0
    assert dhi > 0.0


def test_bundle_night_hour_zeroes_irradiance(hass: HomeAssistant):
    coordinator = _build_coordinator(hass, flag_4d=True)
    _fill_sunny_collector(coordinator)
    coordinator._collector.start_time = NIGHT_START
    bundle = coordinator._hourly_processor._resolve_4d_boundary_overrides(NIGHT_END)
    assert bundle is not None
    assert bundle["override_dni_dhi"] == (0.0, 0.0)


def test_bundle_solar_disabled_zeroes_irradiance(hass: HomeAssistant):
    coordinator = _build_coordinator(hass, flag_4d=True)
    _fill_sunny_collector(coordinator)
    coordinator.solar_enabled = False
    bundle = coordinator._hourly_processor._resolve_4d_boundary_overrides(SUNNY_END)
    assert bundle is not None
    assert bundle["override_dni_dhi"] == (0.0, 0.0)


# =====================================================================
# res_analysis routing
# =====================================================================


@pytest.mark.asyncio
async def test_res_analysis_routes_4d_when_flag_on(hass: HomeAssistant):
    coordinator = _build_coordinator(hass, flag_4d=True)
    _fill_sunny_collector(coordinator)
    _seed_models(coordinator)

    with patch.object(
        coordinator.statistics, "calculate_total_power_4d",
        MagicMock(return_value=_shaped_result()),
    ) as mock_4d, patch.object(
        coordinator.statistics, "_calculate_total_power_3d",
        MagicMock(return_value=_shaped_result()),
    ) as mock_3d:
        await coordinator._process_hourly_data(SUNNY_END)

    assert mock_4d.call_count >= 1, "boundary analysis must route 4D under the flag"
    boundary_kwargs = [c.kwargs for c in mock_4d.call_args_list]
    assert any(
        k.get("override_dni_dhi") is not None for k in boundary_kwargs
    ), "4D boundary call must carry the hour-average DNI/DHI override"
    # No call may smuggle the 3D overrides through the 4D primitive.
    for k in boundary_kwargs:
        assert "override_solar_factor" not in k
        assert "override_solar_vector" not in k
    # The legacy 3D boundary form (override_solar_factor) must be gone.
    for c in mock_3d.call_args_list:
        assert c.kwargs.get("override_solar_factor") is None
        assert c.kwargs.get("override_solar_vector") is None


@pytest.mark.asyncio
async def test_res_analysis_flag_off_unchanged(hass: HomeAssistant):
    coordinator = _build_coordinator(hass, flag_4d=False)
    _fill_sunny_collector(coordinator)
    _seed_models(coordinator)

    with patch.object(
        coordinator.statistics, "calculate_total_power_4d",
        MagicMock(return_value=_shaped_result()),
    ) as mock_4d, patch.object(
        coordinator.statistics, "_calculate_total_power_3d",
        MagicMock(return_value=_shaped_result()),
    ) as mock_3d:
        await coordinator._process_hourly_data(SUNNY_END)

    assert mock_4d.call_count == 0
    assert any(
        c.kwargs.get("override_solar_factor") is not None
        and c.kwargs.get("override_solar_vector") is not None
        for c in mock_3d.call_args_list
    ), "flag off must keep the legacy 3D override form bit-identical"


# =====================================================================
# End-to-end regression: phantom cooling solar / negative gross
# =====================================================================


@pytest.mark.asyncio
async def test_boundary_accounting_uses_4d_coefficients(hass: HomeAssistant):
    """Flag on: a cooling unit whose 4D regime learned ~0 must log ~0
    ``solar_cooling_applied_kwh`` even though its 3D cooling coefficients
    are substantial, and ``thermodynamic_gross_kwh`` must equal actual
    (no phantom negative adjustment)."""
    coordinator = _build_coordinator(hass, flag_4d=True)
    _fill_sunny_collector(coordinator)
    _seed_models(coordinator)

    await coordinator._process_hourly_data(SUNNY_END)

    assert len(coordinator._hourly_log) == 1
    log = coordinator._hourly_log[0]
    assert log["solar_cooling_applied_kwh"] == pytest.approx(0.0, abs=1e-6)
    assert log["thermodynamic_gross_kwh"] == pytest.approx(log["actual_kwh"], abs=1e-6)
    assert log["solar_normalization_delta"] == pytest.approx(0.0, abs=1e-6)


@pytest.mark.asyncio
async def test_boundary_accounting_flag_off_uses_3d_coefficients(hass: HomeAssistant):
    """Flag off: same setup logs the 3D applied split (coeff x potential
    on the hour-average vector) — the pre-fix behaviour, preserved."""
    coordinator = _build_coordinator(hass, flag_4d=False)
    _fill_sunny_collector(coordinator)
    _seed_models(coordinator)

    await coordinator._process_hourly_data(SUNNY_END)

    assert len(coordinator._hourly_log) == 1
    log = coordinator._hourly_log[0]
    # 3D cooling solar: 0.3*0.4 (s) + 0.25*0.1 (w) = 0.145 on the
    # potential==effective vector (correction 100%).
    assert log["solar_cooling_applied_kwh"] == pytest.approx(0.145, abs=0.02)
    # Mode-signed gross adjustment is negative on a cooling-only hour.
    assert log["thermodynamic_gross_kwh"] < log["actual_kwh"]


# =====================================================================
# _close_hour_gap
# =====================================================================


def test_close_hour_gap_uses_4d_bundle(hass: HomeAssistant):
    coordinator = _build_coordinator(hass, flag_4d=True)
    _fill_sunny_collector(coordinator)
    bundle = coordinator._hourly_processor._resolve_4d_boundary_overrides(SUNNY_END)
    assert bundle is not None

    with patch.object(
        coordinator.statistics, "calculate_total_power",
        MagicMock(return_value=_shaped_result()),
    ) as mock_ctp:
        coordinator._close_hour_gap(
            SUNNY_END, 55,
            avg_temp=16.0, avg_wind=2.0, avg_solar=0.3,
            is_aux_active=False, overrides_4d=bundle,
        )

    assert mock_ctp.call_count == 1
    kwargs = mock_ctp.call_args.kwargs
    assert kwargs.get("override_dni_dhi") == bundle["override_dni_dhi"]
    assert "override_solar_factor" not in kwargs
    assert "override_solar_vector" not in kwargs


def test_close_hour_gap_legacy_form_without_bundle(hass: HomeAssistant):
    coordinator = _build_coordinator(hass, flag_4d=False)
    _fill_sunny_collector(coordinator)

    with patch.object(
        coordinator.statistics, "calculate_total_power",
        MagicMock(return_value=_shaped_result()),
    ) as mock_ctp:
        coordinator._close_hour_gap(
            SUNNY_END, 55,
            avg_temp=16.0, avg_wind=2.0, avg_solar=0.3,
            is_aux_active=False, overrides_4d=None,
        )

    kwargs = mock_ctp.call_args.kwargs
    assert kwargs.get("override_solar_factor") == pytest.approx(0.3)
    assert "override_dni_dhi" not in kwargs
