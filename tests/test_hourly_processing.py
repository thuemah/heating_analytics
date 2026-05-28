"""Test hourly processing logic."""
from unittest.mock import MagicMock, patch, AsyncMock
import pytest
from datetime import datetime
from homeassistant.core import HomeAssistant
from custom_components.heating_analytics.coordinator import HeatingDataCoordinator
from custom_components.heating_analytics.const import ATTR_LAST_HOUR_DEVIATION_PCT, ENERGY_GUARD_THRESHOLD

@pytest.mark.asyncio
async def test_hourly_processing_triggers(hass: HomeAssistant):
    """Test that hourly processing is triggered correctly."""
    entry = MagicMock()
    entry.data = {"balance_point": 17.0, "learning_rate": 0.1}

    with patch("custom_components.heating_analytics.storage.Store") as mock_store_cls:
        # Properly mock the store and its async method
        mock_store_instance = mock_store_cls.return_value
        mock_store_instance.async_load = AsyncMock(return_value={})
        mock_store_instance.async_save = AsyncMock()

        coordinator = HeatingDataCoordinator(hass, entry)
        # Mock storage.async_load_data to avoid the real call failing with MagicMock
        coordinator.storage.async_load_data = AsyncMock()
        coordinator._async_save_data = AsyncMock()
        coordinator._process_hourly_data = AsyncMock()
        coordinator.statistics.calculate_temp_stats = MagicMock()

        # 1. Initial Call (No last hour)
        current_time = datetime(2023, 10, 27, 12, 0, 0)
        with patch("custom_components.heating_analytics.coordinator.dt_util.now", return_value=current_time):
             await coordinator._async_update_data()

        # Should initialize last processed, but NOT trigger processing (first run)
        assert coordinator._last_hour_processed == 12
        coordinator._process_hourly_data.assert_not_called()

        # 2. Same Hour (12:30)
        current_time = datetime(2023, 10, 27, 12, 30, 0)
        with patch("custom_components.heating_analytics.coordinator.dt_util.now", return_value=current_time):
             await coordinator._async_update_data()

        coordinator._process_hourly_data.assert_not_called()

        # 3. Next Hour (13:00)
        current_time = datetime(2023, 10, 27, 13, 0, 0)
        with patch("custom_components.heating_analytics.coordinator.dt_util.now", return_value=current_time):
             await coordinator._async_update_data()

        coordinator._process_hourly_data.assert_awaited_once_with(current_time)
        assert coordinator._last_hour_processed == 13
        coordinator.statistics.calculate_temp_stats.assert_called_once()

@pytest.mark.asyncio
async def test_log_retention(hass: HomeAssistant):
    """Test that hourly logs are truncated (retention policy)."""
    entry = MagicMock()
    entry.data = {"balance_point": 17.0}

    with patch("custom_components.heating_analytics.storage.Store"):
        coordinator = HeatingDataCoordinator(hass, entry)
        coordinator._async_save_data = AsyncMock()

        # Fill logs with 2160 entries (90 days)
        coordinator._hourly_log = [{"id": i, "temp": 0.0} for i in range(2160)]

        coordinator._collector.sample_count = 1
        coordinator._collector.wind_values = [0.0]
        coordinator._collector.temp_sum = 0.0

        current_time = datetime(2023, 10, 27, 13, 0, 0)
        await coordinator._process_hourly_data(current_time)

        # Should append 1, total 2161 -> Truncate to 2160
        assert len(coordinator._hourly_log) == 2160
        # The new one should be at the end
        assert coordinator._hourly_log[-1]["timestamp"] == current_time.isoformat()
        # The oldest one (id=0) should be gone. The one at index 0 should be id=1
        assert coordinator._hourly_log[0]["id"] == 1

@pytest.mark.asyncio
async def test_csv_append_trigger(hass: HomeAssistant):
    """Test that CSV appending is called during hourly processing."""
    entry = MagicMock()
    entry.data = {"csv_auto_logging": True}

    with patch("custom_components.heating_analytics.storage.Store") as mock_store_cls:
        coordinator = HeatingDataCoordinator(hass, entry)
        coordinator._async_save_data = AsyncMock()
        coordinator.storage.append_hourly_log_csv = AsyncMock()

        coordinator._collector.sample_count = 1
        coordinator._collector.wind_values = [0.0]
        coordinator._hourly_log = [{"temp": 0.0}] # For inertia

        current_time = datetime(2023, 10, 27, 13, 0, 0)
        await coordinator._process_hourly_data(current_time)

        coordinator.storage.append_hourly_log_csv.assert_awaited_once()


@pytest.mark.asyncio
async def test_log_retention_default_90_days(hass: HomeAssistant):
    """Test that default retention (no config) keeps 2160 entries (90 days)."""
    entry = MagicMock()
    entry.data = {"balance_point": 17.0}  # No hourly_log_retention_days key

    with patch("custom_components.heating_analytics.storage.Store"):
        coordinator = HeatingDataCoordinator(hass, entry)
        assert coordinator._hourly_log_max_entries == 90 * 24  # 2160


@pytest.mark.asyncio
async def test_log_retention_configurable_365_days(hass: HomeAssistant):
    """Test that 365-day retention keeps 8760 entries."""
    entry = MagicMock()
    entry.data = {"balance_point": 17.0, "hourly_log_retention_days": 365}

    with patch("custom_components.heating_analytics.storage.Store"):
        coordinator = HeatingDataCoordinator(hass, entry)
        coordinator._async_save_data = AsyncMock()

        assert coordinator._hourly_log_max_entries == 365 * 24  # 8760

        # Fill log just past the limit
        coordinator._hourly_log = [{"id": i, "temp": 0.0} for i in range(8760)]

        coordinator._collector.sample_count = 1
        coordinator._collector.wind_values = [0.0]
        coordinator._collector.temp_sum = 0.0

        current_time = datetime(2023, 10, 27, 13, 0, 0)
        await coordinator._process_hourly_data(current_time)

        # Should append 1, total 8761 -> Truncate to 8760
        assert len(coordinator._hourly_log) == 8760
        assert coordinator._hourly_log[-1]["timestamp"] == current_time.isoformat()
        assert coordinator._hourly_log[0]["id"] == 1


@pytest.mark.asyncio
async def test_log_entry_records_balance_point(hass: HomeAssistant):
    """Hourly log entries must record the active balance point (#856 A)."""
    entry = MagicMock()
    entry.data = {"balance_point": 14.0}

    with patch("custom_components.heating_analytics.storage.Store"):
        coordinator = HeatingDataCoordinator(hass, entry)
        coordinator._async_save_data = AsyncMock()
        coordinator.balance_point = 14.0

        coordinator._collector.sample_count = 1
        coordinator._collector.wind_values = [0.0]
        coordinator._collector.temp_sum = 0.0

        current_time = datetime(2023, 10, 27, 13, 0, 0)
        await coordinator._process_hourly_data(current_time)

        assert coordinator._hourly_log[-1]["bp_at_log_time"] == 14.0

        # BP change mid-run is reflected on the next entry without touching
        # previous ones — diagnostics reads both and can flag the transition.
        coordinator.balance_point = 16.5
        coordinator._collector.sample_count = 1
        coordinator._collector.wind_values = [0.0]
        coordinator._collector.temp_sum = 0.0

        next_time = datetime(2023, 10, 27, 14, 0, 0)
        await coordinator._process_hourly_data(next_time)

        assert coordinator._hourly_log[-2]["bp_at_log_time"] == 14.0
        assert coordinator._hourly_log[-1]["bp_at_log_time"] == 16.5


@pytest.mark.asyncio
async def test_log_retention_configurable_180_days(hass: HomeAssistant):
    """Test that 180-day retention keeps 4320 entries."""
    entry = MagicMock()
    entry.data = {"balance_point": 17.0, "hourly_log_retention_days": 180}

    with patch("custom_components.heating_analytics.storage.Store"):
        coordinator = HeatingDataCoordinator(hass, entry)
        assert coordinator._hourly_log_max_entries == 180 * 24  # 4320


@pytest.mark.asyncio
async def test_log_retention_preserves_list_identity(hass: HomeAssistant):
    """Test that retention trimming preserves list object identity (ModelState refs)."""
    entry = MagicMock()
    entry.data = {"balance_point": 17.0, "hourly_log_retention_days": 90}

    with patch("custom_components.heating_analytics.storage.Store"):
        coordinator = HeatingDataCoordinator(hass, entry)
        coordinator._async_save_data = AsyncMock()

        coordinator._hourly_log = [{"id": i, "temp": 0.0} for i in range(2160)]
        original_list_id = id(coordinator._hourly_log)

        coordinator._collector.sample_count = 1
        coordinator._collector.wind_values = [0.0]
        coordinator._collector.temp_sum = 0.0

        current_time = datetime(2023, 10, 27, 13, 0, 0)
        await coordinator._process_hourly_data(current_time)

        # List identity must be preserved (in-place deletion)
        assert id(coordinator._hourly_log) == original_list_id


def _make_stats_mock(
    solar_reduction_kwh: float,
    base_kwh: float = 0.0,
    solar_wasted_kwh: float = 0.0,
) -> MagicMock:
    """Return a statistics MagicMock whose calculate_total_power yields controlled values."""
    mock = MagicMock()
    mock.calculate_total_power.return_value = {
        "global_base_kwh": base_kwh,
        "breakdown": {
            "base_kwh": base_kwh,
            "solar_reduction_kwh": solar_reduction_kwh,
            "solar_wasted_kwh": solar_wasted_kwh,
        },
    }
    return mock


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "actual_kwh, expected_kwh, solar_reduction_kwh, solar_wasted_kwh, expected_pct",
    [
        # Normal hour: expected dominates — formula is unchanged from pre-fix.
        (2.5, 2.0, 0.3, 0.0, 25.0),
        # Solar-saturation, large applied (demand ≥ applied): denom = 0.5*(2.0+0) = 1.0.
        (0.187, 0.0, 2.0, 0.0, 18.7),
        # Night, no solar: denom collapses to ENERGY_GUARD_THRESHOLD (0.01).
        (0.05, 0.0, 0.0, 0.0, round(0.05 / ENERGY_GUARD_THRESHOLD * 100, 1)),
        # Heavy saturation: applied≈0 (demand capped it at 0.05), wasted=2.95 (large
        # solar potential fully clipped). denom = 0.5*(0.05+2.95) = 1.5 → 3.3 %.
        # Without the solar_wasted term: denom = max(0, 0.01, 0.025) = 0.025 → 200 %.
        (0.05, 0.0, 0.05, 2.95, round(0.05 / 1.5 * 100, 1)),
    ],
)
async def test_deviation_pct_floored_denominator(
    hass: HomeAssistant,
    actual_kwh,
    expected_kwh,
    solar_reduction_kwh,
    solar_wasted_kwh,
    expected_pct,
):
    """ATTR_LAST_HOUR_DEVIATION_PCT uses a floored denominator so saturation hours
    never produce ∞ or near-∞ percentages. The floor uses applied+wasted solar
    (pre-saturation magnitude) so fully-clipped hours are also bounded."""
    entry = MagicMock()
    entry.data = {"balance_point": 17.0}

    with patch("custom_components.heating_analytics.storage.Store"):
        coordinator = HeatingDataCoordinator(hass, entry)
        coordinator._async_save_data = AsyncMock()

        # Inject controlled actual / expected / solar values.
        coordinator._collector.energy_hour = actual_kwh
        coordinator._collector.expected_energy_hour = expected_kwh
        # sample_count=0 skips learning; deviation_pct is computed before that gate.
        coordinator._collector.sample_count = 0

        coordinator.statistics = _make_stats_mock(
            solar_reduction_kwh, base_kwh=expected_kwh, solar_wasted_kwh=solar_wasted_kwh
        )

        current_time = datetime(2026, 5, 2, 17, 0, 0)
        await coordinator._process_hourly_data(current_time)

        result = coordinator.data[ATTR_LAST_HOUR_DEVIATION_PCT]
        assert result == pytest.approx(expected_pct, abs=0.1)
