"""Test storage corruption resilience."""
from unittest.mock import MagicMock, patch, AsyncMock
import pytest
from homeassistant.core import HomeAssistant
from custom_components.heating_analytics.storage import StorageManager

@pytest.mark.asyncio
async def test_load_data_corrupted_json(hass: HomeAssistant):
    """Test loading data when JSON is corrupted."""
    coordinator = MagicMock()
    # Mock hass manually as the fixture is behaving as async_generator in this context (weird pytest interaction)
    # or just use MagicMock for hass completely to avoid fixture complexity for unit test.
    mock_hass = MagicMock()
    mock_hass.components.persistent_notification = MagicMock()
    mock_hass.components.persistent_notification.create = MagicMock()
    coordinator.hass = mock_hass

    with patch("custom_components.heating_analytics.storage.Store") as mock_store_cls:
        mock_store = mock_store_cls.return_value
        mock_store.async_load = AsyncMock(side_effect=Exception("Corrupted JSON"))

        storage = StorageManager(coordinator)

        # Run load
        await storage.async_load_data()

        # Should catch exception and notify
        mock_hass.components.persistent_notification.create.assert_called_once()
        assert "Heating Analytics: Data Load Error" in mock_hass.components.persistent_notification.create.call_args[1]["title"]

@pytest.mark.asyncio
async def test_load_data_partial_missing(hass: HomeAssistant):
    """Test loading data when some keys are missing."""
    coordinator = MagicMock()

    coordinator.hass = MagicMock()
    coordinator._correlation_data = {}
    coordinator.forecast = MagicMock()
    coordinator.statistics = MagicMock()

    with patch("custom_components.heating_analytics.storage.Store") as mock_store_cls:
        mock_store = mock_store_cls.return_value
        mock_store.async_load = AsyncMock(return_value={
            "correlation_data": {"0": {"normal": 1.0}}
        })

        storage = StorageManager(coordinator)
        await storage.async_load_data()

        # Check that it loaded what was there
        assert coordinator._correlation_data == {"0": {"normal": 1.0}}

        # And defaults for others (not raising KeyError)
        # Verify _daily_history was NOT set back onto the coordinator (since it wasn't in storage)
        # Since coordinator is MagicMock, checking `hasattr` is tricky because it creates attributes on access.
        # Instead, we check if _daily_history equals what StorageManager sets it to (from dict) or if it remains unset/default.
        # But StorageManager sets `self.coordinator._daily_history = loaded_daily_history` where default is {}.
        # So it SHOULD be {} if missing from file.

        assert coordinator._daily_history == {}

@pytest.mark.asyncio
async def test_load_invalid_types(hass: HomeAssistant):
    """Test loading data with wrong types."""
    coordinator = MagicMock()
    coordinator.hass = MagicMock()
    coordinator._correlation_data = {}
    coordinator.forecast = MagicMock()
    coordinator.statistics = MagicMock()

    with patch("custom_components.heating_analytics.storage.Store") as mock_store_cls:
        mock_store = mock_store_cls.return_value
        # Correlation data should be dict, but file has list
        mock_store.async_load = AsyncMock(return_value={
            "correlation_data": [1, 2, 3]
        })

        storage = StorageManager(coordinator)
        await storage.async_load_data()

        # Should remain empty dict (default) because list is rejected
        assert coordinator._correlation_data == {}
