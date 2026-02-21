import pytest
from unittest.mock import MagicMock, patch
from custom_components.heating_analytics.storage import StorageManager

@pytest.fixture
def mock_coordinator():
    coordinator = MagicMock()
    coordinator.hass = MagicMock()
    coordinator.hass.config.path = MagicMock(return_value="/tmp")
    coordinator.balance_point = 17.0
    coordinator.learning_rate = 0.1
    coordinator._correlation_data = {}
    coordinator._aux_coefficients = {}
    coordinator._hourly_log = []
    coordinator._daily_history = {}

    # Mock methods used in import
    coordinator._calculate_effective_wind = MagicMock(return_value=5.0)
    coordinator._get_wind_bucket = MagicMock(return_value="normal")
    coordinator._get_predicted_kwh = MagicMock(return_value=1.0)
    coordinator.learning = MagicMock()
    coordinator.learning.learn_from_historical_import = MagicMock(return_value={"status": "learned"})
    coordinator.solar_enabled = False

    # Use the real aggregation method logic or a simplified version to simulate the bug
    def aggregate_daily_logs_side_effect(day_logs):
        if not day_logs:
            return {}

        # This mimics the logic in HeatingDataCoordinator._aggregate_daily_logs
        # which sums up the 'tdd' field from hourly logs
        total_tdd = sum(e.get("tdd", 0.0) for e in day_logs)
        total_kwh = sum(e.get("actual_kwh", 0.0) for e in day_logs)
        avg_temp = sum(e.get("temp", 0.0) for e in day_logs) / len(day_logs)

        return {
            "kwh": total_kwh,
            "tdd": total_tdd,
            "temp": avg_temp
        }

    coordinator._aggregate_daily_logs = MagicMock(side_effect=aggregate_daily_logs_side_effect)
    coordinator._backfill_daily_from_hourly = MagicMock()

    return coordinator

@pytest.mark.asyncio
async def test_csv_import_tdd_population(mock_coordinator):
    """Test that CSV import properly calculates TDD for hourly logs."""

    storage = StorageManager(mock_coordinator)

    # Mock open and csv reading
    csv_content = (
        "timestamp,temperature,energy\n"
        "2023-01-01T10:00:00,7.0,1.5\n"
        "2023-01-01T11:00:00,7.0,1.5\n"
    )

    # We expect TDD to be calculated.
    # Balance point = 17.0. Temp = 7.0.
    # Diff = 10.0.
    # Hourly TDD contribution = 10.0 / 24.0 = 0.4166...

    mapping = {
        "timestamp": "timestamp",
        "temperature": "temperature",
        "energy": "energy"
    }

    with patch("builtins.open", new_callable=MagicMock) as mock_open:
        mock_file = MagicMock()
        mock_file.__enter__.return_value = filter(None, csv_content.splitlines())
        mock_file.__iter__.return_value = filter(None, csv_content.splitlines())
        mock_open.return_value = mock_file

        # We need to mock csv.DictReader too because we're passing a list of strings not a file handle exactly how csv expects
        # Actually, let's just mock the _process_csv inner function execution by using the real method but mocking the file interaction
        # The storage manager runs _process_csv in an executor. We'll bypass that for testing logic.

        # Better approach: Just invoke import_csv_data and let it use the mocked open/csv
        # We need to ensure async_add_executor_job simply runs the function
        async def execute_job(func, *args):
            return func(*args)

        mock_coordinator.hass.async_add_executor_job = MagicMock(side_effect=execute_job)

        # We also need to mock os.path.exists to return True
        with patch("os.path.exists", return_value=True):
            await storage.import_csv_data("dummy.csv", mapping)

    # Check if hourly logs were populated
    assert len(mock_coordinator._hourly_log) == 2

    first_entry = mock_coordinator._hourly_log[0]

    # Check if TDD is present and correct
    assert "tdd" in first_entry, "TDD field missing from imported hourly log"

    expected_tdd = (17.0 - 7.0) / 24.0 # 0.41666...
    assert abs(first_entry["tdd"] - expected_tdd) < 0.001, f"TDD value incorrect. Expected {expected_tdd}, got {first_entry.get('tdd')}"

    # Check if daily aggregation was called
    mock_coordinator._aggregate_daily_logs.assert_called()

    # Check the result in daily_history (simulated by the side_effect logic we injected)
    # The storage manager updates _daily_history manually using the result of _aggregate_daily_logs

    daily_entry = mock_coordinator._daily_history.get("2023-01-01")
    assert daily_entry is not None

    # If 2 entries of 0.4166, sum is 0.8333
    expected_daily_tdd = expected_tdd * 2
    assert abs(daily_entry["tdd"] - expected_daily_tdd) < 0.001, f"Daily TDD incorrect. Expected {expected_daily_tdd}, got {daily_entry.get('tdd')}"
