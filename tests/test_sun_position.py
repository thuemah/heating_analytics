"""Test sun position calculations using HA sun helpers."""
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone
import pytest
from homeassistant.util import dt as dt_util
from custom_components.heating_analytics.solar import SolarCalculator


# Mock config entry
class MockConfigEntry:
    def __init__(self):
        self.entry_id = "test_entry"
        self.data = {
            "outdoor_temp_sensor": "sensor.outdoor_temp",
            "wind_speed_sensor": "sensor.wind_speed",
            "weather_entity": "weather.test",
            "energy_sensors": [],
        }


@pytest.fixture
def mock_hass():
    hass = MagicMock()
    hass.data = {}
    # Set location (Oslo, Norway)
    hass.config.latitude = 59.9139
    hass.config.longitude = 10.7522
    hass.config.elevation = 0
    return hass


@pytest.fixture
def mock_coordinator(mock_hass):
    coordinator = MagicMock()
    coordinator.hass = mock_hass
    coordinator.solar_window_area = 10.0
    coordinator.solar_azimuth = 180.0  # South
    coordinator.balance_point = 17.0
    coordinator._solar_coefficients = {}
    return coordinator


@pytest.fixture
def solar_calc(mock_coordinator):
    return SolarCalculator(mock_coordinator)


def test_sun_position_uses_astral(solar_calc):
    """Test that sun position is calculated using astral library."""
    # Winter solstice 2024, noon UTC (Oslo: 13:00 local)
    dt_obj = datetime(2024, 12, 21, 12, 0, 0, tzinfo=timezone.utc)

    with patch('custom_components.heating_analytics.solar.Observer', create=True), \
         patch('custom_components.heating_analytics.solar.sun_elevation', create=True) as mock_elev, \
         patch('custom_components.heating_analytics.solar.sun_azimuth', create=True) as mock_azim, \
         patch('custom_components.heating_analytics.solar.HAS_ASTRAL', True):

        mock_elev.return_value = 6.5  # Low winter sun at Oslo
        mock_azim.return_value = 180.0  # South

        elevation, azimuth = solar_calc.get_approx_sun_pos(dt_obj)

        # Verify astral functions were called
        assert mock_elev.called
        assert mock_azim.called

        assert elevation == 6.5
        assert azimuth == 180.0


def test_sun_position_summer_vs_winter(solar_calc):
    """Test that sun position differs correctly between seasons."""
    # Winter: December 21, 2024, noon UTC
    winter_dt = datetime(2024, 12, 21, 12, 0, 0, tzinfo=timezone.utc)

    # Summer: June 21, 2024, noon UTC
    summer_dt = datetime(2024, 6, 21, 12, 0, 0, tzinfo=timezone.utc)

    with patch('custom_components.heating_analytics.solar.Observer', create=True), \
         patch('custom_components.heating_analytics.solar.sun_elevation', create=True) as mock_elev, \
         patch('custom_components.heating_analytics.solar.sun_azimuth', create=True) as mock_azim, \
         patch('custom_components.heating_analytics.solar.HAS_ASTRAL', True):

        # Winter: Low sun
        mock_elev.return_value = 6.5
        mock_azim.return_value = 180.0
        winter_elev, winter_azim = solar_calc.get_approx_sun_pos(winter_dt)

        # Summer: High sun
        mock_elev.return_value = 53.5
        mock_azim.return_value = 180.0
        summer_elev, summer_azim = solar_calc.get_approx_sun_pos(summer_dt)

        # Summer sun should be much higher than winter
        assert summer_elev > winter_elev
        assert summer_elev > 50  # Above 50° in summer at Oslo
        assert winter_elev < 10  # Below 10° in winter at Oslo


def test_sun_position_night(solar_calc):
    """Test sun position at night (below horizon)."""
    # Midnight UTC on winter solstice (Oslo: 01:00 local)
    dt_obj = datetime(2024, 12, 21, 0, 0, 0, tzinfo=timezone.utc)

    with patch('custom_components.heating_analytics.solar.Observer', create=True), \
         patch('custom_components.heating_analytics.solar.sun_elevation', create=True) as mock_elev, \
         patch('custom_components.heating_analytics.solar.sun_azimuth', create=True) as mock_azim, \
         patch('custom_components.heating_analytics.solar.HAS_ASTRAL', True):

        mock_elev.return_value = -42.0  # Well below horizon
        mock_azim.return_value = 0.0

        elevation, azimuth = solar_calc.get_approx_sun_pos(dt_obj)

        # Sun below horizon
        assert elevation < 0


def test_sun_position_no_location(solar_calc):
    """Test sun position returns 0,0 when location is not configured."""
    solar_calc.coordinator.hass.config.latitude = None
    solar_calc.coordinator.hass.config.longitude = None

    dt_obj = datetime(2024, 6, 21, 12, 0, 0, tzinfo=timezone.utc)
    elevation, azimuth = solar_calc.get_approx_sun_pos(dt_obj)

    assert elevation == 0.0
    assert azimuth == 0.0


def test_sun_position_exception_handling(solar_calc):
    """Test that exceptions are handled gracefully."""
    dt_obj = datetime(2024, 6, 21, 12, 0, 0, tzinfo=timezone.utc)

    with patch('custom_components.heating_analytics.solar.sun_elevation', side_effect=Exception("Test error"), create=True), \
         patch('custom_components.heating_analytics.solar.HAS_ASTRAL', True):
        elevation, azimuth = solar_calc.get_approx_sun_pos(dt_obj)

        # Should return 0,0 on error
        assert elevation == 0.0
        assert azimuth == 0.0


def test_sun_position_no_astral(solar_calc):
    """Test that function handles missing astral library."""
    dt_obj = datetime(2024, 6, 21, 12, 0, 0, tzinfo=timezone.utc)

    with patch('custom_components.heating_analytics.solar.HAS_ASTRAL', False):
        elevation, azimuth = solar_calc.get_approx_sun_pos(dt_obj)

        # Should return 0,0 when astral not available
        assert elevation == 0.0
        assert azimuth == 0.0


def test_solar_factor_consistency(solar_calc):
    """Test that solar factor calculation is consistent across time."""
    # Noon on a clear day
    dt_obj = datetime(2024, 6, 21, 12, 0, 0, tzinfo=timezone.utc)

    with patch('custom_components.heating_analytics.solar.Observer', create=True), \
         patch('custom_components.heating_analytics.solar.sun_elevation', create=True) as mock_elev, \
         patch('custom_components.heating_analytics.solar.sun_azimuth', create=True) as mock_azim, \
         patch('custom_components.heating_analytics.solar.HAS_ASTRAL', True):

        mock_elev.return_value = 53.5
        mock_azim.return_value = 180.0  # South

        elevation, azimuth = solar_calc.get_approx_sun_pos(dt_obj)

        # Clear sky (0% cloud)
        solar_factor = solar_calc.calculate_solar_factor(elevation, azimuth, 0.0)

        # Should be > 0 with sun above horizon
        assert solar_factor > 0
        assert solar_factor <= 1.0


def test_estimate_daily_avg_uses_astral(solar_calc):
    """Test that daily average estimation uses astral library."""
    from datetime import date

    test_date = date(2024, 6, 21)

    with patch.object(solar_calc, 'get_approx_sun_pos') as mock_sun_pos:
        # Mock sun position for all 24 hours
        mock_sun_pos.return_value = (30.0, 180.0)

        avg_factor = solar_calc.estimate_daily_avg_solar_factor(test_date, cloud_coverage=50.0)

        # Should call get_approx_sun_pos 24 times (once per hour)
        assert mock_sun_pos.call_count == 24

        # Should return a valid factor
        assert 0.0 <= avg_factor <= 1.0
