"""Forecast Manager Service."""
from __future__ import annotations

import copy
import logging
import math
from datetime import datetime, timedelta, date

from homeassistant.util import dt as dt_util
from homeassistant.const import UnitOfSpeed

from .helpers import convert_speed_to_ms, get_last_year_iso_date
from .explanation import WeatherImpactAnalyzer, ExplanationFormatter
from .const import (
    CONF_FORECAST_CROSSOVER_DAY,
    CONF_SECONDARY_WEATHER_ENTITY,
    DEFAULT_FORECAST_CROSSOVER_DAY,
    DEFAULT_INERTIA_WEIGHTS,
    CLOUD_COVERAGE_MAP,
    ATTR_TEMP_FORECAST_TODAY,
    ATTR_TDD_DAILY_STABLE,
    ATTR_MIDNIGHT_FORECAST,
    ATTR_MIDNIGHT_UNIT_ESTIMATES,
    ATTR_MIDNIGHT_UNIT_MODES,
    ATTR_FORECAST_UNCERTAINTY,
    DEFAULT_UNCERTAINTY_P50,
    DEFAULT_UNCERTAINTY_P95,
    ATTR_DAILY_FORECAST,
    ATTR_WEEKLY_SUMMARY,
    ATTR_FORECAST_RANGE_MIN,
    ATTR_FORECAST_RANGE_MAX,
    ATTR_AVG_TEMP_FORECAST,
    ATTR_AVG_WIND_FORECAST,
    ATTR_COLDEST_DAY,
    ATTR_WARMEST_DAY,
    ATTR_TYPICAL_WEEK_KWH,
    ATTR_VS_TYPICAL_KWH,
    ATTR_VS_TYPICAL_PCT,
    ATTR_PEAK_DAY,
    ATTR_LIGHTEST_DAY,
    ATTR_WEEK_START_DATE,
    ATTR_WEEK_END_DATE,
    DEFAULT_CLOUD_COVERAGE,
)

_LOGGER = logging.getLogger(__name__)

class ForecastManager:
    """Manages weather forecasts and future predictions."""

    def __init__(self, coordinator) -> None:
        """Initialize with reference to coordinator."""
        self.coordinator = coordinator

        # State Variables
        self._reference_forecast = None  # Persisted snapshot (Midnight) - Used for Deviation
        self._primary_reference_forecast = None # Shadow: Primary only (Midnight)
        self._secondary_reference_forecast = None # Shadow: Secondary only (Midnight)
        self._live_forecast = None       # In-memory only (Hourly update) - Used for Predictions

        self._cached_forecast_date = None  # Date when REFERENCE was cached
        self._last_live_update = None      # Timestamp of last live update

        self._cached_forecast_uncertainty = None  # Cache for expensive stats calculation
        self._cached_reference_map = None # Cache for weather deviation calculation {hour: data}
        self._forecast_history = [] # List of {date, forecast_kwh, actual_kwh, error_kwh}
        self._midnight_forecast_snapshot = {} # {date, kwh, ...}

        # Cached Week Ahead Stats
        self._cached_week_ahead_stats = None
        self._cached_week_ahead_timestamp = None

        # Cached Data for Week Projection
        self._cached_long_term_hourly = [] # List of hourly forecast dicts (OpenMeteo style)
        self._cached_long_term_daily = [] # List of daily forecast dicts (Met.no style)
        self._cached_long_term_date = None

        # Performance Cache (Optimized Access)
        self._cached_hourly_by_date = {} # { "YYYY-MM-DD": [items] }

    def _rebuild_optimized_cache(self, source_data: list):
        """Rebuild the optimized date-indexed cache from hourly data."""
        self._cached_hourly_by_date = {}
        if not source_data:
            return

        for f in source_data:
            dt_str = f.get("datetime")
            if dt_str:
                try:
                    f_dt = dt_util.parse_datetime(dt_str)
                    if f_dt:
                        f_dt_local = dt_util.as_local(f_dt)
                        d_key = f_dt_local.date().isoformat()
                        if d_key not in self._cached_hourly_by_date:
                            self._cached_hourly_by_date[d_key] = []
                        self._cached_hourly_by_date[d_key].append(f)
                except (ValueError, TypeError) as e:
                    _LOGGER.debug(f"Failed to parse forecast datetime '{dt_str}': {e}")

    def _build_reference_map(self, source_data: list):
        """Build optimized map for weather deviation calculation (Method B: O(N) -> O(1))."""
        forecast_map = {}
        if not source_data:
            self._cached_reference_map = {}
            return

        weather_wind_unit = self.coordinator._get_weather_wind_unit()

        for f in source_data:
            dt_str = f.get("datetime")
            if dt_str:
                try:
                    f_dt = dt_util.parse_datetime(dt_str)
                    if f_dt:
                        f_dt = dt_util.as_local(f_dt)
                        # We only care about forecast for "Today" in deviation analysis,
                        # but we can store date-keyed map or just all hours if needed.
                        # Since deviation is only for TODAY, we can filter here or map by full datetime.
                        # Mapping by date_iso + hour is safer.
                        date_iso = f_dt.date().isoformat()

                        temp = float(f.get("temperature", 0.0))
                        w_speed = float(f.get("wind_speed", 0.0))
                        w_gust = f.get("wind_gust_speed")

                        w_speed_ms = convert_speed_to_ms(w_speed, weather_wind_unit)
                        if w_gust is not None:
                            try:
                                w_gust = float(w_gust)
                                w_gust_ms = convert_speed_to_ms(w_gust, weather_wind_unit)
                            except (ValueError, TypeError):
                                w_gust_ms = None
                        else:
                            w_gust_ms = None

                        eff_wind = self.coordinator._calculate_effective_wind(w_speed_ms, w_gust_ms)
                        bucket = self.coordinator._get_wind_bucket(eff_wind)

                        # Solar Info for Reference Map
                        solar_factor = 0.0
                        if self.coordinator.solar_enabled:
                            condition = f.get("condition")
                            cloud_cov = DEFAULT_CLOUD_COVERAGE
                            if condition and condition in CLOUD_COVERAGE_MAP:
                                 cloud_cov = float(CLOUD_COVERAGE_MAP[condition])
                            elif f.get("cloud_coverage") is not None:
                                 cloud_cov = float(f.get("cloud_coverage"))

                            elev, azim = self.coordinator.solar.get_approx_sun_pos(f_dt)
                            solar_factor = self.coordinator.solar.calculate_solar_factor(elev, azim, cloud_cov)

                        if date_iso not in forecast_map:
                            forecast_map[date_iso] = {}

                        forecast_map[date_iso][f_dt.hour] = {
                            "temp": temp,
                            "wind": eff_wind,
                            "bucket": bucket,
                            "solar_factor": solar_factor
                        }
                except (ValueError, TypeError):
                    pass

        self._cached_reference_map = forecast_map
        _LOGGER.debug(f"Built reference map for {len(forecast_map)} days")

    async def update_daily_forecast(self):
        """Update daily forecast and predicted energy.

        Manages two forecast types:
        1. Reference Forecast (Persisted): Locked at midnight. Used for "Why did it change?"
        2. Live Forecast (Memory): Updated hourly. Used for "What happens now?"
        """
        if not self.coordinator.weather_entity:
            return

        now = dt_util.now()
        today_str = now.date().isoformat()

        # --- Update Rules ---
        should_update_reference = (
            self._reference_forecast is None or
            self._cached_forecast_date != today_str or
            (now.hour == 0 and now.minute < 20)
        )
        should_update_live = (
            self._live_forecast is None or
            self._last_live_update is None or
            (now - self._last_live_update) > timedelta(hours=1)
        )

        # Runtime Reset for Day Change
        if self._cached_forecast_date and self._cached_forecast_date != today_str:
            _LOGGER.info(f"New day detected in update loop: {today_str}. Resetting stable TDD map.")
            should_update_reference = True
            should_update_live = True

        # Perform Update
        if should_update_reference or should_update_live:
            (
                p_hourly, s_hourly, b_hourly,
                p_daily, s_daily, b_daily
            ) = await self._fetch_and_blend_forecasts()

            if b_hourly:
                self._enrich_forecast_with_sun(b_hourly)
                if p_hourly: self._enrich_forecast_with_sun(p_hourly)
                if s_hourly: self._enrich_forecast_with_sun(s_hourly)

                if should_update_reference:
                    self._reference_forecast = b_hourly
                    self._primary_reference_forecast = p_hourly
                    self._secondary_reference_forecast = s_hourly
                    self._cached_forecast_date = today_str
                    self._build_reference_map(self._reference_forecast)
                    _LOGGER.info(f"Reference Forecast updated for {today_str} with {len(b_hourly)} items.")
                    # Persist the new reference immediately to avoid loss on restart (Baseline Consistency)
                    await self.coordinator.storage.async_save_data()
                if should_update_live:
                    self._live_forecast = b_hourly
                    self._last_live_update = now
                    # Invalidate week ahead cache
                    self._cached_week_ahead_stats = None
                    _LOGGER.debug(f"Live Forecast updated at {now.strftime('%H:%M')}")

                self._cached_long_term_hourly = b_hourly
                self._cached_long_term_date = today_str
                self._rebuild_optimized_cache(b_hourly)

            if b_daily:
                self._cached_long_term_daily = b_daily
        else:
            _LOGGER.debug("Forecasts are fresh, skipping API call.")

        if self._live_forecast is None and self._reference_forecast is not None:
            self._live_forecast = self._reference_forecast
            if not self._cached_long_term_hourly:
                self._cached_long_term_hourly = self._reference_forecast
                self._rebuild_optimized_cache(self._cached_long_term_hourly)

        if self._reference_forecast is None:
            self._reference_forecast = []

        if self._cached_reference_map is None and self._reference_forecast:
            self._build_reference_map(self._reference_forecast)

        self._update_stable_tdd_today()

        if not self._midnight_forecast_snapshot or self._midnight_forecast_snapshot.get("date") != today_str:
            snapshot_data = self._capture_daily_forecast_snapshot()
            full_day_forecast = snapshot_data["kwh"]

            # Determine source of reference forecast for the day
            snapshot_sources = [
                item.get('_source', 'unknown')
                for item in self._reference_forecast
                if dt_util.parse_datetime(item.get("datetime")).date().isoformat() == today_str
            ]
            if not snapshot_sources:
                snapshot_source = "unknown"
            elif all(s == snapshot_sources[0] for s in snapshot_sources):
                snapshot_source = snapshot_sources[0]
            else:
                snapshot_source = "blended"

            if 0 < full_day_forecast < 200:
                self._midnight_forecast_snapshot = {
                    "date": today_str,
                    "kwh": round(full_day_forecast, 2),
                    "unit_estimates": snapshot_data["unit_estimates"],
                    "unit_modes": snapshot_data["unit_modes"],
                    "timestamp": now.isoformat(),
                    "source": snapshot_source,
                    "primary_entity": snapshot_data.get("primary_entity"),
                    "secondary_entity": snapshot_data.get("secondary_entity"),
                    "crossover_day": snapshot_data.get("crossover_day"),
                    "hourly_plan": snapshot_data.get("hourly_plan", [])
                }
                _LOGGER.info(f"Captured Midnight Forecast Snapshot: {full_day_forecast:.2f} kWh (Source: {snapshot_source})")

        if self._midnight_forecast_snapshot and self._midnight_forecast_snapshot.get("date") == today_str:
            self.coordinator.data[ATTR_MIDNIGHT_FORECAST] = self._midnight_forecast_snapshot["kwh"]
            self.coordinator.data[ATTR_MIDNIGHT_UNIT_ESTIMATES] = self._midnight_forecast_snapshot.get("unit_estimates")
            self.coordinator.data[ATTR_MIDNIGHT_UNIT_MODES] = self._midnight_forecast_snapshot.get("unit_modes")

    async def _fetch_and_blend_forecasts(self) -> tuple[list, list, list, list, list, list]:
        """Fetch forecasts from primary and secondary sources and blend them.

        Returns:
            (primary_hourly, secondary_hourly, blended_hourly,
             primary_daily, secondary_daily, blended_daily)
        """
        primary_entity = self.coordinator.weather_entity
        secondary_entity = self.coordinator.entry.data.get(CONF_SECONDARY_WEATHER_ENTITY)

        # Fetch Primary
        primary_hourly = await self._fetch_forecast(primary_entity, "hourly")
        primary_daily = await self._fetch_forecast(primary_entity, "daily")

        # Add source metadata
        for item in primary_hourly: item['_source'] = 'primary'
        for item in primary_daily: item['_source'] = 'primary'

        if not secondary_entity:
            return (
                primary_hourly, [], primary_hourly,
                primary_daily, [], primary_daily
            )

        # Fetch Secondary
        secondary_hourly = await self._fetch_forecast(secondary_entity, "hourly")
        secondary_daily = await self._fetch_forecast(secondary_entity, "daily")

        for item in secondary_hourly: item['_source'] = 'secondary'
        for item in secondary_daily: item['_source'] = 'secondary'

        # Blend
        crossover_day = self.coordinator.entry.data.get(CONF_FORECAST_CROSSOVER_DAY, DEFAULT_FORECAST_CROSSOVER_DAY)
        blended_hourly = self._blend_forecasts(primary_hourly, secondary_hourly, crossover_day, 'hourly')
        blended_daily = self._blend_forecasts(primary_daily, secondary_daily, crossover_day, 'daily')

        return (
            primary_hourly, secondary_hourly, blended_hourly,
            primary_daily, secondary_daily, blended_daily
        )

    def _blend_forecasts(self, primary: list, secondary: list, crossover_day: int, forecast_type: str) -> list:
        """Blend two forecast lists based on a crossover day with robust gap-filling."""
        if not primary and not secondary: return []
        if not secondary:
            for item in primary: item['_source'] = 'primary'
            return primary
        if not primary:
            for item in secondary: item['_source'] = 'secondary'
            return secondary

        blended_map = {}
        now = dt_util.now()
        today = now.date()
        crossover_date = today + timedelta(days=crossover_day)

        # 1. Populate with secondary as a baseline
        for item in secondary:
            try:
                parsed = dt_util.parse_datetime(item["datetime"])
                if not parsed:
                    continue
                f_dt = dt_util.as_local(parsed)
                item['_source'] = 'secondary'
                blended_map[f_dt.isoformat()] = item
            except (ValueError, TypeError):
                continue

        # 2. Overwrite with primary for dates before the crossover
        for item in primary:
            try:
                parsed = dt_util.parse_datetime(item["datetime"])
                if not parsed:
                    continue
                f_dt = dt_util.as_local(parsed)
                if f_dt.date() < crossover_date:
                    item['_source'] = 'primary'
                    blended_map[f_dt.isoformat()] = item
            except (ValueError, TypeError):
                continue

        blended_list = list(blended_map.values())
        blended_list.sort(key=lambda x: x.get("datetime", ""))

        _LOGGER.info(f"Blended {forecast_type} forecast: {len(blended_list)} total items. Crossover at day {crossover_day}.")
        return blended_list


    async def _fetch_forecast(self, entity_id: str, forecast_type: str) -> list:
        """Fetch a specific type of forecast from a weather entity."""
        if not entity_id:
            return []

        forecasts = []
        try:
            response = await self.coordinator.hass.services.async_call(
                "weather",
                "get_forecasts",
                {"entity_id": entity_id, "type": forecast_type},
                blocking=True,
                return_response=True,
            )
            if response and entity_id in response:
                forecasts = response[entity_id].get("forecast", [])
                _LOGGER.debug(f"Fetched {len(forecasts)} {forecast_type} items from {entity_id}")
            else:
                # Fallback for integrations not supporting the service call
                state = self.coordinator.hass.states.get(entity_id)
                if state and state.attributes.get("forecast"):
                    forecasts = state.attributes.get("forecast")
                    _LOGGER.debug(f"Fetched {len(forecasts)} {forecast_type} items from {entity_id} (attribute fallback)")

        except Exception as e:
            if self.coordinator.hass.is_running:
                _LOGGER.warning(f"Could not fetch {forecast_type} forecasts from {entity_id}: {e}")

        return forecasts

    def _enrich_forecast_with_sun(self, forecast_data: list):
        """Add sun position to forecast data in-place."""
        for f in forecast_data:
            dt_str = f.get("datetime")
            if dt_str:
                f_dt = dt_util.parse_datetime(dt_str)
                if f_dt:
                    elev, azim = self.coordinator.solar.get_approx_sun_pos(f_dt)
                    f["elevation"] = elev
                    f["azimuth"] = azim

    def _get_live_forecast_or_ref(self) -> list:
        """Return live forecast if available, else reference."""
        if self._live_forecast:
            return self._live_forecast
        if self._reference_forecast:
            return self._reference_forecast
        return []

    def _update_stable_tdd_today(self):
        """Update Stable TDD calculations for Today using REFERENCE forecast for future."""
        now = dt_util.now()
        end_of_day = now.replace(hour=23, minute=59, second=59)
        start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
        today_iso = now.date().isoformat()

        # Use local variable for calculations instead of persistent state
        temps_for_calculation = {}

        # 1. Backfill with actuals from today's log (Composite Forecast)
        if self.coordinator._hourly_log:
             for entry in self.coordinator._hourly_log:
                 if entry["timestamp"].startswith(today_iso):
                     hour = entry["hour"]
                     temps_for_calculation[hour] = entry["temp"]

        # 2. Fill remaining/future hours with REFERENCE Forecast (Stable Plan)
        # We use reference forecast to avoid "moving goal post" jumps during the day
        forecast_source = self._reference_forecast

        if forecast_source:
            for f in forecast_source:
                dt_str = f.get("datetime")
                f_dt = dt_util.parse_datetime(dt_str)
                if not f_dt: continue
                f_dt = dt_util.as_local(f_dt)

                if start_of_day <= f_dt <= end_of_day:
                    hour_key = f_dt.hour
                    # Only use forecast if we don't have actual log for this hour
                    if hour_key not in temps_for_calculation:
                        temp = float(f.get("temperature", 0.0))
                        temps_for_calculation[hour_key] = temp

        # Calculate Stable Daily Stats
        if temps_for_calculation:
            avg_forecast_temp = sum(temps_for_calculation.values()) / len(temps_for_calculation)
            self.coordinator.data[ATTR_TEMP_FORECAST_TODAY] = round(avg_forecast_temp, 1)

            # Calculate Stable TDD (Thermodynamic Integrity: Integration Method)
            tdd_sum = 0.0
            count = 0
            for t in temps_for_calculation.values():
                tdd_sum += abs(self.coordinator.balance_point - t)
                count += 1

            if count > 0:
                stable_tdd = tdd_sum / count
            else:
                stable_tdd = 0.0

            self.coordinator.data[ATTR_TDD_DAILY_STABLE] = round(stable_tdd, 1)
        else:
             self.coordinator.data[ATTR_TDD_DAILY_STABLE] = 0.0
             self.coordinator.data[ATTR_TEMP_FORECAST_TODAY] = None

    def _capture_daily_forecast_snapshot(self) -> dict:
        """Capture full-day forecast at start of day using REFERENCE forecast.

        Returns:
            dict: {
                "kwh": float,
                "primary_kwh": float,
                "secondary_kwh": float,
                "unit_estimates": dict[str, float],
                "unit_modes": dict[str, str],
                "count": int
            }
        """
        # Use REFERENCE here to lock the midnight view
        if not self._reference_forecast:
            return {
                "kwh": 0.0,
                "primary_kwh": 0.0,
                "secondary_kwh": 0.0,
                "unit_estimates": {},
                "unit_modes": {},
                "count": 0
            }

        today_start = dt_util.start_of_local_day()
        today_end = today_start + timedelta(days=1)

        # Seed thermal inertia from recent actual temperatures (prior to start of day)
        inertia_history = []
        target_iso = today_start.isoformat()

        history_needed = len(DEFAULT_INERTIA_WEIGHTS) - 1

        if self.coordinator._hourly_log:
            recent_logs = []
            for log in reversed(self.coordinator._hourly_log):
                 if log["timestamp"] < target_iso:
                     recent_logs.append(log)
                 if len(recent_logs) >= history_needed:
                     break

            for log in reversed(recent_logs):
                 # Use RAW temp for inertia calculation to prevent double-averaging
                 inertia_history.append(log["temp"])

        res_main = self._sum_forecast_energy_internal(
            start_time=today_start,
            end_time=today_end,
            inertia_history=inertia_history,
            include_start=True,
            source_data=self._reference_forecast,
            ignore_aux=True,
        )
        total_forecast = res_main["total_energy"]
        included_count = res_main["count"]
        unit_estimates = res_main["unit_totals"]
        hourly_plan = res_main["hourly_plan"]

        # Shadow Forecast Calculations
        primary_forecast = 0.0
        if self._primary_reference_forecast:
             primary_forecast, _, _, _ = self.sum_forecast_energy(
                start_time=today_start,
                end_time=today_end,
                inertia_history=inertia_history,
                include_start=True,
                source_data=self._primary_reference_forecast,
                ignore_aux=True,
            )

        secondary_forecast = 0.0
        if self._secondary_reference_forecast:
             secondary_forecast, _, _, _ = self.sum_forecast_energy(
                start_time=today_start,
                end_time=today_end,
                inertia_history=inertia_history,
                include_start=True,
                source_data=self._secondary_reference_forecast,
                ignore_aux=True,
            )

        # Capture current unit modes at the time of snapshot
        unit_modes = {
            eid: self.coordinator.get_unit_mode(eid)
            for eid in self.coordinator.energy_sensors
        }

        _LOGGER.info(f"Midnight forecast calculation: {included_count}/24 hours included. Total: {total_forecast:.2f} kWh (Primary: {primary_forecast:.2f}, Secondary: {secondary_forecast:.2f})")

        # Capture configuration at time of snapshot for provenance tracking
        primary_entity = self.coordinator.weather_entity
        secondary_entity = self.coordinator.entry.data.get(CONF_SECONDARY_WEATHER_ENTITY)
        crossover_day = self.coordinator.entry.data.get(CONF_FORECAST_CROSSOVER_DAY, DEFAULT_FORECAST_CROSSOVER_DAY)

        return {
            "kwh": total_forecast,
            "primary_kwh": primary_forecast,
            "secondary_kwh": secondary_forecast,
            "unit_estimates": unit_estimates,
            "unit_modes": unit_modes,
            "count": included_count,
            "primary_entity": primary_entity,
            "secondary_entity": secondary_entity,
            "crossover_day": crossover_day,
            "hourly_plan": hourly_plan
        }

    def calculate_future_energy(self, start_time: datetime, ignore_aux: bool = False, force_aux: bool = False, screen_override: float | None = None, force_no_wind: bool = False) -> tuple[float, float, dict[str, float]]:
        """Calculate sum of predicted energy for future hours using LIVE forecast."""
        # Prepare for Inertia Calculation on Forecast
        inertia_history = []
        history_needed = len(DEFAULT_INERTIA_WEIGHTS) - 1

        if self.coordinator._hourly_log:
            recent_logs = []
            if history_needed > 0:
                recent_logs = self.coordinator._hourly_log[-history_needed:]
            for log in recent_logs:
                # Use RAW temp for inertia calculation to prevent double-averaging
                inertia_history.append(log["temp"])

        # Append current hour temp estimation
        curr_temp = self.coordinator._get_float_state(self.coordinator.outdoor_temp_sensor)
        if curr_temp is not None:
            inertia_history.append(curr_temp)

        end_of_day = start_time.replace(hour=23, minute=59, second=59)

        # Merge Live and Reference to fill gaps
        # This ensures stability if Live forecast is partial (common with some providers)
        combined_source = self._merge_and_fill_forecast(
            start_time,
            end_of_day,
            self._live_forecast,
            self._reference_forecast
        )

        # Use Combined Forecast
        total_future, _, total_solar, unit_totals = self.sum_forecast_energy(
            start_time=start_time,
            end_time=end_of_day,
            inertia_history=inertia_history,
            include_start=False,
            source_data=combined_source,
            ignore_aux=ignore_aux,
            force_aux=force_aux,
            screen_override=screen_override,
            force_no_wind=force_no_wind,
        )

        return total_future, total_solar, unit_totals

    def get_forecast_for_hour(self, target_dt: datetime, source: str = 'live') -> dict | None:
        """Retrieve forecast data for a specific hour from the best available source.

        Used to "lock" the forecast plan into history when an hour completes.
        Priority: Live Forecast -> Reference Forecast

        Args:
            target_dt: The datetime for the hour to retrieve.
            source: 'live' (default) to use live forecast, 'reference' to use only the midnight snapshot.
        """
        source_data = []
        if source == 'reference':
            source_data = self._reference_forecast or []
        elif source == 'primary_reference':
            source_data = self._primary_reference_forecast or []
        elif source == 'secondary_reference':
            source_data = self._secondary_reference_forecast or []
        else: # Default to 'live' behavior
            source_data = self._get_live_forecast_or_ref()

        return self.get_forecast_from_list(target_dt, source_data)

    def get_forecast_from_list(self, target_dt: datetime, source_data: list) -> dict | None:
        """Find forecast for a specific hour in a given list."""
        if not source_data:
            return None

        # Ensure timezone awareness
        if target_dt.tzinfo is None:
             target_dt = dt_util.as_local(target_dt)

        target_compare = target_dt.replace(minute=0, second=0, microsecond=0)

        for item in source_data:
             dt_str = item.get("datetime")
             if not dt_str: continue

             try:
                 item_dt = dt_util.parse_datetime(dt_str)
                 if item_dt:
                     item_dt_local = dt_util.as_local(item_dt).replace(minute=0, second=0, microsecond=0)
                     if item_dt_local == target_compare:
                         return item
             except (ValueError, TypeError):
                 continue

        return None

    def get_plan_for_hour(self, target_dt: datetime, source: str = 'reference', ignore_aux: bool = False) -> tuple[float, dict]:
        """Calculate the planned energy for a specific hour using the specified forecast source.

        Args:
            target_dt: The hour to calculate for.
            source: 'reference' (default) for stable daily plan, or 'live' for up-to-date prediction.
            ignore_aux: If True, calculates assuming Normal Mode (used for Gross Forecast).
                       If False, respects the current auxiliary heating state.

        Returns:
            (predicted_kwh, unit_breakdown)
        """
        # 1. Get the forecast item for the target hour from the requested source
        forecast_item = self.get_forecast_for_hour(target_dt, source=source)

        if not forecast_item:
            return 0.0, {}

        # 2. Get necessary context for the calculation (inertia, wind unit, cloud cover)
        # Replicates the context setup from other callers.
        inertia_history = self.coordinator._get_inertia_list(target_dt)

        weather_wind_unit = self.coordinator._get_weather_wind_unit()
        current_cloud = self.coordinator._get_cloud_coverage()

        # 3. Call the internal processing function to get the kWh value
        # We pass a copy of inertia_history because _process_forecast_item modifies it.
        predicted_kwh, _, _, _, _, _, unit_breakdown = self._process_forecast_item(
            item=forecast_item,
            inertia_history=list(inertia_history),
            wind_unit=weather_wind_unit,
            default_cloud=current_cloud,
            ignore_aux=ignore_aux
        )

        return predicted_kwh, unit_breakdown

    def get_reference_plan_for_hour(self, target_dt: datetime, ignore_aux: bool = False) -> tuple[float, dict]:
        """Calculate the planned energy for a specific hour using the reference forecast.

        DEPRECATED: Use get_plan_for_hour(source='reference') instead.
        """
        return self.get_plan_for_hour(target_dt, source='reference', ignore_aux=ignore_aux)

    def _merge_and_fill_forecast(
        self,
        start_time: datetime,
        end_time: datetime,
        live_data: list | None,
        reference_data: list | None
    ) -> list:
        """Merge Live forecast with Reference forecast to fill gaps.

        Priority:
        1. Live Data (Most accurate)
        2. Reference Data (Midnight snapshot) - Fills missing hours
        3. (Implicit) Smart Fill - Logic handled elsewhere if needed, but Ref is usually sufficient.
        """
        if not live_data:
            return reference_data if reference_data else []
        if not reference_data:
            return live_data

        merged_map = {}

        # 1. Populate with Reference first (Baseline)
        for item in reference_data:
             dt_str = item.get("datetime")
             if dt_str:
                 try:
                     dt = dt_util.parse_datetime(dt_str)
                     if dt:
                         dt = dt_util.as_local(dt)
                         # Filter to relevant range (optimization)
                         # Relaxed filter: Include full day to avoid edge cases
                         if dt.date() == start_time.date():
                             merged_map[dt.hour] = item
                 except (ValueError, TypeError):
                     pass

        # 2. Overwrite with Live data (Higher priority)
        for item in live_data:
             dt_str = item.get("datetime")
             if dt_str:
                 try:
                     dt = dt_util.parse_datetime(dt_str)
                     if dt:
                         dt = dt_util.as_local(dt)
                         if dt.date() == start_time.date():
                             merged_map[dt.hour] = item
                 except (ValueError, TypeError):
                     pass

        # 3. Convert back to list and sort
        merged_list = list(merged_map.values())
        merged_list.sort(key=lambda x: x.get("datetime", ""))

        # Log gap filling if relevant
        live_hours = [item for item in live_data if dt_util.parse_datetime(item["datetime"]).date() == start_time.date()] if live_data else []
        gaps_filled = len(merged_list) - len(live_hours)
        if gaps_filled > 0:
            _LOGGER.debug(f"Filled {gaps_filled} forecast hours from reference data for {start_time.date()}")

        return merged_list

    def get_future_day_prediction(
        self,
        target_date: date,
        initial_inertia: list[float] | None = None,
        ignore_aux: bool = False
    ) -> tuple[float, float, dict] | None:
        """Get predicted energy for a specific future day using Smart Merge logic.

        Tiers:
        1. Hourly Forecast (Gold) - If >= 18 hours of data exists (or it's Today).
        2. Daily Forecast (Silver) - If daily entry exists.
        3. History (Bronze) - (Caller handles this fallback, returning None here)

        Args:
            target_date: The date to predict for.
            initial_inertia: Optional list of previous temperatures to seed the inertia calculation.
                             If provided, ensures thermodynamic continuity from the previous day.
            ignore_aux: If True, calculates prediction assuming normal heating mode (ignoring aux/fireplace).
                       Defaults to False (respect current state) to maintain backward compat, but
                       should be set to True for future dates.

        Returns:
            (predicted_kwh, predicted_solar_kwh, weather_stats_dict) or None
        """
        target_iso = target_date.isoformat()
        is_today = target_date == dt_util.now().date()
        HOURLY_FULL_THRESHOLD = 18

        # --- 1. Gather Hourly Data (Optimized) ---
        # Try optimized cache first
        if self._cached_hourly_by_date and target_iso in self._cached_hourly_by_date:
             # MUST copy list to prevent cache corruption by smart fill modification
             hourly_items = list(self._cached_hourly_by_date[target_iso])
        else:
            # Fallback (e.g. if cache not built yet or source changed dynamically)
            hourly_items = []
            source_data = self._cached_long_term_hourly or self._get_live_forecast_or_ref()

            if source_data:
                for f in source_data:
                    dt_str = f.get("datetime")
                    if not dt_str: continue
                    try:
                        f_dt = dt_util.parse_datetime(dt_str)
                        if f_dt:
                            f_dt_local = dt_util.as_local(f_dt)
                            if f_dt_local.date().isoformat() == target_iso:
                                hourly_items.append(f)
                    except ValueError:
                        continue

        hourly_count = len(hourly_items)

        # --- 2. Check Daily Availability ---
        daily_item = self._get_daily_forecast_item(target_date)

        # --- 3. Decision Logic ---

        # Case A: Good Hourly Coverage (>= 18 hours)
        # Use Hourly. Smart Fill is technically redundant but harmless here.
        if hourly_count >= HOURLY_FULL_THRESHOLD:
            return self._calculate_from_hourly_forecast(
                hourly_items, target_date, initial_inertia, smart_fill=True, ignore_aux=ignore_aux
            )

        # Case B: Today (Any amount of hourly data)
        # Always use available hourly data. NEVER Smart Fill today.
        if is_today and hourly_count > 0:
            return self._calculate_from_hourly_forecast(
                hourly_items, target_date, initial_inertia, smart_fill=False, ignore_aux=ignore_aux
            )

        # Case C: Future Partial Hourly + Daily Available
        # Prefer Daily Forecast over broken hourly data.
        if hourly_count > 0 and daily_item:
            _LOGGER.debug(f"Tier 2: Using Daily Forecast for {target_date} (Hourly count {hourly_count} < {HOURLY_FULL_THRESHOLD})")
            return self._calculate_from_daily_forecast(daily_item, target_date, ignore_aux=ignore_aux)

        # Case D: Future Partial Hourly + NO Daily
        # Last Resort: Use Hourly with Smart Fill.
        # Requirement: Must have at least SMART_FILL_MIN_HOURS (12) to avoid flat-lining a whole day.
        SMART_FILL_MIN_HOURS = 12
        if hourly_count > 0:
            if hourly_count >= SMART_FILL_MIN_HOURS:
                _LOGGER.debug(f"Tier 3: Last Resort (Smart Fill) for {target_date} (Hourly count {hourly_count} >= {SMART_FILL_MIN_HOURS})")
                return self._calculate_from_hourly_forecast(
                    hourly_items, target_date, initial_inertia, smart_fill=True, ignore_aux=ignore_aux
                )
            else:
                 _LOGGER.debug(f"Tier 3 Skipped: Insufficient data for Smart Fill on {target_date} (Hourly count {hourly_count} < {SMART_FILL_MIN_HOURS}). Falling back to History.")
                 return None

        # Case E: No Hourly -> Try Daily
        if daily_item:
            return self._calculate_from_daily_forecast(daily_item, target_date, ignore_aux=ignore_aux)

        return None

    def _get_daily_forecast_item(self, target_date: date) -> dict | None:
        """Find daily forecast item for a specific date."""
        target_iso = target_date.isoformat()

        _LOGGER.debug(f"Looking for daily forecast for {target_iso}. Cache has {len(self._cached_long_term_daily)} items.")

        if self._cached_long_term_daily:
            for f in self._cached_long_term_daily:
                dt_str = f.get("datetime")
                if not dt_str: continue
                try:
                    f_dt = dt_util.parse_datetime(dt_str)
                    if f_dt:
                        f_dt_local = dt_util.as_local(f_dt)
                        if f_dt_local.date().isoformat() == target_iso:
                            _LOGGER.debug(f"Found daily forecast for {target_iso}")
                            return f
                except ValueError:
                    continue

        _LOGGER.debug(f"No daily forecast found for {target_iso}")
        return None

    def _calculate_from_hourly_forecast(
        self,
        hourly_items: list,
        target_date: date,
        initial_inertia: list[float] | None = None,
        smart_fill: bool = False,
        ignore_aux: bool = False
    ) -> tuple[float, float, dict]:
        """Calculate prediction from detailed hourly items with Thermal Inertia.

        Uses initial_inertia if provided to ensure thermodynamic continuity.
        Otherwise seeds from the first temperature of the day.
        """
        # --- Smart Fill Logic (Fix for partial days) ---
        # If we have < 24 hours, fill the missing hours with the last available hour's data
        # to ensure we don't return a partial sum (e.g. 13 kWh instead of 38 kWh).
        # controlled by smart_fill flag (disabled for Today, used as last resort for Future)
        processed_items = list(hourly_items)
        if smart_fill and 0 < len(processed_items) < 24:
            # 1. Map existing hours
            existing_hours = set()
            last_item = processed_items[-1] # Default to last item if we can't parse times

            for item in processed_items:
                dt_str = item.get("datetime")
                if dt_str:
                    try:
                        dt = dt_util.parse_datetime(dt_str)
                        if dt:
                            dt_local = dt_util.as_local(dt)
                            if dt_local.date() == target_date:
                                existing_hours.add(dt_local.hour)
                                last_item = item # Keep tracking the latest valid item
                    except (ValueError, TypeError):
                        pass

            # 2. Fill missing hours
            if last_item:
                missing_hours = [h for h in range(24) if h not in existing_hours]
                if missing_hours:
                    _LOGGER.debug(f"Smart Fill: Filling {len(missing_hours)} missing hours for {target_date} using last forecast item.")

                    # Create a timezone-aware base date for target_date
                    # Note: We rely on the Coordinator/HA to have set the timezone
                    tz = dt_util.get_time_zone(self.coordinator.hass.config.time_zone)
                    base_dt = datetime.combine(target_date, datetime.min.time()).replace(tzinfo=tz)

                    for h in missing_hours:
                        # Clone the last item
                        new_item = copy.deepcopy(last_item)
                        # Update datetime to the missing hour
                        # We must produce an ISO string that parse_datetime can read
                        synthetic_dt = base_dt.replace(hour=h)
                        new_item["datetime"] = synthetic_dt.isoformat()
                        processed_items.append(new_item)

                    # 3. Sort by time to ensure inertia calculation is correct
                    # Since we use ISO strings, string sort works for same-day items
                    processed_items.sort(key=lambda x: x.get("datetime", ""))

        total_kwh = 0.0
        total_solar = 0.0

        temps = []
        winds = []

        weather_wind_unit = self.coordinator._get_weather_wind_unit()

        # Improved Cloud Fallback: Use Coordinator (live) if available
        current_cloud = self.coordinator._get_cloud_coverage()

        # Inertia Simulation Setup
        history_needed = len(DEFAULT_INERTIA_WEIGHTS) - 1
        if initial_inertia:
            local_inertia_history = list(initial_inertia)
        else:
            local_inertia_history = []
            if processed_items:
                first_temp = float(processed_items[0].get("temperature", 0.0))
                # Seed with N prior hours of same temp to stabilize start
                local_inertia_history = [first_temp] * history_needed

        for f in processed_items:
            predicted, solar_kwh, inertia_val, raw_temp, w_speed, w_speed_ms, _ = self._process_forecast_item(
                f, local_inertia_history, weather_wind_unit, current_cloud, ignore_aux=ignore_aux
            )

            total_kwh += predicted
            total_solar += solar_kwh
            temps.append(raw_temp)
            winds.append(w_speed)

        avg_temp = sum(temps) / len(temps) if temps else 0.0
        avg_wind_raw = sum(winds) / len(winds) if winds else 0.0
        avg_wind_ms = convert_speed_to_ms(avg_wind_raw, weather_wind_unit)

        if history_needed > 0 and local_inertia_history:
            final_inertia = local_inertia_history[-history_needed:]
        else:
            final_inertia = []

        weather_stats = {
            "temp": round(avg_temp, 1),
            "wind": round(avg_wind_ms, 1),
            "source": "hourly_forecast",
            "final_inertia": final_inertia
        }

        return total_kwh, total_solar, weather_stats

    def _process_forecast_item(
        self,
        item: dict,
        inertia_history: list[float],
        wind_unit: str | None,
        default_cloud: float,
        ignore_aux: bool = False,
        force_aux: bool = False,
        screen_override: float | None = None,
        force_no_wind: bool = False
    ) -> tuple[float, float, float, float, float, float, dict]:
        """Process a single forecast item for energy prediction.

        Args:
            item: Forecast item dictionary
            inertia_history: Mutable list of previous temperatures for inertia
            wind_unit: Wind speed unit from weather entity
            default_cloud: Fallback cloud coverage
            ignore_aux: If True, forces normal bucket instead of aux.
            force_aux: If True, forces aux bucket (overrides ignore_aux if both set? No, prefer explicit).
            screen_override: Optional correction percent (0-100). If None, uses SolarOptimizer.
            force_no_wind: If True, forces effective wind to 0.0.

        Returns:
            (predicted_kwh, solar_kwh, inertia_val, raw_temp, wind_speed_raw, wind_speed_ms, unit_breakdown)
        """
        raw_temp = float(item.get("temperature", 0.0))

        # Inertia Calculation
        inertia_history.append(raw_temp)

        # Use coordinator's weighted logic for consistency
        inertia_val = self.coordinator._calculate_weighted_inertia(inertia_history)

        w_speed = float(item.get("wind_speed", 0.0))
        w_speed_ms = convert_speed_to_ms(w_speed, wind_unit)

        w_gust = item.get("wind_gust_speed")
        if w_gust is not None:
            try:
                w_gust = float(w_gust)
                w_gust_ms = convert_speed_to_ms(w_gust, wind_unit)
            except (ValueError, TypeError):
                w_gust_ms = None
        else:
            w_gust_ms = None

        if force_no_wind:
            effective_wind = 0.0
        else:
            effective_wind = self.coordinator._calculate_effective_wind(w_speed_ms, w_gust_ms)
        wind_bucket = self.coordinator._get_wind_bucket(effective_wind)

        # Cloud
        condition = item.get("condition")
        forecast_cloud = default_cloud
        if condition and condition in CLOUD_COVERAGE_MAP:
            forecast_cloud = float(CLOUD_COVERAGE_MAP[condition])
        elif item.get("cloud_coverage") is not None:
            forecast_cloud = float(item.get("cloud_coverage"))

        # Use Inertia Value for Prediction Key
        temp_key = str(int(round(inertia_val)))

        # Determine Solar Factor
        potential_factor = 0.0
        effective_factor = 0.0

        if self.coordinator.solar_enabled:
            elev = item.get("elevation")
            azimuth = item.get("azimuth")

            if elev is None or azimuth is None:
                dt_str = item.get("datetime")
                if dt_str:
                    f_dt = dt_util.parse_datetime(dt_str)
                    if f_dt:
                        elev, azimuth = self.coordinator.solar.get_approx_sun_pos(f_dt)

            if elev is not None:
                potential_factor = self.coordinator.solar.calculate_solar_factor(elev, azimuth, forecast_cloud)

                # Apply Screen Optimization
                correction_percent = 100.0 # Default if optimization fails

                if screen_override is not None:
                    correction_percent = screen_override
                else:
                    # ML Prediction
                    rec_state = self.coordinator.solar_optimizer.get_recommendation_state(inertia_val, potential_factor)
                    current_setting = self.coordinator.solar_correction_percent
                    correction_percent = self.coordinator.solar_optimizer.predict_correction_percent(rec_state, elev, azimuth, current_setting)

                effective_factor = self.coordinator.solar.calculate_effective_solar_factor(potential_factor, correction_percent)

        # Calculate via unified StatisticsManager logic
        # Determine Aux State
        is_aux = False
        if force_aux:
            is_aux = True
        elif not ignore_aux:
            is_aux = self.coordinator.auxiliary_heating_active

        # We pass inertia_val as "temp" to the model calculation
        # But wait, calculate_total_power recalculates temp_key from float temp.
        # It does `temp_key = str(int(round(temp)))`.
        # So passing inertia_val works perfectly.

        res = self.coordinator.statistics.calculate_total_power(
            temp=inertia_val,
            effective_wind=effective_wind,
            solar_impact=0.0, # Unused
            is_aux_active=is_aux,
            unit_modes=None, # Use current coordinator state modes (defaults to internal)
            override_solar_factor=effective_factor
        )

        predicted = res["total_kwh"]
        solar_kwh = res["breakdown"]["solar_reduction_kwh"]
        unit_breakdown = res["unit_breakdown"]

        return predicted, solar_kwh, inertia_val, raw_temp, w_speed, w_speed_ms, unit_breakdown

    def _calculate_from_daily_forecast(
        self,
        daily_item: dict,
        target_date: date,
        ignore_aux: bool = False
    ) -> tuple[float, float, dict]:
        """Calculate prediction from single daily item."""
        temp_high = float(daily_item.get("temperature", 0.0))
        temp_low = float(daily_item.get("templow", temp_high))
        avg_temp = (temp_high + temp_low) / 2.0

        wind_speed = float(daily_item.get("wind_speed", 0.0))
        wind_gust = daily_item.get("wind_gust_speed")

        weather_wind_unit = self.coordinator._get_weather_wind_unit()

        wind_speed_ms = convert_speed_to_ms(wind_speed, weather_wind_unit)

        # Fixed: Wind Gust logic cleanup (removed confusing None assignment)
        wind_gust_ms = None
        if wind_gust is not None:
            try:
                val = float(wind_gust)
                wind_gust_ms = convert_speed_to_ms(val, weather_wind_unit)
            except (ValueError, TypeError):
                wind_gust_ms = None

        effective_wind = self.coordinator._calculate_effective_wind(wind_speed_ms, wind_gust_ms)
        wind_bucket = self.coordinator._get_wind_bucket(effective_wind)
        temp_key = str(int(round(avg_temp)))

        # Determine Daily Solar Factor
        s_factor = 0.0
        if self.coordinator.solar_enabled:
             condition = daily_item.get("condition")
             cloud_cov = DEFAULT_CLOUD_COVERAGE
             if condition and condition in CLOUD_COVERAGE_MAP:
                 cloud_cov = float(CLOUD_COVERAGE_MAP[condition])
             s_factor = self.coordinator.solar.estimate_daily_avg_solar_factor(target_date, cloud_cov)

        # Calculate via unified StatisticsManager logic
        is_aux = False
        if not ignore_aux:
            is_aux = self.coordinator.auxiliary_heating_active

        # calculate_total_power returns Hourly Power (or energy per hour)
        # We need to multiply by 24 for daily

        res = self.coordinator.statistics.calculate_total_power(
            temp=avg_temp,
            effective_wind=effective_wind,
            solar_impact=0.0, # Unused
            is_aux_active=is_aux,
            unit_modes=None,
            override_solar_factor=s_factor
        )

        predicted_24h = res["total_kwh"] * 24.0
        solar_impact_24h = res["breakdown"]["solar_reduction_kwh"] * 24.0

        weather_stats = {
            "temp": round(avg_temp, 1),
            "wind": round(effective_wind, 1), # Use effective wind for daily stats to be consistent
            "source": "daily_forecast"
        }

        return predicted_24h, solar_impact_24h, weather_stats

    def _sum_forecast_energy_internal(
        self,
        start_time: datetime,
        end_time: datetime,
        inertia_history: list[float],
        include_start: bool = False,
        source_data: list | None = None,
        ignore_aux: bool = False,
        force_aux: bool = False,
        screen_override: float | None = None,
        force_no_wind: bool = False
    ) -> dict:
        """Internal helper to sum predicted energy and return detailed plan."""
        # Determine source
        forecast_source = source_data if source_data is not None else self._get_live_forecast_or_ref()

        if not forecast_source:
            return {
                "total_energy": 0.0,
                "count": 0,
                "total_solar": 0.0,
                "unit_totals": {},
                "hourly_plan": []
            }

        total_energy = 0.0
        total_solar = 0.0
        count = 0
        unit_totals = {}
        hourly_plan = []

        local_history = list(inertia_history)

        weather_wind_unit = self.coordinator._get_weather_wind_unit()
        current_cloud = self.coordinator._get_cloud_coverage()

        for f in forecast_source:
            dt_str = f.get("datetime")
            if not dt_str: continue
            f_dt = dt_util.parse_datetime(dt_str)
            if not f_dt: continue
            f_dt = dt_util.as_local(f_dt)

            is_start_ok = (f_dt >= start_time) if include_start else (f_dt > start_time)

            if is_start_ok and f_dt < end_time:
                res = self._process_forecast_item(
                    f, local_history, weather_wind_unit, current_cloud, ignore_aux=ignore_aux, force_aux=force_aux, screen_override=screen_override, force_no_wind=force_no_wind
                )
                predicted = res[0]
                solar_kwh = res[1]
                inertia_val = res[2]
                unit_breakdown = res[6]

                total_energy += predicted
                total_solar += solar_kwh
                count += 1

                # Determine implied aux state used for calculation
                is_aux_used = False
                if force_aux:
                    is_aux_used = True
                elif not ignore_aux:
                    is_aux_used = self.coordinator.auxiliary_heating_active

                # Capture hourly details
                hourly_plan.append({
                    "hour": f_dt.hour,
                    "kwh": round(predicted, 2),
                    "inertia_temp": round(inertia_val, 1),
                    "aux_expected": is_aux_used
                })

                for entity_id, stats in unit_breakdown.items():
                    unit_totals[entity_id] = unit_totals.get(entity_id, 0.0) + stats["net_kwh"]

        return {
            "total_energy": total_energy,
            "count": count,
            "total_solar": total_solar,
            "unit_totals": unit_totals,
            "hourly_plan": hourly_plan
        }

    def sum_forecast_energy(
        self,
        start_time: datetime,
        end_time: datetime,
        inertia_history: list[float],
        include_start: bool = False,
        source_data: list | None = None,
        ignore_aux: bool = False,
        force_aux: bool = False,
        screen_override: float | None = None,
        force_no_wind: bool = False
    ) -> tuple[float, int, float, dict[str, float]]:
        """Sum predicted energy from forecast between start and end times.

        Args:
            source_data: Optional override for source list (defaults to LIVE forecast)
            ignore_aux: If True, forces normal bucket instead of aux.
            force_aux: If True, forces aux bucket.
            screen_override: Optional solar screen percentage (0-100) or None for auto/ML.

        Returns:
            (total_energy, count, total_solar, unit_totals)
        """
        res = self._sum_forecast_energy_internal(
            start_time, end_time, inertia_history, include_start,
            source_data, ignore_aux, force_aux, screen_override, force_no_wind
        )
        return (
            res["total_energy"],
            res["count"],
            res["total_solar"],
            res["unit_totals"]
        )

    def _calculate_uncertainty_from_errors(self, abs_errors: list[float]) -> dict:
        """Helper to calculate percentile stats from a list of absolute errors."""
        count = len(abs_errors)
        if count == 0:
            return {
                "p50_abs_error": DEFAULT_UNCERTAINTY_P50,
                "p95_abs_error": DEFAULT_UNCERTAINTY_P95,
                "samples": 0
            }

        sorted_errors = sorted(abs_errors)

        def percentile(data, p):
            k = (count - 1) * p
            f = math.floor(k)
            c = math.ceil(k)
            if f == c: return data[int(k)]
            d0 = data[int(f)] * (c - k)
            d1 = data[int(c)] * (k - f)
            return d0 + d1

        p50 = percentile(sorted_errors, 0.50)
        p95 = percentile(sorted_errors, 0.95)

        return {
            "p50_abs_error": round(p50, 2),
            "p95_abs_error": round(p95, 2),
            "samples": count
        }

    def calculate_uncertainty_stats(self):
        """Calculate global uncertainty stats from history."""
        if self._cached_forecast_uncertainty is not None:
            return self._cached_forecast_uncertainty

        current_primary = self.coordinator.weather_entity
        all_errors = []

        for h in self._forecast_history:
             # Strict Filtering: Only use history where the Primary Entity matches the current one.
             # Legacy handling: If 'primary_entity' key is missing (None), it is a mismatch.
             hist_primary = h.get("primary_entity")
             if hist_primary == current_primary:
                 all_errors.append(h.get("abs_error_kwh", 0.0))

        stats = self._calculate_uncertainty_from_errors(all_errors)

        self._cached_forecast_uncertainty = stats
        return stats

    def calculate_per_source_uncertainty_stats(self) -> dict:
        """Calculate uncertainty stats split by forecast source using hourly attribution."""
        result = {}

        current_primary = self.coordinator.weather_entity
        current_secondary = self.coordinator.entry.data.get(CONF_SECONDARY_WEATHER_ENTITY)

        for src in ["primary", "secondary"]:
            # Basic Percentile Stats (Global) - Hourly Precision Basis
            abs_errors = []

            # Determine which entity ID to match against for this source
            target_entity = current_primary if src == "primary" else current_secondary

            for h in self._forecast_history:
                # Provenance Filtering
                hist_entity = h.get(f"{src}_entity")

                # Strict Match: If history doesn't match current config, skip it.
                if hist_entity != target_entity:
                    continue

                breakdown = h.get("source_breakdown", {})
                if src in breakdown:
                    abs_errors.append(breakdown[src]["abs_error"])
                elif h.get("source") == src:
                    abs_errors.append(h.get("abs_error_kwh", 0.0))

            # Base stats (P50/P95) go into "hourly" as they use hourly deviations
            hourly_stats = self._calculate_uncertainty_from_errors(abs_errors)

            # Period Stats (7d, 30d) - Split into Hourly and Daily
            period_stats_7d = self._calculate_period_stats(src, 7, target_entity)
            period_stats_30d = self._calculate_period_stats(src, 30, target_entity)

            # Merge stats
            hourly_stats.update(period_stats_7d["hourly"])
            hourly_stats.update(period_stats_30d["hourly"])

            daily_stats = {}
            daily_stats.update(period_stats_7d["daily"])
            daily_stats.update(period_stats_30d["daily"])

            result[src] = {
                "hourly": hourly_stats,
                "daily": daily_stats
            }

        return result

    def _calculate_period_stats(self, source: str, days: int, target_entity: str | None) -> dict:
        """Calculate MAE and MAPE for a specific source over a given period."""
        now_date = dt_util.now().date()
        cutoff_date = (now_date - timedelta(days=days)).isoformat()

        total_hourly_abs_error = 0.0
        total_daily_net_error_abs = 0.0
        total_weather_error_abs = 0.0
        total_weather_error_signed = 0.0
        total_actual = 0.0
        total_hours = 0
        samples = 0
        weather_samples = 0

        for h in reversed(self._forecast_history):
            if h["date"] < cutoff_date:
                break

            # Provenance Filtering
            hist_entity = h.get(f"{source}_entity")
            if hist_entity != target_entity:
                continue

            breakdown = h.get("source_breakdown", {})
            if source in breakdown:
                s_data = breakdown[source]

                # Hourly Basis: Sum of absolute hourly errors
                total_hourly_abs_error += s_data["abs_error"]

                # Daily Basis: Absolute value of the day's net error (Sum of signed hourly errors)
                total_daily_net_error_abs += abs(s_data["error"])

                # Weather Error (Legacy Handling: Only if key exists)
                if "weather_error" in s_data:
                    # weather_error is the signed sum of hourly weather errors
                    # For Daily agg: we take Abs of the Daily Sum (not sum of hourly abs) to avoid time-offset penalty
                    total_weather_error_abs += abs(s_data["weather_error"])
                    total_weather_error_signed += s_data["weather_error"]
                    weather_samples += 1

                total_actual += s_data["actual"]
                total_hours += s_data["hours"]
                samples += 1
            elif h.get("source") == source and "source_breakdown" not in h:
                # Legacy fallback
                hourly_abs = h.get("abs_error_kwh", 0.0)
                total_hourly_abs_error += hourly_abs

                # For legacy, we assume daily error is same as net error if available, else fallback to hourly abs (pessimistic)
                # But we have error_kwh in legacy history
                net_err = h.get("error_kwh", 0.0)
                total_daily_net_error_abs += abs(net_err)

                total_actual += h.get("actual_kwh", 0.0)
                total_hours += 24
                samples += 1

        hourly_mae = total_hourly_abs_error / samples if samples > 0 else 0.0
        hourly_mape = (total_hourly_abs_error / total_actual * 100) if total_actual > 0 else 0.0

        daily_mae = total_daily_net_error_abs / samples if samples > 0 else 0.0
        daily_mape = (total_daily_net_error_abs / total_actual * 100) if total_actual > 0 else 0.0

        weather_mae = total_weather_error_abs / weather_samples if weather_samples > 0 else 0.0
        weather_bias = total_weather_error_signed / weather_samples if weather_samples > 0 else 0.0

        return {
            "hourly": {
                f"mae_{days}d": round(hourly_mae, 2),
                f"mape_{days}d": round(hourly_mape, 1)
            },
            "daily": {
                f"mae_{days}d": round(daily_mae, 2),
                f"mape_{days}d": round(daily_mape, 1),
                f"weather_mae_{days}d": round(weather_mae, 2),
                f"weather_bias_{days}d": round(weather_bias, 2)
            }
        }

    def calculate_plan_revision_impact(self):
        """Calculate impact of actual weather and user actions deviating from the initial plan."""
        now = dt_util.now()
        today_iso = now.date().isoformat()
        today_logs = [log for log in self.coordinator._hourly_log if log["timestamp"].startswith(today_iso)]

        # Ensure reference map is available
        if self._cached_reference_map is None:
            self._build_reference_map(self._reference_forecast)

        forecast_map = self._cached_reference_map.get(today_iso, {})

        total_impact = 0.0
        temp_diffs = []
        hours_analyzed = 0

        # 1. Process Completed Hours
        for log in today_logs:
            hour = log["hour"]
            f_data = forecast_map.get(hour)
            if f_data is None:
                continue

            # Scenario A: What the plan SHOULD have been (Forecast Weather, Normal Mode)
            res_planned = self.coordinator.statistics.calculate_total_power(
                temp=f_data["temp"],
                effective_wind=f_data.get("wind", 0.0),
                solar_impact=0.0,
                is_aux_active=False,  # Crucial: Plan always assumes Normal mode
                override_solar_factor=f_data.get("solar_factor", 0.0)
            )
            planned_kwh = res_planned["total_kwh"]

            # Scenario B: What the model says SHOULD have happened (Actual Weather, Actual Mode)
            res_reality = self.coordinator.statistics.calculate_total_power(
                temp=log["temp"],
                effective_wind=log.get("effective_wind", 0.0),
                solar_impact=0.0,
                is_aux_active=log.get("auxiliary_active", False), # Use actual mode from the log
                override_solar_factor=log.get("solar_factor", 0.0)
            )
            reality_adjusted_kwh = res_reality["total_kwh"]

            hour_impact = reality_adjusted_kwh - planned_kwh
            total_impact += hour_impact

            temp_diffs.append(log["temp"] - f_data["temp"])
            hours_analyzed += 1

        # 2. Process Current, Partial Hour
        current_hour = now.hour
        minutes_passed = now.minute
        if minutes_passed > 0:
            current_temp = self.coordinator._get_float_state(self.coordinator.outdoor_temp_sensor)
            f_data_curr = forecast_map.get(current_hour)

            if current_temp is not None and f_data_curr is not None:
                # Scenario A Rate (Forecast Weather, Normal Mode)
                res_planned_rate = self.coordinator.statistics.calculate_total_power(
                    temp=f_data_curr["temp"],
                    effective_wind=f_data_curr.get("wind", 0.0),
                    solar_impact=0.0,
                    is_aux_active=False,
                    override_solar_factor=f_data_curr.get("solar_factor", 0.0)
                )
                planned_rate = res_planned_rate["total_kwh"]

                # Scenario B Rate (Actual Weather, Actual Mode)
                res_reality_rate = self.coordinator.statistics.calculate_total_power(
                    temp=current_temp,
                    effective_wind=self.coordinator.data.get("effective_wind", 0.0),
                    solar_impact=0.0,
                    is_aux_active=self.coordinator.auxiliary_heating_active,
                    override_solar_factor=self.coordinator.data.get("solar_factor", 0.0)
                )
                reality_adjusted_rate = res_reality_rate["total_kwh"]

                # Pro-rate the impact for the partial hour
                current_hour_impact = (reality_adjusted_rate - planned_rate) * (minutes_passed / 60.0)
                total_impact += current_hour_impact

                temp_diffs.append(current_temp - f_data_curr["temp"])
                hours_analyzed += (minutes_passed / 60.0) # Add fraction for avg calculation

        avg_temp_diff = sum(temp_diffs) / len(temp_diffs) if temp_diffs else 0.0

        has_aux_usage = False
        if any(log.get("auxiliary_active", False) for log in today_logs):
            has_aux_usage = True
        if minutes_passed > 0 and self.coordinator.auxiliary_heating_active:
            has_aux_usage = True

        weather_driver = None
        if avg_temp_diff > 1.5:
            weather_driver = "warmer"
        elif avg_temp_diff < -1.5:
            weather_driver = "colder"

        return {
            "temp_difference_avg": round(avg_temp_diff, 1),
            "estimated_impact_kwh": round(total_impact, 2),
            "hours_analyzed": round(hours_analyzed, 1),
            "explanation": f"Plan revised by {total_impact:+.1f} kWh due to weather/mode changes",
            "has_aux_usage": has_aux_usage,
            "weather_driver": weather_driver,
        }

    def calculate_load_trend(self) -> str:
        """Calculate the energy load trend (Future vs Past).

        Compares the average expected load of the near past (last 3 hours)
        with the near future (next 3 hours) to determine if load is increasing or decreasing.

        Returns:
            "Increasing (Fast)", "Increasing", "Stable", "Easing", "Easing (Fast)", or "Unknown"
        """
        now = dt_util.now()
        current_hour = now.hour

        # 1. Past Data (h-2, h-1) + Current (h)
        past_sum = 0.0
        past_count = 0

        # Current Hour Rate (Best Estimate)
        current_rate = self.coordinator.data.get("current_model_rate")
        if current_rate is not None:
            past_sum += current_rate
            past_count += 1

        # Past 2 hours from log
        # Look for hours: current_hour - 1, current_hour - 2
        target_hours = { (current_hour - 1) % 24, (current_hour - 2) % 24 }

        found_hours = set()
        if self.coordinator._hourly_log:
            # Iterate backwards to find most recent matching hours
            for log in reversed(self.coordinator._hourly_log):
                h = log.get("hour")
                if h in target_hours and h not in found_hours:
                    # Use expected_kwh (Model) not actual (User behavior) for trend
                    past_sum += log.get("expected_kwh", 0.0)
                    past_count += 1
                    found_hours.add(h)
                if len(found_hours) >= 2:
                    break

        if past_count == 0:
            return "Unknown"

        avg_past = past_sum / past_count

        # 2. Future Data (h+1, h+2, h+3)
        # Use _sum_forecast_energy_internal to generate hourly plan
        # We start from now + 1 hour (aligned to start of hour)
        start_future = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        end_future = start_future + timedelta(hours=3) # Covers +1, +2, +3

        # We need recent inertia to seed the future calculation correctly
        inertia_history = self.coordinator._get_inertia_list(now)

        res_future = self._sum_forecast_energy_internal(
            start_time=start_future,
            end_time=end_future,
            inertia_history=list(inertia_history),
            include_start=True,
            ignore_aux=False # Respect current aux state for consistency
        )

        future_plan = res_future.get("hourly_plan", [])

        # If forecast is short (e.g. end of day approaching and no tomorrow data), we might get fewer hours
        # We use what we have.
        future_sum = sum(item["kwh"] for item in future_plan)
        future_count = len(future_plan)

        if future_count == 0:
            return "Unknown"

        avg_future = future_sum / future_count

        # 3. Compare
        if avg_past < 0.1: # Avoid division by zero/noise
             if avg_future > 0.5: return "Increasing (Fast)"
             if avg_future > 0.1: return "Increasing"
             return "Stable"

        delta = avg_future - avg_past
        pct_change = (delta / avg_past) * 100

        if pct_change > 20.0:
            return "Increasing (Fast)"
        elif pct_change > 5.0:
            return "Increasing"
        elif pct_change < -20.0:
            return "Easing (Fast)"
        elif pct_change < -5.0:
            return "Easing"
        else:
            return "Stable"

    def backfill_history_from_logs(self):
        """Generate synthetic forecast history from hourly logs."""
        _LOGGER.info("Backfilling forecast history from hourly logs (Option B)...")
        daily_groups = {}
        for entry in self.coordinator._hourly_log:
            # Skip legacy entries without provenance tracking
            if "primary_entity" not in entry:
                continue

            ts = entry["timestamp"]
            date_key = ts.split("T")[0]
            if date_key not in daily_groups:
                daily_groups[date_key] = {
                    "actual": 0.0,
                    "expected": 0.0,
                    "sources": {},
                    # Capture provenance from the first valid entry of the day
                    "primary_entity": entry.get("primary_entity"),
                    "secondary_entity": entry.get("secondary_entity"),
                    "crossover_day": entry.get("crossover_day"),
                }
            daily_groups[date_key]["actual"] += entry.get("actual_kwh", 0.0)
            daily_groups[date_key]["expected"] += entry.get("expected_kwh", 0.0)

            src = entry.get("forecast_source")
            if src:
                daily_groups[date_key]["sources"][src] = daily_groups[date_key]["sources"].get(src, 0) + 1

        for date_key, vals in daily_groups.items():
            if date_key == dt_util.now().date().isoformat():
                continue

            actual = round(vals["actual"], 2)
            expected = round(vals["expected"], 2)
            error = round(actual - expected, 2)

            # Determine dominant source from logs
            sources = vals.get("sources", {})
            if sources:
                dominant = max(sources, key=sources.get)
            else:
                # Fallback for legacy logs: Assume primary if crossover is active
                crossover = self.coordinator.entry.data.get(CONF_FORECAST_CROSSOVER_DAY, DEFAULT_FORECAST_CROSSOVER_DAY)
                dominant = "primary" if crossover > 0 else "unknown"

            self._forecast_history.append({
                "date": date_key,
                "forecast_kwh": expected,
                "actual_kwh": actual,
                "error_kwh": error,
                "abs_error_kwh": abs(error),
                "source": dominant,
                "primary_entity": vals.get("primary_entity"),
                "secondary_entity": vals.get("secondary_entity"),
                "crossover_day": vals.get("crossover_day"),
            })

        self._cached_forecast_uncertainty = None
        _LOGGER.info(f"Backfilled {len(self._forecast_history)} days of history.")

    def reset_forecast_history(self):
        """Reset the forecast history."""
        self._forecast_history = []
        self._cached_forecast_uncertainty = None
        # We also reset the midnight snapshot for consistency
        self._midnight_forecast_snapshot = {}
        _LOGGER.info("Forecast history reset.")

    def log_accuracy(self, date_key: str, actual_kwh: float, aux_impact_kwh: float = 0.0, modeled_net_kwh: float | None = None, guest_impact_kwh: float = 0.0):
        """Log forecast accuracy for the day with per-source breakdown."""
        if self._midnight_forecast_snapshot and self._midnight_forecast_snapshot.get("date") == date_key:
            forecast = self._midnight_forecast_snapshot.get("kwh", 0.0)

            # Apple-to-Apple Comparison:
            # Forecast is Thermodynamic Demand (Gross, Aux-Unaware) of the BASE House.
            # Actual is Meter Reading (Net, Aux-Aware) + Guest Mode (Unmodeled).
            # We must convert Actual to Thermodynamic Base Actual (Gross Base) by:
            # 1. Adding back Aux Impact (Aux reduces Net).
            # 2. Subtracting Guest Impact (Guest increases Net but is not in Forecast).
            gross_actual_kwh = actual_kwh + aux_impact_kwh - guest_impact_kwh
            modeled_gross_kwh = (modeled_net_kwh + aux_impact_kwh) if modeled_net_kwh is not None else None

            error = gross_actual_kwh - forecast

            # Calculate per-source breakdown from hourly logs (Shadow Forecasting Support)
            source_breakdown = {}

            # Seed primary and secondary to ensure they are always tracked if logs exist
            for src in ['primary', 'secondary']:
                source_breakdown[src] = {
                    "hours": 0,
                    "forecast": 0.0,
                    "actual": 0.0,
                    "error": 0.0,
                    "abs_error": 0.0,
                    "weather_error": 0.0,
                    "abs_weather_error": 0.0
                }

            # Time-of-Day Analysis Variables
            day_error_sum = 0.0
            day_count = 0
            night_error_sum = 0.0
            night_count = 0

            for entry in self.coordinator._hourly_log:
                if entry["timestamp"].startswith(date_key):
                    # Data Preparation
                    h_aux = entry.get("aux_impact_kwh", 0.0)
                    h_guest = entry.get("guest_impact_kwh", 0.0)
                    h_actual_net = entry.get("actual_kwh", 0.0)
                    # Thermodynamic Base Actual = Net + Aux - Guest
                    h_actual_gross = h_actual_net + h_aux - h_guest
                    h_modeled_gross = entry.get("expected_kwh", 0.0) + h_aux # Modeled Gross (Truth)

                    # Error Calculation: Hybrid Approach
                    # New Logs (Gross Forecast Available): Compare Gross Actual vs Gross Forecast.
                    # Legacy Logs (Net Forecast Only): Compare Net Actual vs Net Forecast.
                    # Rationale: Using Gross Actual vs Net Forecast (on old logs) would wrongly interpret
                    # the entire Aux reduction as a forecast error. Net vs Net is the best proxy.

                    h_forecast_gross = entry.get("forecasted_kwh_gross")

                    if h_forecast_gross is not None:
                        h_actual_for_error = h_actual_gross
                        h_forecast_for_error = h_forecast_gross
                    else:
                        # Legacy/Fallback: Compare Base Net vs Forecast Net
                        # Actual Net (Meter) includes Guest, so subtract it to get Base Net
                        h_actual_for_error = h_actual_net - h_guest
                        h_forecast_for_error = entry.get("forecasted_kwh", 0.0) # Net

                    h_error = h_actual_for_error - h_forecast_for_error
                    h_abs_error = abs(h_error)
                    h_hour = entry["hour"]

                    # Breakdown by Time of Day (Day: 06:00 - 22:00)
                    if 6 <= h_hour < 22:
                        day_error_sum += h_abs_error
                        day_count += 1
                    else:
                        night_error_sum += h_abs_error
                        night_count += 1

                    # 1. Track Primary Shadow
                    h_f_p_gross = entry.get("forecasted_kwh_gross_primary")
                    h_f_p_net = entry.get("forecasted_kwh_primary")

                    if h_f_p_net is not None: # Use Net as presence check for legacy
                        if h_f_p_gross is not None:
                            h_f_p = h_f_p_gross
                            h_a_p = h_actual_gross
                        else:
                            h_f_p = h_f_p_net
                            h_a_p = h_actual_net

                        source_breakdown['primary']["hours"] += 1
                        source_breakdown['primary']["forecast"] += h_f_p
                        # Use Gross Actual for accumulation to fix MAPE baseline,
                        # even if error was calculated Net vs Net.
                        source_breakdown['primary']["actual"] += h_actual_gross
                        source_breakdown['primary']["error"] += (h_a_p - h_f_p)
                        source_breakdown['primary']["abs_error"] += abs(h_a_p - h_f_p)

                        if h_f_p_gross is not None:
                            h_w_err = h_modeled_gross - h_f_p_gross
                            source_breakdown['primary']["weather_error"] += h_w_err
                            source_breakdown['primary']["abs_weather_error"] += abs(h_w_err)

                    # 2. Track Secondary Shadow
                    h_f_s_gross = entry.get("forecasted_kwh_gross_secondary")
                    h_f_s_net = entry.get("forecasted_kwh_secondary")

                    if h_f_s_net is not None:
                        if h_f_s_gross is not None:
                            h_f_s = h_f_s_gross
                            h_a_s = h_actual_gross
                        else:
                            h_f_s = h_f_s_net
                            h_a_s = h_actual_net

                        source_breakdown['secondary']["hours"] += 1
                        source_breakdown['secondary']["forecast"] += h_f_s
                        source_breakdown['secondary']["actual"] += h_actual_gross
                        source_breakdown['secondary']["error"] += (h_a_s - h_f_s)
                        source_breakdown['secondary']["abs_error"] += abs(h_a_s - h_f_s)

                        if h_f_s_gross is not None:
                            h_w_err = h_modeled_gross - h_f_s_gross
                            source_breakdown['secondary']["weather_error"] += h_w_err
                            source_breakdown['secondary']["abs_weather_error"] += abs(h_w_err)

            # Cleanup empty sources and round values
            final_breakdown = {}
            for src, data in source_breakdown.items():
                if data["hours"] > 0:
                    data["forecast"] = round(data["forecast"], 2)
                    data["actual"] = round(data["actual"], 2)
                    data["error"] = round(data["error"], 2)
                    data["abs_error"] = round(data["abs_error"], 2)
                    data["weather_error"] = round(data["weather_error"], 2)
                    data["abs_weather_error"] = round(data["abs_weather_error"], 2)
                    final_breakdown[src] = data

            # Determine dominant source for legacy support (based on blended usage)
            dominant_source = self._midnight_forecast_snapshot.get("source", "unknown")

            day_mae = round(day_error_sum / day_count, 2) if day_count > 0 else None
            night_mae = round(night_error_sum / night_count, 2) if night_count > 0 else None

            entry = {
                "date": date_key,
                "forecast_kwh": round(forecast, 2),
                "actual_kwh": round(gross_actual_kwh, 2), # Storing Base Gross to align with Forecast
                "net_actual_kwh": round(actual_kwh, 2),   # Preserving Net for reference
                "aux_impact_kwh": round(aux_impact_kwh, 2),
                "guest_impact_kwh": round(guest_impact_kwh, 2),
                "modeled_kwh": round(modeled_gross_kwh, 2) if modeled_gross_kwh is not None else None,
                "error_kwh": round(error, 2),
                "abs_error_kwh": round(abs(error), 2),
                "day_mae": day_mae,
                "night_mae": night_mae,
                "source": dominant_source,
                "source_breakdown": final_breakdown,
                "primary_entity": self._midnight_forecast_snapshot.get("primary_entity"),
                "secondary_entity": self._midnight_forecast_snapshot.get("secondary_entity"),
                "crossover_day": self._midnight_forecast_snapshot.get("crossover_day"),
            }
            self._forecast_history.append(entry)

            if len(self._forecast_history) > 365:
                self._forecast_history.pop(0)

            self._cached_forecast_uncertainty = None
            _LOGGER.info(f"Forecast Accuracy Logged for {date_key}: Error={error:.2f} kWh (Source: {dominant_source})")

    def calculate_week_ahead_stats(self) -> dict:
        """Get week ahead stats with caching to prevent expensive recalculation."""
        now = dt_util.now()

        # Cache check (5 minutes TTL)
        if (self._cached_week_ahead_stats is not None and
            self._cached_week_ahead_timestamp is not None and
            (now - self._cached_week_ahead_timestamp).total_seconds() < 300):
            return self._cached_week_ahead_stats

        stats = self._calculate_week_ahead_stats_internal()
        self._cached_week_ahead_stats = stats
        self._cached_week_ahead_timestamp = now
        return stats

    def _calculate_week_ahead_stats_internal(self) -> dict:
        """Calculate detailed 7-day forecast statistics using LIVE forecast."""
        now = dt_util.now()
        today = now.date()
        week_end = today + timedelta(days=6)

        stats = {
            "total_kwh": 0.0,
            ATTR_AVG_TEMP_FORECAST: None,
            ATTR_AVG_WIND_FORECAST: None,
            ATTR_COLDEST_DAY: None,
            ATTR_WARMEST_DAY: None,
            ATTR_PEAK_DAY: None,
            ATTR_LIGHTEST_DAY: None,
            ATTR_WEEKLY_SUMMARY: "Generating forecast...",
            ATTR_TYPICAL_WEEK_KWH: 0.0,
            ATTR_VS_TYPICAL_KWH: 0.0,
            ATTR_VS_TYPICAL_PCT: 0.0,
            ATTR_WEEK_START_DATE: today.isoformat(),
            ATTR_WEEK_END_DATE: week_end.isoformat(),
            "confidence_level": "low",
        }

        daily_details = []
        total_kwh = 0.0
        temp_sum = 0.0
        wind_sum = 0.0
        temp_days = 0
        wind_days = 0

        min_temp = 999.0
        max_temp = -999.0
        min_temp_day = None
        max_temp_day = None

        max_kwh = -1.0
        min_kwh = 99999.0
        max_kwh_day = None
        min_kwh_day = None

        # Analysis Lists
        current_days_analysis = []
        baseline_days_analysis = []

        # Optimization: Pre-fetch Last Year logs for the entire week range.
        # Handle 53-week year anomalies by calculating the true Min/Max of the mapped dates.
        ly_dates = []
        for i in range(7):
            day_in_week = today + timedelta(days=i)
            ly_dates.append(get_last_year_iso_date(day_in_week))

        ly_start_iso = min(ly_dates)
        ly_end_iso = max(ly_dates)

        ly_logs_map = self.coordinator.statistics._get_daily_log_map(ly_start_iso, ly_end_iso)

        # Initialize running inertia from recent history for Today
        # Use centralized helper to avoid duplication
        running_inertia = self.coordinator._get_inertia_list(now)

        # Fallback if list is empty (rare: restart with no history/sensors)
        if not running_inertia:
             current_inertia_avg = self.coordinator._calculate_inertia_temp()
             if current_inertia_avg:
                 history_needed = len(DEFAULT_INERTIA_WEIGHTS) - 1
                 running_inertia = [current_inertia_avg] * history_needed

        current = today
        while current <= week_end:
            day_stats = {
                "date": current.isoformat(),
                "kwh": 0.0,
                "temp": None,
                "wind": None,
                "solar_impact": 0.0,
                "source": "forecast"
            }

            # Use Smart Merge Prediction with Continuous Inertia
            # FORCE IGNORE AUX FOR ALL DAYS (INCLUDING TODAY) to provide Pure Model forecast
            prediction = self.get_future_day_prediction(current, running_inertia, ignore_aux=True)

            if prediction:
                p_kwh, p_solar, w_stats = prediction
                day_stats["kwh"] = round(p_kwh, 1)

                # Fix for Today's Total: Use Midnight Snapshot if available
                # Live forecast (prediction) often shrinks throughout the day (remaining hours).
                # To show the full day expectation in the week-ahead view, we override with the static midnight plan.
                if self._midnight_forecast_snapshot and self._midnight_forecast_snapshot.get("date") == current.isoformat():
                    day_stats["kwh"] = round(self._midnight_forecast_snapshot["kwh"], 1)

                day_stats["solar_impact"] = round(p_solar, 1)
                day_stats["temp"] = w_stats.get("temp")
                day_stats["wind"] = w_stats.get("wind")
                day_stats["source"] = w_stats.get("source", "forecast")

                # Update inertia for next iteration
                if "final_inertia" in w_stats:
                    running_inertia = w_stats["final_inertia"]
            else:
                # Fallback to Last Year
                # Note: History fallback breaks the inertia chain.
                running_inertia = []

                ly_date = get_last_year_iso_date(current)

                ly_kwh, ly_solar, ly_temp, ly_wind, _ = self.coordinator.calculate_modeled_energy(
                    ly_date, ly_date, pre_fetched_logs=ly_logs_map
                )
                day_stats["kwh"] = round(ly_kwh, 1)
                day_stats["solar_impact"] = round(ly_solar, 1)
                day_stats["temp"] = ly_temp
                day_stats["wind"] = ly_wind
                day_stats["source"] = "history_fallback"

            source = day_stats.get("source")
            if source == "hourly_forecast":
                 day_stats["reliability"] = "high"
            elif source == "daily_forecast":
                 day_stats["reliability"] = "medium"
            elif source == "today_forecast":
                 day_stats["reliability"] = "high"
            else:
                 day_stats["reliability"] = "low"

            # Check if model covers this temperature/wind combination (Thermodynamic Integrity)
            # If not, we are extrapolating, so downgrade reliability
            if day_stats["temp"] is not None:
                # Determine forecast bucket
                f_wind = day_stats["wind"] if day_stats["wind"] is not None else 0.0
                f_bucket = self.coordinator._get_wind_bucket(f_wind)
                f_temp_key = str(int(round(day_stats["temp"])))

                is_covered = self.coordinator._is_model_covered(f_temp_key, f_bucket)
                if not is_covered:
                    day_stats["reliability"] = "low"
                    # Optional: Add explanation?
                    # day_stats["note"] = "Extrapolated (No Model Data)"

            if day_stats["temp"] is not None:
                temp_sum += day_stats["temp"]
                temp_days += 1
                if day_stats["temp"] < min_temp:
                    min_temp = day_stats["temp"]
                    min_temp_day = day_stats
                if day_stats["temp"] > max_temp:
                    max_temp = day_stats["temp"]
                    max_temp_day = day_stats

            if day_stats["wind"] is not None:
                wind_sum += day_stats["wind"]
                wind_days += 1

            if day_stats["kwh"] > max_kwh:
                max_kwh = day_stats["kwh"]
                max_kwh_day = day_stats

            if day_stats["kwh"] < min_kwh:
                min_kwh = day_stats["kwh"]
                min_kwh_day = day_stats

            # --- Analysis Data Collection ---
            # Calculate Baseline (Last Year Model) for this day
            ly_date = get_last_year_iso_date(current)

            ly_kwh, _, ly_temp, ly_wind, _ = self.coordinator.calculate_modeled_energy(
                ly_date, ly_date, pre_fetched_logs=ly_logs_map
            )

            # Determine buckets
            curr_wind = day_stats.get("wind") or 0.0
            curr_bucket = self.coordinator._get_wind_bucket(curr_wind)

            base_wind = ly_wind if ly_wind is not None else 0.0
            base_bucket = self.coordinator._get_wind_bucket(base_wind)

            current_days_analysis.append({
                "date": day_stats["date"],
                "temp": day_stats.get("temp"),
                "wind": curr_wind,
                "wind_bucket": curr_bucket,
                "kwh": day_stats.get("kwh", 0.0)
            })

            baseline_days_analysis.append({
                "date": ly_date.isoformat(),
                "temp": ly_temp,
                "wind": base_wind,
                "wind_bucket": base_bucket,
                "kwh": ly_kwh
            })

            total_kwh += day_stats["kwh"]
            daily_details.append(day_stats)
            current += timedelta(days=1)

        avg_temp = round(temp_sum / temp_days, 1) if temp_days > 0 else None
        avg_wind = round(wind_sum / wind_days, 1) if wind_days > 0 else None

        # Use Explanation Module for Analysis & Summary
        analyzer = WeatherImpactAnalyzer(self.coordinator)
        analysis = analyzer.analyze_period(current_days_analysis, baseline_days_analysis, context='week_ahead')

        formatter = ExplanationFormatter()
        summary_text = formatter.format_week_ahead(analysis)

        # Extract calculated values from analysis to ensure consistency
        typical_kwh = analysis['baseline_kwh']
        vs_typical_kwh = analysis['delta_kwh']
        vs_typical_pct = analysis['delta_pct']

        uncertainty = self.calculate_uncertainty_stats()
        p95_error_per_day = uncertainty.get("p95_abs_error", 2.0)
        p50_error_per_day = uncertainty.get("p50_abs_error", 1.0)
        samples = uncertainty.get("samples", 0)

        week_margin = p95_error_per_day * 7

        if samples < 7:
            confidence_level = "low"
        elif p50_error_per_day < 2.0 and samples >= 14:
            confidence_level = "high"
        elif p50_error_per_day < 4.0:
            confidence_level = "medium"
        else:
            confidence_level = "low"

        stats["total_kwh"] = round(total_kwh, 1)
        stats[ATTR_AVG_TEMP_FORECAST] = avg_temp
        stats[ATTR_AVG_WIND_FORECAST] = avg_wind

        if min_temp_day and "temp" in min_temp_day:
            stats[ATTR_COLDEST_DAY] = {"date": min_temp_day["date"], "temp": min_temp_day["temp"]}
        else:
            stats[ATTR_COLDEST_DAY] = None

        if max_temp_day and "temp" in max_temp_day:
            stats[ATTR_WARMEST_DAY] = {"date": max_temp_day["date"], "temp": max_temp_day["temp"]}
        else:
            stats[ATTR_WARMEST_DAY] = None

        stats[ATTR_PEAK_DAY] = {"date": max_kwh_day["date"], "kwh": max_kwh_day["kwh"]} if max_kwh_day else None
        stats[ATTR_LIGHTEST_DAY] = {"date": min_kwh_day["date"], "kwh": min_kwh_day["kwh"]} if min_kwh_day else None
        stats[ATTR_WEEKLY_SUMMARY] = summary_text
        stats[ATTR_TYPICAL_WEEK_KWH] = round(typical_kwh, 1)
        stats[ATTR_VS_TYPICAL_KWH] = round(vs_typical_kwh, 1)
        stats[ATTR_VS_TYPICAL_PCT] = round(vs_typical_pct, 1)
        stats[ATTR_FORECAST_RANGE_MIN] = round(total_kwh - week_margin, 1)
        stats[ATTR_FORECAST_RANGE_MAX] = round(total_kwh + week_margin, 1)
        stats["confidence_level"] = confidence_level

        stats[ATTR_DAILY_FORECAST] = daily_details

        return stats
