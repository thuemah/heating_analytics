"""Model Comparison sensors for Heating Analytics.

State is served from a precomputed snapshot: the heavy period computation
(past-day model evaluations, hybrid projection, period day-lists, explanation
formatting) runs in an executor thread and the property getters only read the
stored result.  The getters must never compute — the cold path takes 0.5-1.6 s
and stalls the event loop when run inline (slow-state-update warnings after
restart).  The first build is deferred to EVENT_HOMEASSISTANT_STARTED so it
never competes with startup; until then the sensors report unknown.
"""
from __future__ import annotations

import logging
from datetime import date, timedelta
import calendar

from ..helpers import get_last_year_iso_date

from homeassistant.components.sensor import SensorStateClass
from homeassistant.const import EVENT_HOMEASSISTANT_STARTED, UnitOfEnergy
from homeassistant.util import dt as dt_util

from ..const import (
    ATTR_ENERGY_TODAY,
    ATTR_PREDICTED,
    ATTR_SOLAR_PREDICTED,
    ATTR_TEMP_ACTUAL_TODAY,
    ATTR_WIND_ACTUAL_TODAY,
    SENSOR_MODEL_COMPARISON_DAY,
    SENSOR_MODEL_COMPARISON_WEEK,
    SENSOR_MODEL_COMPARISON_MONTH,
)
from ..explanation import WeatherImpactAnalyzer, ExplanationFormatter

from .base import HeatingAnalyticsBaseSensor

_LOGGER = logging.getLogger(__name__)


# Session-level dedup for "missing historical data" warnings.  ``extra_state_attributes``
# is recomputed on every coordinator update (~1/min), but the underlying condition
# (no reference-temperature data for last-year's same period) is stable until new
# history accumulates — so the warning is deterministic noise at per-minute
# cadence.  Dedup at module level: one log per unique missing period per HA
# restart.  Cleared on restart by design — user gets a reminder after each boot
# that comparison data is still missing.
_WARNED_WEEKS: set[int] = set()
_WARNED_MONTHS: set[str] = set()


def weighted_avg(val1, w1, val2, w2):
    """Calculate weighted average of two values."""
    if val1 is None and val2 is None:
        return None
    if val1 is None:
        return val2
    if val2 is None:
        return val1
    total_w = w1 + w2
    if total_w == 0:
        return 0.0
    return (val1 * w1 + val2 * w2) / total_w


class HeatingModelComparisonBaseSensor(HeatingAnalyticsBaseSensor):
    """Base class for Model Comparison Sensors.

    See the module docstring for the snapshot architecture.  Subclasses
    implement `_compute_native_value` / `_compute_extra_attributes` (the
    former property bodies); the base owns scheduling and the getters.
    """

    _attr_native_unit_of_measurement = UnitOfEnergy.KILO_WATT_HOUR
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_icon = "mdi:chart-timeline-variant"

    def __init__(self, coordinator, entry) -> None:
        super().__init__(coordinator, entry)
        self._snapshot: dict | None = None
        self._refresh_in_flight = False
        self._cached_ly_days: list[dict] | None = None
        self._cached_ly_days_key: tuple | None = None

    @property
    def native_value(self) -> float | None:
        """Return the snapshot state; None until the first background build."""
        if self._snapshot is None:
            return None
        return self._snapshot["native_value"]

    @property
    def extra_state_attributes(self) -> dict | None:
        """Return the snapshot attributes; None until the first background build."""
        if self._snapshot is None:
            return None
        return self._snapshot["attributes"]

    async def async_added_to_hass(self) -> None:
        """Register listeners and schedule the first snapshot build."""
        await super().async_added_to_hass()
        if self.hass.is_running:
            self._schedule_snapshot_refresh()
        else:
            # Defer the cold-path build until startup congestion has passed.
            # The sensor reports unknown until the first snapshot lands.
            self.async_on_remove(
                self.hass.bus.async_listen_once(
                    EVENT_HOMEASSISTANT_STARTED, self._on_hass_started
                )
            )

    async def _on_hass_started(self, _event) -> None:
        # Must be async: the event bus runs plain sync listeners as executor
        # jobs, and _schedule_snapshot_refresh uses loop-only APIs.  A
        # coroutine listener is scheduled on the event loop.
        self._schedule_snapshot_refresh()

    def _handle_coordinator_update(self) -> None:
        """Schedule an off-loop snapshot rebuild instead of computing inline.

        Overrides CoordinatorEntity's default, which writes state (and with
        it evaluates the properties) synchronously.  State is written by
        `_async_refresh_snapshot` when the executor build completes.
        Pre-start ticks are skipped; the EVENT_HOMEASSISTANT_STARTED
        listener schedules the first build.
        """
        if self.hass.is_running:
            self._schedule_snapshot_refresh()

    def _schedule_snapshot_refresh(self) -> None:
        if self._refresh_in_flight:
            # A build is already running; this tick's data lands on the next
            # coordinator update, at most a minute away.
            return
        self._refresh_in_flight = True
        try:
            self.hass.async_create_background_task(
                self._async_refresh_snapshot(),
                name=f"heating_analytics.{type(self).__name__}.snapshot",
            )
        except Exception:
            # A flag stuck True would coalesce-skip every future refresh and
            # leave the sensor unknown permanently.
            self._refresh_in_flight = False
            raise

    async def _async_refresh_snapshot(self) -> None:
        try:
            snapshot = await self.hass.async_add_executor_job(self._build_snapshot)
        except Exception as err:  # noqa: BLE001
            # Parity with the previous inline flow, where a raising property
            # aborted the state write and HA logged it: keep the last good
            # snapshot and report.
            _LOGGER.error(
                "Comparison snapshot build failed for %s: %s",
                self._attr_name, err, exc_info=True,
            )
            return
        finally:
            self._refresh_in_flight = False
        self._snapshot = snapshot
        self.async_write_ha_state()

    def _build_snapshot(self) -> dict:
        """Compute the full state snapshot.  Runs in an executor thread.

        Pure read of coordinator state — must not mutate anything outside
        this sensor's private caches, which only this builder touches and
        which `_refresh_in_flight` serializes.  Attributes are computed
        first so the period day-list they build primes the hour-bucket
        stats cache; the native value is then a cache hit.
        """
        attributes = self._compute_extra_attributes()
        return {
            "native_value": self._compute_native_value(),
            "attributes": attributes,
        }

    def _get_last_year_period_days(self, ly_start: date, ly_end: date) -> list[dict]:
        """Per-calendar-day cached wrapper around `_build_last_year_period_days`.

        The rows read only last-year history (daily_history plus year-old
        hourly-log entries), which is stable within a calendar day —
        rebuilding every tick re-scans the hourly log for nothing.
        Consumers must not mutate the returned rows.
        """
        today = dt_util.now().date()
        key = (today, ly_start, ly_end)
        if self._cached_ly_days is not None and self._cached_ly_days_key == key:
            return self._cached_ly_days
        days = self._build_last_year_period_days(ly_start, ly_end)
        self._cached_ly_days = days
        self._cached_ly_days_key = key
        return days

    def _get_or_calculate_stats(self, start_date, period_type="day", total_days_in_period=1, current_period_days=None):
        """Get cached stats or calculate them."""
        now = dt_util.now()

        # Check cache
        if (
            self._cached_stats
            and self._cached_time
            and self._cached_time.hour == now.hour
            and self._cached_time.date() == now.date()
        ):
            return self._cached_stats

        try:
            stats = self._calculate_period_stats(
                start_date, period_type, total_days_in_period, current_period_days
            )
            self._cached_stats = stats
            self._cached_time = now
            return stats
        except (TypeError, KeyError, ZeroDivisionError) as e:
            _LOGGER.error("Error calculating model stats: %s", e, exc_info=True)
            # Return proper structure with all expected keys (5-tuple)
            empty_weather_stats = {
                "ref_temp": None,
                "ref_wind": None,
                "ref_solar": None,
                "curr_temp": None,
                "curr_wind": None,
                "curr_solar": None
            }
            # Fallback: Current Hybrid=0, Last Model=0, Last Actual=0, Current Debug=0, Metadata
            return 0.0, 0.0, 0.0, 0.0, empty_weather_stats

    def _calculate_period_stats(self, start_date, period_type, total_days_in_period, current_period_days=None):
        """Calculate stats for a period (Current vs Last Year) using the iterative modeled energy.

        Args:
            current_period_days: optional pre-built day list for the current
                period (from `_build_current_period_days`), so a snapshot
                build that already made the list for the attribute path does
                not build it twice.  None → built internally.

        Returns:
            (model_curr_total, model_last_total, last_year_actual_kwh, current_model_kwh, metadata)
            metadata is a dict containing average temp, wind, and solar totals for reference vs current.
        """
        now = dt_util.now()
        today = now.date()

        # --- 1. PAST DATA (CACHEABLE) ---
        if self._cached_past_date == today and self._cached_past_data:
            (model_past, solar_past, temp_past, wind_past,
             model_last_so_far, solar_last_so_far, temp_last_so_far, wind_last_so_far,
             model_last_remaining, solar_last_remaining, temp_last_remaining, wind_last_remaining,
             days_past, ly_total_days, last_year_actual_kwh) = self._cached_past_data
        else:
            # Calculate Past Days (Completed)
            if today > start_date:
                yesterday = today - timedelta(days=1)
                calc_end = max(start_date, yesterday)
                if calc_end >= start_date:
                    model_past, solar_past, temp_past, wind_past, _ = self.coordinator.calculate_modeled_energy(start_date, calc_end)
                    days_past = (calc_end - start_date).days + 1
                else:
                    model_past, solar_past, temp_past, wind_past = 0.0, 0.0, None, None
                    days_past = 0
            else:
                model_past, solar_past, temp_past, wind_past = 0.0, 0.0, None, None
                days_past = 0

            # --- Last Year Period & Remaining Projection ---
            if period_type == "week":
                curr_year, curr_week, _ = start_date.isocalendar()
                try:
                    ly_start = date.fromisocalendar(curr_year - 1, curr_week, 1)
                except ValueError:
                    ly_start = date.fromisocalendar(curr_year - 1, 52, 1)
            elif period_type == "day":
                ly_start = start_date - timedelta(days=365)
            else:
                try:
                    ly_start = start_date.replace(year=start_date.year - 1)
                except ValueError:
                    ly_start = start_date.replace(year=start_date.year - 1, day=28)

            if period_type == "month":
                ly_month = ly_start.month
                ly_year = ly_start.year
                ly_is_leap = (ly_year % 4 == 0 and ly_year % 100 != 0) or (ly_year % 400 == 0)
                ly_days = 31 if ly_month in [1,3,5,7,8,10,12] else \
                        30 if ly_month in [4,6,9,11] else \
                        (29 if ly_is_leap else 28)
                ly_end = ly_start + timedelta(days=ly_days - 1)
                ly_total_days = ly_days
            else:
                ly_total_days = total_days_in_period
                ly_end = ly_start + timedelta(days=ly_total_days - 1)

            days_so_far = days_past + 1
            ly_days_so_far = min(days_so_far, ly_total_days)
            ly_so_far_end = ly_start + timedelta(days=ly_days_so_far - 1)

            # Get Last Year "So Far"
            (model_last_so_far, solar_last_so_far,
             temp_last_so_far, wind_last_so_far, _) = self.coordinator.calculate_modeled_energy(ly_start, ly_so_far_end)

            # Get Last Year "Remaining"
            if ly_days_so_far < ly_total_days:
                ly_rem_start = ly_so_far_end + timedelta(days=1)
                (model_last_remaining, solar_last_remaining,
                 temp_last_remaining, wind_last_remaining, _) = self.coordinator.calculate_modeled_energy(ly_rem_start, ly_end)
            else:
                model_last_remaining, solar_last_remaining, temp_last_remaining, wind_last_remaining = 0.0, 0.0, None, None

            # Get Last Year Actuals
            last_year_actual_kwh = self.coordinator.statistics.calculate_historical_actual_sum(ly_start, ly_end)

            # Store in cache
            self._cached_past_data = (model_past, solar_past, temp_past, wind_past,
                                      model_last_so_far, solar_last_so_far, temp_last_so_far, wind_last_so_far,
                                      model_last_remaining, solar_last_remaining, temp_last_remaining, wind_last_remaining,
                                      days_past, ly_total_days, last_year_actual_kwh)
            self._cached_past_date = today

        # Calculate ly_days_so_far after cache check (needed for weather stats calculation)
        # This must be recalculated even when using cache since it depends on current time
        days_so_far = days_past + 1
        ly_days_so_far = min(days_so_far, ly_total_days)

        # --- 2. TODAY DATA (DYNAMIC) ---
        # Guard against None coordinator.data
        if self.coordinator.data is None:
            _LOGGER.warning("coordinator.data is None in _calculate_period_stats")
            model_today = 0.0
            solar_today = 0.0
            temp_today = None
            wind_today = None
        else:
            model_today = self.coordinator.data.get(ATTR_PREDICTED, 0.0)
            solar_today = self.coordinator.data.get(ATTR_SOLAR_PREDICTED, 0.0)
            temp_today = self.coordinator.data.get(ATTR_TEMP_ACTUAL_TODAY)
            wind_today = self.coordinator.data.get(ATTR_WIND_ACTUAL_TODAY)

        model_last_total = model_last_so_far + model_last_remaining
        solar_last_total = solar_last_so_far + solar_last_remaining

        # --- 3. AGGREGATE WEATHER STATS ---
        # Last Year Average (Full Period)
        ly_days_remaining = ly_total_days - ly_days_so_far
        ly_avg_temp = weighted_avg(temp_last_so_far, ly_days_so_far, temp_last_remaining, ly_days_remaining)
        ly_avg_wind = weighted_avg(wind_last_so_far, ly_days_so_far, wind_last_remaining, ly_days_remaining)

        # Calculate Full Period Weighted Average (Past + Today + Future)
        # Use helper to build the full period data list (handles Past, Today, and Future fallback internally)
        end_date = start_date + timedelta(days=total_days_in_period - 1)
        if current_period_days is None:
            current_period_days = self._build_current_period_days(start_date, end_date)

        temps = [d['temp'] for d in current_period_days if d.get('temp') is not None]
        winds = [d['wind'] for d in current_period_days if d.get('wind') is not None]

        curr_avg_temp = sum(temps) / len(temps) if temps else None
        curr_avg_wind = sum(winds) / len(winds) if winds else None

        # --- 4. FINALIZE CURRENT PERIOD TOTAL ---
        # USE HYBRID PROJECTION FOR ALL PERIODS (Generalized)
        # Note: total_days_in_period is passed correctly by subclasses, including correct month length
        model_curr_total, solar_curr_total = self.coordinator.statistics.calculate_hybrid_projection(start_date, end_date)

        metadata = {
            "ref_temp": ly_avg_temp,
            "ref_wind": ly_avg_wind,
            "ref_solar": solar_last_total,
            "curr_temp": curr_avg_temp,
            "curr_wind": curr_avg_wind,
            "curr_solar": solar_curr_total
        }

        # Return 5 values: Current Hybrid, Last Year Model, Last Year Actual, Current Hybrid (Debug), Metadata
        # current_model_kwh (4th element) is rounded to 3 decimals as requested.
        ly_actual = round(last_year_actual_kwh, 3) if last_year_actual_kwh is not None else None
        return round(model_curr_total, 1), round(model_last_total, 1), ly_actual, round(model_curr_total, 3), metadata

    def _build_current_period_days(self, start_date: date, end_date: date) -> list[dict]:
        """
        Build daily data list for current period.

        Strategy:
        - Past days (start_date to yesterday): Use _daily_history
        - Today: Use current actuals
        - Future days (tomorrow to end_date): Use forecast from ForecastManager

        Returns:
            List of dicts: [{'date': date, 'temp': float, 'wind': float,
                            'wind_bucket': str, 'kwh': float, 'solar_kwh': float}, ...]
        """
        days = []
        current = start_date
        now = dt_util.now()
        today = now.date()

        # Pre-fetch the hourly-log -> date map once for the whole period.  Without
        # this, _get_historical_day -> calculate_modeled_energy re-scans the
        # reversed hourly_log per day, which blocks the event loop for ~0.7 s on
        # month-sensor refreshes against year-old dates.
        prefetch_start = min(start_date, today)
        prefetch_end = max(end_date, today)
        pre_fetched_logs = self.coordinator.statistics._get_daily_log_map(prefetch_start, prefetch_end)

        while current <= end_date:
            if current < today:
                # Past: Use daily_history
                day_data = self._get_historical_day(current, pre_fetched_logs=pre_fetched_logs)
            elif current == today:
                # Today: Use current actuals (from coordinator.data)
                day_data = self._get_today_data(current)
            else:
                # Future: Use forecast
                day_data = self._get_forecast_day(current)

            days.append(day_data)
            current += timedelta(days=1)

        return days

    def _build_last_year_period_days(self, ly_start: date, ly_end: date) -> list[dict]:
        """
        Build daily data list for last year same period.

        Uses _daily_history exclusively (all past data).
        """
        days = []
        current = ly_start

        pre_fetched_logs = self.coordinator.statistics._get_daily_log_map(ly_start, ly_end)

        while current <= ly_end:
            day_data = self._get_historical_day(current, pre_fetched_logs=pre_fetched_logs)
            days.append(day_data)
            current += timedelta(days=1)

        return days

    def _get_historical_day(self, date_obj: date, pre_fetched_logs: dict | None = None) -> dict:
        """Extract day data from _daily_history and calculate Model value."""
        day_str = date_obj.isoformat()

        if day_str in self.coordinator.model.daily_history:
            entry = self.coordinator.model.daily_history[day_str]
            # Guard against None entries in legacy storage
            if entry is None:
                # Treat as missing data
                return {
                    'date': date_obj.isoformat(),
                    'temp': None,
                    'wind': None,
                    'wind_bucket': None,
                    'kwh': 0.0,
                    'solar_kwh': 0.0
                }

            temp = entry.get('temp')
            wind = entry.get('wind', 0.0)

            # Use calculate_modeled_energy to get Model value (Base - Solar)
            # This serves as a fallback or for Model Comparison if actuals are missing
            model_kwh, solar_kwh, _, _, _ = self.coordinator.calculate_modeled_energy(date_obj, date_obj, pre_fetched_logs)

            # Use actual kwh if available (Hybrid), otherwise fallback to model
            actual_kwh = entry.get('kwh')
            if actual_kwh is None:
                actual_kwh = model_kwh

            # Determine wind bucket
            if wind is not None:
                wind_bucket = self.coordinator._get_wind_bucket(wind)
            else:
                wind_bucket = 'normal'

            return {
                'date': date_obj.isoformat(),
                'temp': temp,
                'wind': wind,
                'wind_bucket': wind_bucket,
                'kwh': round(actual_kwh, 2),
                'solar_kwh': round(solar_kwh, 2)
            }
        else:
            # Missing data - return None values
            return {
                'date': date_obj.isoformat(),
                'temp': None,
                'wind': None,
                'wind_bucket': None,
                'kwh': 0.0,
                'solar_kwh': 0.0
            }

    def _get_today_data(self, date_obj: date) -> dict:
        """Get today's data using Hybrid calculation (Actual + Forecast)."""
        temp = self.coordinator.data.get(ATTR_TEMP_ACTUAL_TODAY)
        wind = self.coordinator.data.get(ATTR_WIND_ACTUAL_TODAY)
        solar = self.coordinator.data.get(ATTR_SOLAR_PREDICTED, 0.0)

        # Use Actuals So Far + Forecast Remaining (Hybrid)
        actual_so_far = self.coordinator.data.get(ATTR_ENERGY_TODAY, 0.0)

        # Forecast remaining (Model)
        future_kwh, _, _ = self.coordinator.forecast.calculate_future_energy(dt_util.now())

        kwh = actual_so_far + future_kwh

        wind_bucket = self.coordinator._get_wind_bucket(wind or 0.0)

        return {
            'date': date_obj.isoformat(),
            'temp': temp,
            'wind': wind,
            'wind_bucket': wind_bucket,
            'kwh': round(kwh, 2),
            'solar_kwh': round(solar, 2)
        }

    def _get_forecast_day(self, date_obj: date) -> dict:
        """Get forecast data for future day from ForecastManager."""
        # Use ForecastManager.get_future_day_prediction
        prediction = self.coordinator.forecast.get_future_day_prediction(date_obj)

        if prediction:
            p_kwh, p_solar, w_stats = prediction

            return {
                'date': date_obj.isoformat(),
                'temp': w_stats.get('temp'),
                'wind': w_stats.get('wind'),
                'wind_bucket': self.coordinator._get_wind_bucket(w_stats.get('wind', 0.0)),
                'kwh': p_kwh,
                'solar_kwh': p_solar
            }
        else:
            # Fallback: No forecast available (beyond forecast horizon)
            # Use last year's same date as proxy for expected energy
            ly_date = get_last_year_iso_date(date_obj)
            model_kwh, solar_kwh, avg_temp, avg_wind, _ = self.coordinator.calculate_modeled_energy(ly_date, ly_date)
            wind_bucket = None
            if avg_wind is not None:
                wind_bucket = self.coordinator._get_wind_bucket(avg_wind)

            return {
                'date': date_obj.isoformat(),
                'temp': avg_temp,
                'wind': avg_wind,
                'wind_bucket': wind_bucket,
                'kwh': round(model_kwh, 2),
                'solar_kwh': round(solar_kwh, 2)
            }

    def _generate_fallback_summary(self, curr, last):
        """Fallback summary if explanation module fails."""
        kwh_diff = curr - last
        if abs(kwh_diff) > 5:
            if kwh_diff > 0:
                return f"Higher consumption: +{kwh_diff:.1f} kWh vs last year"
            else:
                return f"Lower consumption: -{abs(kwh_diff):.1f} kWh vs last year"
        else:
            return "Consumption similar to last year"


class HeatingModelComparisonDaySensor(HeatingModelComparisonBaseSensor):
    """Sensor for Daily Model Comparison."""

    _attr_name = SENSOR_MODEL_COMPARISON_DAY

    def _compute_native_value(self) -> float:
        now = dt_util.now()
        today = now.date()
        curr, last, _, _, _ = self._get_or_calculate_stats(today, "day", 1)
        return round(curr - last, 1)

    def _compute_extra_attributes(self):
        now = dt_util.now()
        today = now.date()
        curr, last, actual, model, w_stats = self._get_or_calculate_stats(today, "day", 1)

        # Calculate Deltas
        t_delta = None
        if w_stats["curr_temp"] is not None and w_stats["ref_temp"] is not None:
            t_delta = round(w_stats["curr_temp"] - w_stats["ref_temp"], 1)

        w_delta = None
        if w_stats["curr_wind"] is not None and w_stats["ref_wind"] is not None:
            w_delta = round(w_stats["curr_wind"] - w_stats["ref_wind"], 1)

        s_delta = None
        if w_stats["curr_solar"] is not None and w_stats["ref_solar"] is not None:
            s_delta = round(w_stats["curr_solar"] - w_stats["ref_solar"], 1)

        # === NEW: Use explanation module ===
        try:
            # Reconstruct day objects
            ly_date = today - timedelta(days=365)

            day_curr = self._get_today_data(today)
            # Override with Pure Model values for consistent explanation
            day_curr["kwh"] = model
            if w_stats.get("curr_solar") is not None:
                day_curr["solar_kwh"] = w_stats["curr_solar"]

            # Per-day cached + routed through the prefetched hourly-log map;
            # an un-prefetched lookup re-scans the log every tick for a
            # year-old date.
            day_last = self._get_last_year_period_days(ly_date, ly_date)[0]

            # Analyze
            analyzer = WeatherImpactAnalyzer(self.coordinator)
            analysis = analyzer.analyze_day(day_curr, day_last)

            # Format
            formatter = ExplanationFormatter()
            daily_summary = formatter.format_day_comparison(analysis)

        except (TypeError, AttributeError, KeyError) as e:
            _LOGGER.warning(f"Failed to generate daily explanation: {e}")
            daily_summary = self._generate_fallback_summary(curr, last)

        return {
            "daily_summary": daily_summary,
            "current_model_kwh": model,
            "last_year_model_kwh": last,
            "last_year_actual_kwh": round(actual, 3) if actual is not None else None,
            # Comparison Attributes
            "reference_temperature": round(w_stats["ref_temp"], 1) if w_stats["ref_temp"] is not None else None,
            "current_temperature": round(w_stats["curr_temp"], 1) if w_stats["curr_temp"] is not None else None,
            "temperature_delta": t_delta,
            "reference_effective_wind": round(w_stats["ref_wind"], 1) if w_stats["ref_wind"] is not None else None,
            "current_effective_wind": round(w_stats["curr_wind"], 1) if w_stats["curr_wind"] is not None else None,
            "wind_delta": w_delta,
            "reference_solar_kwh": round(w_stats["ref_solar"], 1) if w_stats["ref_solar"] is not None else None,
            "current_solar_kwh": round(w_stats["curr_solar"], 1) if w_stats["curr_solar"] is not None else None,
            "solar_delta": s_delta,
        }

    @property
    def unique_id(self) -> str:
        return f"{self.entry.entry_id}_model_comparison_day"


class HeatingModelComparisonWeekSensor(HeatingModelComparisonBaseSensor):
    """Sensor for Weekly Model Comparison."""

    _attr_name = SENSOR_MODEL_COMPARISON_WEEK

    def _compute_native_value(self) -> float:
        now = dt_util.now()
        today = now.date()
        start_week = today - timedelta(days=today.weekday())
        curr, last, _, _, _ = self._get_or_calculate_stats(start_week, "week", 7)
        return round(curr - last, 1)

    def _compute_extra_attributes(self):
        now = dt_util.now()
        today = now.date()
        start_week = today - timedelta(days=today.weekday())

        # Get ISO week number
        _, week_num, _ = now.isocalendar()

        # === Build data lists for comparison ===
        # Last year ISO week
        curr_year, curr_week, _ = start_week.isocalendar()
        try:
            ly_start = date.fromisocalendar(curr_year - 1, curr_week, 1)
        except ValueError:
            ly_start = date.fromisocalendar(curr_year - 1, 52, 1)

        ly_end = ly_start + timedelta(days=6)
        end_week = start_week + timedelta(days=6)

        try:
            current_days = self._build_current_period_days(start_week, end_week)
            last_year_days = self._get_last_year_period_days(ly_start, ly_end)
        except (TypeError, ValueError, KeyError, AttributeError) as e:
            # TypeError covers round(None, 2) on degenerate last-year lookups.
            _LOGGER.warning(f"Failed to build period data for week comparison: {e}")
            # If we can't build the lists, use empty lists for hybrid calculation
            current_days = []
            last_year_days = []

        # The day list doubles as the stats input (built once per snapshot);
        # on build failure the stats path falls back to its own internal build.
        curr, last, actual, model, w_stats = self._get_or_calculate_stats(
            start_week, "week", 7, current_period_days=current_days or None
        )

        # Calculate Deltas
        t_delta = None
        if w_stats["curr_temp"] is not None and w_stats["ref_temp"] is not None:
            t_delta = round(w_stats["curr_temp"] - w_stats["ref_temp"], 1)

        w_delta = None
        if w_stats["curr_wind"] is not None and w_stats["ref_wind"] is not None:
            w_delta = round(w_stats["curr_wind"] - w_stats["ref_wind"], 1)

        s_delta = None
        if w_stats["curr_solar"] is not None and w_stats["ref_solar"] is not None:
            s_delta = round(w_stats["curr_solar"] - w_stats["ref_solar"], 1)

        # === Generate explanation ===
        try:
            # Analyze (using modeled totals for accurate comparison)
            analyzer = WeatherImpactAnalyzer(self.coordinator)
            analysis = analyzer.analyze_period(
                current_days,
                last_year_days,
                'week_comparison',
                current_total_kwh=curr,
                last_year_total_kwh=last
            )

            # Format
            formatter = ExplanationFormatter()
            weekly_summary = formatter.format_period_comparison(analysis)

        except (TypeError, AttributeError, KeyError) as e:
            _LOGGER.warning(f"Failed to generate explanation: {e}")
            # Fallback to existing logic (keep current implementation as backup)
            weekly_summary = self._generate_fallback_summary(curr, last)

        # Warning for missing data (as per PR feedback).  Session-deduped:
        # the condition is deterministic per restart, repeating every
        # coordinator tick adds no information.
        if w_stats["ref_temp"] is None and week_num not in _WARNED_WEEKS:
            _WARNED_WEEKS.add(week_num)
            _LOGGER.warning(f"Missing historical data for week {week_num}, comparison may be inaccurate.")

        # Calculate hybrid projection totals for comparison
        # Current: Actual (past) + Budget (today) + Forecast (future)
        # This matches what user sees in real-time (actionable comparison)
        current_hybrid_kwh = sum(d.get('kwh', 0.0) for d in current_days)

        # Last year: Actual consumption for same period
        ly_actual_kwh = sum(d.get('kwh', 0.0) for d in last_year_days)

        hybrid_delta_kwh = current_hybrid_kwh - ly_actual_kwh

        return {
            "weekly_summary": weekly_summary,
            "week_number": week_num,
            "current_model_kwh": model,
            "last_year_model_kwh": last,
            "last_year_actual_kwh": round(actual, 3) if actual is not None else None,
            # Hybrid projection comparison (actionable real-time comparison)
            "current_hybrid_kwh": round(current_hybrid_kwh, 1),
            "hybrid_delta_kwh": round(hybrid_delta_kwh, 1),
            # Comparison Attributes
            "reference_temperature": round(w_stats["ref_temp"], 1) if w_stats["ref_temp"] is not None else None,
            "current_temperature": round(w_stats["curr_temp"], 1) if w_stats["curr_temp"] is not None else None,
            "temperature_delta": t_delta,
            "reference_effective_wind": round(w_stats["ref_wind"], 1) if w_stats["ref_wind"] is not None else None,
            "current_effective_wind": round(w_stats["curr_wind"], 1) if w_stats["curr_wind"] is not None else None,
            "wind_delta": w_delta,
            "reference_solar_kwh": round(w_stats["ref_solar"], 1) if w_stats["ref_solar"] is not None else None,
            "current_solar_kwh": round(w_stats["curr_solar"], 1) if w_stats["curr_solar"] is not None else None,
            "solar_delta": s_delta,
        }

    @property
    def unique_id(self) -> str:
        return f"{self.entry.entry_id}_model_comparison_week"


class HeatingModelComparisonMonthSensor(HeatingModelComparisonBaseSensor):
    """Sensor for Monthly Model Comparison."""

    _attr_name = SENSOR_MODEL_COMPARISON_MONTH

    def _compute_native_value(self) -> float:
        now = dt_util.now()
        today = now.date()
        start_month = today.replace(day=1)

        month = now.month
        year = now.year
        is_leap = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
        days_in_month = 31 if month in [1,3,5,7,8,10,12] else 30 if month in [4,6,9,11] else (29 if is_leap else 28)

        curr, last, _, _, _ = self._get_or_calculate_stats(start_month, "month", days_in_month)
        return round(curr - last, 1)

    def _compute_extra_attributes(self):
        now = dt_util.now()
        today = now.date()
        start_month = today.replace(day=1)

        month = now.month
        year = now.year
        is_leap = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
        days_in_month = 31 if month in [1,3,5,7,8,10,12] else 30 if month in [4,6,9,11] else (29 if is_leap else 28)

        # === Build data lists for comparison ===
        # Last year month start
        try:
            ly_start = start_month.replace(year=start_month.year - 1)
        except ValueError:
            ly_start = start_month.replace(year=start_month.year - 1, day=28)

        # Calculate LY end date (full month) using calendar module
        ly_month = ly_start.month
        ly_year = ly_start.year
        _, ly_days_count = calendar.monthrange(ly_year, ly_month)
        ly_end = ly_start + timedelta(days=ly_days_count - 1)

        # Current end date
        end_month = start_month + timedelta(days=days_in_month - 1)

        try:
            current_days = self._build_current_period_days(start_month, end_month)
            last_year_days = self._get_last_year_period_days(ly_start, ly_end)
        except (TypeError, ValueError, KeyError, AttributeError) as e:
            # TypeError covers round(None, 2) on degenerate last-year lookups.
            _LOGGER.warning(f"Failed to build period data for month comparison: {e}")
            # If we can't build the lists, use empty lists for hybrid calculation
            current_days = []
            last_year_days = []

        # The day list doubles as the stats input (built once per snapshot);
        # on build failure the stats path falls back to its own internal build.
        curr, last, actual, model, w_stats = self._get_or_calculate_stats(
            start_month, "month", days_in_month, current_period_days=current_days or None
        )

        # Calculate Deltas
        t_delta = None
        if w_stats["curr_temp"] is not None and w_stats["ref_temp"] is not None:
            t_delta = round(w_stats["curr_temp"] - w_stats["ref_temp"], 1)

        w_delta = None
        if w_stats["curr_wind"] is not None and w_stats["ref_wind"] is not None:
            w_delta = round(w_stats["curr_wind"] - w_stats["ref_wind"], 1)

        s_delta = None
        if w_stats["curr_solar"] is not None and w_stats["ref_solar"] is not None:
            s_delta = round(w_stats["curr_solar"] - w_stats["ref_solar"], 1)

        # === Generate explanation ===
        try:
            # Analyze (using modeled totals for accurate comparison)
            analyzer = WeatherImpactAnalyzer(self.coordinator)
            analysis = analyzer.analyze_period(
                current_days,
                last_year_days,
                'month_comparison',
                current_total_kwh=curr,
                last_year_total_kwh=last
            )

            # Format (Using generic period formatter)
            formatter = ExplanationFormatter()
            monthly_summary = formatter.format_period_comparison(analysis)

        except (TypeError, AttributeError, KeyError) as e:
            _LOGGER.warning(f"Failed to generate monthly explanation: {e}")
            monthly_summary = self._generate_fallback_summary(curr, last)

        _month_key = f"{year}-{month:02d}"
        if w_stats["ref_temp"] is None and _month_key not in _WARNED_MONTHS:
            _WARNED_MONTHS.add(_month_key)
            _LOGGER.warning(f"Missing historical data for month comparison, summary may be inaccurate.")

        # Calculate hybrid projection totals for comparison
        # Current: Actual (past) + Budget (today) + Forecast (future)
        # This matches what user sees in real-time (actionable comparison)
        current_hybrid_kwh = sum(d.get('kwh', 0.0) for d in current_days)

        # Last year: Actual consumption for same period
        ly_actual_kwh = sum(d.get('kwh', 0.0) for d in last_year_days)

        hybrid_delta_kwh = current_hybrid_kwh - ly_actual_kwh

        return {
            "monthly_summary": monthly_summary,
            "days_in_month": days_in_month,
            "current_model_kwh": model,
            "last_year_model_kwh": last,
            "last_year_actual_kwh": round(actual, 3) if actual is not None else None,
            # Hybrid projection comparison (actionable real-time comparison)
            "current_hybrid_kwh": round(current_hybrid_kwh, 1),
            "hybrid_delta_kwh": round(hybrid_delta_kwh, 1),
            # Comparison Attributes
            "reference_temperature": round(w_stats["ref_temp"], 1) if w_stats["ref_temp"] is not None else None,
            "current_temperature": round(w_stats["curr_temp"], 1) if w_stats["curr_temp"] is not None else None,
            "temperature_delta": t_delta,
            "reference_effective_wind": round(w_stats["ref_wind"], 1) if w_stats["ref_wind"] is not None else None,
            "current_effective_wind": round(w_stats["curr_wind"], 1) if w_stats["curr_wind"] is not None else None,
            "wind_delta": w_delta,
            "reference_solar_kwh": round(w_stats["ref_solar"], 1) if w_stats["ref_solar"] is not None else None,
            "current_solar_kwh": round(w_stats["curr_solar"], 1) if w_stats["curr_solar"] is not None else None,
            "solar_delta": s_delta,
        }

    @property
    def unique_id(self) -> str:
        return f"{self.entry.entry_id}_model_comparison_month"
