"""Model Comparison sensors for Heating Analytics."""
from __future__ import annotations

import logging
from datetime import date, timedelta
import calendar

from ..helpers import get_last_year_iso_date

from homeassistant.components.sensor import SensorStateClass
from homeassistant.const import UnitOfEnergy
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
    """Base class for Model Comparison Sensors."""

    _attr_native_unit_of_measurement = UnitOfEnergy.KILO_WATT_HOUR
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_icon = "mdi:chart-timeline-variant"

    def _get_or_calculate_stats(self, start_date, period_type="day", total_days_in_period=1):
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
            stats = self._calculate_period_stats(start_date, period_type, total_days_in_period)
            self._cached_stats = stats
            self._cached_time = now
            return stats
        except Exception as e:
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

    def _calculate_period_stats(self, start_date, period_type, total_days_in_period):
        """Calculate stats for a period (Current vs Last Year) using the iterative modeled energy.

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

        while current <= end_date:
            if current < today:
                # Past: Use daily_history
                day_data = self._get_historical_day(current)
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

        while current <= ly_end:
            day_data = self._get_historical_day(current)
            days.append(day_data)
            current += timedelta(days=1)

        return days

    def _get_historical_day(self, date_obj: date) -> dict:
        """Extract day data from _daily_history and calculate Model value."""
        day_str = date_obj.isoformat()

        if day_str in self.coordinator._daily_history:
            entry = self.coordinator._daily_history[day_str]
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
            model_kwh, solar_kwh, _, _, _ = self.coordinator.calculate_modeled_energy(date_obj, date_obj)

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
            model_kwh, solar_kwh, _, _, _ = self.coordinator.calculate_modeled_energy(ly_date, ly_date)
            return {
                'date': date_obj.isoformat(),
                'temp': None,
                'wind': None,
                'wind_bucket': None,
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

    @property
    def native_value(self) -> float:
        now = dt_util.now()
        today = now.date()
        curr, last, _, _, _ = self._get_or_calculate_stats(today, "day", 1)
        return round(curr - last, 1)

    @property
    def extra_state_attributes(self):
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

            day_last = self._get_historical_day(ly_date)

            # Analyze
            analyzer = WeatherImpactAnalyzer(self.coordinator)
            analysis = analyzer.analyze_day(day_curr, day_last)

            # Format
            formatter = ExplanationFormatter()
            daily_summary = formatter.format_day_comparison(analysis)

        except Exception as e:
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

    @property
    def native_value(self) -> float:
        now = dt_util.now()
        today = now.date()
        start_week = today - timedelta(days=today.weekday())
        curr, last, _, _, _ = self._get_or_calculate_stats(start_week, "week", 7)
        return round(curr - last, 1)

    @property
    def extra_state_attributes(self):
        now = dt_util.now()
        today = now.date()
        start_week = today - timedelta(days=today.weekday())
        curr, last, actual, model, w_stats = self._get_or_calculate_stats(start_week, "week", 7)

        # Get ISO week number
        _, week_num, _ = now.isocalendar()

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
            last_year_days = self._build_last_year_period_days(ly_start, ly_end)
        except Exception as e:
            _LOGGER.warning(f"Failed to build period data for week comparison: {e}")
            # If we can't build the lists, use empty lists for hybrid calculation
            current_days = []
            last_year_days = []

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

        except Exception as e:
            _LOGGER.warning(f"Failed to generate explanation: {e}")
            # Fallback to existing logic (keep current implementation as backup)
            weekly_summary = self._generate_fallback_summary(curr, last)

        # Warning for missing data (as per PR feedback)
        if w_stats["ref_temp"] is None:
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

    @property
    def native_value(self) -> float:
        now = dt_util.now()
        today = now.date()
        start_month = today.replace(day=1)

        month = now.month
        year = now.year
        is_leap = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
        days_in_month = 31 if month in [1,3,5,7,8,10,12] else 30 if month in [4,6,9,11] else (29 if is_leap else 28)

        curr, last, _, _, _ = self._get_or_calculate_stats(start_month, "month", days_in_month)
        return round(curr - last, 1)

    @property
    def extra_state_attributes(self):
        now = dt_util.now()
        today = now.date()
        start_month = today.replace(day=1)

        month = now.month
        year = now.year
        is_leap = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
        days_in_month = 31 if month in [1,3,5,7,8,10,12] else 30 if month in [4,6,9,11] else (29 if is_leap else 28)

        curr, last, actual, model, w_stats = self._get_or_calculate_stats(start_month, "month", days_in_month)

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
            last_year_days = self._build_last_year_period_days(ly_start, ly_end)
        except Exception as e:
            _LOGGER.warning(f"Failed to build period data for month comparison: {e}")
            # If we can't build the lists, use empty lists for hybrid calculation
            current_days = []
            last_year_days = []

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

        except Exception as e:
            _LOGGER.warning(f"Failed to generate monthly explanation: {e}")
            monthly_summary = self._generate_fallback_summary(curr, last)

        if w_stats["ref_temp"] is None:
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
