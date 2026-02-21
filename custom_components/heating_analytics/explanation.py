"""Explanation module for Weather Impact Analysis.

Handles impact categorization, causality analysis, and natural language explanation generation.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, Any

from .const import (
    DEFAULT_TEMP_EXTREME,
    DEFAULT_TEMP_SIGNIFICANT,
    DEFAULT_TEMP_MODERATE,
    DEFAULT_WIND_RELEVANCE,
    DEFAULT_SOLAR_RELEVANCE,
    DEFAULT_CONTRADICTION_TEMP_DELTA,
    DEFAULT_CONTRADICTION_WIND_DELTA,
    DEFAULT_CONTRADICTION_SOLAR_KWH,
    DEFAULT_SOLAR_SIGNIFICANT_KWH,
    DEFAULT_SOLAR_MODERATE_KWH,
    DEFAULT_WIND_THRESHOLD,
    DEFAULT_EXTREME_WIND_THRESHOLD,
)

# Thresholds for categorization
# These match the design spec but can be adjusted later via config if needed.
class CategoryThresholds:
    """Configuration for impact categorization."""

    # Temperature impact (relative to baseline)
    TEMP_EXTREME = DEFAULT_TEMP_EXTREME
    TEMP_SIGNIFICANT = DEFAULT_TEMP_SIGNIFICANT
    TEMP_MODERATE = DEFAULT_TEMP_MODERATE

    # Relevance Thresholds (For Causality Check)
    WIND_RELEVANCE = DEFAULT_WIND_RELEVANCE
    SOLAR_RELEVANCE = DEFAULT_SOLAR_RELEVANCE

    # Contradiction Significance Thresholds (User specified / Display)
    CONTRADICTION_TEMP_DELTA = DEFAULT_CONTRADICTION_TEMP_DELTA
    CONTRADICTION_WIND_DELTA = DEFAULT_CONTRADICTION_WIND_DELTA
    CONTRADICTION_SOLAR_KWH = DEFAULT_CONTRADICTION_SOLAR_KWH

    # Solar impact (kWh delta)
    # Solar gain reduces consumption, so positive solar delta means LESS consumption
    # But usually we talk about "Sunny" (High solar) or "Cloudy" (Low solar)
    SOLAR_SIGNIFICANT_KWH = DEFAULT_SOLAR_SIGNIFICANT_KWH
    SOLAR_MODERATE_KWH = DEFAULT_SOLAR_MODERATE_KWH

    # Wind impact (bucket-based)
    # Using existing coordinator buckets: normal, high_wind, extreme_wind
    WIND_BUCKET_ORDER = ['normal', 'high_wind', 'extreme_wind']

    @staticmethod
    def get_wind_impact(current_bucket: str, baseline_bucket: str) -> str:
        """Calculate wind impact based on bucket change.

        Returns:
            'extreme': 2+ levels change (e.g. normal -> extreme_wind OR extreme_wind -> normal)
            'significant': 1 level change (e.g. normal -> high_wind OR high_wind -> normal)
            'moderate': same bucket but with significant speed difference (handled in analyzer)
            'normal': same bucket
        """
        if current_bucket == baseline_bucket:
            return 'normal'

        try:
            curr_idx = CategoryThresholds.WIND_BUCKET_ORDER.index(current_bucket)
            base_idx = CategoryThresholds.WIND_BUCKET_ORDER.index(baseline_bucket)
        except ValueError:
            # Unknown bucket (e.g. "with_auxiliary_heating"), treat as normal for now or ignore
            return 'normal'

        diff = curr_idx - base_idx
        abs_diff = abs(diff)

        if abs_diff >= 2:
            return 'extreme'
        elif abs_diff == 1:
            return 'significant'
        else:
            return 'normal'

    @staticmethod
    def get_combined_severity(temp_impact: str, wind_impact: str, solar_impact: str = 'normal') -> str:
        """Calculate combined severity score.

        Score mapping: extreme=3, significant=2, moderate=1, normal=0

        Returns:
            'severe': score >= 5 (e.g. extreme + significant)
            'challenging': score >= 3 (e.g. sig + sig, or ext + norm)
            'notable': score >= 1
            'normal': score 0
        """
        scores = {'extreme': 3, 'significant': 2, 'moderate': 1, 'normal': 0}

        t_score = scores.get(temp_impact, 0)
        w_score = scores.get(wind_impact, 0)
        s_score = scores.get(solar_impact, 0)

        total = t_score + w_score + s_score

        if total >= 5:
            return 'severe'
        elif total >= 3:
            return 'challenging'
        elif total >= 1:
            return 'notable'
        else:
            return 'normal'


class WeatherImpactAnalyzer:
    """Universal analyzer for weather impact on energy consumption."""

    def __init__(self, coordinator=None):
        self.coordinator = coordinator
        self.thresholds = CategoryThresholds()

    def analyze_day(self, day_data: Dict, baseline_data: Dict) -> Dict:
        """Analyze single day's weather impact vs baseline.

        Args:
            day_data: {temp, wind, wind_bucket, solar_kwh, kwh, ...}
            baseline_data: {temp, wind, solar_kwh, kwh, ...}

        Returns:
            Analysis dict with impacts and deltas.
        """
        # Temperature Analysis
        curr_temp = day_data.get('temp')
        base_temp = baseline_data.get('temp')

        temp_delta = 0.0
        temp_impact = 'normal'

        if curr_temp is not None and base_temp is not None:
            temp_delta = curr_temp - base_temp
            abs_delta = abs(temp_delta)

            if abs_delta >= self.thresholds.TEMP_EXTREME:
                temp_impact = 'extreme'
            elif abs_delta >= self.thresholds.TEMP_SIGNIFICANT:
                temp_impact = 'significant'
            elif abs_delta >= self.thresholds.TEMP_MODERATE:
                temp_impact = 'moderate'

        # Wind Analysis
        curr_wind = day_data.get('wind') or 0.0
        base_wind = baseline_data.get('wind') or 0.0
        wind_delta = curr_wind - base_wind

        curr_bucket = day_data.get('wind_bucket', 'normal')

        # If baseline bucket isn't provided, infer from wind speed using coordinator logic (if available)
        # or simplified fallback
        base_bucket = baseline_data.get('wind_bucket')
        if not base_bucket and self.coordinator:
            # Use coordinator to get bucket for baseline wind
            base_bucket = self.coordinator._get_wind_bucket(base_wind)
        elif not base_bucket:
             # Fallback if no coordinator attached (e.g. tests)
             # Mimic default thresholds using constants
             if base_wind >= DEFAULT_EXTREME_WIND_THRESHOLD: base_bucket = 'extreme_wind'
             elif base_wind >= DEFAULT_WIND_THRESHOLD: base_bucket = 'high_wind'
             else: base_bucket = 'normal'

        wind_impact = self.thresholds.get_wind_impact(curr_bucket, base_bucket)

        # Refine wind impact: if buckets are same but speed diff is large?
        # Let's say if bucket matches but speed is significantly higher (+2.5 m/s), treat as moderate
        if wind_impact == 'normal' and wind_delta >= 2.5:
             wind_impact = 'moderate'

        # Solar Analysis
        curr_solar = day_data.get('solar_kwh') or 0.0
        base_solar = baseline_data.get('solar_kwh') or 0.0
        solar_delta = curr_solar - base_solar
        abs_solar_delta = abs(solar_delta)

        solar_impact = 'normal'
        if abs_solar_delta >= self.thresholds.SOLAR_SIGNIFICANT_KWH:
            solar_impact = 'significant'
        elif abs_solar_delta >= self.thresholds.SOLAR_MODERATE_KWH:
            solar_impact = 'moderate'

        combined = self.thresholds.get_combined_severity(temp_impact, wind_impact, solar_impact)

        # kWh Delta
        curr_kwh = day_data.get('kwh') or 0.0
        base_kwh = baseline_data.get('kwh') or 0.0
        delta_kwh = curr_kwh - base_kwh

        # Check Causality (Did weather cause this?)
        causality = self.check_causality(delta_kwh, temp_delta, wind_delta, solar_delta)

        return {
            'date': day_data.get('date'),
            'temp_delta': temp_delta,
            'temp_impact': temp_impact,
            'wind_delta': wind_delta,
            'wind_impact': wind_impact,
            'solar_delta': solar_delta,
            'solar_impact': solar_impact,
            'combined_severity': combined,
            'kwh_delta': delta_kwh,
            'delta_kwh': delta_kwh, # Alias for consistency with period analysis
            'causality': causality
        }

    def check_causality(self, kwh_delta: float, temp_delta: float, wind_delta: float, solar_delta: float) -> Dict:
        """Check if weather changes explain consumption change."""
        # Logic:
        # Colder (negative temp_delta) -> Should increase consumption (positive kwh_delta)
        # Warmer (positive temp_delta) -> Should decrease consumption (negative kwh_delta)
        # More Wind (positive wind_delta) -> Should increase consumption
        # More Solar (positive solar_delta) -> Should decrease consumption (negative kwh_delta)

        temp_driver = False
        temp_contradicts = False

        if abs(temp_delta) >= self.thresholds.TEMP_MODERATE:
            if temp_delta < 0 and kwh_delta > 0: temp_driver = True
            elif temp_delta > 0 and kwh_delta < 0: temp_driver = True
            elif temp_delta > 0 and kwh_delta > 0: temp_contradicts = True
            elif temp_delta < 0 and kwh_delta < 0: temp_contradicts = True

        wind_driver = False
        wind_contradicts = False

        if abs(wind_delta) >= self.thresholds.WIND_RELEVANCE:
            if wind_delta > 0 and kwh_delta > 0: wind_driver = True
            elif wind_delta < 0 and kwh_delta < 0: wind_driver = True
            elif wind_delta < 0 and kwh_delta > 0: wind_contradicts = True
            elif wind_delta > 0 and kwh_delta < 0: wind_contradicts = True

        solar_driver = False
        solar_contradicts = False

        if abs(solar_delta) >= self.thresholds.SOLAR_RELEVANCE:
            if solar_delta < 0 and kwh_delta > 0: solar_driver = True # Less sun -> More energy
            elif solar_delta > 0 and kwh_delta < 0: solar_driver = True # More sun -> Less energy
            elif solar_delta > 0 and kwh_delta > 0: solar_contradicts = True
            elif solar_delta < 0 and kwh_delta < 0: solar_contradicts = True

        return {
            'temp_explains': temp_driver,
            'temp_contradicts': temp_contradicts,
            'wind_explains': wind_driver,
            'wind_contradicts': wind_contradicts,
            'solar_explains': solar_driver,
            'solar_contradicts': solar_contradicts
        }

    def analyze_period(self, period_days: List[Dict], baseline_days: List[Dict], context: str = 'week_ahead',
                      current_total_kwh: Optional[float] = None, last_year_total_kwh: Optional[float] = None) -> Dict:
        """Analyze entire period with aggregated insights."""

        total_kwh = current_total_kwh if current_total_kwh is not None else sum(d.get('kwh', 0.0) for d in period_days)
        base_kwh = last_year_total_kwh if last_year_total_kwh is not None else sum(d.get('kwh', 0.0) for d in baseline_days)
        delta_kwh = total_kwh - base_kwh
        delta_pct = (delta_kwh / base_kwh * 100) if base_kwh > 0 else 0.0

        day_counts = {'severe': 0, 'challenging': 0, 'notable': 0, 'normal': 0}

        # Drivers tracking
        driver_counts = {
            'temp': {'extreme': 0, 'significant': 0, 'moderate': 0},
            'wind': {'extreme': 0, 'significant': 0, 'moderate': 0},
            'solar': {'significant': 0, 'moderate': 0}
        }

        contrasts = []
        daily_analysis = []

        # Calculate aggregate weather deltas
        # Avoid division by zero
        p_len = len(period_days)
        b_len = len(baseline_days)
        valid_days = min(p_len, b_len)

        avg_temp_curr = sum((d.get('temp') or 0.0) for d in period_days[:valid_days]) / valid_days if valid_days > 0 else 0.0
        avg_temp_base = sum((d.get('temp') or 0.0) for d in baseline_days[:valid_days]) / valid_days if valid_days > 0 else 0.0
        period_temp_delta = avg_temp_curr - avg_temp_base

        avg_wind_curr = sum((d.get('wind') or 0.0) for d in period_days[:valid_days]) / valid_days if valid_days > 0 else 0.0
        avg_wind_base = sum((d.get('wind') or 0.0) for d in baseline_days[:valid_days]) / valid_days if valid_days > 0 else 0.0
        period_wind_delta = avg_wind_curr - avg_wind_base

        avg_solar_curr = sum((d.get('solar_kwh') or 0.0) for d in period_days[:valid_days]) / valid_days if valid_days > 0 else 0.0
        avg_solar_base = sum((d.get('solar_kwh') or 0.0) for d in baseline_days[:valid_days]) / valid_days if valid_days > 0 else 0.0
        period_solar_delta = avg_solar_curr - avg_solar_base

        count = valid_days

        for i in range(count):
            day = period_days[i]
            base = baseline_days[i]

            res = self.analyze_day(day, base)
            daily_analysis.append(res)

            # Severity Count
            sev = res['combined_severity']
            day_counts[sev] += 1

            # Driver Counting (only if it matches global trend direction)
            # FIX: Only count days that align with the global consumption trend
            day_delta_kwh = res.get('delta_kwh', 0.0)
            potential_drivers = []

            # Check if this day actually contributed to the period trend
            is_aligned = False
            if delta_kwh > 0 and day_delta_kwh > 0:
                 is_aligned = True
            elif delta_kwh < 0 and day_delta_kwh < 0:
                 is_aligned = True

            if is_aligned:
                if delta_kwh > 0:
                    # Using more: look for Cold, Wind, or Low Solar
                    if res['causality']['temp_explains']: potential_drivers.append(('temp', res['temp_impact']))
                    if res['causality']['wind_explains']: potential_drivers.append(('wind', res['wind_impact']))
                    if res['causality']['solar_explains']: potential_drivers.append(('solar', res['solar_impact']))
                else:
                    # Using less: look for Warm, Calm, or High Solar
                    if res['causality']['temp_explains']: potential_drivers.append(('temp', res['temp_impact']))
                    if res['causality']['wind_explains']: potential_drivers.append(('wind', res['wind_impact']))
                    if res['causality']['solar_explains']: potential_drivers.append(('solar', res['solar_impact']))

            if potential_drivers:
                # Sort by severity
                rank = {'extreme': 3, 'significant': 2, 'moderate': 1, 'normal': 0}
                potential_drivers.sort(key=lambda x: rank.get(x[1], 0), reverse=True)

                primary_factor, primary_impact = potential_drivers[0]
                self._increment_driver(driver_counts, primary_factor, primary_impact)

        # Structure Drivers List
        drivers_list = []

        # Summarize Factors
        for factor in ['temp', 'wind', 'solar']:
            count = sum(driver_counts[factor].values())
            if count > 0:
                imp = 'moderate'
                if driver_counts[factor].get('extreme', 0) > 0: imp = 'extreme'
                elif driver_counts[factor].get('significant', 0) > 0: imp = 'significant'

                drivers_list.append({
                    'factor': factor,
                    'impact': imp,
                    'affected_days': count,
                    'details': driver_counts[factor]
                })

        # Sort drivers by impact severity then count
        impact_rank = {'extreme': 3, 'significant': 2, 'moderate': 1}
        drivers_list.sort(key=lambda x: (impact_rank[x['impact']], x['affected_days']), reverse=True)

        # Variability
        variability = 'low'
        if day_counts['severe'] > 0: variability = 'high'
        elif day_counts['challenging'] > 0: variability = 'medium'
        elif day_counts['notable'] > 2: variability = 'medium'

        # Characterization (for summary text)
        characterization = "Similar to last year"
        if delta_pct > 5.0:
            if drivers_list:
                top_driver = drivers_list[0]
                top = top_driver['factor']
                count = top_driver['affected_days']

                # Check for contradiction with aggregate weather
                contradiction = False
                if top == 'temp':
                    # Driven by Cold, but Period is Warmer (or not clearly colder)
                    if period_temp_delta > -0.5: contradiction = True
                    characterization = "Significantly Colder"
                elif top == 'wind':
                    # Driven by Wind, but Period is Calmer
                    if period_wind_delta < 0.5: contradiction = True
                    characterization = "Windier period"
                elif top == 'solar':
                    # Driven by Cloud (low solar), but Period is Sunnier (high solar)
                    if period_solar_delta > 0.5: contradiction = True
                    characterization = "Cloudier period"
                else:
                    characterization = "Higher consumption"

                if contradiction:
                    day_word = "day" if count == 1 else "days"
                    characterization = f"{count} of {valid_days} {day_word} {'colder' if top == 'temp' else 'windier' if top == 'wind' else 'cloudier'}"
            else:
                characterization = "Higher consumption"
        elif delta_pct < -5.0:
            if drivers_list:
                top_driver = drivers_list[0]
                top = top_driver['factor']
                count = top_driver['affected_days']

                # Check for contradiction with aggregate weather
                contradiction = False
                if top == 'temp':
                    # Driven by Warmth, but Period is Colder
                    if period_temp_delta < 0.5: contradiction = True
                    characterization = "Significantly Warmer"
                elif top == 'wind':
                    # Driven by Calm, but Period is Windier
                    if period_wind_delta > -0.5: contradiction = True
                    characterization = "Calmer period"
                elif top == 'solar':
                    # Driven by Sun, but Period is Cloudier
                    if period_solar_delta < -0.5: contradiction = True
                    characterization = "Sunnier period"
                else:
                    characterization = "Lower consumption"

                if contradiction:
                    day_word = "day" if count == 1 else "days"
                    characterization = f"{count} of {valid_days} {day_word} {'warmer' if top == 'temp' else 'calmer' if top == 'wind' else 'sunnier'}"
            else:
                 characterization = "Lower consumption"

        # Overwrite characterization for forecast context if needed
        if context == 'week_ahead':
             if variability == 'high': characterization = "Variable week"
             elif variability == 'medium': characterization = "Challenging week"
             else: characterization = "Steady week"

        return {
            'total_kwh': round(total_kwh, 1),
            'baseline_kwh': round(base_kwh, 1),
            'delta_kwh': round(delta_kwh, 1),
            'delta_pct': round(delta_pct, 1),
            'day_counts': day_counts,
            'drivers': drivers_list,
            'variability': variability,
            'characterization': characterization,
            'daily_analysis': daily_analysis
        }

    def _increment_driver(self, counts, factor, impact):
        if factor in counts and impact in counts[factor]:
            counts[factor][impact] += 1


class ExplanationFormatter:
    """Generate human-readable explanations from analysis data."""

    def format_behavioral_deviation(self, deviation_kwh: float, deviation_pct: float,
                                    top_contributor: Optional[Dict], weather_impact: Optional[Dict],
                                    guest_impact_kwh: float = 0.0) -> str:
        """Format behavioral deviation (Actual vs Model) for Deviation Today sensor."""
        # Guest Mode Explanation Logic
        # Positive deviation (using more than expected)
        if deviation_kwh > 0.5 and guest_impact_kwh > 0.5:
            guest_ratio = guest_impact_kwh / deviation_kwh

            # Dominant guest impact (>50%)
            if guest_ratio > 0.5:
                return f"Usage is {deviation_kwh:.1f} kWh higher than expected, primarily due to guest heaters consuming {guest_impact_kwh:.1f} kWh."

            # Significant but not dominant (30-50%)
            elif guest_ratio > 0.3 and top_contributor:
                contrib_dev = top_contributor.get('deviation', 0.0)
                return f"Usage is {deviation_kwh:.1f} kWh higher. Guest heaters account for {guest_impact_kwh:.1f} kWh, with {top_contributor['name']} contributing {contrib_dev:+.1f} kWh."

        # Negative deviation but guest heaters are active
        elif deviation_kwh < -0.5 and guest_impact_kwh > 0.5:
            return f"Using {abs(deviation_kwh):.1f} kWh less than expected despite guest heaters consuming {guest_impact_kwh:.1f} kWh - excellent efficiency!"

        # Standard explanation (no significant guest impact)
        parts = []

        # 1. Main Statement
        if abs(deviation_kwh) <= 1.0:
            parts.append("Consumption matches expectations")
        elif deviation_kwh > 0:
            parts.append(f"Using {deviation_kwh:.1f} kWh ({deviation_pct:+.1f}%) more than typical")
        else:
            parts.append(f"Using {abs(deviation_kwh):.1f} kWh ({abs(deviation_pct):.1f}%) less than typical - good job!")

        # 2. Contributor Context (Only if using MORE)
        if deviation_kwh > 1.0 and top_contributor:
            contrib_dev = top_contributor.get('deviation', 0.0)
            if contrib_dev > 0.5:
                parts.append(f"mainly from {top_contributor['name']}")

        return " - ".join(parts)

    def format_week_ahead(self, analysis: Dict) -> str:
        """Format for Week Ahead sensor."""
        kwh = analysis['total_kwh']
        delta_pct = analysis['delta_pct']
        delta_kwh = analysis['delta_kwh']

        # 1. Characterization
        char = analysis['characterization']
        sign = "+" if delta_pct > 0 else ""

        # "Steady week: 150 kWh (+10% vs typical)."
        summary = f"{char}: {kwh:.0f} kWh ({sign}{delta_pct:.0f}% vs typical)."

        # 2. Drivers
        driver_text = self._build_driver_summary(analysis['day_counts'], analysis['drivers'], delta_kwh)
        if driver_text:
            summary += f" {driver_text}"

        return summary

    def format_period_comparison(self, analysis: Dict) -> str:
        """Format for Week/Month Comparison sensor (Unified Style)."""
        delta_kwh = analysis['delta_kwh']
        delta_pct = analysis['delta_pct']
        char = analysis['characterization']

        # 1. Main Summary
        # "Significantly Colder: +50 kWh (+20% vs last year)."
        sign = "+" if delta_kwh > 0 else ""

        if abs(delta_kwh) <= 5.0:
            return "Consumption similar to last year"

        summary = f"{char}: {sign}{delta_kwh:.0f} kWh ({sign}{delta_pct:.0f}% vs last year)."

        # 2. Drivers
        # "Driven by 2 cold days and 1 windy day."
        driver_text = self._build_driver_cause(analysis['drivers'], delta_kwh)
        if driver_text:
            summary += f" {driver_text}."

        return summary

    def format_day_comparison(self, analysis: Dict) -> str:
        """Format for Day Comparison sensor."""
        delta = analysis.get('delta_kwh', 0.0)

        if abs(delta) <= 3.0:
             return "Consumption similar to last year"

        causality = analysis.get('causality', {})
        drivers = []
        contradictions = []

        # Temp Analysis
        temp_imp = analysis.get('temp_impact')
        temp_delta = analysis.get('temp_delta', 0.0)

        if causality.get('temp_explains'):
            if delta > 0: drivers.append("extreme cold" if temp_imp == 'extreme' else "colder weather")
            else: drivers.append("extreme warmth" if temp_imp == 'extreme' else "warmer weather")
        elif causality.get('temp_contradicts'):
            # Check significance for contradiction
            if abs(temp_delta) >= CategoryThresholds.CONTRADICTION_TEMP_DELTA:
                if delta > 0: contradictions.append("warmer weather") # Usage UP, despite warmth
                else: contradictions.append("colder weather") # Usage DOWN, despite cold

        # Wind Analysis
        wind_imp = analysis.get('wind_impact')
        wind_delta = analysis.get('wind_delta', 0.0)

        if causality.get('wind_explains'):
            if delta > 0:
                if wind_imp == 'extreme': drivers.append("stormy weather")
                elif wind_imp == 'significant': drivers.append("high wind")
                else: drivers.append("windy weather")
            else:
                if wind_imp == 'extreme': drivers.append("very calm weather")
                else: drivers.append("calm weather")
        elif causality.get('wind_contradicts'):
            # Check significance
            if abs(wind_delta) >= CategoryThresholds.CONTRADICTION_WIND_DELTA:
                if delta > 0: contradictions.append("calmer weather") # Usage UP, despite calm
                else:
                    if wind_imp == 'extreme': contradictions.append("stormy weather")
                    elif wind_imp == 'significant': contradictions.append("high wind")
                    else: contradictions.append("windy weather") # Usage DOWN, despite wind

        # Solar Analysis
        solar_delta = analysis.get('solar_delta', 0.0)

        if causality.get('solar_explains'):
            solar_imp = analysis.get('solar_impact')
            if delta > 0: drivers.append("cloudier weather")
            else: drivers.append("sunny weather")
        elif causality.get('solar_contradicts'):
             # Check significance for contradiction
             if abs(solar_delta) >= CategoryThresholds.CONTRADICTION_SOLAR_KWH:
                 if delta > 0: contradictions.append("sunny weather") # Usage UP, despite sun
                 else: contradictions.append("cloudy weather") # Usage DOWN, despite clouds

        sign = "+" if delta > 0 else ""

        # Assemble string
        if not drivers:
             main_text = "Higher consumption" if delta > 0 else "Lower consumption"
        else:
             main_text = " + ".join(drivers)
             main_text = main_text[0].upper() + main_text[1:]

        # Append contradictions if any
        if contradictions:
            contra_text = " + ".join(contradictions)
            full_text = f"{main_text}, offset by {contra_text}"
        else:
            full_text = main_text

        return f"{full_text} ({sign}{delta:.1f} kWh vs last year)"

    def _build_driver_summary(self, day_counts: Dict, drivers: List[Dict], delta_kwh: float) -> str:
        """Build 'Driven by...' text for Week Ahead."""
        relevant_days = []
        if day_counts['severe'] > 0: relevant_days.append(f"{day_counts['severe']} severe days")
        if day_counts['challenging'] > 0: relevant_days.append(f"{day_counts['challenging']} challenging days")

        if not relevant_days:
            return ""

        day_text = " and ".join(relevant_days)

        reasons = []
        for d in drivers:
            reasons.append(self._get_factor_description(d['factor'], d['impact'], delta_kwh))

        reason_text = " + ".join(reasons)
        return f"Driven by {day_text} ({reason_text})."

    def _build_driver_cause(self, drivers: List[Dict], delta_kwh: float) -> str:
        """Build 'Driven by...' text for Comparison."""
        if not drivers:
            return ""

        parts = []
        for d in drivers:
            count = d['affected_days']
            desc = self._get_factor_description(d['factor'], d['impact'], delta_kwh)
            day_word = "day" if count == 1 else "days"
            parts.append(f"{count} {desc} {day_word}")

        return "Driven by " + " and ".join(parts)

    def _get_factor_description(self, factor, impact, delta_kwh):
        """Get description string for factor/impact."""
        if factor == 'temp':
            if delta_kwh > 0: return "extreme cold" if impact == 'extreme' else "cold"
            else: return "extreme warmth" if impact == 'extreme' else "warm"
        elif factor == 'wind':
            if delta_kwh > 0: return "stormy" if impact == 'extreme' else "windy"
            else: return "very calm" if impact == 'extreme' else "calm"
        elif factor == 'solar':
            if delta_kwh > 0: return "cloudy" # More usage -> Less sun
            else: return "sunny" # Less usage -> More sun
        return factor

    def format_comparison_summary(self, comparison: Dict) -> str:
        """Format summary for Period Comparison sensor.

        Deltas are P1 - P2 (positive = current period is higher).
        """
        p1 = comparison.get("period_1", {})
        p2 = comparison.get("period_2", {})
        delta_kwh = comparison.get("delta_actual_kwh")
        delta_temp = comparison.get("delta_temp")
        delta_wind = comparison.get("delta_wind")
        cross_kwh = comparison.get("actual_vs_reference_model_kwh")
        cross_pct = comparison.get("actual_vs_reference_model_pct")

        # Use cross-comparison if actuals are missing for reference period
        p2_actual = p2.get("actual_kwh")
        use_cross = (p2_actual is None or p2_actual == 0) and cross_kwh is not None

        if use_cross:
            headline_kwh = cross_kwh
            headline_pct = cross_pct
            label = "vs reference model"
        elif delta_kwh is not None:
            headline_kwh = delta_kwh
            p2_val = p2_actual if p2_actual and p2_actual > 0.1 else None
            headline_pct = round((delta_kwh / p2_val) * 100, 1) if p2_val else None
            label = "vs reference"
        else:
            return "Insufficient data for comparison"

        if abs(headline_kwh) < 1.0:
            return "Similar consumption between periods"

        sign = "+" if headline_kwh > 0 else ""
        summary = f"{sign}{headline_kwh:.1f} kWh"
        if headline_pct is not None:
            summary += f" ({sign}{headline_pct:.0f}%)"
        summary += f" {label}"

        # Weather context — neutral phrasing with signed values
        context = []
        if delta_temp is not None and abs(delta_temp) >= 1.0:
            context.append(f"temp {delta_temp:+.1f}°C")
        if delta_wind is not None and abs(delta_wind) >= 1.0:
            context.append(f"wind {delta_wind:+.1f} m/s")

        # Aux/solar — show per-period values for clarity
        p1_aux = p1.get("aux_impact_kwh", 0.0)
        p2_aux = p2.get("aux_impact_kwh", 0.0)
        if p1_aux > 0.5 or p2_aux > 0.5:
            context.append(f"aux savings {p1_aux:.0f} vs {p2_aux:.0f} kWh")

        p1_solar = p1.get("solar_impact_kwh", 0.0)
        p2_solar = p2.get("solar_impact_kwh", 0.0)
        if abs(p1_solar) > 0.5 or abs(p2_solar) > 0.5:
            context.append(f"solar {p1_solar:.1f} vs {p2_solar:.1f} kWh")

        if context:
            summary += ". " + ", ".join(context).capitalize()

        return summary

    def format_last_hour_summary(self, kwh: float, top_consumer_name: Optional[str], top_consumer_pct: Optional[float]) -> str:
        """Format summary for Last Hour Actual sensor."""
        if kwh <= 0: return "No consumption recorded"
        summary = f"{kwh:.1f} kWh consumed"
        if top_consumer_name and top_consumer_pct is not None:
            summary += f" - led by {top_consumer_name} ({top_consumer_pct:.0f}%)"
        return summary

    def format_forecast_weather_context(self, temp: Optional[float], wind: Optional[float],
                                        wind_high_threshold: Optional[float] = None,
                                        wind_extreme_threshold: Optional[float] = None) -> str:
        """Format absolute weather context for Forecast Today.

        Args:
            temp: Temperature in Celsius.
            wind: Wind speed in m/s.
            wind_high_threshold: Threshold for 'strong wind' (default: system default).
            wind_extreme_threshold: Threshold for 'stormy conditions' (default: system default).
        """
        # Set defaults if not provided (allows caller to override with config)
        if wind_high_threshold is None:
            wind_high_threshold = DEFAULT_WIND_THRESHOLD
        if wind_extreme_threshold is None:
            wind_extreme_threshold = DEFAULT_EXTREME_WIND_THRESHOLD

        if temp is None: return "Weather data unavailable"

        temp_category = "mild"
        if temp < -10: temp_category = "extreme cold"
        elif temp < 0: temp_category = "very cold"
        elif temp < 5: temp_category = "cold"
        elif temp < 12: temp_category = "chilly"
        elif temp > 22: temp_category = "hot"
        elif temp > 17: temp_category = "warm"

        wind_category = None
        if wind is not None:
            if wind >= wind_extreme_threshold: wind_category = "stormy conditions"
            elif wind >= wind_high_threshold: wind_category = "strong wind"
            elif wind >= 3.0: wind_category = "breezy conditions"

        is_significant = temp < 12 or (wind is not None and wind >= 3.0)
        if is_significant:
            parts = [f"{temp_category} ({temp:.1f}°C)"]
            if wind_category: parts.append(wind_category)
            return "Driven by " + " and ".join(parts)
        else:
            return f"Mild conditions ({temp:.1f}°C)"
