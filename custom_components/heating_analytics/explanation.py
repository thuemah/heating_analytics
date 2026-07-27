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

# Expected sign of the correlation between a weather delta and the resulting
# consumption delta, per thermal regime (#1051).
#
#   -1  the two move oppositely (colder -> more heating; less sun -> more heating)
#   +1  the two move together   (warmer -> more cooling; more sun -> more cooling)
# None no directional claim may be made
#
# One map serves both the causality test and the prose, so the two cannot
# drift into disagreeing about which direction counts as an explanation.
#
# Wind is None under cooling deliberately.  Physically it relieves cooling
# demand, but there is no signal to back a claim either way: per-unit cooling
# samples route to COOLING_WIND_BUCKET regardless of actual wind, so the
# cooling model is wind-independent by construction (see CLAUDE.md, and the
# matching exclusion from the cooling driver set in sensor.py).
#
# "mixed" carries no entry at all: with both regimes running at comparable
# scale, neither set of signs describes the building, so every directional
# claim is withheld rather than resolved to the larger side.  Magnitude
# categorisation is unaffected — it never depended on direction.
REGIME_WEATHER_CORRELATION: dict[str, dict[str, int | None]] = {
    "heating": {"temp": -1, "wind": +1, "solar": -1},
    "cooling": {"temp": +1, "wind": None, "solar": +1},
    "mixed": {"temp": None, "wind": None, "solar": None},
}

# Regime assumed when none can be resolved — an unrecorded historical day, an
# install with no coordinator attached, or an idle day.  Matches the
# integration's default-mode-is-heating convention and keeps pre-#1051
# behaviour intact.
DEFAULT_THERMAL_REGIME = "heating"


def _weather_correlation(regime: str | None, factor: str) -> int | None:
    """Expected weather/consumption correlation for a factor under a regime.

    Only ``"mixed"`` suppresses claims wholesale, and it does so because the
    building genuinely ran both ways.  Anything unrecognised — ``None``,
    ``"idle"``, a value from some future release — falls back to heating
    rather than muting the explanation layer.  Silently withholding every
    directional claim on an unknown label would be a failure mode that hides:
    the prose degrades to bare "Higher consumption" with nothing to indicate
    why.
    """
    if regime not in REGIME_WEATHER_CORRELATION:
        regime = DEFAULT_THERMAL_REGIME
    return REGIME_WEATHER_CORRELATION[regime].get(factor)


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

    def _resolve_regime(self, date_str: str | None) -> str | None:
        """Thermal regime of the day being explained, or None if unknown.

        The regime that matters is the one for the period being *described*,
        not the baseline it is measured against — the baseline is a reference
        point, not a thing whose consumption we are attributing.  That matters
        practically: the current side is always inside the hourly-log
        retention window, while a year-ago baseline usually is not.

        Falls back to live coordinator state for today, which has no
        ``daily_history`` entry until midnight.  Returns None — not a guess —
        for any other unrecorded day, and callers treat None as the default
        heating framing.
        """
        if not self.coordinator:
            return None

        regime = None
        if date_str:
            regime = self.coordinator.thermal_regime_for_day(date_str)

        if regime is None and date_str:
            try:
                from homeassistant.util import dt as dt_util

                if date_str == dt_util.now().date().isoformat():
                    regime = self.coordinator.thermal_regime
            except Exception:  # pragma: no cover - defensive, HA always present
                pass

        return regime

    def analyze_day(self, day_data: Dict, baseline_data: Dict) -> Dict:
        """Analyze single day's weather impact vs baseline.

        Args:
            day_data: {temp, wind, wind_bucket, solar_kwh, kwh, ...}
            baseline_data: {temp, wind, solar_kwh, kwh, ...}

        Returns:
            Analysis dict with impacts and deltas.
        """
        regime = self._resolve_regime(day_data.get('date'))
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
        base_bucket = baseline_data.get('wind_bucket')

        if baseline_data.get('wind') is None and not base_bucket:
            # A baseline day with no recorded weather carries wind=None and
            # wind_bucket=None (see sensors/comparison.py).  Inferring a bucket
            # from the `or 0.0` default above would fabricate a dead-calm
            # baseline and attribute the whole of the current day's wind as a
            # change — reporting "wind was significantly higher than last year"
            # when the truth is that last year has no data.  Mirror the
            # temperature path, which already requires both sides to be present.
            wind_delta = 0.0
            wind_impact = 'normal'
        else:
            # If baseline bucket isn't provided, infer from wind speed using coordinator logic (if available)
            # or simplified fallback
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
        causality = self.check_causality(
            delta_kwh, temp_delta, wind_delta, solar_delta, regime=regime
        )

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
            'causality': causality,
            'thermal_regime': regime,
        }

    def check_causality(
        self,
        kwh_delta: float,
        temp_delta: float,
        wind_delta: float,
        solar_delta: float,
        regime: str | None = None,
    ) -> Dict:
        """Check if weather changes explain consumption change.

        Direction is resolved per thermal regime via
        ``REGIME_WEATHER_CORRELATION``.  Under heating, colder / windier /
        darker raise consumption; under cooling, warmer and sunnier raise it
        while wind carries no defensible direction.  Under ``mixed`` — and
        for any factor whose correlation is ``None`` — no claim is made in
        either direction.

        Bit-identical to the pre-#1051 logic when ``regime`` resolves to
        heating, which is also what an unresolvable regime falls back to.
        """

        def _classify(delta: float, threshold: float, factor: str) -> tuple[bool, bool]:
            correlation = _weather_correlation(regime, factor)
            if correlation is None or abs(delta) < threshold:
                return False, False
            # Aligned when the weather moved in the direction that this regime
            # says produces the observed consumption change.
            if delta * correlation * kwh_delta > 0:
                return True, False
            if delta != 0 and kwh_delta != 0:
                return False, True
            return False, False

        temp_driver, temp_contradicts = _classify(
            temp_delta, self.thresholds.TEMP_MODERATE, "temp"
        )
        wind_driver, wind_contradicts = _classify(
            wind_delta, self.thresholds.WIND_RELEVANCE, "wind"
        )
        solar_driver, solar_contradicts = _classify(
            solar_delta, self.thresholds.SOLAR_RELEVANCE, "solar"
        )

        return {
            'temp_explains': temp_driver,
            'temp_contradicts': temp_contradicts,
            'wind_explains': wind_driver,
            'wind_contradicts': wind_contradicts,
            'solar_explains': solar_driver,
            'solar_contradicts': solar_contradicts
        }

    def analyze_period(self, period_days: List[Dict], baseline_days: List[Dict], context: str = 'week_ahead',
                      current_total_kwh: Optional[float] = None, last_year_total_kwh: Optional[float] = None,
                      current_basis: str = "actual", reference_basis: str = "actual") -> Dict:
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

        # Period regime: the days' own regimes, collapsed.  Two kinds of day
        # abstain rather than voting:
        #
        #   None  — no recorded regime, so no evidence either way.
        #   idle  — the building did no heating or cooling that day.  An idle
        #           day carries no sign convention to contribute, and letting
        #           it count would collapse the period to "mixed" and suppress
        #           the characterization for every other day: a single day
        #           away over a cold week would hide "Significantly Colder".
        #
        # Disagreement among the days that *do* carry a convention means the
        # period ran both ways, and no single set of signs describes it —
        # "mixed", which withholds directional claims.
        voting_regimes = {
            res.get('thermal_regime')
            for res in daily_analysis
            if res.get('thermal_regime') not in (None, 'idle')
        }
        if not voting_regimes:
            period_regime = None
        elif len(voting_regimes) == 1:
            period_regime = voting_regimes.pop()
        else:
            period_regime = "mixed"

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
                    'details': driver_counts[factor],
                    # Aggregate weather delta for this factor, so the prose can
                    # name the weather from its own sign rather than inferring
                    # it from the consumption delta (which only works under
                    # heating).  See format_day_comparison for the same fix.
                    'weather_delta': {
                        'temp': period_temp_delta,
                        'wind': period_wind_delta,
                        'solar': period_solar_delta,
                    }[factor],
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
        #
        # The direction a driver is *expected* to point is `correlation *
        # consumption direction`, resolved per regime.  Under heating with
        # consumption up that reproduces the original hard-coded framing
        # exactly (cold, windier, cloudier); under cooling it flips to warmth
        # and sun, and wind drops out entirely because it carries no
        # defensible direction there.
        #
        # The contradiction margins are deliberately NOT unified: temp and
        # wind require the expected direction to be clearly present (0.5),
        # while solar only objects when the opposite direction is clearly
        # present (-0.5).  That asymmetry predates this change and is
        # preserved verbatim rather than tidied — normalising it would move
        # heating-install output, which is out of scope here.
        contradiction_margin = {'temp': 0.5, 'wind': 0.5, 'solar': -0.5}
        period_deltas = {
            'temp': period_temp_delta,
            'wind': period_wind_delta,
            'solar': period_solar_delta,
        }
        # Long form for the headline, short form for the "N of M days" fallback.
        direction_words = {
            ('temp', -1): ("Significantly Colder", "colder"),
            ('temp', +1): ("Significantly Warmer", "warmer"),
            ('wind', -1): ("Calmer period", "calmer"),
            ('wind', +1): ("Windier period", "windier"),
            ('solar', -1): ("Cloudier period", "cloudier"),
            ('solar', +1): ("Sunnier period", "sunnier"),
        }

        characterization = "Similar to last year"
        if abs(delta_pct) > 5.0:
            fallback = "Higher consumption" if delta_pct > 0 else "Lower consumption"
            characterization = fallback

            if drivers_list:
                top_driver = drivers_list[0]
                top = top_driver['factor']
                count = top_driver['affected_days']

                consumption_direction = 1 if delta_pct > 0 else -1
                correlation = _weather_correlation(period_regime, top)

                if correlation is not None and top in period_deltas:
                    expected_sign = correlation * consumption_direction
                    long_word, short_word = direction_words[(top, expected_sign)]
                    characterization = long_word

                    # Contradiction: the aggregate weather does not actually
                    # move the way the per-day drivers claim.
                    if period_deltas[top] * expected_sign < contradiction_margin[top]:
                        day_word = "day" if count == 1 else "days"
                        characterization = (
                            f"{count} of {valid_days} {day_word} {short_word}"
                        )

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
            'daily_analysis': daily_analysis,
            'current_basis': current_basis,
            'reference_basis': reference_basis,
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
        reference_basis = analysis.get('reference_basis', 'actual')

        # Label reflects what the reference actually is
        if reference_basis == "actual":
            ref_label = "vs last year"
            similar_label = "Consumption similar to last year"
        elif reference_basis == "hybrid":
            ref_label = "vs last year (partial)"
            similar_label = "Consumption similar to last year (partial data)"
        else:  # "modeled"
            ref_label = "vs last year's model"
            similar_label = "Consumption similar to last year's model"

        sign = "+" if delta_kwh > 0 else ""

        if abs(delta_kwh) <= 5.0:
            return similar_label

        summary = f"{char}: {sign}{delta_kwh:.0f} kWh ({sign}{delta_pct:.0f}% {ref_label})."

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

        # Wording describes the *weather*, so it is derived from the weather
        # delta's own sign rather than from the consumption delta.  The
        # weather does not change meaning with the regime — only whether it
        # explains the consumption does, and that is already settled by
        # `causality`.  Under heating this is bit-identical to deriving from
        # `delta`, because the causality rule only fires when the two signs
        # correspond; under cooling, deriving from `delta` would have called
        # a hot day "colder weather".
        colder = temp_delta < 0

        if causality.get('temp_explains'):
            if colder: drivers.append("extreme cold" if temp_imp == 'extreme' else "colder weather")
            else: drivers.append("extreme warmth" if temp_imp == 'extreme' else "warmer weather")
        elif causality.get('temp_contradicts'):
            # Check significance for contradiction
            if abs(temp_delta) >= CategoryThresholds.CONTRADICTION_TEMP_DELTA:
                if colder: contradictions.append("colder weather")
                else: contradictions.append("warmer weather")

        # Wind Analysis
        wind_imp = analysis.get('wind_impact')
        wind_delta = analysis.get('wind_delta', 0.0)

        windier = wind_delta > 0

        def _windy_phrase() -> str:
            if wind_imp == 'extreme': return "stormy weather"
            if wind_imp == 'significant': return "high wind"
            return "windy weather"

        if causality.get('wind_explains'):
            if windier: drivers.append(_windy_phrase())
            else: drivers.append("very calm weather" if wind_imp == 'extreme' else "calm weather")
        elif causality.get('wind_contradicts'):
            # Check significance
            if abs(wind_delta) >= CategoryThresholds.CONTRADICTION_WIND_DELTA:
                # Comparative form on the calm side reads better in the
                # "offset by ..." clause, and is the pre-#1051 wording.
                if windier: contradictions.append(_windy_phrase())
                else: contradictions.append("calmer weather")

        # Solar Analysis
        solar_delta = analysis.get('solar_delta', 0.0)
        sunnier = solar_delta > 0

        if causality.get('solar_explains'):
            drivers.append("sunny weather" if sunnier else "cloudier weather")
        elif causality.get('solar_contradicts'):
             # Check significance for contradiction
             if abs(solar_delta) >= CategoryThresholds.CONTRADICTION_SOLAR_KWH:
                 contradictions.append("sunny weather" if sunnier else "cloudy weather")

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
            reasons.append(self._get_factor_description(
                d['factor'], d['impact'], delta_kwh, d.get('weather_delta')
            ))

        reason_text = " + ".join(reasons)
        return f"Driven by {day_text} ({reason_text})."

    def _build_driver_cause(self, drivers: List[Dict], delta_kwh: float) -> str:
        """Build 'Driven by...' text for Comparison."""
        if not drivers:
            return ""

        parts = []
        for d in drivers:
            count = d['affected_days']
            desc = self._get_factor_description(
                d['factor'], d['impact'], delta_kwh, d.get('weather_delta')
            )
            day_word = "day" if count == 1 else "days"
            parts.append(f"{count} {desc} {day_word}")

        return "Driven by " + " and ".join(parts)

    def _get_factor_description(self, factor, impact, delta_kwh, weather_delta=None):
        """Get description string for factor/impact.

        Names the weather from its own aggregate delta when available.  Under
        heating that is equivalent to reading it off ``delta_kwh`` — the
        causality rule only counts a factor as a driver when the two signs
        correspond — but under cooling the consumption delta points the other
        way, and inferring from it would describe a hot week as "cold".

        ``weather_delta=None`` keeps the legacy consumption-delta inference
        for any caller that has not been updated.
        """
        if weather_delta is None:
            colder = delta_kwh > 0
            windier = delta_kwh > 0
            sunnier = delta_kwh <= 0
        else:
            colder = weather_delta < 0
            windier = weather_delta > 0
            sunnier = weather_delta > 0

        if factor == 'temp':
            if colder: return "extreme cold" if impact == 'extreme' else "cold"
            else: return "extreme warmth" if impact == 'extreme' else "warm"
        elif factor == 'wind':
            if windier: return "stormy" if impact == 'extreme' else "windy"
            else: return "very calm" if impact == 'extreme' else "calm"
        elif factor == 'solar':
            if sunnier: return "sunny"
            else: return "cloudy"
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

        # Use cross-comparison (P1 actual vs P2 modeled) when reference has no measurements
        p2_basis = comparison.get("period_2_basis", "actual")
        use_cross = p2_basis != "actual" and cross_kwh is not None

        if use_cross:
            headline_kwh = cross_kwh
            headline_pct = cross_pct
            label = "vs reference model" if p2_basis == "modeled" else "vs reference (partial)"
        elif delta_kwh is not None:
            headline_kwh = delta_kwh
            p2_actual = p2.get("actual_kwh")
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
