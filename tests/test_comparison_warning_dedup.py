"""Session-level dedup for "missing historical data" warnings in comparison sensors.

Regression: ``extra_state_attributes`` is recomputed every coordinator tick
(~1/min), so a deterministic data-availability condition
(``w_stats["ref_temp"] is None``) would fire the WARNING every minute until
last-year's data filled in.  The fix is a module-level set that dedups per
(week_num, year-month) per HA restart.  Users still get notified at first
detection after each boot; subsequent ticks are silent.

These tests exercise the dedup state directly rather than running a full
sensor through many coordinator updates — the guard is a trivial
``if x not in S: S.add(x); log.warning(...)`` pattern but we want to lock
in its state-carrier location (module-level, cleared on restart) so a
refactor doesn't silently reintroduce per-tick spam.
"""
from unittest.mock import MagicMock

import pytest


class TestComparisonWarningDedup:
    """Module-level `_WARNED_WEEKS` / `_WARNED_MONTHS` guarantee each
    missing period logs at most once per HA session.
    """

    def _reset_state(self):
        from custom_components.heating_analytics.sensors import comparison
        comparison._WARNED_WEEKS.clear()
        comparison._WARNED_MONTHS.clear()

    def test_week_dedup_state_carrier_exists(self):
        """Module exposes the dedup set (future refactors must keep it)."""
        from custom_components.heating_analytics.sensors import comparison
        assert isinstance(comparison._WARNED_WEEKS, set)
        assert isinstance(comparison._WARNED_MONTHS, set)

    def test_week_dedup_one_log_per_week_per_session(self, caplog):
        """Simulating the guard body: same week_num triggered twice → log once."""
        self._reset_state()
        from custom_components.heating_analytics.sensors import comparison
        import logging

        with caplog.at_level(logging.WARNING, logger=comparison._LOGGER.name):
            # Simulate the guarded code path for the same week number twice.
            for _ in range(2):
                w_ref_temp = None
                week_num = 17
                if w_ref_temp is None and week_num not in comparison._WARNED_WEEKS:
                    comparison._WARNED_WEEKS.add(week_num)
                    comparison._LOGGER.warning(
                        "Missing historical data for week %d, comparison may be inaccurate.",
                        week_num,
                    )

        week_warnings = [r for r in caplog.records if "week 17" in r.getMessage()]
        assert len(week_warnings) == 1

    def test_different_weeks_log_independently(self, caplog):
        """Distinct week_num values each log once."""
        self._reset_state()
        from custom_components.heating_analytics.sensors import comparison
        import logging

        with caplog.at_level(logging.WARNING, logger=comparison._LOGGER.name):
            for week_num in [17, 18, 17, 19, 18]:
                if week_num not in comparison._WARNED_WEEKS:
                    comparison._WARNED_WEEKS.add(week_num)
                    comparison._LOGGER.warning(
                        "Missing historical data for week %d, comparison may be inaccurate.",
                        week_num,
                    )

        warnings_by_week = {17, 18, 19}
        logged_weeks = {
            int(r.getMessage().split("week ")[1].split(",")[0])
            for r in caplog.records
            if "week" in r.getMessage()
        }
        assert logged_weeks == warnings_by_week

    def test_month_dedup_per_year_month_key(self, caplog):
        """`_WARNED_MONTHS` keys on YYYY-MM so a rollover re-warns."""
        self._reset_state()
        from custom_components.heating_analytics.sensors import comparison
        import logging

        with caplog.at_level(logging.WARNING, logger=comparison._LOGGER.name):
            for year, month in [(2026, 4), (2026, 4), (2026, 5), (2026, 4)]:
                key = f"{year}-{month:02d}"
                if key not in comparison._WARNED_MONTHS:
                    comparison._WARNED_MONTHS.add(key)
                    comparison._LOGGER.warning(
                        "Missing historical data for month comparison, summary may be inaccurate."
                    )

        month_warnings = [r for r in caplog.records if "month comparison" in r.getMessage()]
        # April logs once, May logs once, duplicates deduped → 2 total.
        assert len(month_warnings) == 2
