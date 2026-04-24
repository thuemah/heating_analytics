"""Tests for diagnose_model.base_model_health (dark-hour replay).

For each (temp_key, wind_bucket) pair in the stored correlation model, we
replay dark (solar_factor < 0.05) non-aux all-heating hours from the hourly
log and compare the empirical mean actual_kwh to the stored bucket value.
A stored bucket that sits significantly above the dark-hour mean is flagged
inflated — the classic signature of solar-contamination in winter learning
that the user ran into manually and corrected by hand.

Independent signal from solar diagnostics: dark hours have no solar impact
to normalise, so the empirical mean is a ground-truth the solar-learning
loop cannot have biased.
"""
from unittest.mock import MagicMock

import pytest

from custom_components.heating_analytics.coordinator import HeatingDataCoordinator
from custom_components.heating_analytics.diagnostics import DiagnosticsEngine


def _base_entry(ts, *, temp_key="10", wind_bucket="normal", actual_kwh=0.4,
                solar_factor=0.0, aux=False, unit_modes=None):
    """Build an hourly-log entry with the fields diagnose_model reads."""
    return {
        "timestamp": ts,
        "temp_key": temp_key,
        "wind_bucket": wind_bucket,
        "actual_kwh": actual_kwh,
        "solar_factor": solar_factor,
        "auxiliary_active": aux,
        "unit_modes": unit_modes or {"sensor.heater1": "heating"},
        "unit_breakdown": {"sensor.heater1": actual_kwh},
        "expected_kwh": actual_kwh,
    }


def _make_coord(correlation_data, hourly_log, daily_learning_mode=False,
                daily_history=None, track_c_enabled=False,
                mpc_managed_sensor=None):
    """Minimal coord mock for calling diagnose_model as unbound method.

    Track C wiring defaults to off so existing tests stay on the raw-actual
    dark-replay path.  Set ``track_c_enabled=True`` + ``mpc_managed_sensor``
    + ``daily_history`` with a ``track_c_distribution`` per day to exercise
    the Track C-aware reference path.
    """
    coord = MagicMock()
    coord._correlation_data = correlation_data
    coord._hourly_log = hourly_log
    coord._learning_buffer_global = {}
    coord._daily_history = daily_history or {}
    coord.daily_learning_mode = daily_learning_mode
    coord.track_c_enabled = track_c_enabled
    coord.mpc_managed_sensor = mpc_managed_sensor
    return coord


def _dark_hours(temp_key, wind_bucket, actual_kwh, n=15, start="2026-04-10"):
    """Generate n dark qualifying hours for a bucket."""
    from datetime import datetime, timedelta
    base_dt = datetime.fromisoformat(start + "T00:00:00")
    return [
        _base_entry(
            (base_dt + timedelta(hours=i)).isoformat(),
            temp_key=temp_key, wind_bucket=wind_bucket,
            actual_kwh=actual_kwh, solar_factor=0.0,
        )
        for i in range(n)
    ]


class TestBaseModelHealthInflation:
    """Inflated bucket: stored > dark-hour actual mean."""

    def test_inflated_bucket_flagged(self):
        """Stored 0.40, dark-hour mean 0.25 → flagged inflated at +60%."""
        stored = {"10": {"normal": 0.40}}
        log = _dark_hours("10", "normal", actual_kwh=0.25, n=15)
        coord = _make_coord(stored, log)
        result = DiagnosticsEngine(coord).diagnose_model(days_back=30)

        health = result["base_model_health"]
        bucket = health["buckets"]["10"]["normal"]
        assert bucket["stored_kwh"] == pytest.approx(0.40)
        assert bucket["actual_dark_mean_kwh"] == pytest.approx(0.25)
        assert bucket["verdict"] == "inflated"
        assert bucket["delta_pct"] == pytest.approx(60.0, abs=0.5)
        assert health["summary"]["inflated_count"] == 1

    def test_deflated_bucket_flagged(self):
        """Stored 0.10, dark-hour mean 0.40 → flagged deflated."""
        stored = {"5": {"normal": 0.10}}
        log = _dark_hours("5", "normal", actual_kwh=0.40, n=15)
        coord = _make_coord(stored, log)
        result = DiagnosticsEngine(coord).diagnose_model(days_back=30)

        bucket = result["base_model_health"]["buckets"]["5"]["normal"]
        assert bucket["verdict"] == "deflated"
        assert bucket["delta_pct"] == pytest.approx(-75.0, abs=0.5)

    def test_ok_bucket_within_threshold(self):
        """Stored 0.30, dark-hour mean 0.32 (~6% off) → verdict ok."""
        stored = {"10": {"normal": 0.30}}
        log = _dark_hours("10", "normal", actual_kwh=0.32, n=15)
        coord = _make_coord(stored, log)
        result = DiagnosticsEngine(coord).diagnose_model(days_back=30)

        bucket = result["base_model_health"]["buckets"]["10"]["normal"]
        assert bucket["verdict"] == "ok"
        assert result["base_model_health"]["summary"]["inflated_count"] == 0
        assert result["base_model_health"]["summary"]["deflated_count"] == 0


class TestBaseModelHealthUnverifiable:
    """Insufficient dark-hour sample → unverifiable verdict."""

    def test_few_dark_hours_unverifiable(self):
        stored = {"10": {"normal": 0.40}}
        log = _dark_hours("10", "normal", actual_kwh=0.25, n=5)  # under threshold
        coord = _make_coord(stored, log)
        result = DiagnosticsEngine(coord).diagnose_model(days_back=30)

        bucket = result["base_model_health"]["buckets"]["10"]["normal"]
        assert bucket["verdict"] == "unverifiable"
        assert bucket["n_dark_hours"] == 5
        assert "actual_dark_mean_kwh" not in bucket
        # Inflation NOT flagged because we can't verify
        assert result["base_model_health"]["summary"]["inflated_count"] == 0
        assert result["base_model_health"]["summary"]["unverifiable_count"] == 1


class TestBaseModelHealthFiltering:
    """Replay correctly excludes contaminated sources."""

    def test_solar_hours_excluded(self):
        """Hours with solar_factor >= threshold are excluded even if count is high."""
        stored = {"10": {"normal": 0.40}}
        # 20 hours at high-solar-factor with suppressed actual (would otherwise
        # inflate inflation signal).  No dark hours → unverifiable.
        log = [
            _base_entry(f"2026-04-{(i // 24) + 10:02d}T{i % 24:02d}:00:00",
                        temp_key="10", wind_bucket="normal",
                        actual_kwh=0.10, solar_factor=0.5)
            for i in range(20)
        ]
        coord = _make_coord(stored, log)
        result = DiagnosticsEngine(coord).diagnose_model(days_back=30)

        bucket = result["base_model_health"]["buckets"]["10"]["normal"]
        assert bucket["verdict"] == "unverifiable"
        assert bucket["n_dark_hours"] == 0

    def test_aux_hours_excluded(self):
        """Hours with auxiliary_active=True are excluded."""
        stored = {"10": {"normal": 0.40}}
        log = [
            _base_entry(f"2026-04-{(i // 24) + 10:02d}T{i % 24:02d}:00:00",
                        temp_key="10", wind_bucket="normal",
                        actual_kwh=0.25, solar_factor=0.0, aux=True)
            for i in range(20)
        ]
        coord = _make_coord(stored, log)
        result = DiagnosticsEngine(coord).diagnose_model(days_back=30)

        bucket = result["base_model_health"]["buckets"]["10"]["normal"]
        assert bucket["verdict"] == "unverifiable"

    def test_excluded_mode_hours_filtered(self):
        """DHW / guest / off mode contamination excludes the hour."""
        stored = {"10": {"normal": 0.40}}
        # 15 hours with DHW mode contamination — should NOT count toward dark-hour mean
        log = [
            _base_entry(f"2026-04-{(i // 24) + 10:02d}T{i % 24:02d}:00:00",
                        temp_key="10", wind_bucket="normal",
                        actual_kwh=0.25, solar_factor=0.0,
                        unit_modes={"sensor.heater1": "heating",
                                    "sensor.heater2": "dhw"})
            for i in range(15)
        ]
        coord = _make_coord(stored, log)
        result = DiagnosticsEngine(coord).diagnose_model(days_back=30)

        bucket = result["base_model_health"]["buckets"]["10"]["normal"]
        assert bucket["verdict"] == "unverifiable"
        assert bucket["n_dark_hours"] == 0


class TestBaseModelHealthMultipleBuckets:
    """Realistic scenario reproducing the user's manual calibration case."""

    def test_contamination_pattern_matches_user_case(self):
        """Cold buckets clean, mild buckets inflated — mirrors observed Glob_nå vs Glob_ber.

        At 6-7 °C the stored value matches dark-hour reality; at 11-15 °C
        the stored is 50-80% above the dark-hour reality because winter
        solar contamination inflated mild buckets.  Verifies the flags
        surface exactly the contaminated buckets, not the clean ones.
        """
        stored = {
            "6":  {"normal": 0.56},  # matches reality
            "7":  {"normal": 0.50},  # matches reality
            "11": {"normal": 0.40},  # inflated: real is 0.26
            "12": {"normal": 0.33},  # inflated: real is 0.21
            "15": {"normal": 0.14},  # inflated: real is 0.08 (83%)
        }
        log = (
            _dark_hours("6",  "normal", 0.56, n=20, start="2026-04-01") +
            _dark_hours("7",  "normal", 0.50, n=20, start="2026-04-02") +
            _dark_hours("11", "normal", 0.26, n=15, start="2026-04-03") +
            _dark_hours("12", "normal", 0.21, n=15, start="2026-04-04") +
            _dark_hours("15", "normal", 0.08, n=15, start="2026-04-05")
        )
        coord = _make_coord(stored, log)
        result = DiagnosticsEngine(coord).diagnose_model(days_back=30)

        health = result["base_model_health"]
        # Clean buckets
        assert health["buckets"]["6"]["normal"]["verdict"] == "ok"
        assert health["buckets"]["7"]["normal"]["verdict"] == "ok"
        # Contaminated buckets
        assert health["buckets"]["11"]["normal"]["verdict"] == "inflated"
        assert health["buckets"]["12"]["normal"]["verdict"] == "inflated"
        assert health["buckets"]["15"]["normal"]["verdict"] == "inflated"
        assert health["summary"]["inflated_count"] == 3
        # Flags sorted worst-first → 15 °C (~75% inflated) before 11 °C (~54%)
        flag_temps = [f["temp_key"] for f in health["flags"]["inflated"]]
        assert flag_temps[0] == "15"


class TestBaseModelHealthConfig:
    """Config block reflects constants used for the analysis."""

    def test_config_block_present(self):
        coord = _make_coord({"10": {"normal": 0.30}}, [])
        result = DiagnosticsEngine(coord).diagnose_model(days_back=30)
        cfg = result["base_model_health"]["config"]
        assert cfg["dark_solar_factor_threshold"] == 0.05
        assert cfg["min_dark_hours_for_verdict"] == 10
        assert cfg["bucket_deviation_threshold_pct"] == 15.0


# -----------------------------------------------------------------------------
# Track C-aware dark replay
# -----------------------------------------------------------------------------

def _track_c_dist(synthetic_kwh_per_hour: dict[int, float]) -> list[dict]:
    """Build a track_c_distribution-shaped list for the given hours."""
    return [
        {
            "datetime": f"2026-04-15T{h:02d}:00:00",
            "synthetic_kwh_el": kwh,
        }
        for h, kwh in synthetic_kwh_per_hour.items()
    ]


def _track_c_entry(hour, *, temp_key="10", wind_bucket="normal", actual_kwh=0.05,
                   solar_factor=0.0, mpc_sid="sensor.heater_mpc",
                   non_mpc_breakdown=None):
    """Hourly entry for a Track C day on 2026-04-15."""
    breakdown = {mpc_sid: actual_kwh}
    if non_mpc_breakdown:
        breakdown.update(non_mpc_breakdown)
    return {
        "timestamp": f"2026-04-15T{hour:02d}:00:00",
        "hour": hour,
        "temp_key": temp_key,
        "wind_bucket": wind_bucket,
        "actual_kwh": actual_kwh + sum((non_mpc_breakdown or {}).values()),
        "solar_factor": solar_factor,
        "auxiliary_active": False,
        "unit_modes": {mpc_sid: "heating", **{
            sid: "heating" for sid in (non_mpc_breakdown or {})
        }},
        "unit_breakdown": breakdown,
        "expected_kwh": 0.0,
    }


class TestTrackCAwareDarkReplay:
    """When Track C is enabled and a distribution exists, dark-replay uses
    synthetic_kwh_el for the MPC sensor instead of raw actual_kwh.

    The Antwerpen scenario: stored bucket is written from MPC's thermal/COP
    synthesis; comparing against a partial electrical meter would produce
    spurious "inflated" verdicts.  This is what motivated the fix.
    """

    def test_no_track_c_uses_raw_actual_unchanged(self):
        """track_c_enabled=False → behaviour identical to pre-fix."""
        stored = {"10": {"normal": 0.40}}
        log = _dark_hours("10", "normal", actual_kwh=0.25, n=15)
        coord = _make_coord(stored, log)  # track_c_enabled=False default
        result = DiagnosticsEngine(coord).diagnose_model(days_back=30)
        bucket = result["base_model_health"]["buckets"]["10"]["normal"]
        assert bucket["actual_dark_mean_kwh"] == pytest.approx(0.25)
        assert bucket["track_c_aware_hours"] == 0

    def test_track_c_uses_synthetic_when_distribution_present(self):
        """Stored 0.50 vs synthetic 0.50 → ok (not inflated despite raw=0.05)."""
        mpc = "sensor.heater_mpc"
        # 12 dark hours on 2026-04-15, all at temp=10, with low raw electrical
        # (sensor partial) but synthetic kWh from MPC distribution at 0.50
        log = [_track_c_entry(h, actual_kwh=0.05, mpc_sid=mpc) for h in range(12)]
        daily_history = {
            "2026-04-15": {
                "track_c_distribution": _track_c_dist({h: 0.50 for h in range(12)}),
            }
        }
        stored = {"10": {"normal": 0.50}}
        coord = _make_coord(stored, log, daily_history=daily_history,
                             track_c_enabled=True, mpc_managed_sensor=mpc)
        result = DiagnosticsEngine(coord).diagnose_model(days_back=30)

        bucket = result["base_model_health"]["buckets"]["10"]["normal"]
        assert bucket["track_c_aware_hours"] == 12
        # Dark mean uses synthetic (0.50), not raw (0.05) → ok verdict
        assert bucket["actual_dark_mean_kwh"] == pytest.approx(0.50)
        assert bucket["verdict"] == "ok"
        # Crucially: the 900%+ inflated verdict pre-fix would have fired here
        assert result["base_model_health"]["summary"]["inflated_count"] == 0

    def test_track_c_falls_back_to_raw_when_distribution_missing(self):
        """Track B fallback day (no track_c_distribution stored) → raw actual."""
        mpc = "sensor.heater_mpc"
        log = [_track_c_entry(h, actual_kwh=0.40, mpc_sid=mpc) for h in range(15)]
        # No track_c_distribution → Track B fallback for this day
        daily_history = {"2026-04-15": {"track_c_kwh": 5.0}}
        stored = {"10": {"normal": 0.50}}
        coord = _make_coord(stored, log, daily_history=daily_history,
                             track_c_enabled=True, mpc_managed_sensor=mpc)
        result = DiagnosticsEngine(coord).diagnose_model(days_back=30)

        bucket = result["base_model_health"]["buckets"]["10"]["normal"]
        # Raw actual was 0.40 → mean 0.40 → close to stored 0.50 (25 % gap)
        assert bucket["actual_dark_mean_kwh"] == pytest.approx(0.40)
        assert bucket["track_c_aware_hours"] == 0  # nothing used Track C path

    def test_track_c_adds_non_mpc_sensor_contributions(self):
        """For multi-unit installs: synthetic (MPC) + raw actual (non-MPC)."""
        mpc = "sensor.heater_mpc"
        # Each hour: MPC contributes synthetic 0.30, non-MPC contributes raw 0.10
        log = [
            _track_c_entry(h, actual_kwh=0.05, mpc_sid=mpc,
                           non_mpc_breakdown={"sensor.heater_aux": 0.10})
            for h in range(12)
        ]
        daily_history = {
            "2026-04-15": {
                "track_c_distribution": _track_c_dist({h: 0.30 for h in range(12)}),
            }
        }
        stored = {"10": {"normal": 0.40}}  # = synthetic 0.30 + non-MPC 0.10
        coord = _make_coord(stored, log, daily_history=daily_history,
                             track_c_enabled=True, mpc_managed_sensor=mpc)
        result = DiagnosticsEngine(coord).diagnose_model(days_back=30)

        bucket = result["base_model_health"]["buckets"]["10"]["normal"]
        # synthetic + non-MPC = 0.30 + 0.10 = 0.40 → ok verdict
        assert bucket["actual_dark_mean_kwh"] == pytest.approx(0.40)
        assert bucket["verdict"] == "ok"
        assert bucket["track_c_aware_hours"] == 12

    def test_track_c_per_hour_missing_falls_back_to_raw(self):
        """Day has distribution but THIS hour missing → that hour uses raw."""
        mpc = "sensor.heater_mpc"
        # 10 hours, distribution covers only hours 0-4 (5 hours)
        log = [_track_c_entry(h, actual_kwh=0.05, mpc_sid=mpc) for h in range(10)]
        daily_history = {
            "2026-04-15": {
                "track_c_distribution": _track_c_dist({h: 0.40 for h in range(5)}),
            }
        }
        stored = {"10": {"normal": 0.40}}
        coord = _make_coord(stored, log, daily_history=daily_history,
                             track_c_enabled=True, mpc_managed_sensor=mpc)
        result = DiagnosticsEngine(coord).diagnose_model(days_back=30)

        bucket = result["base_model_health"]["buckets"]["10"]["normal"]
        # 5 hours used synthetic (0.40), 5 hours used raw (0.05)
        # mean = (5*0.40 + 5*0.05) / 10 = 0.225
        assert bucket["actual_dark_mean_kwh"] == pytest.approx(0.225)
        assert bucket["track_c_aware_hours"] == 5

    def test_track_c_enabled_without_mpc_sensor_uses_raw(self):
        """track_c_enabled=True but mpc_managed_sensor=None → no Track C path."""
        log = _dark_hours("10", "normal", actual_kwh=0.25, n=15)
        daily_history = {
            "2026-04-15": {
                "track_c_distribution": _track_c_dist({h: 0.50 for h in range(15)}),
            }
        }
        stored = {"10": {"normal": 0.40}}
        coord = _make_coord(stored, log, daily_history=daily_history,
                             track_c_enabled=True, mpc_managed_sensor=None)
        result = DiagnosticsEngine(coord).diagnose_model(days_back=30)

        bucket = result["base_model_health"]["buckets"]["10"]["normal"]
        assert bucket["actual_dark_mean_kwh"] == pytest.approx(0.25)
        assert bucket["track_c_aware_hours"] == 0

    def test_track_c_aware_hours_field_present_for_non_track_c(self):
        """track_c_aware_hours is always present, 0 for non-Track-C installs."""
        stored = {"10": {"normal": 0.30}}
        log = _dark_hours("10", "normal", actual_kwh=0.30, n=15)
        coord = _make_coord(stored, log)
        result = DiagnosticsEngine(coord).diagnose_model(days_back=30)
        bucket = result["base_model_health"]["buckets"]["10"]["normal"]
        assert "track_c_aware_hours" in bucket
        assert bucket["track_c_aware_hours"] == 0
