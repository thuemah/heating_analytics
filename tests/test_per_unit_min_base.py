"""Tests for per-unit min-base thresholds.

Covers:

1. ``_calibrate_per_unit_min_base_thresholds`` logic: p10 computation,
   safety guards (log length, sample count, floor/ceiling, rate-of-change).
2. ``_resolve_min_base`` resolver semantics.
3. ``detect_solar_shutdown_entities`` honours per-unit overrides.
4. ``count_active_learnable_units`` honours per-unit overrides.
5. ``LearningConfig.unit_min_base`` contract: None/empty → legacy fallback.
6. Per-unit threading in the NLMS + inequality gates of
   ``_process_per_unit_learning`` and ``replay_solar_nlms``.
"""
from datetime import timedelta

import pytest
from homeassistant.util import dt as dt_util

from tests.helpers import CoordinatorModelMixin

from custom_components.heating_analytics.diagnostics import DiagnosticsEngine
from custom_components.heating_analytics.const import (
    MODE_HEATING,
    MODE_OFF,
    PER_UNIT_MIN_BASE_CEILING,
    PER_UNIT_MIN_BASE_FLOOR,
    PER_UNIT_MIN_BASE_MIN_SAMPLES,
    SOLAR_LEARNING_MIN_BASE,
    SOLAR_SHUTDOWN_MIN_BASE,
)
from custom_components.heating_analytics.learning import (
    _resolve_min_base,
    count_active_learnable_units,
)
from custom_components.heating_analytics.observation import (
    detect_solar_shutdown_entities,
)


# Trampoline so we can call the calibration method on a mock.
# Post-#877 the implementation lives on DiagnosticsEngine.
def _calibrate(coord, **kwargs):
    return DiagnosticsEngine(coord).calibrate_per_unit_min_base_thresholds(**kwargs)


class MockCoord(CoordinatorModelMixin):
    """Minimal coordinator for calibration tests."""

    def __init__(self, sensors):
        self.energy_sensors = sensors
        self._hourly_log = []
        self._per_unit_min_base_thresholds = {}


def _dark_entry(ts_iso, unit_actuals, solar_factor=0.0, aux=False):
    """Build a dark-hour log entry for calibration input."""
    return {
        "timestamp": ts_iso,
        "hour": int(ts_iso[11:13]),
        "solar_factor": solar_factor,
        "auxiliary_active": aux,
        "unit_modes": {sid: MODE_HEATING for sid in unit_actuals},
        "unit_breakdown": dict(unit_actuals),
    }


def _fill_dark_samples(coord, unit_samples, days_spread=20):
    """Populate _hourly_log with dark-hour samples per unit.

    ``unit_samples`` is {sensor_id: [actual_kwh, ...]}.  Padding entries
    are SUNNY (solar_factor=0.5) so they pass the log-length gate without
    contaminating the dark-hour sample window that drives p10.
    """
    now = dt_util.now()
    max_n = max(len(s) for s in unit_samples.values())
    # Sunny padding — does not contribute to dark-hour samples.
    pad_entries = max(0, 14 * 24 + 10 - max_n)
    for i in range(pad_entries):
        ts = (now - timedelta(hours=pad_entries + max_n - i)).isoformat()
        coord._hourly_log.append(
            _dark_entry(ts, {sid: 0.5 for sid in unit_samples}, solar_factor=0.5)
        )
    for i in range(max_n):
        ts = (now - timedelta(hours=max_n - i)).isoformat()
        row = {}
        for sid, samples in unit_samples.items():
            if i < len(samples):
                row[sid] = samples[i]
        coord._hourly_log.append(_dark_entry(ts, row))


# ---------------------------------------------------------------------------
# 1. _resolve_min_base resolver semantics
# ---------------------------------------------------------------------------


class TestResolveMinBase:

    def test_none_overrides_fall_back_to_constant(self):
        assert _resolve_min_base("u1", None, 0.15) == 0.15

    def test_empty_overrides_fall_back_to_constant(self):
        assert _resolve_min_base("u1", {}, 0.15) == 0.15

    def test_override_wins_when_present(self):
        assert _resolve_min_base("u1", {"u1": 0.05}, 0.15) == 0.05

    def test_missing_entity_falls_back(self):
        assert _resolve_min_base("u1", {"u2": 0.05}, 0.15) == 0.15

    def test_zero_or_negative_override_ignored(self):
        assert _resolve_min_base("u1", {"u1": 0.0}, 0.15) == 0.15
        assert _resolve_min_base("u1", {"u1": -0.1}, 0.15) == 0.15


# ---------------------------------------------------------------------------
# 2. count_active_learnable_units honours per-unit overrides
# ---------------------------------------------------------------------------


class TestCountActiveLearnableUnitsPerUnit:

    def test_override_enables_small_load(self):
        """Small load below global 0.15 qualifies when its override permits."""
        n = count_active_learnable_units(
            energy_sensors=["small"],
            unit_modes={"small": MODE_HEATING},
            expected_base_per_unit={"small": 0.05},
            unit_min_base={"small": 0.03},  # per-unit floor
        )
        assert n == 1

    def test_override_can_also_tighten(self):
        """Override above global tightens the gate for that unit.

        Uses a mixed-unit setup so the defensive fallback (which returns
        the full count when nothing qualifies) does not mask the tighten.
        """
        n = count_active_learnable_units(
            energy_sensors=["vp_small", "vp_big"],
            unit_modes={"vp_small": MODE_HEATING, "vp_big": MODE_HEATING},
            expected_base_per_unit={"vp_small": 0.20, "vp_big": 1.5},
            unit_min_base={"vp_small": 0.25},  # stricter than 0.15 for vp_small
        )
        # vp_small filtered (0.20 < 0.25), vp_big qualifies (1.5 > 0.15)
        assert n == 1

    def test_mixed_fallback_and_override(self):
        n = count_active_learnable_units(
            energy_sensors=["vp", "small"],
            unit_modes={"vp": MODE_HEATING, "small": MODE_HEATING},
            expected_base_per_unit={"vp": 1.5, "small": 0.05},
            unit_min_base={"small": 0.03},  # small → override, vp → fallback 0.15
        )
        assert n == 2


# ---------------------------------------------------------------------------
# 3. detect_solar_shutdown_entities honours per-unit overrides
# ---------------------------------------------------------------------------


class TestShutdownDetectionPerUnit:

    def _base_kwargs(self):
        return dict(
            solar_enabled=True,
            is_aux_dominant=False,
            potential_vector=(2.0, 0.0, 0.0),
            unit_modes={"vp": MODE_HEATING, "small": MODE_HEATING},
            unit_actual_kwh={"vp": 0.01, "small": 0.005},
            unit_expected_base_kwh={"vp": 0.138, "small": 0.05},
            energy_sensors=["vp", "small"],
        )

    def test_fallback_blocks_sub_global_units(self):
        """Without overrides both vp (0.138) and small (0.05) are below 0.15 gate."""
        flagged = detect_solar_shutdown_entities(**self._base_kwargs())
        assert flagged == ()

    def test_override_enables_toshiba_class_detection(self):
        """Toshiba-class vp at 0.138 kWh becomes eligible when its threshold drops."""
        flagged = detect_solar_shutdown_entities(
            **self._base_kwargs(),
            unit_min_base={"vp": 0.10},  # < 0.138
        )
        assert "vp" in flagged
        # Small load still blocked (no override + sub-global)
        assert "small" not in flagged

    def test_override_enables_small_load_detection(self):
        flagged = detect_solar_shutdown_entities(
            **self._base_kwargs(),
            unit_min_base={"small": 0.03, "vp": 0.10},
        )
        assert "small" in flagged
        assert "vp" in flagged


# ---------------------------------------------------------------------------
# 4. Calibration logic — guards
# ---------------------------------------------------------------------------


class TestCalibrateGuards:

    def test_insufficient_log_length_skips(self):
        coord = MockCoord(["u1"])
        coord._hourly_log = []  # zero entries
        result = _calibrate(coord)
        assert result["status"] == "insufficient_log_data"
        assert coord._per_unit_min_base_thresholds == {}

    def test_under_sampled_unit_skipped(self):
        coord = MockCoord(["u1"])
        # Pad log to pass the length gate but only give 5 dark samples.
        now = dt_util.now()
        for i in range(14 * 24 + 10):
            ts = (now - timedelta(hours=14 * 24 + 10 - i)).isoformat()
            # Sunny hours (solar_factor=0.5) — NOT dark, not sampled.
            coord._hourly_log.append({
                "timestamp": ts,
                "hour": int(ts[11:13]),
                "solar_factor": 0.5,
                "auxiliary_active": False,
                "unit_modes": {"u1": MODE_HEATING},
                "unit_breakdown": {"u1": 1.0},
            })
        # Add 5 dark samples (below the 20-sample minimum).
        for i in range(5):
            ts = (now - timedelta(hours=i)).isoformat()
            coord._hourly_log.append(_dark_entry(ts, {"u1": 0.07}))

        result = _calibrate(coord)
        assert result["status"] == "ok"
        assert "u1" in result["skipped"]
        assert result["skipped"]["u1"]["status"] == "skipped_low_samples"
        assert "u1" not in coord._per_unit_min_base_thresholds

    def test_p10_with_sufficient_samples_calibrates(self):
        coord = MockCoord(["u1"])
        # 30 samples: p10 ≈ 0.07 (3rd-lowest value).
        samples = [0.05, 0.06, 0.07] + [0.10] * 27
        _fill_dark_samples(coord, {"u1": samples})
        result = _calibrate(coord)
        assert result["status"] == "ok"
        assert "u1" in result["updated"]
        effective = coord._per_unit_min_base_thresholds["u1"]
        # p10 is 0.07, above floor → use p10.
        assert effective == pytest.approx(0.07, abs=1e-3)

    def test_floor_clamps_very_low_p10(self):
        coord = MockCoord(["u1"])
        # Low-tail p10 = 0.005, main mass at 0.05 so ratio-guard sees
        # p10/median = 0.1 (well below 0.9) and accepts the distribution.
        # Floor clamp then raises stored threshold from 0.005 to 0.03.
        samples = [0.005] * 3 + [0.05] * 27
        _fill_dark_samples(coord, {"u1": samples})
        _calibrate(coord)
        assert coord._per_unit_min_base_thresholds["u1"] == PER_UNIT_MIN_BASE_FLOOR

    def test_ratio_guard_rejects_constant_load(self):
        """Always-on load (electric boiler mislabeled as heat-pump heating,
        sensor scoped to a shared circuit) has p10 ≈ median, which cannot
        represent a modulating noise floor.  Ratio-guard rejects before
        the absolute ceiling is tested.
        """
        coord = MockCoord(["u1"])
        # All 30 samples at 0.5 → p10 = median = 0.5 → ratio 1.0 > 0.9.
        # Also below the 1.5 ceiling, so only the ratio-guard catches it.
        samples = [0.5] * 30
        _fill_dark_samples(coord, {"u1": samples})
        result = _calibrate(coord)
        assert "u1" in result["rejected"]
        assert result["rejected"]["u1"]["status"] == "rejected_constant_load"
        assert result["rejected"]["u1"]["p10_over_median"] == pytest.approx(1.0, abs=1e-3)
        assert "u1" not in coord._per_unit_min_base_thresholds

    def test_ceiling_rejects_above_max(self):
        """Absolute ceiling safety-net rejects p10 > 1.5 kWh when the
        ratio-guard alone would not fire.  Construction: p10 = 2.0 but
        median = 4.0 → ratio = 0.5 (ratio-guard accepts) → ceiling catches.
        """
        coord = MockCoord(["u1"])
        # Sorted: [2.0, 2.0, 2.0, 4.0, 4.0, …].  p10 idx = 2 → 2.0.
        # Median idx = 15 → 4.0.  ratio = 0.5 < 0.9, so ratio-guard
        # accepts; absolute ceiling 1.5 rejects on p10 = 2.0.
        samples = [2.0] * 3 + [4.0] * 27
        _fill_dark_samples(coord, {"u1": samples})
        result = _calibrate(coord)
        assert "u1" in result["rejected"]
        assert result["rejected"]["u1"]["status"] == "rejected_above_ceiling"
        assert "u1" not in coord._per_unit_min_base_thresholds

    def test_large_vp_cycling_accepted(self):
        """Larger residential heat pump (~800 W min modulation) with
        typical cycling is accepted under the 1.5 kWh ceiling — the
        0.30 ceiling used before v1.3.3 would have rejected this class.

        Distribution: a mix of off-periods (near-zero), modulating
        intermediate values, and sustained at-min-modulation samples.
        p10 falls in the off-period tail; median falls in the running
        mass.  Ratio stays low; ceiling is nowhere near hit.
        """
        coord = MockCoord(["u1"])
        # 30 samples spanning a realistic cycling distribution.
        samples = [0.05] * 5 + [0.10] * 5 + [0.30] * 5 + [0.60] * 5 + [0.80] * 10
        _fill_dark_samples(coord, {"u1": samples})
        result = _calibrate(coord)
        assert "u1" in result["updated"]
        # p10 at idx 2 = 0.05, above 0.03 floor → stored at 0.05.
        assert coord._per_unit_min_base_thresholds["u1"] == pytest.approx(0.05, abs=1e-3)

    def test_rate_of_change_clamps(self):
        coord = MockCoord(["u1"])
        coord._per_unit_min_base_thresholds["u1"] = 0.10  # prior value
        # p10 = 0.20 would be a 100 % jump; clamp to +50 % = 0.15.
        # Distribution has variance so the ratio-guard accepts (p10=0.20,
        # median=0.40, ratio=0.5 < 0.9).
        samples = [0.20] * 3 + [0.40] * 27
        _fill_dark_samples(coord, {"u1": samples})
        _calibrate(coord)
        assert coord._per_unit_min_base_thresholds["u1"] == pytest.approx(
            0.15, abs=1e-3
        )

    @pytest.mark.parametrize(
        "n,expected_idx",
        [
            # Nearest-rank p10: idx = ceil(0.10 × n) - 1 (0-indexed).
            # Regression: an earlier implementation used int(round(...)) and
            # diverged for n ∈ {21..25} (picked idx=1 instead of 2), biasing
            # the calibrated threshold one rank lower than documented.
            # n=25 also tripped Python's banker's rounding (2.5 → 2).
            (20, 1),  # 0.10 × 20 = 2.0 → idx 1. round matches ceil.
            (21, 2),  # 0.10 × 21 = 2.1 → idx 2. round=idx 1 → divergence.
            (22, 2),  # 0.10 × 22 = 2.2 → idx 2. round=idx 1 → divergence.
            (23, 2),  # 0.10 × 23 = 2.3 → idx 2. round=idx 1 → divergence.
            (24, 2),  # 0.10 × 24 = 2.4 → idx 2. round=idx 1 → divergence.
            (25, 2),  # 0.10 × 25 = 2.5 → idx 2. banker's rounding → idx 1.
            (26, 2),  # 0.10 × 26 = 2.6 → idx 2. round matches ceil.
            (30, 2),  # 0.10 × 30 = 3.0 → idx 2. round matches ceil.
            (40, 3),  # 0.10 × 40 = 4.0 → idx 3.
            (100, 9),  # 0.10 × 100 = 10.0 → idx 9.
        ],
    )
    def test_p10_ceiling_rank_at_boundary_sample_counts(self, n, expected_idx):
        """_p10 must use ceiling-based nearest-rank, not round.

        Each sample list is distinct ascending floats so the returned p10
        equals sample[expected_idx] exactly.  Rate-of-change and ceiling
        guards are bypassed by constructing sensible samples.
        """
        coord = MockCoord(["u1"])
        # Distinct ascending samples above the 0.03 floor and below the
        # 0.30 ceiling so neither safety clamp fires.
        samples = [0.05 + 0.001 * i for i in range(n)]
        expected = round(samples[expected_idx], 5)
        _fill_dark_samples(coord, {"u1": samples})
        result = _calibrate(coord)
        assert result["status"] == "ok"
        # Reported p10 (pre-floor) matches the ceiling-rank element.
        assert result["updated"]["u1"]["p10_actual"] == pytest.approx(
            samples[expected_idx], abs=1e-5
        )
        # Stored threshold matches (no rate clamp applies — first run).
        assert coord._per_unit_min_base_thresholds["u1"] == expected

    def test_aux_hours_excluded_from_samples(self):
        """Aux-active hours must not contribute to the dark-hour sample set.

        Construction: 25 heating samples spanning a realistic modulating
        distribution (5 low-tail at 0.08, 20 main-mass at 0.15) plus 30
        aux-active samples at 0.005.  Dark-heating p10 = 0.08 (stored).
        If aux leaked in, the 30 aux samples at 0.005 would dominate the
        low tail and shift p10 to 0.005 → floor-clamped to 0.03 — so the
        stored threshold moves visibly on any aux-filter regression.
        """
        coord = MockCoord(["u1"])
        now = dt_util.now()
        # 30 aux-active entries at 0.005 — must be excluded.
        for i in range(30):
            ts = (now - timedelta(hours=30 - i)).isoformat()
            entry = _dark_entry(ts, {"u1": 0.005})
            entry["auxiliary_active"] = True
            coord._hourly_log.append(entry)
        # 25 legitimate heating samples: 5 low-tail at 0.08, 20 at 0.15.
        heating = [0.08] * 5 + [0.15] * 20
        for i, kwh in enumerate(heating):
            ts = (now - timedelta(hours=100 + i)).isoformat()
            coord._hourly_log.append(_dark_entry(ts, {"u1": kwh}))
        # Sunny padding to clear the 14-day log-length gate without
        # contaminating dark-hour sampling.
        for i in range(14 * 24):
            ts = (now - timedelta(hours=14 * 24 + 200 + i)).isoformat()
            entry = _dark_entry(ts, {"u1": 0.10}, solar_factor=0.5)
            coord._hourly_log.append(entry)
        result = _calibrate(coord)
        assert result["status"] == "ok"
        # p10 of 25 heating samples, idx = ceil(2.5)-1 = 2 → 0.08.
        # If aux leaked: 55 samples, p10 idx 5 falls in the 0.005 tail.
        assert "u1" in coord._per_unit_min_base_thresholds
        assert coord._per_unit_min_base_thresholds["u1"] == pytest.approx(
            0.08, abs=1e-3
        )

    def test_sunny_hours_excluded_from_samples(self):
        coord = MockCoord(["u1"])
        now = dt_util.now()
        # 50 sunny entries with low actual (should NOT be counted as dark samples).
        for i in range(50):
            ts = (now - timedelta(hours=50 - i)).isoformat()
            entry = _dark_entry(ts, {"u1": 0.01}, solar_factor=0.5)
            coord._hourly_log.append(entry)
        # Only 10 dark samples — below the 20-sample minimum.
        for i in range(10):
            ts = (now - timedelta(hours=60 + i)).isoformat()
            coord._hourly_log.append(_dark_entry(ts, {"u1": 0.10}))
        # Padding to pass the length gate.
        for i in range(14 * 24):
            ts = (now - timedelta(hours=14 * 24 + 100 + i)).isoformat()
            entry = _dark_entry(ts, {"u1": 0.01}, solar_factor=0.8)
            coord._hourly_log.append(entry)
        result = _calibrate(coord)
        # Despite 50 + padding sunny entries, only 10 dark → under-sampled.
        assert result["skipped"].get("u1", {}).get("status") == "skipped_low_samples"

    def test_non_heating_mode_excluded(self):
        """Non-heating-mode dark samples must not contribute to p10.

        Construction: 25 heating samples with realistic modulating shape
        (5 at 0.08 low-tail, 20 at 0.15 main mass) plus 50 OFF-mode
        samples at 0.005.  Both have ``u1`` populated in
        ``unit_breakdown`` so only the mode filter (not the missing-key
        short-circuit) decides whether each sample contributes.
        Expected p10 = 0.08; any regression in the mode filter moves it
        to the 0.005 tail → floor-clamped to 0.03.
        """
        coord = MockCoord(["u1"])
        now = dt_util.now()
        # 50 OFF-mode dark entries at 0.005 — would crater p10 if leaked.
        # Note: u1 IS in unit_breakdown; only the mode filter saves us.
        for i in range(50):
            ts = (now - timedelta(hours=i)).isoformat()
            entry = _dark_entry(ts, {"u1": 0.005})
            entry["unit_modes"]["u1"] = MODE_OFF
            coord._hourly_log.append(entry)
        # 25 heating samples with realistic modulating shape.
        heating = [0.08] * 5 + [0.15] * 20
        for i, kwh in enumerate(heating):
            ts = (now - timedelta(hours=100 + i)).isoformat()
            coord._hourly_log.append(_dark_entry(ts, {"u1": kwh}))
        # Sunny padding to satisfy log-length gate.
        for i in range(14 * 24):
            ts = (now - timedelta(hours=14 * 24 + 200 + i)).isoformat()
            entry = _dark_entry(ts, {"u1": 0.10}, solar_factor=0.5)
            coord._hourly_log.append(entry)
        result = _calibrate(coord)
        assert result["status"] == "ok"
        # If mode filter works: p10 idx 2 of 25 heating samples = 0.08.
        # If mode filter broken: 75 mixed, p10 idx 7 falls in 0.005 tail.
        assert "u1" in coord._per_unit_min_base_thresholds
        assert coord._per_unit_min_base_thresholds["u1"] == pytest.approx(
            0.08, abs=1e-3
        )

    def test_multiple_units_independently_calibrated(self):
        """Each unit's p10 is derived from its own distribution.

        Distributions mirror realistic modulating shapes (low-tail +
        main-mass) rather than constant values so the ratio-guard
        accepts each.
        """
        coord = MockCoord(["toshiba", "mitsubishi", "small"])
        _fill_dark_samples(coord, {
            "toshiba":    [0.14] * 3 + [0.25] * 27,  # p10 = 0.14
            "mitsubishi": [0.25] * 3 + [0.50] * 27,  # p10 = 0.25
            "small":      [0.04] * 3 + [0.08] * 27,  # p10 = 0.04
        })
        result = _calibrate(coord)
        assert result["status"] == "ok"
        thr = coord._per_unit_min_base_thresholds
        assert thr["toshiba"] == pytest.approx(0.14, abs=1e-3)
        assert thr["mitsubishi"] == pytest.approx(0.25, abs=1e-3)
        # 'small' at 0.04 (above 0.03 floor) stays at p10 value.
        assert thr["small"] == pytest.approx(0.04, abs=1e-3)


# ---------------------------------------------------------------------------
# 5. Calibration status shape
# ---------------------------------------------------------------------------


class TestCalibrationReport:

    def test_report_has_expected_keys(self):
        coord = MockCoord(["u1"])
        result = _calibrate(coord)
        for key in ("total_log_hours", "required_log_hours", "status", "units",
                    "updated", "rejected", "skipped"):
            assert key in result

    def test_ok_status_when_log_sufficient(self):
        coord = MockCoord(["u1"])
        _fill_dark_samples(coord, {"u1": [0.08] * 3 + [0.15] * 27})
        result = _calibrate(coord)
        assert result["status"] == "ok"
        assert result["total_log_hours"] >= 14 * 24

    def test_unit_report_method_field(self):
        coord = MockCoord(["u1"])
        _fill_dark_samples(coord, {"u1": [0.08] * 3 + [0.15] * 27})
        result = _calibrate(coord)
        assert result["units"]["u1"]["method"] == "auto"
