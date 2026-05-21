"""Routing tests for the ``experimental_4d_primary`` flag (#962).

Validates that the five live solar read sites route through the 4D
shadow pipeline when the flag is on, that the flag=False path is
bit-identical to today's behaviour, and that the per-site opt-out
(``force_3d=True``, forecast TODOs, diagnose blocks) holds.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------
# Site 1 — Prediction dispatch
# ---------------------------------------------------------------------


class _StubStats:
    """Minimal StatisticsManager-shaped stub used to test the dispatcher
    contract on ``calculate_total_power`` without booting the full
    coordinator stack.

    Mirrors the public signature and exposes ``calls_3d`` /
    ``calls_4d`` counters so the test can assert which primitive was
    invoked under each flag state.
    """

    def __init__(self, coordinator):
        self.coordinator = coordinator
        self.calls_3d = 0
        self.calls_4d = 0

    # Copied from the real StatisticsManager.calculate_total_power
    # dispatcher (post-#962); kept inline here so this test does not
    # need to construct the real coordinator graph.
    def calculate_total_power(self, *args, force_3d: bool = False, **kwargs):
        flag = getattr(self.coordinator, "experimental_4d_primary", False) is True
        has_3d_only_overrides = (
            kwargs.get("override_solar_factor") is not None
            or kwargs.get("override_solar_vector") is not None
            or kwargs.get("carryover_state_override") is not None
        )
        if not flag or force_3d or has_3d_only_overrides:
            return self._calculate_total_power_3d(*args, **kwargs)
        return self.calculate_total_power_4d(*args, **kwargs)

    def _calculate_total_power_3d(self, *args, **kwargs):
        self.calls_3d += 1
        return {"total_kwh": 3.0, "breakdown": {"solar_reduction_kwh": 0.3}}

    def calculate_total_power_4d(self, *args, **kwargs):
        self.calls_4d += 1
        return {"total_kwh": 4.0, "breakdown": {"solar_reduction_kwh": 0.4}}


def test_dispatcher_flag_off_routes_3d():
    coord = MagicMock()
    coord.experimental_4d_primary = False
    stats = _StubStats(coord)

    result = stats.calculate_total_power(10.0, 0.0, 0.0, False)

    assert result["total_kwh"] == 3.0
    assert stats.calls_3d == 1
    assert stats.calls_4d == 0


def test_dispatcher_flag_on_routes_4d():
    coord = MagicMock()
    coord.experimental_4d_primary = True
    stats = _StubStats(coord)

    result = stats.calculate_total_power(10.0, 0.0, 0.0, False)

    assert result["total_kwh"] == 4.0
    assert stats.calls_3d == 0
    assert stats.calls_4d == 1


def test_dispatcher_force_3d_overrides_flag():
    coord = MagicMock()
    coord.experimental_4d_primary = True
    stats = _StubStats(coord)

    result = stats.calculate_total_power(10.0, 0.0, 0.0, False, force_3d=True)

    assert result["total_kwh"] == 3.0
    assert stats.calls_3d == 1
    assert stats.calls_4d == 0


def test_dispatcher_strict_is_true_check():
    """MagicMock auto-attribute is truthy but not ``is True`` — must route 3D."""
    coord = MagicMock()  # bare MagicMock; ``experimental_4d_primary`` is auto-MagicMock
    stats = _StubStats(coord)

    result = stats.calculate_total_power(10.0, 0.0, 0.0, False)

    assert result["total_kwh"] == 3.0
    assert stats.calls_4d == 0


def test_dispatcher_auto_falls_back_to_3d_for_3d_only_overrides():
    """``override_solar_factor`` / ``override_solar_vector`` /
    ``carryover_state_override`` are 3D-only kwargs.  Under flag=True,
    callers that pass any of them (forecast, diagnostics replay,
    sensitivity attributes) must be auto-routed to the 3D primitive so
    the injected overrides are honoured instead of silently dropped.
    """
    for override_kwargs in (
        {"override_solar_factor": 0.5},
        {"override_solar_vector": (0.1, 0.2, 0.3)},
        {"carryover_state_override": 0.05},
    ):
        coord = MagicMock()
        coord.experimental_4d_primary = True
        called = {"3d": 0, "4d": 0}

        class _RecordingStats(_StubStats):
            def _calculate_total_power_3d(self, *args, **kwargs):
                called["3d"] += 1
                return {}

            def calculate_total_power_4d(self, *args, **kwargs):
                called["4d"] += 1
                return {}

        stats = _RecordingStats(coord)
        stats.calculate_total_power(10.0, 0.0, 0.0, False, **override_kwargs)
        assert called["3d"] == 1, f"expected 3D route for {override_kwargs}"
        assert called["4d"] == 0, f"unexpected 4D route for {override_kwargs}"


# ---------------------------------------------------------------------
# Site 1 — Forecast intentionally NOT flipped
# ---------------------------------------------------------------------


def test_forecast_daily_site_no_longer_pinned_to_3d():
    """Counterpart to the original ``force_3d=True`` pin test.  After
    Agent B's follow-up (#978), site #2 (``_calculate_from_daily_forecast``)
    routes through 24 hourly 4D calls under ``experimental_4d_primary``
    instead of a single 3D call multiplied by 24.

    Enforced by grep-style inspection of ``forecast.py``: the
    transitional ``TODO(#978-daily-4d)`` marker MUST be gone, and the
    daily-forecast site MUST not contain ``force_3d=True``.
    """
    from pathlib import Path
    forecast_path = Path(
        "custom_components/heating_analytics/forecast.py"
    )
    text = forecast_path.read_text(encoding="utf-8")
    assert "TODO(#978-daily-4d)" not in text, (
        "TODO(#978-daily-4d) marker should have been removed once the "
        "24-hour 4D loop replaces the single-call-times-24 pin."
    )
    assert "force_3d=True" not in text, (
        "site #2's force_3d=True pin must be gone; the daily-forecast "
        "path now branches on experimental_4d_primary natively."
    )


# ---------------------------------------------------------------------
# Site 2 — Base EMA delta source
# ---------------------------------------------------------------------


def test_learning_config_has_experimental_4d_primary_field():
    """The flag rides on LearningConfig so the learner sees it per-hour."""
    from custom_components.heating_analytics.observation import LearningConfig
    cfg = LearningConfig(
        learning_enabled=True,
        solar_enabled=True,
        learning_rate=0.1,
        balance_point=12.0,
        energy_sensors=[],
        aux_impact=0.0,
        experimental_4d_primary=True,
    )
    assert cfg.experimental_4d_primary is True

    cfg_default = LearningConfig(
        learning_enabled=True,
        solar_enabled=True,
        learning_rate=0.1,
        balance_point=12.0,
        energy_sensors=[],
        aux_impact=0.0,
    )
    assert cfg_default.experimental_4d_primary is False


def test_compute_4d_normalization_delta_returns_none_without_state():
    """Helper short-circuits to None when the 4D path is not exercisable."""
    from custom_components.heating_analytics.learning import LearningManager
    from custom_components.heating_analytics.observation import LearningConfig

    mgr = LearningManager()
    cfg = LearningConfig(
        learning_enabled=True,
        solar_enabled=True,
        learning_rate=0.1,
        balance_point=12.0,
        energy_sensors=["hp.vp"],
        aux_impact=0.0,
        experimental_4d_primary=True,
    )

    # No solar_calculator — must return None
    assert mgr._compute_4d_normalization_delta(MagicMock(timestamp=None), cfg, {}) is None

    # solar_calculator present but solar disabled — must return None
    cfg_solar_off = LearningConfig(
        learning_enabled=True,
        solar_enabled=False,
        learning_rate=0.1,
        balance_point=12.0,
        energy_sensors=["hp.vp"],
        aux_impact=0.0,
        solar_calculator=MagicMock(),
        experimental_4d_primary=True,
    )
    obs = MagicMock()
    obs.timestamp = MagicMock()
    assert mgr._compute_4d_normalization_delta(obs, cfg_solar_off, {}) is None


# ---------------------------------------------------------------------
# Site 2 — Retrain fallback ladder
# ---------------------------------------------------------------------


def test_retrain_prefers_4d_delta_when_present_regardless_of_flag():
    """#968 — retrain.py's Track B/C aggregation path prefers
    ``solar_normalization_delta_4d`` whenever the log entry carries it,
    falling back to the 3D field otherwise.  Independent of
    ``experimental_4d_primary`` — the flag gates the live read-path only.
    """
    from pathlib import Path
    src = Path("custom_components/heating_analytics/retrain.py").read_text()
    # 4D field must still be referenced as the preferred signal.
    assert "solar_normalization_delta_4d" in src, (
        "retrain.py must reference the 4D delta field with fallback to 3D"
    )
    # The old per-site flag gate must be gone — the delta read should not
    # branch on ``experimental_4d_primary``.  Strip comments before checking
    # so documentation references to the flag (explaining the de-gating) do
    # not trip the assertion; only executable code should be inspected.
    code_lines = []
    for line in src.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("#"):
            continue
        # Drop trailing inline comments.
        if "#" in line:
            line = line.split("#", 1)[0]
        code_lines.append(line)
    code_only = "\n".join(code_lines)
    idx = code_only.index("solar_normalization_delta_4d")
    window = code_only[max(0, idx - 400):idx + 400]
    assert "experimental_4d_primary" not in window, (
        "retrain.py: the per-entry delta read must no longer be gated by "
        "experimental_4d_primary (#968 de-gated aggregation paths)."
    )
    assert "flag_4d" not in code_only, (
        "retrain.py: the now-unused flag_4d local must be removed."
    )


# ---------------------------------------------------------------------
# Site 3 — Carryover charging paused under flag
# ---------------------------------------------------------------------


def test_carryover_charging_guard_present():
    """Under flag=True, ``carryover_input`` must be forced to 0 regardless
    of ``solar_heating_wasted`` so the unbounded-growth invariant holds
    (4D's release is hardcoded to 0)."""
    from pathlib import Path
    src = Path("custom_components/heating_analytics/hourly_processor.py").read_text()
    # Find the carryover_input assignment block.
    assert "carryover_input" in src
    # The flag gate must appear in the same neighbourhood.
    idx = src.index("carryover_input = (")
    window = src[max(0, idx - 800):idx + 400]
    assert "experimental_4d_primary" in window, (
        "hourly_processor.py: carryover_input assignment must be gated by "
        "experimental_4d_primary so charging pauses while flag is on."
    )


# ---------------------------------------------------------------------
# Site 5 — Display fallback ladder
# ---------------------------------------------------------------------


def test_hourly_solar_impact_kwh_helper_routes_4d_when_flag_on():
    """When the flag is on AND the log entry has ``solar_impact_4d_kwh``,
    the helper returns the 4D value; else falls back to 3D."""
    from custom_components.heating_analytics.coordinator import HeatingDataCoordinator

    # Build a minimal stub exposing only the bits the helper reads.
    class _Stub:
        def __init__(self, flag):
            self.experimental_4d_primary = flag

    # Flag off: always 3D
    entry_both = {"solar_impact_kwh": 0.3, "solar_impact_4d_kwh": 0.4}
    val_off = HeatingDataCoordinator.hourly_solar_impact_kwh(_Stub(False), entry_both)
    assert val_off == pytest.approx(0.3)

    # Flag on + 4D field present: 4D
    val_on = HeatingDataCoordinator.hourly_solar_impact_kwh(_Stub(True), entry_both)
    assert val_on == pytest.approx(0.4)

    # Flag on + 4D field absent: fall back to 3D
    entry_3d_only = {"solar_impact_kwh": 0.5}
    val_on_no_4d = HeatingDataCoordinator.hourly_solar_impact_kwh(_Stub(True), entry_3d_only)
    assert val_on_no_4d == pytest.approx(0.5)

    # Strict ``is True`` — truthy non-bool values still route 3D.
    val_truthy = HeatingDataCoordinator.hourly_solar_impact_kwh(_Stub(1), entry_both)
    assert val_truthy == pytest.approx(0.3)


# ---------------------------------------------------------------------
# Bit-identity smoke
# ---------------------------------------------------------------------


def test_flag_default_is_false_on_coordinator():
    """The coordinator property defaults to False when no config value is set."""
    from custom_components.heating_analytics.coordinator import HeatingDataCoordinator
    from custom_components.heating_analytics.const import (
        CONF_EXPERIMENTAL_4D_PRIMARY,
        DEFAULT_EXPERIMENTAL_4D_PRIMARY,
    )

    assert DEFAULT_EXPERIMENTAL_4D_PRIMARY is False

    stub_entry = MagicMock()
    stub_entry.data = {}
    stub = MagicMock()
    stub.entry = stub_entry

    val = HeatingDataCoordinator.experimental_4d_primary.fget(stub)
    assert val is False

    # Explicit True
    stub_entry.data = {CONF_EXPERIMENTAL_4D_PRIMARY: True}
    val_on = HeatingDataCoordinator.experimental_4d_primary.fget(stub)
    assert val_on is True
