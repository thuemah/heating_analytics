"""Config-sourced battery scalars are coerced at the coordinator boundary.

``solar_battery_decay`` genuinely arrives from ``entry.data`` —
``diagnose_solar``'s ``apply_battery_decay: true`` writes it there — so a
JSON round-trip or a hand-edited config entry can present it as a string.
It then feeds bare arithmetic (``state * decay``) in the battery replays
with no guard downstream, and ``forecast.py`` already defends it at its
own read site, which tells you the exposure is real and was being papered
over one consumer at a time.

Coercing at the boundary covers every consumer instead of the one that
noticed.  ``bool`` is excluded deliberately: it is a subclass of ``int``,
so ``True`` would otherwise become a decay of ``1.0`` — a battery that
never discharges.
"""
from __future__ import annotations

import pytest

from custom_components.heating_analytics.const import SOLAR_BATTERY_DECAY
from custom_components.heating_analytics.diagnostics import _coerce_scalar


class TestCoerceScalar:
    """The shared helper the diagnose_solar read sites route through."""

    def test_a_float_passes_through(self):
        assert _coerce_scalar(0.7, 0.0) == 0.7

    def test_an_int_becomes_a_float(self):
        out = _coerce_scalar(1, 0.0)
        assert out == 1.0 and isinstance(out, float)

    @pytest.mark.parametrize("bad", ["0.7", "", None, [], {}, object()])
    def test_non_numerics_fall_back(self, bad):
        assert _coerce_scalar(bad, 0.5) == 0.5

    @pytest.mark.parametrize("flag", [True, False])
    def test_bool_falls_back_rather_than_becoming_one_or_zero(self, flag):
        """``True`` as a decay is a battery that never discharges."""
        assert _coerce_scalar(flag, 0.5) == 0.5

    def test_the_default_is_returned_unchanged(self):
        assert _coerce_scalar("nonsense", SOLAR_BATTERY_DECAY) == SOLAR_BATTERY_DECAY


class TestCoordinatorDecayBoundary:
    """``solar_battery_decay`` must be numeric by the time anything reads it.

    Exercises the coercion expression directly rather than standing up a
    coordinator: ``HeatingDataCoordinator.__init__`` pulls in the whole
    HA entity stack, and the property under test is a two-line boundary
    guard, not an integration concern.
    """

    @staticmethod
    def _boundary(raw):
        return (
            float(raw)
            if isinstance(raw, (int, float)) and not isinstance(raw, bool)
            else SOLAR_BATTERY_DECAY
        )

    def test_a_configured_float_survives(self):
        assert self._boundary(0.82) == 0.82

    def test_a_stringified_value_falls_back_to_the_default(self):
        """The JSON round-trip case: entry.data carrying "0.82"."""
        assert self._boundary("0.82") == SOLAR_BATTERY_DECAY

    def test_a_missing_value_is_the_default(self):
        assert self._boundary(SOLAR_BATTERY_DECAY) == SOLAR_BATTERY_DECAY

    def test_the_result_is_always_usable_in_arithmetic(self):
        for raw in (0.5, 1, "0.5", None, True, [0.5]):
            assert isinstance(self._boundary(raw) * 2.0, float)

    def test_it_matches_the_shared_helper(self):
        """Boundary and helper must not diverge on the same input."""
        for raw in (0.82, "0.82", None, True, 1, object()):
            assert self._boundary(raw) == _coerce_scalar(raw, SOLAR_BATTERY_DECAY)


def test_coordinator_source_uses_the_guard():
    """Pins the boundary itself, so the guard cannot be quietly dropped.

    The expression above mirrors production; this checks production still
    carries it, which is the half a mirrored test cannot prove.
    """
    from pathlib import Path

    source = (
        Path(__file__).parent.parent
        / "custom_components"
        / "heating_analytics"
        / "coordinator.py"
    ).read_text()
    assert "_decay_raw = entry.data.get(\"solar_battery_decay\"" in source
    assert "isinstance(_decay_raw, (int, float))" in source
    assert "not isinstance(_decay_raw, bool)" in source
