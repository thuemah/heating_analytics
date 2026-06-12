"""Call-path regression tests for ``compute_tobit_for_diagnose``.

A refactor mirrored the ``solar_coefficients_per_unit=...`` kwarg line
from the ``batch_fit_solar_coefficients`` call site into
``compute_tobit_for_diagnose`` — where no such local exists.  Every call
raised ``NameError``, which the blanket ``except Exception`` in
``diagnose_solar``'s per-unit block swallowed into
``skip_reason: "exception"`` with ``failure_reason: null``.  Symptom:
``implied_coefficient_tobit_30d`` silently null on ALL units of ALL
installs, and a green test suite — no test exercised the real call path
(diagnose tests run with ``coord.learning`` as an auto-MagicMock).

Covers:
- the function is callable for real (no NameError) and returns the
  documented skip shape on empty input,
- the resolved live-coefficient dict is threaded into the sample
  collector (MAD outlier baseline parity with batch_fit), with the
  model-view -> private-attribute -> None fallback ladder,
- ``diagnose_solar`` reaches the real learning manager without tripping
  the exception guard, and
- when the guard DOES fire, ``failure_reason`` now carries the exception
  class/message instead of null.
"""
from unittest.mock import MagicMock

import pytest

from custom_components.heating_analytics.learning import LearningManager
from custom_components.heating_analytics.solar import SolarCalculator
from custom_components.heating_analytics.diagnostics import DiagnosticsEngine

from tests.test_solar_diagnose import _make_coord, _hour_entry


def _make_learning_coord(coeffs=None):
    """Minimal coordinator mock for direct compute_tobit_for_diagnose calls."""
    coord = MagicMock()
    coord.screen_config = (False, False, False)
    coord.screen_config_for_entity = MagicMock(return_value=(False, False, False))
    coord._solar_coefficients_per_unit = coeffs if coeffs is not None else {}
    coord.solar = MagicMock()
    coord.solar.reconstruct_potential_vector = MagicMock(
        side_effect=SolarCalculator.reconstruct_potential_vector
    )
    return coord


def test_callable_without_nameerror_empty_log():
    """The original bug: every call raised NameError before any gate ran.

    An empty log must reach the ``insufficient_uncensored`` skip path —
    not blow up resolving a non-existent local.
    """
    mgr = LearningManager()
    result = mgr.compute_tobit_for_diagnose(
        [], "sensor.heater1", "heating", _make_learning_coord(), days_back=30
    )
    assert result["skip_reason"] == "insufficient_uncensored"
    assert result["coefficient"] is None
    assert result["n_uncensored"] == 0


def test_live_coefficients_threaded_into_collector():
    """The kwarg must carry the coordinator's live 3D coefficient dict
    (MAD baseline parity with the batch_fit caller), resolved via the
    model view with private-attribute fallback."""
    mgr = LearningManager()
    coeffs = {"sensor.heater1": {"heating": {"s": 0.3, "e": 0.1, "w": 0.2}}}
    coord = _make_learning_coord(coeffs)
    coord.model.solar_coefficients_per_unit = coeffs

    captured = {}

    def _spy(**kwargs):
        captured.update(kwargs)
        return [], [], {}

    mgr._collect_batch_fit_samples = _spy
    mgr.compute_tobit_for_diagnose(
        [], "sensor.heater1", "heating", coord, days_back=30
    )
    assert captured["solar_coefficients_per_unit"] is coeffs


def test_coefficient_resolution_fallback_ladder():
    """model view missing/non-dict -> private attribute -> None."""
    mgr = LearningManager()
    captured = {}

    def _spy(**kwargs):
        captured.update(kwargs)
        return [], [], {}

    mgr._collect_batch_fit_samples = _spy

    # Private attribute only (model view returns a non-dict MagicMock).
    coeffs = {"sensor.heater1": {"heating": {"s": 0.3}}}
    coord = _make_learning_coord(coeffs)
    mgr.compute_tobit_for_diagnose([], "sensor.heater1", "heating", coord)
    assert captured["solar_coefficients_per_unit"] == coeffs

    # Neither surface exposes a dict -> None (MAD skipped, no raise).
    bare = _make_learning_coord()
    bare._solar_coefficients_per_unit = None
    mgr.compute_tobit_for_diagnose([], "sensor.heater1", "heating", bare)
    assert captured["solar_coefficients_per_unit"] is None


def test_diagnose_solar_real_learning_manager_no_exception():
    """End-to-end through the real bug site: diagnose_solar's per-unit
    block calling the REAL learning manager must not trip the exception
    guard.  (Few samples -> honest 'insufficient_uncensored', never
    'exception'.)"""
    entries = [
        _hour_entry(f"2026-06-{day:02d}T12:00:00+00:00")
        for day in range(1, 6)
    ]
    coord = _make_coord(entries)
    coord.learning = LearningManager()
    coord.solar.reconstruct_potential_vector = MagicMock(
        side_effect=SolarCalculator.reconstruct_potential_vector
    )
    coord._per_unit_min_base_thresholds = {}
    # Real dict so the for_tobit base lookup inside the sample collector
    # does numeric comparisons (auto-MagicMock would TypeError on '<').
    base_buckets = {"sensor.heater1": {"10": {"normal": 2.0}}}
    coord._correlation_data_per_unit = base_buckets
    coord.model.correlation_data_per_unit = base_buckets

    result = DiagnosticsEngine(coord).diagnose_solar(days_back=30)

    diag = result["per_unit"]["sensor.heater1"]["tobit_diagnostics"]
    assert diag["skip_reason"] != "exception"
    assert diag["skip_reason"] == "insufficient_uncensored"


def test_diagnose_solar_exception_guard_records_failure_reason():
    """When the guard fires it must say WHAT failed — a bare 'exception'
    label hid the NameError for weeks."""
    entries = [
        _hour_entry(f"2026-06-{day:02d}T12:00:00+00:00")
        for day in range(1, 6)
    ]
    coord = _make_coord(entries)
    coord.learning.compute_tobit_for_diagnose = MagicMock(
        side_effect=ValueError("boom")
    )
    coord._per_unit_min_base_thresholds = {}

    result = DiagnosticsEngine(coord).diagnose_solar(days_back=30)

    diag = result["per_unit"]["sensor.heater1"]["tobit_diagnostics"]
    assert diag["skip_reason"] == "exception"
    assert diag["failure_reason"] == "ValueError: boom"
