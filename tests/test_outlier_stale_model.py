import pytest
from unittest.mock import MagicMock
from custom_components.heating_analytics.learning import LearningManager
from custom_components.heating_analytics.const import (
    HARD_OUTLIER_CAP_FACTOR,
    HARD_OUTLIER_SANITY_MULTIPLIER,
)

def test_stale_model_does_not_drop_all_samples():
    """Verify that a stale model (large residual) doesn't drop all samples in small windows (#919)."""
    lm = LearningManager()
    entity_id = "sensor.test_stale"
    regime = "heating"
    
    coordinator = MagicMock()
    # expected_base = 20.0.
    # actual_impact = 12.0. (Residual = 12.0 if c=0).
    # This exceeds HARD_OUTLIER_CAP_FACTOR (10.0) if centered at 0.
    # But it's within sanity: abs(actual - base) = 12.0 < 10 * 20.0 = 200.0.
    # And actual_impact (12.0) < saturation_threshold (0.95 * 20.0 = 19.0), so they are uncensored.
    coordinator.model.correlation_data_per_unit = {entity_id: {"10": {"normal": 20.0}}}

    hourly_log = []
    entry_potentials = []
    for i in range(5):
        hourly_log.append({
            "unit_modes": {entity_id: "heating"},
            "unit_breakdown": {entity_id: 8.0}, # impact = base(20.0) - actual(8.0) = 12.0
            "temp_key": "10",
            "wind_bucket": "normal",
        })
        entry_potentials.append((1.0, 0.0, 0.0, 1.0))
    samples, censored, drop_counts = lm._collect_batch_fit_samples(
        entity_id=entity_id,
        regime=regime,
        hourly_log=hourly_log,
        entry_potentials=entry_potentials,
        coordinator=coordinator,
        unit_threshold=0.1,
        screen_affected_entities=None,
        for_tobit=True,
        solar_coefficients_per_unit={},
    )
    
    assert len(samples) == 5, f"Expected 5 samples, got {len(samples)}. Drop counts: {drop_counts}"
    assert drop_counts.get("outlier", 0) == 0

def test_match_diagnose_bypasses_outlier_filter():
    """Verify that match_diagnose=True bypasses the outlier filter for consistency (#919)."""
    lm = LearningManager()
    entity_id = "sensor.test_diagnose"
    regime = "heating"
    
    coordinator = MagicMock()
    coordinator.model.correlation_data_per_unit = {entity_id: {"10": {"normal": 1.0}}}
    
    # 25 good samples (residual ~0)
    hourly_log = []
    entry_potentials = []
    for i in range(25):
        hourly_log.append({
            "unit_modes": {entity_id: "heating"},
            "unit_breakdown": {entity_id: 0.5}, # impact 0.5
            "temp_key": "10",
            "wind_bucket": "normal",
            "unit_expected_breakdown": {entity_id: 1.0},
        })
        entry_potentials.append((1.0, 0.0, 0.0, 1.0))
        
    # One extreme outlier (impact 80.0) that won't be flagged as shutdown.
    # expected_base = 100.0, actual = 20.0 -> impact = 80.0. ratio = 0.20 (not < 0.20)
    hourly_log.append({
        "unit_modes": {entity_id: "heating"},
        "unit_breakdown": {entity_id: 20.0}, # actual 20.0, impact 80.0
        "temp_key": "10",
        "wind_bucket": "normal",
        "unit_expected_breakdown": {entity_id: 100.0},
    })
    entry_potentials.append((1.0, 0.0, 0.0, 1.0))
    
    # We also need to update the correlation_data_per_unit to 100.0 so match_diagnose=False uses it
    coordinator.model.correlation_data_per_unit = {entity_id: {"10": {"normal": 100.0}}}
    
    # And update the good samples to have base 100.0 as well so median works out
    for entry in hourly_log[:-1]:
        entry["unit_expected_breakdown"] = {entity_id: 100.0}
        entry["unit_breakdown"] = {entity_id: 99.5} # impact 0.5
    
    # With match_diagnose=False, outlier is dropped
    samples_f, _, drops_f = lm._collect_batch_fit_samples(
        entity_id=entity_id, regime=regime, hourly_log=hourly_log,
        entry_potentials=entry_potentials, coordinator=coordinator,
        unit_threshold=0.1, screen_affected_entities=None, for_tobit=False,
        solar_coefficients_per_unit={},
        match_diagnose=False
    )
    assert len(samples_f) == 25
    assert drops_f.get("outlier") == 1
    
    # With match_diagnose=True, MAD-based outlier is KEPT (bypassed)
    samples_t, _, drops_t = lm._collect_batch_fit_samples(
        entity_id=entity_id, regime=regime, hourly_log=hourly_log,
        entry_potentials=entry_potentials, coordinator=coordinator,
        unit_threshold=0.1, screen_affected_entities=None, for_tobit=False,
        solar_coefficients_per_unit={},
        match_diagnose=True
    )
    assert len(samples_t) == 26
    assert drops_t.get("outlier", 0) == 0
    
    # BUT: a massive glitch (100x base) is still dropped via sanity check
    # even when match_diagnose=True.
    massive_glitch_log = hourly_log + [{
        "unit_modes": {entity_id: "heating"},
        "unit_breakdown": {entity_id: -9900.0}, # impact 10000.0, actual 10100.0
        "temp_key": "10",
        "wind_bucket": "normal",
        "unit_expected_breakdown": {entity_id: 100.0},
    }]
    massive_glitch_pots = entry_potentials + [(1.0, 0.0, 0.0, 1.0)]
    
    samples_m, _, drops_m = lm._collect_batch_fit_samples(
        entity_id=entity_id, regime=regime, hourly_log=massive_glitch_log,
        entry_potentials=massive_glitch_pots, coordinator=coordinator,
        unit_threshold=0.1, screen_affected_entities=None, for_tobit=False,
        solar_coefficients_per_unit={},
        match_diagnose=True
    )
    # The massive glitch (actual 10100 vs base 100) is dropped.
    # Total samples should be 26 (from before) - 0 (kept) = ... wait.
    # original 25 + 1 (kept) + 1 (dropped) = 26.
    assert len(samples_m) == 26
    assert drops_m.get("outlier") == 1

if __name__ == "__main__":
    pass
