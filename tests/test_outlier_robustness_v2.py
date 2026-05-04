import pytest
import logging
from custom_components.heating_analytics.learning import LearningManager
from custom_components.heating_analytics.const import (
    OUTLIER_MIN_SAMPLES,
    OUTLIER_PROMOTION_THRESHOLD,
    HARD_OUTLIER_CAP_FACTOR,
    HARD_OUTLIER_SANITY_MULTIPLIER,
    OUTLIER_RESIDUAL_WINDOW,
)

def test_recovery_from_genuine_shift_v2():
    """Verify that we eventually recover if there is a genuine physical shift (#919 Part 2)."""
    lm = LearningManager()
    entity_id = "sensor.test_shift"
    regime = "heating"
    
    # 1. Prime baseline at residual 0.0
    # Sigma will be 0.02 (floor)
    for _ in range(OUTLIER_MIN_SAMPLES + 5):
        lm._is_outlier_residual(entity_id, regime, 0.0, lm._outlier_state)
    
    # 2. Huge shift: residual 0.5. 
    # Threshold is 5 * 0.02 = 0.1.
    # Initially it's an outlier.
    assert lm._is_outlier_residual(entity_id, regime, 0.5, lm._outlier_state) is True
    
    # 3. Repeat the shift. Should eventually be promoted.
    # Need OUTLIER_PROMOTION_THRESHOLD consecutive rejected samples.
    for _ in range(OUTLIER_PROMOTION_THRESHOLD - 2):
        assert lm._is_outlier_residual(entity_id, regime, 0.5, lm._outlier_state) is True
        
    # The last one should trigger promotion and return False (allow learning)
    assert lm._is_outlier_residual(entity_id, regime, 0.5, lm._outlier_state) is False
    
    # VERIFICATION: baseline should now have OUTLIER_PROMOTION_THRESHOLD (10) samples
    key = (regime, False)
    assert len(lm._outlier_state[entity_id][key]["baseline"]) == OUTLIER_PROMOTION_THRESHOLD
    
    # Now it should be part of the baseline
    assert lm._is_outlier_residual(entity_id, regime, 0.5, lm._outlier_state) is False

def test_batch_fit_fallback_insufficient_uncensored():
    """Verify batch fit outlier fallback when uncensored samples < OUTLIER_MIN_SAMPLES (#919)."""
    from unittest.mock import MagicMock
    lm = LearningManager()
    entity_id = "sensor.test_fallback"
    regime = "heating"
    
    coordinator = MagicMock()
    coordinator.model.correlation_data_per_unit = {entity_id: {"10": {"normal": 1.0}}}
    
    # 5 good uncensored samples + 50 censored samples + 1 outlier uncensored
    hourly_log = []
    entry_potentials = []
    
    # 5 good uncensored
    for i in range(5):
        hourly_log.append({
            "unit_modes": {entity_id: "heating"},
            "unit_breakdown": {entity_id: 0.5},
            "temp_key": "10",
            "wind_bucket": "normal",
        })
        entry_potentials.append((1.0, 0.0, 0.0, 1.0))
        
    # 50 censored
    for i in range(50):
        hourly_log.append({
            "unit_modes": {entity_id: "heating"},
            "unit_breakdown": {entity_id: 0.0}, # impact = 1.0 (saturated)
            "temp_key": "10",
            "wind_bucket": "normal",
        })
        entry_potentials.append((1.0, 0.0, 0.0, 1.0))
        
    # 1 Outlier uncensored (impact 15.0)
    hourly_log.append({
        "unit_modes": {entity_id: "heating"},
        "unit_breakdown": {entity_id: -14.0}, # impact 15.0
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
    
    # Since uncensored (6) < 20, it should use HARD_OUTLIER_CAP_FACTOR = 10.0
    # The outlier (15.0) should be dropped.
    # Total samples should be 5 + 50 = 55.
    assert len(samples) == 55, f"Expected 55 samples, got {len(samples)}"
    assert drop_counts.get("outlier") == 1

def test_warm_up_hard_cap():
    """Verify that we have a hard cap during warm-up (#919 Part 3)."""
    lm = LearningManager()
    entity_id = "sensor.test_warmup"
    regime = "heating"
    
    # During warm-up (n < OUTLIER_MIN_SAMPLES), residuals > 20 are rejected.
    assert lm._is_outlier_residual(entity_id, regime, 10.0, lm._outlier_state) is False
    assert lm._is_outlier_residual(entity_id, regime, 25.0, lm._outlier_state) is True

def test_sigma_floor_custom_min_base():
    """Verify that sigma floor scales with unit_min_base (#919 Part 4)."""
    lm = LearningManager()
    entity_id = "sensor.test_floor"
    regime = "heating"
    
    # unit_min_base = 2.0. Floor = 0.05 * 2.0 = 0.1. 
    # sigma_robust = max(0.1, 1.48 * mad).
    # If mad=0, sigma_robust=0.1. Threshold = 5 * 0.1 = 0.5.
    
    unit_min_base = {entity_id: 2.0}
    
    # Prime baseline with 0 noise
    for _ in range(OUTLIER_MIN_SAMPLES + 5):
        lm._is_outlier_residual(entity_id, regime, 0.0, lm._outlier_state, unit_min_base=unit_min_base)
        
    # Residual 0.4 should be fine
    assert lm._is_outlier_residual(entity_id, regime, 0.4, lm._outlier_state, unit_min_base=unit_min_base) is False
    # Residual 0.6 should be rejected
    assert lm._is_outlier_residual(entity_id, regime, 0.6, lm._outlier_state, unit_min_base=unit_min_base) is True

def test_prior_free_sanity_check():
    """Verify sanity check against expected base (#919 Part 5)."""
    lm = LearningManager()
    entity_id = "sensor.test_sanity"
    regime = "heating"
    
    # expected_base = 1.0. multiplier = 10. Max diff = 10.0.
    # actual = 12.0. diff = 11.0. Reject.
    assert lm._is_outlier_residual(entity_id, regime, 0.0, lm._outlier_state, 
                                 actual=12.0, expected_base=1.0) is True
    
    # actual = 5.0. diff = 4.0. Accept (no baseline yet, hard cap is 20).
    assert lm._is_outlier_residual(entity_id, regime, 4.0, lm._outlier_state, 
                                 actual=5.0, expected_base=1.0) is False

def test_batch_fit_pre_fit_mad_pass():
    """Verify that batch fit has a pre-fit MAD pass (#919 Part 1)."""
    # This requires a more complex setup with a mock coordinator
    from unittest.mock import MagicMock
    lm = LearningManager()
    
    entity_id = "sensor.test_batch"
    regime = "heating"
    
    # Mock coordinator and its model
    coordinator = MagicMock()
    coordinator.model.correlation_data_per_unit = {
        entity_id: {"10": {"normal": 1.0}}
    }
    
    # 25 good samples (residual ~0) + 1 outlier
    hourly_log = []
    entry_potentials = []
    for i in range(25):
        entry = {
            "unit_modes": {entity_id: "heating"},
            "unit_breakdown": {entity_id: 0.5}, # impact = base(1.0) - actual(0.5) = 0.5
            "temp_key": "10",
            "wind_bucket": "normal",
        }
        hourly_log.append(entry)
        entry_potentials.append((1.0, 0.0, 0.0, 1.0)) # (S, E, W, Mag)
        
    # Outlier: impact = 50.0
    entry = {
        "unit_modes": {entity_id: "heating"},
        "unit_breakdown": {entity_id: -49.0}, # impact = 1.0 - (-49.0) = 50.0
        "temp_key": "10",
        "wind_bucket": "normal",
    }
    hourly_log.append(entry)
    entry_potentials.append((1.0, 0.0, 0.0, 1.0))

    solar_coeffs = {entity_id: {"heating": {"s": 0.0, "e": 0.0, "w": 0.0}}}
    
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
    
    assert len(samples) == 25, f"Expected 25 samples, got {len(samples)}. Drop counts: {drop_counts}"
    assert drop_counts.get("outlier") == 1

def test_biased_prior_warmup():
    """Verify that the filter handles a biased prior during warm-up (#919)."""
    lm = LearningManager()
    entity_id = "sensor.test_bias_warmup"
    regime = "heating"
    
    # 1. Start with a significantly biased prior:
    # Sannhet er S=1.0. Prior (gammel state) er S=3.0.
    
    # 2. Mat inn "sanne" data (impact = 1.0)
    # Med S=3.0 prior blir residual = 1.0 - 3.0 = -2.0.
    for _ in range(OUTLIER_MIN_SAMPLES + 10):
        # Dette vil føre til abs(-2.0) = 2.0.
        # Dette er under HARD_OUTLIER_CAP_FACTOR (10.0), så det slipper inn i baselinen.
        # Testen skal bekrefte at vi takler dette og bygger en gyldig baseline rundt -2.0.
        res = 1.0 - 3.0
        is_outlier = lm._is_outlier_residual(entity_id, regime, res, lm._outlier_state)
        # Siden abs(-2.0) < 10.0 (warm-up) og etterhvert MAD tilpasser seg, skal is_outlier = False
        assert is_outlier is False
    
    # Verifiser at baselinen er bygget (median rundt -2.0) og at en ekte glitch blokkeres.
    glitch_res = 100.0 - 3.0 # impact 100.0, prior 3.0 -> res 97.0
    assert lm._is_outlier_residual(entity_id, regime, glitch_res, lm._outlier_state) is True
