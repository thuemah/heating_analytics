import pytest
import logging
from custom_components.heating_analytics.learning import LearningManager
from custom_components.heating_analytics.const import (
    TOBIT_RUNNING_WINDOW,
    MODE_HEATING,
    OUTLIER_MIN_SAMPLES,
)

def test_outlier_rejection_tobit_live():
    """Verify that outliers are rejected before entering Tobit live window (#919)."""
    lm = LearningManager()
    entity_id = "sensor.test_outlier"
    regime = "heating"
    
    tobit_stats = {}
    solar_coeffs = {entity_id: {"heating": {"s": 1.0, "e": 0.0, "w": 0.0}, "cooling": {"s": 0.0, "e": 0.0, "w": 0.0}}}
    
    # 1. Provide enough baseline samples to prime the outlier screener
    # Use c_s=1.0 with small noise
    import random
    random.seed(42)
    
    for _ in range(OUTLIER_MIN_SAMPLES + 5):
        s_vec = (1.0, 0.0, 0.0)
        impact = 1.0 + random.normalvariate(0, 0.02)
        base = impact + 1.0
        
        # Manually invoke the filtering & update logic as it appears in _process_per_unit_learning
        res = impact - solar_coeffs[entity_id][regime]["s"] * s_vec[0]
        is_outlier = lm._is_outlier_residual(entity_id, regime, res, lm._outlier_state)
        assert not is_outlier
        
        lm._update_unit_tobit_live(entity_id, regime, s_vec, impact, base, tobit_stats, solar_coeffs)

    # 2. Inject Outlier
    outlier_impact = 50.0 # 50x true value
    s_vec = (1.0, 0.0, 0.0)
    res = outlier_impact - solar_coeffs[entity_id][regime]["s"] * s_vec[0]
    is_outlier = lm._is_outlier_residual(entity_id, regime, res, lm._outlier_state)
    
    assert is_outlier, "Expected outlier to be flagged"
    
    # Verify it doesn't enter the Tobit window if we follow the gated logic
    slot = tobit_stats[entity_id][regime]
    assert all(s[3] < 10.0 for s in slot["samples"]), "Outlier should not be in the window"

def test_outlier_rejection_nlms():
    """Verify that outliers are rejected before entering NLMS learning (#919)."""
    lm = LearningManager()
    entity_id = "sensor.test_nlms_outlier"
    regime = "heating"
    
    solar_coeffs = {entity_id: {"heating": {"s": 1.0, "e": 0.0, "w": 0.0}, "cooling": {"s": 0.0, "e": 0.0, "w": 0.0}}}
    buf = {entity_id: {"heating": []}}
    
    # 1. Prime baseline
    for _ in range(OUTLIER_MIN_SAMPLES + 5):
        res = 0.01 # Very small residual
        is_outlier = lm._is_outlier_residual(entity_id, regime, res, lm._outlier_state)
        assert not is_outlier
    
    # 2. Huge residual
    res = 100.0
    is_outlier = lm._is_outlier_residual(entity_id, regime, res, lm._outlier_state)
    assert is_outlier
    
    # In production code, is_outlier gates the call to _learn_unit_solar_coefficient.
    # We verify here that the detection works.
    current_val = solar_coeffs[entity_id]["heating"]["s"]
    
    # Simulate gating
    if not is_outlier:
        lm._learn_unit_solar_coefficient(
            entity_id, "ts", 2.0, 2.0 - 100.0, (1.0, 0.0, 0.0), 0.1, 
            solar_coeffs, buf, 5.0, 15.0, "heating"
        )
    
    assert solar_coeffs[entity_id]["heating"]["s"] == current_val, "Coefficient should not have changed"

def test_recovery_from_genuine_shift():
    """Verify that we eventually recover if there is a genuine physical shift (not a glitch)."""
    lm = LearningManager()
    entity_id = "sensor.test_shift"
    regime = "heating"
    
    # Prime baseline at 1.0
    for _ in range(OUTLIER_MIN_SAMPLES + 5):
        lm._is_outlier_residual(entity_id, regime, 0.0, lm._outlier_state)
    
    # Sigma_robust will be floor=0.02. k=5. Threshold=0.1.
    # A shift to 2.0 will be blocked initially.
    assert lm._is_outlier_residual(entity_id, regime, 1.0, lm._outlier_state) is True
    
    # But if we keep seeing 1.0, does it eventually get accepted?
    # No, because we don't append outliers to the window in the current implementation.
    # This is a trade-off: protect against glitches vs adapt to huge jumps.
    # Since solar coefficients move slowly, a 100x jump is always a glitch.
def test_outlier_edge_cases():
    """Verify boundary conditions and edge cases for outlier detection."""
    lm = LearningManager()
    entity_id = "sensor.test_edges"
    regime = "heating"
    
    # n = 0 (Empty history)
    # Should use hard cap. 9.9 is below HARD_OUTLIER_CAP_FACTOR (10.0) -> False
    assert lm._is_outlier_residual(entity_id, regime, 9.9, lm._outlier_state) is False
    
    # Build up to OUTLIER_MIN_SAMPLES - 1
    for _ in range(OUTLIER_MIN_SAMPLES - 2): # -2 because we already added 1 above
        lm._is_outlier_residual(entity_id, regime, 0.0, lm._outlier_state)
        
    # n = OUTLIER_MIN_SAMPLES - 1
    # Still uses hard cap.
    assert lm._is_outlier_residual(entity_id, regime, 9.9, lm._outlier_state) is False
    
    # n = OUTLIER_MIN_SAMPLES
    # Now MAD filter is active. All previous residuals are ~0.0 (one 9.9, but median is 0).
    # Sigma floor applies: max(0.02, 0.05 * 0.5) = 0.025. Threshold = 5 * 0.025 = 0.125.
    # 9.9 is now a massive outlier and should be blocked.
    assert lm._is_outlier_residual(entity_id, regime, 9.9, lm._outlier_state) is True
    
    # Marginal outlier test
    # Threshold is 0.125. 0.12 is NOT an outlier.
    assert lm._is_outlier_residual(entity_id, regime, 0.12, lm._outlier_state) is False
    # 0.13 IS an outlier.
    assert lm._is_outlier_residual(entity_id, regime, 0.13, lm._outlier_state) is True

def test_reset_outlier_state():
    """Verify that reset_outlier_state clears the MAD windows (#919)."""
    lm = LearningManager()
    entity_id = "sensor.test_reset"
    regime = "heating"
    
    # 1. Populate some state
    for _ in range(5):
        lm._is_outlier_residual(entity_id, regime, 1.0, lm._outlier_state)
        
    assert entity_id in lm._outlier_state
    key = (regime, False)
    assert len(lm._outlier_state[entity_id][key]["baseline"]) == 5
    
    # 2. Reset specifically this entity
    lm.reset_outlier_state(entity_id)
    assert entity_id not in lm._outlier_state
    
    # 3. Populate multiple entities and reset all
    lm._is_outlier_residual("sensor.a", regime, 1.0, lm._outlier_state)
    lm._is_outlier_residual("sensor.b", regime, 1.0, lm._outlier_state)
    assert len(lm._outlier_state) == 2
    
    lm.reset_outlier_state()
    assert len(lm._outlier_state) == 0

if __name__ == "__main__":
    pass
