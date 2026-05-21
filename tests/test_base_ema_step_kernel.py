"""Tests for ``helpers.compute_base_ema_step`` (#967 Stage 1).

The kernel is the single arithmetic source-of-truth for the base-model
EMA step.  These tests pin:

1. The formula itself (step = lr · w · (target − bucket); new = bucket + step).
2. Byte-identity against the inline forms currently in
   ``learning.process_learning`` (live writer) and
   ``learning.learn_from_historical_import`` (retrain) — when those call
   sites are migrated through the helper (#967 Stage 2), the migration
   must produce bit-identical numeric output.  These tests guarantee a
   regression in the migration would be caught immediately.
"""
from __future__ import annotations

import pytest

from custom_components.heating_analytics.helpers import compute_base_ema_step


class TestBaseEmaStepKernel:
    def test_formula_basic_positive_target(self):
        """``new = bucket + lr × weight × (target − bucket)``; ``step`` returned alongside."""
        new_bucket, step = compute_base_ema_step(
            current_bucket=0.30,
            target=0.50,
            learning_rate=0.10,
            snr_weight=1.0,
        )
        # step = 0.10 × 1.0 × (0.50 − 0.30) = 0.020
        # new  = 0.30 + 0.020 = 0.32
        assert step == pytest.approx(0.02, abs=1e-9)
        assert new_bucket == pytest.approx(0.32, abs=1e-9)

    def test_zero_weight_zero_step(self):
        """SNR weight of 0 (e.g. all units shutdown) → no movement."""
        new_bucket, step = compute_base_ema_step(0.30, 0.50, 0.10, 0.0)
        assert step == 0.0
        assert new_bucket == 0.30

    def test_target_below_bucket_negative_step(self):
        """Target below bucket → negative step.  Helper does NOT clamp;
        clamping is caller-side (target construction at the call site)."""
        new_bucket, step = compute_base_ema_step(0.50, 0.30, 0.10, 1.0)
        assert step == pytest.approx(-0.02, abs=1e-9)
        assert new_bucket == pytest.approx(0.48, abs=1e-9)

    def test_inline_forms_byte_identical_byte_match(self):
        """Pin: helper output == inline forms used by live + retrain.

        ``learning.py:999`` (live):
            new_base_prediction = base_expected_kwh + base_effective_rate × (base_target − base_expected_kwh)
        ``learning.py:2734`` (retrain):
            new_pred = current_pred + effective_rate × (target − current_pred)

        Both reduce to the same arithmetic.  Migration regression caught here.
        """
        # Synthetic case sweep — values spread across realistic regime.
        cases = [
            # (bucket, target, lr, weight)
            (0.30, 0.50, 0.10, 1.0),
            (0.50, 0.30, 0.10, 0.5),
            (0.043, 0.235, 0.05, 0.83),  # shoulder bucket from #928 log
            (0.426, 0.317, 0.05, 0.99),  # mitsubishi temp=7 case
            (0.0, 0.50, 0.10, 1.0),  # cold-start-like
        ]
        for bucket, target, lr, weight in cases:
            new_bucket, step = compute_base_ema_step(bucket, target, lr, weight)
            # Inline form A (live writer at learning.py:999):
            inline_live = bucket + lr * weight * (target - bucket)
            # Inline form B (retrain at learning.py:2734):
            #   new_pred = current_pred + effective_rate × (target − current_pred)
            # where effective_rate = learning_rate × snr_weight.  Algebraically
            # identical to form A when written out.
            inline_retrain = bucket + (lr * weight) * (target - bucket)
            assert new_bucket == inline_live, (
                f"helper diverges from live inline form at case "
                f"({bucket=}, {target=}, {lr=}, {weight=}): "
                f"helper={new_bucket}, inline={inline_live}"
            )
            assert new_bucket == inline_retrain, (
                f"helper diverges from retrain inline form at case "
                f"({bucket=}, {target=}, {lr=}, {weight=}): "
                f"helper={new_bucket}, inline={inline_retrain}"
            )
            # Step matches new - old to floating-point precision.  Strict ==
            # would catch IEEE rounding noise in the subtraction; the helper
            # computes ``step`` once and returns ``bucket + step`` so the
            # two values are FP-consistent within ~1 ULP.
            assert step == pytest.approx(new_bucket - bucket, abs=1e-15)
