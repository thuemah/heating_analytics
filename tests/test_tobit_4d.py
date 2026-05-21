"""Focused tests for the 4D Tobit MLE solver (#954 commit 8).

The 4D path of ``_solve_tobit`` generalises the 3D solver
to the ``(s, e, w, diffuse)`` shadow potential.  Same Mills-ratio
likelihood, same active-set projected Newton, same trust-region clip.

Stage-1 acceptance: solver recovers known coefficients from synthetic
data within ±0.05 (uncensored) / ±0.10 (censored), pins free directions
correctly at the boundary, rejects singular warm-starts, and meets
``TOBIT_MAX_ITER`` on well-conditioned inputs.
"""
from __future__ import annotations

import random

import pytest

from custom_components.heating_analytics.const import TOBIT_MAX_ITER
from custom_components.heating_analytics.learning import LearningManager


def _gen_4d_samples(
    n: int,
    true_c: tuple[float, float, float, float],
    sigma: float,
    rng: random.Random,
    *,
    drop_diffuse: bool = False,
) -> list[tuple[float, float, float, float, float]]:
    """Generate uncensored 4D Tobit samples.

    Each row carries (s_dir, e_dir, w_dir, diffuse, value).  Diffuse
    spans a smaller range than direct components — diffuse irradiance
    is typically 20-40 % of GHI on clear days, larger on cloudy days,
    but the magnitude is comparable.
    """
    out: list[tuple[float, float, float, float, float]] = []
    for _ in range(n):
        s = rng.uniform(0.0, 0.8)
        e = rng.uniform(0.0, 0.6)
        w = rng.uniform(0.0, 0.7)
        d = 0.0 if drop_diffuse else rng.uniform(0.0, 0.5)
        val = (
            true_c[0] * s + true_c[1] * e + true_c[2] * w + true_c[3] * d
            + rng.gauss(0.0, sigma)
        )
        out.append((s, e, w, d, val))
    return out


class TestTobit4DRecovery:

    def test_tobit_4d_recovers_true_coefficient(self):
        """No censoring → solver should recover true coefficient ± 0.05."""
        rng = random.Random(1234)
        true_c = (0.4, 0.2, 0.15, 0.08)
        samples = _gen_4d_samples(60, true_c, sigma=0.05, rng=rng)
        censored = [False] * len(samples)

        fit = LearningManager._solve_tobit(samples, censored, components=('s', 'e', 'w', 'diffuse'))
        assert fit is not None
        assert fit["converged"], (
            f"did not converge: {fit.get('failure_reason')}"
        )
        for k, expected in zip(("s", "e", "w", "diffuse"), true_c):
            assert abs(fit[k] - expected) < 0.05, (
                f"{k}: {fit[k]:.4f} vs true {expected:.4f}"
            )

    def test_tobit_4d_recovers_with_censoring(self):
        """30 % right-censoring → Tobit beats OLS-on-uncensored-only.

        Censor the highest-impact rows so OLS-on-uncensored has a
        systematically depressed slope estimate.  Tobit's Mills-ratio
        likelihood recovers slope information from the censoring
        point itself.  Assert Tobit MSE < OLS MSE.
        """
        rng = random.Random(4242)
        true_c = (0.5, 0.3, 0.2, 0.1)
        raw = _gen_4d_samples(80, true_c, sigma=0.04, rng=rng)
        # Pick censoring threshold T at ~70th percentile of predictions
        preds = [
            true_c[0] * s + true_c[1] * e + true_c[2] * w + true_c[3] * d
            for s, e, w, d, _ in raw
        ]
        T = sorted(preds)[int(0.7 * len(preds))]
        samples: list[tuple[float, float, float, float, float]] = []
        mask: list[bool] = []
        for (s, e, w, d, v), p in zip(raw, preds):
            if p >= T:
                samples.append((s, e, w, d, T))
                mask.append(True)
            else:
                samples.append((s, e, w, d, v))
                mask.append(False)

        fit = LearningManager._solve_tobit(samples, mask, components=('s', 'e', 'w', 'diffuse'))
        assert fit is not None
        assert fit["converged"]
        tobit_mse = sum(
            (fit[k] - tv) ** 2
            for k, tv in zip(("s", "e", "w", "diffuse"), true_c)
        )

        # OLS using only uncensored rows (the LS warm-start path
        # would do this with the censored rows excluded).
        unc = [samples[i] for i in range(len(samples)) if not mask[i]]
        X = [(r[0], r[1], r[2], r[3]) for r in unc]
        y = [r[4] for r in unc]
        ls = LearningManager._solve_4x4_normal_equations(X, y)
        assert ls is not None
        ols_mse = sum(
            (ls[i] - tv) ** 2 for i, tv in enumerate(true_c)
        )
        assert tobit_mse < ols_mse, (
            f"tobit_mse={tobit_mse:.5f} not less than ols_mse={ols_mse:.5f}"
        )
        # Loose tolerance: with 30% censoring Tobit should still get
        # close to truth on every component.
        for k, tv in zip(("s", "e", "w", "diffuse"), true_c):
            assert abs(fit[k] - tv) < 0.10, f"{k}: {fit[k]:.4f} vs {tv}"

    def test_tobit_4d_active_set_pinning(self):
        """True diffuse coefficient = 0 → solver lands at 0 (or near)."""
        rng = random.Random(99)
        true_c = (0.4, 0.2, 0.15, 0.0)
        samples = _gen_4d_samples(60, true_c, sigma=0.03, rng=rng)
        mask = [False] * len(samples)

        fit = LearningManager._solve_tobit(samples, mask, components=('s', 'e', 'w', 'diffuse'))
        assert fit is not None
        assert fit["converged"]
        # diffuse should be very close to zero — active-set should pin
        # it on the boundary or noise-level just above.
        assert fit["diffuse"] < 0.05, (
            f"diffuse={fit['diffuse']:.5f} not pinned near zero"
        )

    def test_tobit_4d_singular_returns_none(self):
        """Degenerate samples (diffuse identically 0) → singular LS → None."""
        rng = random.Random(7)
        # All rows have diffuse=0, AND s=e=w (one rank-1 column block)
        # → 4×4 normal equations matrix is singular.
        samples = [(0.5, 0.5, 0.5, 0.0, 0.5 + rng.gauss(0, 0.01))
                   for _ in range(20)]
        mask = [False] * len(samples)

        fit = LearningManager._solve_tobit(samples, mask, components=('s', 'e', 'w', 'diffuse'))
        assert fit is None

    def test_tobit_4d_minimum_uncensored(self):
        """|U| < 4 → return None (4D needs ≥ 4 for warm-start LS)."""
        rng = random.Random(11)
        samples = _gen_4d_samples(3, (0.3, 0.2, 0.1, 0.05), sigma=0.02, rng=rng)
        mask = [False] * 3
        fit = LearningManager._solve_tobit(samples, mask, components=('s', 'e', 'w', 'diffuse'))
        assert fit is None

    def test_tobit_4d_convergence_within_max_iter(self):
        rng = random.Random(2024)
        true_c = (0.35, 0.25, 0.2, 0.1)
        samples = _gen_4d_samples(50, true_c, sigma=0.04, rng=rng)
        mask = [False] * len(samples)
        fit = LearningManager._solve_tobit(samples, mask, components=('s', 'e', 'w', 'diffuse'))
        assert fit is not None
        assert fit["converged"]
        assert fit["iterations"] <= TOBIT_MAX_ITER
