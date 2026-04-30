"""Tests for ``apply_implied_coefficient`` (#884 follow-up).

This service writes ``diagnose_solar``'s implied LS-fit into the live
solar coefficient for one (unit, mode regime), with per-direction
stability guards.  Use case: Track C / MPC-managed sensors that NLMS,
inequality, and batch_fit all skip — they're stuck on default
coefficients with no automatic learning path.

Covers:

1. **Per-direction stability assessment** — sign-flip, spread > 3x,
   near-zero consensus, insufficient windows.
2. **Apply path** — stable directions written, unstable directions
   preserve current value.
3. **``force=True``** — overrides per-direction guard, writes everything.
4. **``dry_run=True``** — analysis runs, nothing written.
5. **Mode routing** — heating/cooling regimes write the right slot;
   guest modes route to their respective regimes; OFF/DHW reject.
6. **Insufficient samples** — below ``APPLY_IMPLIED_MIN_QUALIFYING_HOURS``
   returns ``no_data``.
"""
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from custom_components.heating_analytics.const import (
    APPLY_IMPLIED_MAX_SPREAD,
    APPLY_IMPLIED_MIN_QUALIFYING_HOURS,
    APPLY_IMPLIED_NEAR_ZERO,
    MODE_COOLING,
    MODE_DHW,
    MODE_GUEST_COOLING,
    MODE_GUEST_HEATING,
    MODE_HEATING,
    MODE_OFF,
    SOLAR_COEFF_CAP,
)
from custom_components.heating_analytics.coordinator import HeatingDataCoordinator
from custom_components.heating_analytics.learning import LearningManager
from tests.helpers import stratified_coeff


# -----------------------------------------------------------------------------
# Stability assessment (pure function, no coordinator required)
# -----------------------------------------------------------------------------

class TestAssessStability:
    """``LearningManager.assess_apply_implied_stability`` is a static
    classifier on stability windows.  Tests cover the four reason codes:
    ``ok``, ``sign_flip``, ``spread_exceeds_threshold``,
    ``near_zero_consensus``, ``insufficient_windows``.
    """

    def _windows(self, *vals_per_dir):
        """Build n stability windows from per-direction value tuples.

        ``_windows((0.3, 0.4, 0.0), (0.32, 0.41, -0.01))`` builds two
        windows where ``s/e/w`` carry the corresponding tuple positions.
        """
        return [
            {
                "coefficient": {"s": s, "e": e, "w": w},
                "qualifying_hours": 20,
            }
            for s, e, w in vals_per_dir
        ]

    def test_stable_components_pass_with_ok_reason(self):
        # w values strictly below APPLY_IMPLIED_NEAR_ZERO (0.05) →
        # near_zero_consensus.  Boundary value 0.05 is treated as a
        # small-but-real signal (strict ``<`` check).
        windows = self._windows((0.30, 0.40, 0.02), (0.32, 0.41, 0.01))
        result = LearningManager.assess_apply_implied_stability(windows)
        assert result["s"]["stable"] is True
        assert result["s"]["reason"] == "ok"
        assert result["e"]["stable"] is True
        assert result["e"]["reason"] == "ok"
        assert result["w"]["stable"] is True
        assert result["w"]["reason"] == "near_zero_consensus"

    def test_sign_flip_marks_unstable(self):
        """The user's reported case: w window 1 = +0.5756, window 2 = -0.035.
        Sign-flip with at least one non-trivial value → unstable.  Note
        that even though -0.035 is below NEAR_ZERO (0.05), the sign-flip
        check covers ALL non-zero values — the user-facing red flag is
        "windows disagree on direction", which a +0.58 vs -0.04 split
        clearly does."""
        windows = self._windows((0.30, 0.40, 0.5756), (0.32, 0.41, -0.035))
        result = LearningManager.assess_apply_implied_stability(windows)
        assert result["s"]["stable"] is True
        assert result["e"]["stable"] is True
        assert result["w"]["stable"] is False
        assert result["w"]["reason"] == "sign_flip"

    def test_spread_exceeds_threshold_marks_unstable(self):
        """4x spread on a non-zero direction with consistent sign."""
        windows = self._windows((0.10, 0.40, 0.05), (0.45, 0.41, 0.05))
        result = LearningManager.assess_apply_implied_stability(windows)
        # s: 0.45 / 0.10 = 4.5x → above 3.0 default threshold
        assert result["s"]["stable"] is False
        assert result["s"]["reason"] == "spread_exceeds_threshold"

    def test_near_zero_consensus_is_stable(self):
        """All values below NEAR_ZERO threshold → consistently zero.
        Returning stable=True lets the caller clear stale priors."""
        windows = self._windows(
            (0.30, 0.40, 0.001),
            (0.32, 0.41, -0.002),
            (0.31, 0.39, 0.003),
        )
        result = LearningManager.assess_apply_implied_stability(windows)
        assert result["w"]["stable"] is True
        assert result["w"]["reason"] == "near_zero_consensus"

    def test_insufficient_windows_returns_unstable(self):
        """Fewer than 2 non-empty windows → can't assess."""
        windows = [
            {"coefficient": {"s": 0.3, "e": 0.4, "w": 0.0}, "qualifying_hours": 10},
            None,
            None,
        ]
        result = LearningManager.assess_apply_implied_stability(windows)
        for d in ("s", "e", "w"):
            assert result[d]["stable"] is False
            assert result[d]["reason"] == "insufficient_windows"

    def test_near_zero_consensus_wins_over_sign_flip_for_tiny_values(self):
        """Documents the order-of-checks decision: when ALL values are
        below NEAR_ZERO (0.05), they're classified as near_zero_consensus
        regardless of sign — the windows agree the component is
        effectively zero, and the sign of noise around zero is
        irrelevant.  Sign-flip detection only fires when at least one
        value is above NEAR_ZERO.

        Reference case: ``(0.04, -0.03)`` could be read as "windows
        disagree" (sign-flip semantics) or as "both windows say zero
        within noise" (near_zero_consensus).  The implementation chose
        the latter — write 0.0, clear stale priors, accept the noise.
        """
        windows = self._windows((0.30, 0.40, 0.04), (0.32, 0.41, -0.03))
        result = LearningManager.assess_apply_implied_stability(windows)
        assert result["w"]["stable"] is True
        assert result["w"]["reason"] == "near_zero_consensus"

    def test_explicit_thresholds_override_defaults(self):
        """Caller-provided ``max_spread`` and ``near_zero`` win over
        the const defaults — useful for tests and for tuning per-call."""
        # Spread of 2.5x — under the strict 2.0 we pass, so it should
        # reject; default 3.0 would let it through.
        windows = self._windows((0.20, 0.40, 0.0), (0.50, 0.41, 0.0))
        strict = LearningManager.assess_apply_implied_stability(
            windows, max_spread=2.0
        )
        loose = LearningManager.assess_apply_implied_stability(windows)
        assert strict["s"]["stable"] is False
        assert loose["s"]["stable"] is True


# -----------------------------------------------------------------------------
# Coordinator wrapper integration
# -----------------------------------------------------------------------------

def _entry(ts, *, sensor_id="sensor.heater1", solar_s=0.5, actual_kwh=1.5,
           expected_base=2.0, mode=MODE_HEATING):
    """Minimal hourly-log entry.

    ``expected_base`` populates ``unit_expected_breakdown`` — the
    log-time per-unit base that diagnose_solar (and apply_implied via
    ``match_diagnose=True``) reads as the ground truth for the LS
    target.  Without this field, the diagnose-match path skips the
    entry as ``below_min_base``.

    Note on timestamps: ``ts`` is passed through verbatim.  Production
    timestamps from ``coordinator.py`` are always tz-aware
    (``dt_util.utcnow().isoformat()`` → ``+00:00`` suffix), so the
    days_back filter's lex-string comparison works correctly there.
    Tests that exercise the days_back filter must build tz-aware
    timestamps explicitly (see
    ``test_days_back_filters_log_to_recent_window`` for the pattern);
    naive ISO strings would lex-compare against tz-aware cutoffs in a
    way that excludes timestamps at the exact boundary.
    """
    return {
        "timestamp": ts,
        "hour": int(ts[11:13]),
        "temp": 10.0,
        "temp_key": "10",
        "wind_bucket": "normal",
        "solar_factor": solar_s,
        "solar_vector_s": solar_s,
        "solar_vector_e": 0.0,
        "solar_vector_w": 0.0,
        "correction_percent": 100.0,
        "auxiliary_active": False,
        "actual_kwh": actual_kwh,
        "unit_modes": {sensor_id: mode},
        "unit_breakdown": {sensor_id: actual_kwh},
        "unit_expected_breakdown": {sensor_id: expected_base},
        "solar_dominant_entities": [],
        "learning_status": "logged",
    }


def _make_coord(*, n_samples: int = 60, true_coeff_s: float = 1.0,
                 stratified_seed: dict | None = None):
    """Real HeatingDataCoordinator with synthetic log + base."""
    hass = MagicMock()
    hass.config = MagicMock()
    hass.config.latitude = 60.0
    hass.config.longitude = 10.0
    hass.is_running = True
    hass.states = MagicMock()
    hass.data = {"heating_analytics": {}}
    hass.services = MagicMock()
    hass.async_add_executor_job = AsyncMock()

    entry = MagicMock()
    entry.entry_id = "test_entry"
    entry.data = {
        "energy_sensors": ["sensor.heater1"],
        "balance_point": 15.0,
        "solar_enabled": True,
        "csv_auto_logging": False,
    }

    with patch("custom_components.heating_analytics.storage.Store"):
        coord = HeatingDataCoordinator(hass, entry)
        # Build a synthetic log where solar_s varies enough to give
        # stable windows: morning low-S, midday high-S, afternoon
        # mid-S — chronologically ordered so chunked stability windows
        # see different sample distributions but same underlying coeff.
        entries = []
        vectors = [(0.4,), (0.6,), (0.5,), (0.7,), (0.55,)]
        for i in range(n_samples):
            sv = vectors[i % len(vectors)][0]
            impact = true_coeff_s * sv
            actual = max(0.0, 2.0 - impact)
            entries.append(_entry(
                f"2026-04-{10 + i // 24:02d}T{i % 24:02d}:00:00",
                solar_s=sv,
                actual_kwh=actual,
            ))
        coord._hourly_log = entries
        coord._correlation_data_per_unit = {
            "sensor.heater1": {"10": {"normal": 2.0}}
        }
        if stratified_seed is not None:
            coord._solar_coefficients_per_unit = {"sensor.heater1": stratified_seed}
        coord._async_save_data = AsyncMock()
        return coord


@pytest.mark.asyncio
async def test_apply_writes_stable_components_to_heating_regime():
    """Synthetic uniform log → all three directions stable → all written."""
    coord = _make_coord()
    result = await coord.async_apply_implied_coefficient(
        entity_id="sensor.heater1", mode=MODE_HEATING
    )
    assert result["status"] == "ok"
    assert result["regime"] == "heating"
    # All three components classified — synthetic data is stable on s.
    # e and w are near-zero across windows (no signal) → consensus stable.
    for d in ("s", "e", "w"):
        assert d in result["applied_components"]
    assert result["after"]["s"] == pytest.approx(1.0, abs=0.05)
    # Cooling regime preserved at zero.
    coeff = coord._solar_coefficients_per_unit["sensor.heater1"]
    assert coeff["cooling"] == {"s": 0.0, "e": 0.0, "w": 0.0}
    coord._async_save_data.assert_awaited_once()


@pytest.mark.asyncio
async def test_dry_run_reports_but_does_not_write():
    """``dry_run=True`` runs the analysis but writes nothing."""
    coord = _make_coord(stratified_seed=stratified_coeff(s=0.42))
    before = dict(coord._solar_coefficients_per_unit["sensor.heater1"]["heating"])
    result = await coord.async_apply_implied_coefficient(
        entity_id="sensor.heater1", mode=MODE_HEATING, dry_run=True
    )
    assert result["status"] == "ok"
    assert result["dry_run"] is True
    # ``after`` reports the would-be value.
    assert result["after"]["s"] == pytest.approx(1.0, abs=0.05)
    # But on disk / in-memory, the coefficient is unchanged.
    actual = coord._solar_coefficients_per_unit["sensor.heater1"]["heating"]
    assert actual == before
    coord._async_save_data.assert_not_awaited()


@pytest.mark.asyncio
async def test_force_overrides_per_direction_guard():
    """``force=True`` writes every direction even when stability would
    skip.  Construct a log + manual stability override that produces
    sign-flip on w; force should still write the implied value."""
    coord = _make_coord()
    # Patch the stability check to mark w as unstable.
    real_assess = LearningManager.assess_apply_implied_stability

    def _fake_assess(windows, **kwargs):
        result = real_assess(windows, **kwargs)
        result["w"] = {"stable": False, "reason": "sign_flip", "values": [0.5, -0.1]}
        return result

    with patch.object(
        LearningManager, "assess_apply_implied_stability", staticmethod(_fake_assess)
    ):
        no_force = await coord.async_apply_implied_coefficient(
            entity_id="sensor.heater1", mode=MODE_HEATING, dry_run=True
        )
        assert "w" in no_force["skipped_components"]
        assert "w" not in no_force["applied_components"]

        with_force = await coord.async_apply_implied_coefficient(
            entity_id="sensor.heater1", mode=MODE_HEATING, dry_run=True, force=True
        )
        assert "w" in with_force["applied_components"]
        assert "w" not in with_force["skipped_components"]


@pytest.mark.asyncio
async def test_no_stable_components_returns_status_and_skips_save():
    """When every direction is unstable AND force is False → no write."""
    coord = _make_coord()

    def _all_unstable(windows, **kwargs):
        return {
            d: {"stable": False, "reason": "sign_flip", "values": [0.5, -0.5]}
            for d in ("s", "e", "w")
        }

    with patch.object(
        LearningManager, "assess_apply_implied_stability",
        staticmethod(_all_unstable),
    ):
        result = await coord.async_apply_implied_coefficient(
            entity_id="sensor.heater1", mode=MODE_HEATING
        )
    assert result["status"] == "no_stable_components"
    assert result["applied_components"] == []
    assert result["skipped_components"] == ["s", "e", "w"]
    assert result["after"] == result["before"]
    coord._async_save_data.assert_not_awaited()


@pytest.mark.asyncio
async def test_insufficient_samples_returns_no_data():
    """Below ``APPLY_IMPLIED_MIN_QUALIFYING_HOURS`` → status ``no_data``."""
    coord = _make_coord(n_samples=APPLY_IMPLIED_MIN_QUALIFYING_HOURS - 1)
    result = await coord.async_apply_implied_coefficient(
        entity_id="sensor.heater1", mode=MODE_HEATING
    )
    assert result["status"] == "no_data"
    assert result["skip_reason"] == "insufficient_samples"
    coord._async_save_data.assert_not_awaited()


@pytest.mark.asyncio
async def test_unknown_entity_returns_no_data():
    coord = _make_coord()
    result = await coord.async_apply_implied_coefficient(
        entity_id="sensor.does_not_exist", mode=MODE_HEATING
    )
    assert result["status"] == "no_data"
    assert result["skip_reason"] == "unknown_entity"


@pytest.mark.asyncio
async def test_off_and_dhw_modes_raise():
    """OFF / DHW are not learnable solar regimes — service rejects."""
    coord = _make_coord()
    with pytest.raises(ValueError, match="learnable regime"):
        await coord.async_apply_implied_coefficient(
            entity_id="sensor.heater1", mode=MODE_OFF
        )
    with pytest.raises(ValueError, match="learnable regime"):
        await coord.async_apply_implied_coefficient(
            entity_id="sensor.heater1", mode=MODE_DHW
        )


@pytest.mark.asyncio
async def test_guest_modes_route_to_correct_regime():
    """GUEST_HEATING → heating regime, GUEST_COOLING → cooling regime.
    Mirrors the regime mapping used elsewhere in the codebase."""
    coord = _make_coord()
    res_h = await coord.async_apply_implied_coefficient(
        entity_id="sensor.heater1", mode=MODE_GUEST_HEATING, dry_run=True
    )
    assert res_h["regime"] == "heating"

    coord2 = _make_coord()
    # Cooling regime needs cooling-mode entries to fit, otherwise no data.
    # Build a cooling-mode log.
    cooling_entries = []
    for i in range(40):
        sv = 0.5 + (i % 5) * 0.05
        impact = 1.0 * sv  # cooling: actual = base + impact
        actual = 2.0 + impact
        cooling_entries.append(_entry(
            f"2026-04-{10 + i // 24:02d}T{i % 24:02d}:00:00",
            solar_s=sv,
            actual_kwh=actual,
            mode=MODE_COOLING,
        ))
    # Cooling samples are routed to the cooling wind-bucket per #885;
    # provide a base there so the LS has something to fit against.
    coord2._hourly_log = cooling_entries
    coord2._correlation_data_per_unit = {
        "sensor.heater1": {"10": {"cooling": 2.0}}
    }
    res_c = await coord2.async_apply_implied_coefficient(
        entity_id="sensor.heater1", mode=MODE_GUEST_COOLING, dry_run=True
    )
    assert res_c["regime"] == "cooling"


@pytest.mark.asyncio
async def test_track_c_mpc_sensor_apply_path():
    """Track C / MPC reference case — the actual motivation for this feature.

    The CHANGELOG and commit messages center this feature on Track C
    installations whose MPC-managed sensors get skipped by NLMS,
    inequality, and batch_fit_solar (no coherent dark-equivalent
    baseline).  This test wires up a ``WeightedSmear(use_synthetic=True)``
    strategy and runs the apply path on synthetic MPC-shaped data.

    Two outcomes are acceptable:

    1. **Apply succeeds** with a coefficient close to the synthetic
       truth → the diagnose-match LS extracts a usable signal from MPC
       data even though the rationale at ``learning.py:_collect_batch_fit``
       (via the WeightedSmear-skip in ``batch_fit_solar_coefficients``)
       calls it incoherent.  In this case the apply-implied path is
       genuinely the Track C escape hatch.
    2. **Apply produces a clearly-wrong coefficient** (zero, inverted
       sign, off by an order of magnitude) → MPC data is incoherent
       for LS as the original rationale claims, and the CHANGELOG
       narrative needs reframing ("Track C users still set coefficients
       manually until MPC exposes a dark baseline").

    This test pins outcome (1) — recovery within tolerance — and is
    the source of truth for whether the Track C use case actually
    works.  If a future code change breaks recovery, this test fails
    and the maintainer must decide between fixing the code or
    reframing the feature scope.
    """
    from custom_components.heating_analytics.observation import WeightedSmear

    coord = _make_coord()
    # Wire up a Track C strategy.  The strategy itself doesn't gate
    # the apply path (apply does NOT skip WeightedSmear the way batch
    # does); but the test confirms the end-to-end shape works.
    coord._unit_strategies = {
        "sensor.heater1": WeightedSmear(
            sensor_id="sensor.heater1", use_synthetic=True
        ),
    }
    result = await coord.async_apply_implied_coefficient(
        entity_id="sensor.heater1", mode=MODE_HEATING
    )
    # The synthetic log builder (`_make_coord`) bakes
    # ``actual = base − true_coeff_s × pot_s`` so the LS recovers
    # ``true_coeff_s ≈ 1.0`` regardless of which sensor strategy is
    # configured.  If a future change introduces an MPC-specific
    # filter in compute_implied_for_apply that skips this entity, the
    # status would be ``no_data`` and the user can't recover the
    # coefficient via this service.
    assert result["status"] == "ok", (
        f"Apply rejected MPC sensor (status={result['status']}, "
        f"skip_reason={result.get('skip_reason')}).  Either fix the "
        f"path or reframe the CHANGELOG: Track C apply does not work."
    )
    # Recovery is within the same tolerance batch_fit's synthetic test
    # uses (5 %).
    assert "s" in result["applied_components"], result
    assert result["after"]["s"] == pytest.approx(1.0, abs=0.05)


@pytest.mark.asyncio
async def test_clamp_to_solar_coeff_cap_invariant_4():
    """Implied values exceeding ``SOLAR_COEFF_CAP`` get clamped on apply
    (invariant #4: non-negative AND ≤ cap).  Construct a synthetic log
    that would naively fit a coefficient > cap, verify the apply clamps."""
    # Inflate the synthetic data so the LS fits coeff = 100 (> 5.0 cap).
    coord = _make_coord(true_coeff_s=10.0)
    # Need more base headroom so 10 × pot_s impact stays < base (otherwise
    # samples saturate and get filtered).
    coord._correlation_data_per_unit = {
        "sensor.heater1": {"10": {"normal": 100.0}}
    }
    # Rebuild log with base=100 so impacts in [4, 7] are well under base.
    # Update both ``unit_breakdown`` (actual) and ``unit_expected_breakdown``
    # (the log-time base diagnose-match reads); otherwise the LS would see
    # actual=92, expected=2 → negative impact → filter as non_positive.
    new_entries = []
    for i, e in enumerate(coord._hourly_log):
        sv = e["solar_vector_s"]
        impact = 10.0 * sv
        e["actual_kwh"] = 100.0 - impact
        e["unit_breakdown"]["sensor.heater1"] = e["actual_kwh"]
        e["unit_expected_breakdown"]["sensor.heater1"] = 100.0
        new_entries.append(e)
    coord._hourly_log = new_entries

    result = await coord.async_apply_implied_coefficient(
        entity_id="sensor.heater1", mode=MODE_HEATING
    )
    assert result["status"] == "ok"
    # The LS produced ~10.0 but the clamp caps at 5.0 (SOLAR_COEFF_CAP).
    assert result["after"]["s"] <= SOLAR_COEFF_CAP
    assert result["after"]["s"] >= 0.0


# -----------------------------------------------------------------------------
# days_back parameter
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_days_back_filters_log_to_recent_window():
    """``days_back`` restricts the input log to the last N days.  The
    implementation calls ``dt_util.utcnow() - timedelta(days=N)`` and
    keeps entries with ``timestamp >= cutoff``.

    Build a coordinator whose log spans ~5 days starting from a known
    ``utcnow``.  With days_back=2, only the last ~2 days of entries
    qualify; sample_count drops accordingly.
    """
    from datetime import datetime, timedelta, timezone
    from unittest.mock import patch as _patch

    coord = _make_coord(n_samples=60)
    # Rewrite timestamps so the log spans 5 days ending "now" — entries
    # 0-23 are 5 days ago, 24-47 are 4 days ago, ..., 48-59 are within
    # the last day.
    now = datetime(2026, 5, 1, 12, 0, 0, tzinfo=timezone.utc)
    new_entries = []
    for i, e in enumerate(coord._hourly_log):
        days_ago = 5 - (i // 12)  # 12 entries per virtual "day"
        ts = (now - timedelta(days=days_ago)).isoformat()
        e["timestamp"] = ts
        new_entries.append(e)
    coord._hourly_log = new_entries

    with _patch(
        "homeassistant.util.dt.utcnow", return_value=now
    ):
        # Full log: 60 samples.
        result_full = await coord.async_apply_implied_coefficient(
            entity_id="sensor.heater1",
            mode=MODE_HEATING,
            dry_run=True,
            days_back=None,
        )
        # Last 2 days: ~24 samples (48-59 + some of 36-47 depending on
        # exact cutoff).  Strict assertion: fewer than the full count
        # AND non-zero.
        result_recent = await coord.async_apply_implied_coefficient(
            entity_id="sensor.heater1",
            mode=MODE_HEATING,
            dry_run=True,
            days_back=2,
        )

    assert result_full["sample_count"] == 60
    assert result_recent["sample_count"] < result_full["sample_count"]
    assert result_recent["sample_count"] > 0
    assert result_recent["days_back"] == 2
    assert result_full["days_back"] is None


@pytest.mark.asyncio
async def test_days_back_returned_in_response_contract():
    """All return paths surface ``days_back`` so downstream consumers
    (dashboards, automations) can show which window was used."""
    coord = _make_coord()
    # Below-threshold case (no_data path).
    coord._hourly_log = []
    result = await coord.async_apply_implied_coefficient(
        entity_id="sensor.heater1",
        mode=MODE_HEATING,
        days_back=14,
    )
    assert result["status"] == "no_data"
    assert result["days_back"] == 14


@pytest.mark.asyncio
async def test_days_back_propagates_coordinator_to_helper():
    """Regression: a future refactor that drops ``days_back`` from the
    coordinator → helper kwarg list (or passes it under the wrong name)
    would silently use full-log fits.  Stub the helper's return value
    and assert it received the value the coordinator was called with.
    """
    coord = _make_coord()
    captured: dict = {}
    # Return an empty-result stub so the coordinator's downstream
    # branching exits cleanly (no_data path).
    stub_response = {
        "sample_count": 0,
        "drop_counts": {},
        "days_back": 14,
        "implied": None,
        "windows": [None, None, None],
    }

    def _capture(*args, **kwargs):
        captured.update(kwargs)
        return stub_response

    with patch.object(coord.learning, "compute_implied_for_apply",
                       side_effect=_capture):
        await coord.async_apply_implied_coefficient(
            entity_id="sensor.heater1",
            mode=MODE_HEATING,
            dry_run=True,
            days_back=14,
        )
    assert captured.get("days_back") == 14


@pytest.mark.asyncio
async def test_days_back_zero_or_negative_treated_as_none():
    """``days_back <= 0`` falls through to "use full log" — consistent
    with batch_fit_solar's filter logic.  Defensive against bad inputs
    that bypass the schema (e.g. programmatic callers, or a future
    schema change).  Schema's ``min: 7`` enforces sane values via the
    service-call path; this test exercises the coordinator-method API
    where any int (or None) is accepted.
    """
    coord = _make_coord(n_samples=60)
    result_full = await coord.async_apply_implied_coefficient(
        entity_id="sensor.heater1",
        mode=MODE_HEATING,
        dry_run=True,
        days_back=None,
    )
    result_zero = await coord.async_apply_implied_coefficient(
        entity_id="sensor.heater1",
        mode=MODE_HEATING,
        dry_run=True,
        days_back=0,
    )
    result_neg = await coord.async_apply_implied_coefficient(
        entity_id="sensor.heater1",
        mode=MODE_HEATING,
        dry_run=True,
        days_back=-5,
    )
    # Both fall through to full-log path → identical sample counts.
    # Strict assertion: both paths actually hit the log (non-zero
    # counts) — guards against a fixture bug where both legs return
    # 0 samples and the equality assertion vacuously passes.
    assert result_full["sample_count"] > 0, "fixture should produce qualifying samples"
    assert result_zero["sample_count"] == result_full["sample_count"]
    assert result_neg["sample_count"] == result_full["sample_count"]


# -----------------------------------------------------------------------------
# Timestamp parsing — Codex P2 finding
# -----------------------------------------------------------------------------

class TestDaysBackTimestampParsing:
    """Regression for the Codex review finding: ``days_back`` filter
    initially compared timestamps as plain strings (lex compare).  ISO
    strings with different tz-offsets are NOT chronologically ordered
    by lex — e.g. ``"2026-03-27T11:00:00+02:00"`` (= 09:00 UTC) lex-
    compares greater than ``"2026-03-27T10:33:53+00:00"`` even though
    the first is chronologically earlier.

    Production hourly log entries may carry non-UTC offsets depending
    on how they're written; the fix parses both sides into datetime
    objects before comparing.
    """

    def test_filter_excludes_chronologically_older_entry_with_later_local_time(self):
        """Boundary case: an entry written in CEST that's chronologically
        BEFORE the UTC cutoff was incorrectly INCLUDED by lex compare.
        With the parsed-datetime fix, it's correctly EXCLUDED.
        """
        from datetime import datetime, timedelta, timezone
        from custom_components.heating_analytics.learning import (
            _filter_log_by_days_back,
        )

        # Mock dt_util.utcnow to a fixed value, then build entries
        # whose timestamps are tz-aware but use different offsets.
        now_utc = datetime(2026, 4, 26, 10, 33, 53, tzinfo=timezone.utc)
        cest = timezone(timedelta(hours=2))

        # Entry A: written in CEST, chronologically OLDER than the
        # days_back=30 cutoff (2026-03-27T10:33:53Z).  Local timestamp
        # "2026-03-27T11:00:00+02:00" = "2026-03-27T09:00:00Z" — that's
        # 1.5 hours BEFORE the cutoff.  Should be excluded.
        entry_a = {
            "timestamp": datetime(2026, 3, 27, 11, 0, 0, tzinfo=cest).isoformat(),
        }
        # Entry B: written in CEST, chronologically AFTER the cutoff.
        # "2026-03-27T13:00:00+02:00" = "2026-03-27T11:00:00Z" — 0.5h
        # AFTER the cutoff.  Should be included.
        entry_b = {
            "timestamp": datetime(2026, 3, 27, 13, 0, 0, tzinfo=cest).isoformat(),
        }
        # Entry C: pre-cutoff by half a day, plain UTC.  Should be excluded.
        entry_c = {
            "timestamp": datetime(2026, 3, 26, 22, 0, 0, tzinfo=timezone.utc).isoformat(),
        }

        with patch(
            "homeassistant.util.dt.utcnow", return_value=now_utc
        ):
            result = _filter_log_by_days_back(
                [entry_a, entry_b, entry_c], days_back=30
            )

        ts_kept = {e["timestamp"] for e in result}
        assert entry_b["timestamp"] in ts_kept, (
            "Entry chronologically AFTER cutoff must be kept"
        )
        assert entry_a["timestamp"] not in ts_kept, (
            "Pre-fix lex compare: '2026-03-27T11+02:00' > '2026-03-27T10:33:53+00:00' "
            "→ INCLUDED (wrong).  Post-fix parsed compare: 09:00 UTC < 10:33:53 UTC → "
            "EXCLUDED (right)."
        )
        assert entry_c["timestamp"] not in ts_kept

    def test_naive_timestamps_treated_as_utc(self):
        """Legacy log entries without tz-offset are interpreted as UTC.
        Naive entries must still filter chronologically; the helper
        attaches UTC tzinfo before comparing."""
        from datetime import datetime, timedelta, timezone
        from custom_components.heating_analytics.learning import (
            _filter_log_by_days_back,
        )

        now_utc = datetime(2026, 4, 26, 10, 33, 53, tzinfo=timezone.utc)
        # Naive timestamps (no offset suffix) — typical of older logs.
        recent = {"timestamp": "2026-04-25T15:00:00"}  # 1 day before now → included
        old = {"timestamp": "2026-03-26T15:00:00"}      # 31 days before now → excluded

        with patch("homeassistant.util.dt.utcnow", return_value=now_utc):
            result = _filter_log_by_days_back([recent, old], days_back=30)

        assert recent in result
        assert old not in result

    def test_unparseable_timestamps_dropped_silently(self):
        """Garbage timestamps (corrupt log entry, missing field) are
        dropped rather than raising.  Downstream sample collector would
        have dropped them anyway; doing it here avoids the parse error
        propagating."""
        from custom_components.heating_analytics.learning import (
            _filter_log_by_days_back,
        )

        entries = [
            {"timestamp": "not a timestamp"},
            {"timestamp": ""},
            {},  # no timestamp key at all
            {"timestamp": "2026-04-26T10:00:00+00:00"},  # valid, recent
        ]
        from datetime import datetime, timezone
        with patch(
            "homeassistant.util.dt.utcnow",
            return_value=datetime(2026, 4, 26, 10, 33, 53, tzinfo=timezone.utc),
        ):
            result = _filter_log_by_days_back(entries, days_back=30)

        # Only the valid recent entry survives.
        assert len(result) == 1
        assert result[0]["timestamp"] == "2026-04-26T10:00:00+00:00"

    def test_days_back_none_or_zero_returns_input_unchanged(self):
        """No filtering when days_back is None / 0 / negative."""
        from custom_components.heating_analytics.learning import (
            _filter_log_by_days_back,
        )
        log = [{"timestamp": "2020-01-01T00:00:00+00:00"}]
        assert _filter_log_by_days_back(log, None) is log
        assert _filter_log_by_days_back(log, 0) is log
        assert _filter_log_by_days_back(log, -5) is log
