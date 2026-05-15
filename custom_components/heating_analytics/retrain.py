"""RetrainEngine — hosts retrain_from_history() extracted from coordinator.py.

Thin-delegate pattern: the engine holds a reference to the coordinator
and reaches back for state.  Public methods are called via delegates
on the coordinator so the external API is unchanged.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import date, datetime, timedelta

from homeassistant.util import dt as dt_util

from .const import (
    DEFAULT_DAILY_LEARNING_RATE,
    MODE_COOLING,
    MODE_DHW,
    MODE_GUEST_COOLING,
    MODE_GUEST_HEATING,
    MODE_HEATING,
    MODE_OFF,
    MODES_EXCLUDED_FROM_GLOBAL_LEARNING,
    SNR_WEIGHT_FLOOR,
    SNR_WEIGHT_K,
)
from .learning import compute_snr_weight, count_active_learnable_units
from .observation import DirectMeter, WeightedSmear

_LOGGER = logging.getLogger(__name__)


def _screen_affected_set_or_none(coordinator) -> frozenset[str] | None:
    """Return coordinator._screen_affected_set if it is a real set/frozenset.

    Guards against MagicMock-based test coordinators where ``getattr`` would
    return a MagicMock (truthy, `__contains__` returns False) and silently
    route every entity down the "unscreened" branch inside
    :meth:`learning.replay_solar_nlms` — masking inequality-path regressions
    in tests that don't explicitly set the attribute.
    """
    value = getattr(coordinator, "_screen_affected_set", None)
    if isinstance(value, (frozenset, set)):
        return value
    return None


def _solar_affected_set_or_none(coordinator) -> frozenset[str] | None:
    """Return coordinator._solar_affected_set if it is a real set/frozenset (#962).

    MagicMock-safe sibling of :func:`_screen_affected_set_or_none`.
    """
    value = getattr(coordinator, "_solar_affected_set", None)
    if isinstance(value, (frozenset, set)):
        return value
    return None


class RetrainEngine:
    """Hosts the retrain_from_history service implementation."""

    def __init__(self, coordinator) -> None:
        self.coordinator = coordinator

    async def retrain_from_history(self, days_back: int | None = None, reset_first: bool = False, experimental_cop_smear: bool = False) -> dict:
        """Retrain the learning model from existing hourly log data.

        Track A (daily_learning_mode=False): replays each logged hour through
        learn_from_historical_import(), honouring aux/base routing.

        Track B (daily_learning_mode=True): groups hours by day and applies the
        same midnight EMA logic as the live calibration. Thermal mass correction
        is applied when an indoor_temp_sensor is configured AND the hourly log
        contains 'indoor_temp' entries; otherwise it is skipped gracefully.
        """
        if reset_first:
            self.coordinator._correlation_data.clear()
            self.coordinator._correlation_data_per_unit.clear()
            self.coordinator._aux_coefficients.clear()
            self.coordinator._aux_coefficients_per_unit.clear()
            self.coordinator._learning_buffer_global.clear()
            self.coordinator._learning_buffer_per_unit.clear()
            self.coordinator._learning_buffer_aux_per_unit.clear()
            self.coordinator._solar_coefficients_per_unit.clear()
            self.coordinator._learning_buffer_solar_per_unit.clear()
            self.coordinator._observation_counts.clear()
            self.coordinator._learned_u_coefficient = None
            _LOGGER.info("retrain_from_history: Model reset before retraining.")

        if days_back is not None:
            from homeassistant.util import dt as dt_util
            from datetime import timedelta
            cutoff_str = (dt_util.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
            entries = [e for e in self.coordinator._hourly_log if e.get("timestamp", "") >= cutoff_str]
        else:
            entries = list(self.coordinator._hourly_log)

        if not entries:
            return {"status": "no_data", "entries_processed": 0, "days_processed": 0, "learning_count": 0}

        learning_count = 0

        def _is_poisoned(entry: dict, daily_mode: bool = False) -> bool:
            s = entry.get("learning_status", "unknown")
            # In daily_learning_mode, "disabled" is the normal state for every hour
            # (Track A is blocked from writing to correlation_data). These hours
            # contain valid sensor data and must NOT be filtered out for Track B/C
            # daily aggregation.  Only genuine data-quality issues are poisoned.
            if daily_mode and s == "disabled":
                return False
            return s == "disabled" or s.startswith("skipped_") or s == "cooldown_post_aux"

        if self.coordinator.daily_learning_mode:
            # Daily learning: batch aggregation per calendar day using strategy dispatch (#776).
            # Poisoned hours are excluded before grouping so the <22-hour guard
            # automatically rejects days with too many bad hours.
            daily_batches: dict[str, list] = {}
            for entry in entries:
                if not _is_poisoned(entry, daily_mode=True):
                    daily_batches.setdefault(entry["timestamp"][:10], []).append(entry)

            # NLMS-replay 3-pass EM-lite (#847) for reset_first=True.
            # Only the Track B flat daily branch depends on the solar delta;
            # Track C and Track B COP-smear use pre-computed distributions
            # (synthetic_kwh_el) that are NOT biased by stored delta.  The
            # ``_b_flat_delta_source`` below controls how flat-daily
            # calculates retrain_solar_delta per pass:
            #   "stored"     — sum(e.solar_normalization_delta) from log
            #                  (current behaviour; use when reset_first=False)
            #   "none"       — 0 (priming pass; avoids reintroducing old
            #                  solar contamination before NLMS has re-run)
            #   "on_the_fly" — recompute from current solar coefficients
            #                  (refinement pass after NLMS replay)
            async def _run_daily_pass(_b_flat_delta_source: str) -> tuple[int, int, int]:
                """One pass of the Track B/C daily-replay loop.

                ``_b_flat_delta_source`` in {"stored","none","on_the_fly"}
                selects which solar_normalization_delta is used for the
                Track B flat-daily path.  Track C and Track B COP-smear
                paths use pre-computed distributions and are unaffected by
                this parameter.
                Returns (days_processed, learning_count, days_skipped_mpc).
                """
                days_processed_local = 0
                learning_count_local = 0
                days_skipped_mpc_local = 0  # #855 Option B
                for date_str, day_entries in sorted(daily_batches.items()):
                    if len(day_entries) < 22:
                        _LOGGER.debug(f"retrain_from_history: Skipping {date_str} — {len(day_entries)}/24 hours")
                        continue

                    # #855 Option B: on Track C installs, a day without
                    # ``track_c_distribution`` is either an MPC outage OR
                    # pre-Track-C history.  Both cases are unsafe to process
                    # because the Track B fallback would write Track-B-semantic
                    # values (raw electrical ± thermal-mass correction) into
                    # the same correlation_data buckets that Track C fills
                    # with MPC-synthetic thermal-per-hour.  Skip entirely;
                    # users who want pre-Track-C days included can run
                    # retrain with a narrower ``days_back`` that excludes
                    # the pre-Track-C period, or temporarily disable Track C.
                    if self.coordinator.track_c_enabled:
                        if not self.coordinator._daily_history.get(date_str, {}).get("track_c_distribution"):
                            _LOGGER.warning(
                                "retrain: skipping %s — no track_c_distribution "
                                "(MPC outage or pre-Track-C history).",
                                date_str,
                            )
                            days_skipped_mpc_local += 1
                            continue

                    total_kwh = sum(e.get("actual_kwh", 0.0) for e in day_entries)
                    daily_tdd = sum(e.get("tdd", 0.0) for e in day_entries)

                    if daily_tdd < 0.5 or total_kwh <= 0:
                        continue

                    # Mode filtering (#789): exclude OFF/DHW/Guest/Cooling from retrain.
                    excluded_kwh = self.coordinator._compute_excluded_mode_energy(day_entries)
                    total_kwh -= excluded_kwh

                    if total_kwh <= 0:
                        continue

                    # Determine q_adjusted for U-coefficient.
                    track_c_daily = self.coordinator._daily_history.get(date_str, {}).get("track_c_kwh")
                    if self.coordinator.track_c_enabled and track_c_daily is not None:
                        q_adjusted = track_c_daily
                        # Backward compat: days stored before non-MPC inclusion.
                        if self.coordinator.mpc_managed_sensor and "track_c_kwh_non_mpc" not in self.coordinator._daily_history.get(date_str, {}):
                            non_mpc_retrain = 0.0
                            for log_entry in day_entries:
                                breakdown = log_entry.get("unit_breakdown", {})
                                for sid in self.coordinator.energy_sensors:
                                    if sid != self.coordinator.mpc_managed_sensor:
                                        non_mpc_retrain += breakdown.get(sid, 0.0)
                            q_adjusted += non_mpc_retrain
                    else:
                        q_adjusted = total_kwh
                        if self.coordinator.thermal_mass_kwh_per_degree > 0.0:
                            from datetime import date as _date, timedelta as _td
                            prev_day_str = (
                                _date.fromisoformat(date_str) - _td(days=1)
                            ).isoformat()
                            end_temp = self.coordinator._daily_history.get(date_str, {}).get("midnight_indoor_temp")
                            start_temp = self.coordinator._daily_history.get(prev_day_str, {}).get("midnight_indoor_temp")
                            if end_temp is not None and start_temp is not None:
                                delta_t_indoor = end_temp - start_temp
                                q_adjusted = total_kwh - (self.coordinator.thermal_mass_kwh_per_degree * delta_t_indoor)

                    if q_adjusted <= 0:
                        continue

                    # U-coefficient update.
                    observed_u = q_adjusted / daily_tdd
                    if self.coordinator._learned_u_coefficient is None:
                        self.coordinator._learned_u_coefficient = observed_u
                    else:
                        self.coordinator._learned_u_coefficient += DEFAULT_DAILY_LEARNING_RATE * (
                            observed_u - self.coordinator._learned_u_coefficient
                        )

                    # Bucket learning: Track C uses strategy dispatch, Track B uses flat daily.
                    track_c_dist = self.coordinator._daily_history.get(date_str, {}).get("track_c_distribution")
                    if track_c_dist:
                        self.coordinator._apply_strategies_to_global_model(day_entries, track_c_dist)
                    else:
                        # Check for stored COP-smeared distribution, or generate
                        # on-the-fly if experimental_cop_smear is active (#793).
                        track_b_cop_dist = self.coordinator._daily_history.get(date_str, {}).get("track_b_cop_distribution")
                        if not track_b_cop_dist and experimental_cop_smear:
                            cop_smeared = await self.coordinator._try_track_b_cop_smearing(
                                day_entries, q_adjusted, date_str,
                            )
                            if cop_smeared is not None:
                                _LOGGER.info(f"retrain COP-smear (#793): {date_str} -> {cop_smeared} bucket updates")
                                # Per-unit replay + continue — bucket writing already done
                                self.coordinator._replay_per_unit_models(day_entries)
                                learning_count_local += 1
                                days_processed_local += 1
                                continue
                        if track_b_cop_dist:
                            self.coordinator._apply_strategies_to_global_model(day_entries, track_b_cop_dist)
                        else:
                            # Track B flattened daily bucket learning.
                            #
                            # Compute a DAY-LEVEL SNR weight: average the
                            # hour-level solar factors across the day and
                            # apply the same (FLOOR, K) mapping used on
                            # Track A.  Mostly-dark days retain full rate;
                            # sunny days down-weighted; all-shutdown zeroed.
                            # The target is raw ``q_adjusted / 24``.  The
                            # dark-equivalent bucket semantics are preserved
                            # (prediction consumers subtract solar impact
                            # from the base) because dark/overcast days
                            # dominate the weighted EMA.
                            avg_temp_retrain = sum(e.get("temp", 0.0) for e in day_entries) / len(day_entries)
                            daily_wind_retrain = sum(e.get("effective_wind", 0.0) for e in day_entries) / len(day_entries)
                            flat_temp_key = str(int(round(avg_temp_retrain)))
                            flat_wind_bucket = self.coordinator._get_wind_bucket(daily_wind_retrain)

                            avg_solar_factor = sum(
                                e.get("solar_factor", 0.0) for e in day_entries
                            ) / len(day_entries)
                            # Reuse the same shape as hour-level weight.
                            # Shutdown fraction at day level: count entries
                            # where ALL units shut down.  In practice this
                            # is rare at day granularity, but the check
                            # mirrors hour-level semantics.
                            n_all_shutdown = sum(
                                1 for e in day_entries
                                if len(e.get("solar_dominant_entities") or []) >= len(self.coordinator.energy_sensors)
                                and self.coordinator.energy_sensors
                            )
                            clean_fraction = (
                                (len(day_entries) - n_all_shutdown) / len(day_entries)
                                if day_entries else 1.0
                            )
                            day_weight = max(
                                SNR_WEIGHT_FLOOR,
                                1.0 - SNR_WEIGHT_K * max(0.0, avg_solar_factor),
                            ) * clean_fraction
                            q_hourly_avg = q_adjusted / 24.0
                            effective_rate = self.coordinator.learning_rate * day_weight

                            if flat_temp_key not in self.coordinator._correlation_data:
                                self.coordinator._correlation_data[flat_temp_key] = {}
                            current_pred = self.coordinator._correlation_data[flat_temp_key].get(flat_wind_bucket, 0.0)

                            if current_pred == 0.0:
                                # Seed from the raw hourly average.  A
                                # zero-weight day (all shutdown) is skipped
                                # to avoid seeding with actual ≈ 0.
                                if effective_rate > 0.0:
                                    self.coordinator._correlation_data[flat_temp_key][flat_wind_bucket] = round(q_hourly_avg, 5)
                            else:
                                new_pred = current_pred + effective_rate * (q_hourly_avg - current_pred)
                                self.coordinator._correlation_data[flat_temp_key][flat_wind_bucket] = round(new_pred, 5)

                    # Per-unit model replay for DirectMeter sensors (needed for isolate_sensor).
                    self.coordinator._replay_per_unit_models(day_entries)

                    learning_count_local += 1
                    days_processed_local += 1
                return days_processed_local, learning_count_local, days_skipped_mpc_local

            # Single-pass: base learns with day-level SNR weighting, then
            # NLMS replay refreshes solar coefficients orthogonally.  The
            # ``_b_flat_delta_source`` argument no longer affects base
            # learning (SNR weighting ignores the delta) and is kept as
            # "none" for signature stability in the inner helper.
            days_processed, learning_count, days_skipped_mpc = await _run_daily_pass("none")
            solar_replay_diagnostics = self.coordinator.learning.replay_solar_nlms(
                entries,
                solar_calculator=self.coordinator.solar,
                screen_config=getattr(self.coordinator, "screen_config", None),
                correlation_data_per_unit=self.coordinator._correlation_data_per_unit,
                solar_coefficients_per_unit=self.coordinator._solar_coefficients_per_unit,
                learning_buffer_solar_per_unit=self.coordinator._learning_buffer_solar_per_unit,
                energy_sensors=self.coordinator.energy_sensors,
                learning_rate=self.coordinator.learning_rate,
                balance_point=self.coordinator.balance_point,
                aux_affected_entities=self.coordinator.aux_affected_entities,
                unit_strategies=self.coordinator._unit_strategies,
                daily_history=self.coordinator._daily_history,
                unit_min_base=self.coordinator._per_unit_min_base_thresholds or None,
                screen_affected_entities=_screen_affected_set_or_none(self.coordinator),
                solar_affected_entities=_solar_affected_set_or_none(self.coordinator),
                return_diagnostics=True,
            )
            solar_replay_updates = solar_replay_diagnostics.get("updates", 0)
            em_passes = 1

            self.coordinator.data["learned_u_coefficient"] = self.coordinator._learned_u_coefficient
            await self.coordinator.storage.async_save_data(force=True)

            _LOGGER.info(
                f"retrain_from_history: Completed. "
                f"{days_processed} days learned, "
                f"U-coefficient={self.coordinator._learned_u_coefficient}, "
                f"{solar_replay_updates} NLMS updates."
            )
            return {
                "status": "completed",
                "mode": "strategy_dispatch",
                "entries_processed": len(entries),
                "days_processed": days_processed,
                "learning_count": learning_count,
                # #855 Option B: days skipped because Track C was enabled but
                # ``track_c_distribution`` was missing for the day (MPC outage
                # or pre-Track-C history).  Always 0 on non-Track-C installs.
                "days_skipped_mpc_unavailable": days_skipped_mpc,
                "learned_u_coefficient": round(self.coordinator._learned_u_coefficient, 4) if self.coordinator._learned_u_coefficient is not None else None,
                "solar_replay_updates": solar_replay_updates,
                "solar_replay_diagnostics": solar_replay_diagnostics,
                "em_passes": em_passes,
            }

        else:
            # Track A: per-hour replay.
            #
            # When reset_first=True we run a three-pass EM-lite to break the
            # base ↔ solar circularity (#847 NLMS-replay fix):
            #   Pass 1  base priming        — replay with solar_norm = 0 to
            #                                 get shape without importing
            #                                 prior solar contamination via
            #                                 the stored delta
            #   Pass 1b per-unit replay     — populate correlation_data_per_unit
            #                                 so NLMS replay has unit-level
            #                                 base reference
            #   Pass 2  NLMS replay         — re-learn solar coefficients
            #                                 from raw historical vectors
            #                                 using the primed base
            #   Pass 3  base refinement     — clear base, replay with
            #                                 on-the-fly solar_norm computed
            #                                 from the re-learned coefficients
            #   Pass 3b per-unit replay     — final per-unit alignment
            #
            # When reset_first=False we keep the existing single-pass base
            # replay (uses stored delta) and append an NLMS replay at the
            # end so post-retrain solar coefficients are refined against
            # whatever base the user currently has.
            def _run_base_pass() -> tuple[int, int, list[dict]]:
                """Run the Track A base-replay pass over ``entries``.

                The base EMA is driven by the per-hour SNR weight; the
                aux path reads each entry's stored
                ``solar_normalization_delta`` to attribute aux reductions
                on hours where aux and sun overlapped.
                """
                temp_history_local: list[float] = []
                skipped_local = 0
                processed_local: list[dict] = []
                learning_count_local = 0

                for entry in entries:
                    actual_kwh = entry.get("actual_kwh")
                    if actual_kwh is None:
                        skipped_local += 1
                        continue

                    # Mode filtering (#789 parity with live + Track B retrain):
                    # stored ``actual_kwh`` is total_energy_kwh (all units,
                    # all modes).  Live Track A learning uses
                    # ``learning_energy_kwh`` (OFF/DHW/Guest subtracted).
                    # Track B retrain has always subtracted via
                    # ``_compute_excluded_mode_energy``.  Track A retrain
                    # was the outlier — on hours where one unit was in
                    # DHW / OFF but others were heating, retrain inflated
                    # the base bucket by the excluded unit's kWh.  Fix:
                    # subtract excluded-mode energy per entry before
                    # feeding into learn_from_historical_import.
                    excluded_entry_kwh = 0.0
                    entry_unit_modes = entry.get("unit_modes", {}) or {}
                    entry_unit_breakdown = entry.get("unit_breakdown", {}) or {}
                    for _sid, _kwh in entry_unit_breakdown.items():
                        _mode = entry_unit_modes.get(_sid, MODE_HEATING)
                        if _mode in MODES_EXCLUDED_FROM_GLOBAL_LEARNING:
                            excluded_entry_kwh += _kwh
                    actual_kwh_filtered = max(0.0, actual_kwh - excluded_entry_kwh)

                    temp = entry.get("temp", 0.0)
                    wind_bucket = entry.get("wind_bucket", "normal")
                    is_aux = entry.get("auxiliary_active", False)

                    if len(temp_history_local) >= 4:
                        temp_history_local.pop(0)
                    temp_history_local.append(temp)

                    # Skip poisoned hours after updating temp_history so the
                    # inertia sliding window stays accurate for subsequent hours.
                    if _is_poisoned(entry):
                        skipped_local += 1
                        continue

                    inertia_avg = sum(temp_history_local) / len(temp_history_local)
                    temp_key_local = str(int(round(inertia_avg)))

                    # The base EMA inside ``learn_from_historical_import``
                    # uses ``snr_weight`` to scale the step size.  The
                    # aux path still reads ``solar_normalization_delta``
                    # (pulled from the stored entry) so aux-active hours
                    # overlapping with sun are attributed correctly.
                    delta = entry.get("solar_normalization_delta", 0.0)

                    # Active-units count proxy for retrain: use the
                    # entry's unit_breakdown (units with non-zero
                    # consumption this hour) instead of a per-unit
                    # base lookup, which would change pass-to-pass.
                    # This matches the spirit of count_active_learnable_units
                    # — a unit with no consumption isn't a signal-bearer
                    # this hour.  Mode filter still applies via entry's
                    # unit_modes.
                    breakdown_proxy = {
                        sid: float(kwh)
                        for sid, kwh in (entry.get("unit_breakdown") or {}).items()
                        if kwh and float(kwh) > 0.0
                    }
                    snr_w = compute_snr_weight(
                        entry.get("solar_factor", 0.0),
                        entry.get("solar_dominant_entities", []) or [],
                        total_units=count_active_learnable_units(
                            self.coordinator.energy_sensors,
                            entry.get("unit_modes", {}) or {},
                            breakdown_proxy,
                            min_base=0.0,  # presence > 0 in breakdown is enough
                        ),
                    )

                    status = self.coordinator.learning.learn_from_historical_import(
                        temp_key=temp_key_local,
                        wind_bucket=wind_bucket,
                        actual_kwh=actual_kwh_filtered,
                        is_aux_active=is_aux,
                        correlation_data=self.coordinator._correlation_data,
                        aux_coefficients=self.coordinator._aux_coefficients,
                        learning_rate=self.coordinator.learning_rate,
                        get_predicted_kwh_fn=self.coordinator._get_predicted_kwh,
                        actual_temp=temp,
                        solar_normalization_delta=delta,
                        snr_weight=snr_w,
                        solar_coefficients_per_unit=self.coordinator._solar_coefficients_per_unit,
                        energy_sensors=self.coordinator.energy_sensors,
                        unit_modes=entry.get("unit_modes"),
                    )
                    if "skipped" not in status:
                        learning_count_local += 1
                        processed_local.append(entry)
                    else:
                        skipped_local += 1
                return learning_count_local, skipped_local, processed_local

            solar_replay_updates = 0
            solar_replay_diagnostics: dict = {}
            skipped = 0
            learning_count = 0
            processed_entries: list[dict] = []

            # Single-pass Track A retrain + orthogonal NLMS replay.
            # Rationale: base learning uses the SNR-weighted per-hour rate;
            # dark hours dominate the weighted EMA so the resulting base
            # is solar-clean by construction.  NLMS replay then refreshes
            # solar coefficients against the final base.  The aux path
            # inside ``learn_from_historical_import`` still consumes the
            # stored ``solar_normalization_delta`` to attribute aux
            # reductions correctly on hours where aux and sun overlap.
            learning_count, skipped, processed_entries = _run_base_pass()
            if processed_entries:
                self.coordinator._replay_per_unit_models(processed_entries)
            solar_replay_diagnostics = self.coordinator.learning.replay_solar_nlms(
                entries,
                solar_calculator=self.coordinator.solar,
                screen_config=getattr(self.coordinator, "screen_config", None),
                correlation_data_per_unit=self.coordinator._correlation_data_per_unit,
                solar_coefficients_per_unit=self.coordinator._solar_coefficients_per_unit,
                learning_buffer_solar_per_unit=self.coordinator._learning_buffer_solar_per_unit,
                energy_sensors=self.coordinator.energy_sensors,
                learning_rate=self.coordinator.learning_rate,
                balance_point=self.coordinator.balance_point,
                aux_affected_entities=self.coordinator.aux_affected_entities,
                unit_strategies=self.coordinator._unit_strategies,
                daily_history=self.coordinator._daily_history,
                unit_min_base=self.coordinator._per_unit_min_base_thresholds or None,
                screen_affected_entities=_screen_affected_set_or_none(self.coordinator),
                solar_affected_entities=_solar_affected_set_or_none(self.coordinator),
                return_diagnostics=True,
            )
            solar_replay_updates = solar_replay_diagnostics.get("updates", 0)
            em_passes = 1

            await self.coordinator.storage.async_save_data(force=True)

            _LOGGER.info(
                f"retrain_from_history Track A: Completed. "
                f"{learning_count} entries learned, {skipped} skipped, "
                f"{solar_replay_updates} NLMS updates."
            )
            return {
                "status": "completed",
                "mode": "track_a_hourly",
                "entries_processed": len(entries),
                "days_processed": len({e["timestamp"][:10] for e in entries}),
                "learning_count": learning_count,
                "skipped": skipped,
                "solar_replay_updates": solar_replay_updates,
                "solar_replay_diagnostics": solar_replay_diagnostics,
                "em_passes": em_passes,
            }

