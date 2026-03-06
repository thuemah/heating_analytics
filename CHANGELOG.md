# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.2.4] - 2026-03-06

### Added
- Added `test_asymmetric` flag to `calibrate_inertia` service. When enabled, evaluates a regime-based asymmetric inertia model alongside the standard symmetric scan: slow profile (4 h) when outdoor temperature is falling (heat shedding), fast profile (2 h) when rising (heat gaining), and a stable 3 h profile otherwise. Returns comparative R², RMSE, delta_r2 vs best symmetric, and a regime breakdown (shedding/gaining/stable hour counts).
- Added `test_delta_t_scaling` flag to `calibrate_inertia` service. Bins pure hours by (balance_point − outdoor_temp) in 5 °C steps and finds the optimal exponential τ per bin (`window = min(5τ, 168)`), consistent with the coordinator's runtime model. Tests the hypothesis that thermal inertia scales with temperature differential: low ΔT (mild weather) → short τ, high ΔT (deep winter) → long τ. A monotonic `best_tau` trend across bins confirms the hypothesis and would support a future ΔT-adaptive inertia model.

### Fixed
- **Solar learning buffer is no longer partitioned by temperature bucket.** The per-unit solar coefficient (`c_s`, `c_e`) models window area, orientation, and shading — physical properties that do not depend on outdoor temperature. Partitioning the cold-start buffer by `temp_key` caused it to accumulate at most 1–2 samples per bucket before the temperature drifted to the next bucket, so `LEARNING_BUFFER_THRESHOLD = 4` was never reached in any single bucket and jump-start never fired. The buffer is now a single flat list per unit (pooling solar observations across all temperatures), and the learned coefficient is stored globally per unit rather than per `(unit, temp_key)`. The `calculate_unit_coefficient` lookup in `solar.py` is simplified accordingly. On first load, storage migration flattens existing temp-stratified buffers into the new format and picks the first valid entry from any existing temp-stratified coefficient store.
- **Thermal inertia kernel corrected from Gaussian to causal exponential decay.** The Gaussian kernel `exp(−½((t − μ)/σ)²)` places its peak weight at the midpoint of the lookback window (e.g. 2 hours ago for a 4-hour setting), not at the current observation. A building's thermal mass follows first-order (RC-circuit) dynamics: the indoor heat content decays toward the outdoor temperature at a rate governed by the time constant τ = C_a / UA, where C_a is the effective thermal capacitance and UA is the total heat loss coefficient. The correct weighting function is therefore a causal exponential decay `e^(−t/τ)`, which assigns maximum weight to the present state and monotonically less to older samples — physically equivalent to the impulse response of an RC low-pass filter. The Gaussian has no analogue in building thermodynamics and systematically biased the effective temperature toward historical observations, effectively overstating inertia and producing recommendations that were too conservative. Replaced with `generate_exponential_kernel(tau)` throughout.
- **Kernel window capped at 5τ hours (captures 99.3 % of cumulative weight).** Previously the exponential kernel used the full 168-hour default window regardless of τ. For τ = 4 h this meant `_get_inertia_parameters` requested 167 hours of history and weighted 99 % of that data at negligible values. Window is now `min(5τ, 168)`, consistent with the standard RC-circuit approximation where five time constants represent effective convergence to steady state.
- **Stale-data gap detection decoupled from kernel window size.** `_get_inertia_parameters` derived `max_gap` from `len(inertia_weights)`, which after the window cap became 5τ instead of τ. This disabled the thermal-discontinuity guard: after a system restart or sensor outage, temperature logs from up to 20 hours ago (for τ = 4) were treated as continuous rather than discarded. `max_gap` is now `int(tau)`, preserving the original behaviour that only consecutive hourly data within one time constant is considered thermodynamically continuous.
- **`calibrate_inertia` now recommends τ directly applicable to the config-flow slider.** The service previously returned `best_overall.hours` from a Gaussian sweep. Because the coordinator uses an exponential kernel, the Gaussian recommendation was not directly comparable: the Gaussian peak at h = 6 compensated for its non-causal weighting by artificially widening the window, while the physical time constant is τ = 4 h. The primary output is now `recommended_tau` from an exponential sweep (τ = 1–24 h, `window = min(5τ, 168)`), matching the coordinator's runtime model exactly. The Gaussian sweep is retained as `gaussian_best_hours` / `gaussian_best_r2` for comparison. Weekly stability analysis and the optional extended sweep (`test_exponential_kernel`, τ = 1–72 h) are also converted to exponential.
- **Inertia weights attribute now shows normalised values (sum = 1.0).** The `weights` field on the Thermal State sensor previously exposed raw kernel fractions normalised over the full window (e.g. sum ≈ 0.64 when 10 of 45 kernel positions were active). The internal calculation already re-normalises within the active subset, but the displayed values did not reflect this, making manual verification (`Σ history[i] × weight[i] ≈ effective_temperature`) impossible without dividing by the displayed sum. Weights are now re-normalised to the active sample count before display.

### Changed
- **Cold-start solar coefficient estimates are now dampened by 50% before publishing.** The 4-sample least squares estimate used for jump-start is computed while the base model is itself in early learning, making `actual_impact` noisy. This systematically inflates the initial coefficient, which at the 3 % EMA cap takes 50–70 observations to correct downward. A `COLD_START_SOLAR_DAMPING = 0.5` factor is applied to both the 2D and 1D collinear jump-start paths so EMA converges from a conservative underestimate rather than correcting an overshoot. The damping value is a named constant and can be tuned as more real-world data is collected.
- Lowered aux cooldown convergence threshold from 95% to 92%. Radiant heat sources (e.g. wood stoves) warm walls and furniture in ways that linger well beyond the active burn period; waiting for 95% normalisation was unnecessarily conservative. 92% exits the learning lock earlier while still preventing thermal-lag bias in the model.

## [1.2.3] - 2026-03-04

### Added
- Added `calibrate_wind_thresholds` service that analyses historical "pure" hours (no aux, no solar) and performs a brute-force grid search over `high_wind` (3–10 m/s) and `extreme_wind` (high+2 to high+8 m/s) threshold pairs. For each candidate pair, hours are reclassified and compared against the existing global model's expected consumption to compute MAE. Returns the recommended thresholds, current MAE, a warning if fewer than 30 windy hours were found, and the top-10 candidates. Accepts an optional `days` parameter (default 60, max 180).

### Fixed
- Fixed `TypeError: cannot unpack non-iterable float object` on startup when calculating potential solar impact. The coordinator was passing `potential_solar_factor` (scalar) instead of `potential_solar_vector` (tuple) to `calculate_unit_solar_impact()`.
- Fixed solar optimizer model (screen recommendations) being silently reset to empty on every restart. `async_load_data` was missing the call to `solar_optimizer.set_data()`, so the learned insulate/maximize_solar model was discarded and overwritten on first save.

### Changed
- Removed **Window Orientation (Azimuth)** from the configuration UI. With the 2D solar vector system, window orientation is learned empirically through per-unit `(s, e)` coefficient vectors and no longer requires manual input. The internal 180° south default is retained for cold-start initialization and historical log reconstruction.

## [1.2.2] - 2026-03-03

### Added
- Added `reset_solar_learning` service to clear per-unit solar coefficients and learning buffers without affecting the base model or aux learning data. Accepts an optional `entity_id` to reset a single unit; omitting it resets all units.

## [1.2.1] - 2026-03-02

### Added
- Added `calibrate_inertia` service that analyses historical "pure" hours (no aux, no solar, learning active) and tests Gaussian kernels from 1–24 hours, returning R² and RMSE for each. The service returns the best overall profile, a top-5 ranking, a per-week stability breakdown, and a stability score. Intended to replace the arbitrary 4-hour default with a data-driven recommendation.
- Added `centered_energy_average` parameter to `calibrate_inertia` (default `false`). When enabled, actual energy consumption is smoothed with a 3-hour centred moving average before correlation is calculated, reducing noise from heat pump defrost cycles or partial-hour meter delays without shifting the identified inertia peak.
- Added `generate_gaussian_kernel(hours)` helper in `helpers.py`, shared between the coordinator's live inertia calculation and the calibration service.

### Changed
- Thermal inertia configuration replaced the fast / normal / slow dropdown with a numeric slider (1–24 h). The coordinator generates the Gaussian kernel dynamically from the chosen value. Existing configurations are migrated automatically: `fast → 2 h`, `normal → 4 h`, `slow → 12 h`.
- Removed `hourly` sub-section from the `forecast_accuracy_by_source` attribute on the Weather Plan Today sensor. The hourly temperature error metrics (`p50_abs_error`, `p95_abs_error`, `mae_7d`, etc.) are internal inputs to the source-blending logic and not meaningful to end users. Only the `daily` energy accuracy metrics are now exposed.

### Fixed
- Fixed `calibrate_inertia` incorrectly excluding hours logged with `learning_status: "active"`. Only `"success"` and `"model_updated"` were accepted, discarding the majority of normal operating hours.
- Fixed a restart-at-hourly-boundary race condition where `_last_hour_processed` was not persisted to storage. After a restart or config reload at the top of an hour the coordinator re-initialised the tracker to the current hour, silently skipping `_process_hourly_data` for the missed hour and leaving a gap in `_hourly_log`. The gap appeared as fewer inertia samples than the configured profile width (e.g. 3 samples instead of 4). A duplicate-entry guard is also added to `_process_hourly_data` as a safe fallback if the crash occurred after the log entry was already appended.
- Fixed "Deviation Today" percentage drifting with weather forecast revisions. The numerator now compares `forecast_today_gross` against `predicted_gross` — both in the aux-unaware (gross) domain — so live forecast updates cancel out in the difference and only genuine consumption deviations are surfaced. The denominator remains the frozen `midnight_forecast` to prevent the percentage from scaling with cold-weather revisions mid-day.

### Docs
- Updated DESIGN.md section 3B–3C with precise definitions of all per-source accuracy metrics exposed in `forecast_accuracy_by_source`: `mae_Xd`/`mape_Xd` (net daily energy error, kWh/day), `weather_mae_Xd` (mean absolute net daily temperature deviation, °C), and `weather_bias_Xd` (signed mean net daily temperature deviation, °C). Clarifies sign convention, why net daily error is used over sum of hourly absolutes, and that temperature metrics serve as internal blend-selection inputs rather than end-user KPIs.

## [1.2.0] - 2026-03-01

### Fixed
- Fixed historical weather data (temperature, wind, wind bucket) being discarded in forecast fallback paths. When outside the forecast horizon, `calculate_modeled_energy()` already computed last year's weather values, but both `comparison.py` and `statistics.py` returned `None` for these fields instead of the actual historical values.
- Replaced heuristic 5°–10° solar elevation cutoff/fade with a physics-based air mass transmittance model (`intensity = 0.7 ** (1 / sin(elevation))`). The model naturally attenuates solar factor at low sun angles without an arbitrary threshold.

## [1.1.3] - 2026-02-28

### Fixed
- Fixed thermal inertia using a raw instantaneous sensor reading as the current-hour temperature instead of the hourly rolling average. Historical inertia points (H-3, H-2, H-1) are stored as hourly averages, so the current point must use the same representation for a consistent weighted calculation. The fix prefers `hourly_temp_sum / hourly_sample_count` when samples exist, falling back to the raw sensor reading only at the very start of a new hour.

### Docs
- Added documentation regarding thermal hysteresis limitation in DESIGN.md

## [1.1.2] - 2026-02-27

### Fixed
- Fixed `actual_kwh` inflation after a restart spanning an hour boundary. Stale energy meter baselines (`_last_energy_values`) were previously retained across hour changes on restore, causing the first delta after restart to absorb the entire missed gap into the next logged hour. Baselines are now cleared when the restored data belongs to a different hour, so the first reading after restart establishes a clean baseline.
- Atmospheric attenuation fade zone to 5°–10° solar elevation. the system overestimated solar heating effect when the sun is low on the horizon. The mathematical model now includes an atmospheric attenuation factor. Cutoff: Solar factor is forced to 0.0 when elevation is < 5 degrees.

## [1.1.1] - 2026-02-24

### Fixed
- Fixed midnight forecast showing as "Unknown" after restart until next hour boundary. Snapshot values are now published to sensor data immediately during storage restore.
- Restored default solar coefficients to 0.35/0.40 (heating/cooling). The previous reduction to 0.15/0.17 was based on COP-adjusted estimates that proved too conservative in practice, causing solar gain to be under-attributed and base model drift during sunny hours.

## [1.1.0] - 2026-02-21

### Added
- Added `get_forecast` service for retrieving detailed hourly heating plans.
- Added support for configurable Thermal Inertia Profiles (Fast, Normal, Slow) to better match different building types.

## [1.0.0] - 2026-02-21

- Initial release.
