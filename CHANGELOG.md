# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
