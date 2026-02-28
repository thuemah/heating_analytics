# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
