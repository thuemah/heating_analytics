# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
