# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-12-08

### Added
- Initial stable release
- Machine learning-based heating prediction using EMA (Exponential Moving Average)
- Temperature and wind correlation analysis
- Wind compensation with effective wind calculation (wind speed + gust factor)
- Multi-bucket learning model (normal, high_wind, extreme_wind, auxiliary_heating)
- Hourly and daily energy tracking with configurable spike filtering
- Import/Export functionality for historical data (CSV format)
- Auxiliary heating detection (wood stove/fireplace tracking)
- Heating Degree Days (HDD) calculations
- Real-time deviation detection and efficiency metrics
- Config flow UI for easy setup
- Number entities for runtime configuration (wind thresholds, learning rate, etc.)
- Binary sensors for efficiency status and auxiliary heating detection
- Services for data import/export
- Comprehensive dashboard configuration with Plotly graphs
- Persistent storage with automatic recovery after restarts
- Detailed logging for debugging and analysis

### Features
- **Spike Protection**: Configurable max energy delta to filter out sensor errors
- **Negative Change Filtering**: Automatically handles meter resets
- **Baseline Updates**: Smart baseline tracking when spikes are detected
- **30-day Rolling Buffer**: Hourly logs automatically pruned to last 720 hours
- **Weather Integration**: Uses outdoor temperature and wind sensors
- **Multi-sensor Support**: Tracks multiple energy meters simultaneously
- **Individual Unit Tracking**: Per-device energy consumption monitoring

### Documentation
- Comprehensive README with installation instructions
- DESIGN.md with architecture overview
- Dashboard configuration examples
- Import/Export documentation with CSV format examples

### Testing
- Unit tests for coordinator logic
- Binary sensor tests
- Persistence and data integrity tests
- Budget and inertia calculation tests

## [Unreleased]

### Planned Features
- Support for multiple zones/thermostats
- Advanced weather integration (humidity, pressure)
- Cost calculations with dynamic electricity pricing
- Predictive maintenance alerts
- Integration with Home Assistant Energy Dashboard
- Mobile app dashboard templates
