"""Solar Optimizer Service."""
from __future__ import annotations

import logging
import math

from .const import (
    RECOMMENDATION_MAXIMIZE_SOLAR,
    RECOMMENDATION_INSULATE,
    RECOMMENDATION_MITIGATE_SOLAR,
)

_LOGGER = logging.getLogger(__name__)

class SolarOptimizer:
    """Optimizes solar screen usage based on learned user preferences."""

    def __init__(self, coordinator) -> None:
        """Initialize."""
        self.coordinator = coordinator
        self._model = {} # { state: { azimuth_bucket: { elevation_bucket: percent } } }
        self._learning_rate = 0.1 # Simple EMA

    def get_recommendation_state(self, temp: float, potential_solar_factor: float) -> str:
        """Determine recommendation state based on physical conditions."""
        balance_point = self.coordinator.balance_point

        if temp < balance_point:
            if potential_solar_factor > 0.1:
                return RECOMMENDATION_MAXIMIZE_SOLAR
            else:
                return RECOMMENDATION_INSULATE
        else:
            if potential_solar_factor > 0.1:
                return RECOMMENDATION_MITIGATE_SOLAR

        # Default fallback (e.g. warm and dark)
        return "none"

    def _get_elevation_bucket(self, elevation: float) -> str:
        """Get bucket for sun elevation (10 degree steps)."""
        if elevation <= 0:
            return "0"
        bucket = int(elevation / 10) * 10
        return str(bucket)

    def _get_azimuth_bucket(self, azimuth: float) -> str:
        """Get bucket for sun azimuth (30 degree steps)."""
        # Normalize to 0-360
        normalized = azimuth % 360
        bucket = int(normalized / 30) * 30
        return str(bucket)

    def predict_correction_percent(self, state: str, elevation: float, azimuth: float, default_percent: float) -> float:
        """Predict the correction percent for a given state, elevation and azimuth."""
        if state == "none":
            return default_percent

        elev_bucket = self._get_elevation_bucket(elevation)
        az_bucket = self._get_azimuth_bucket(azimuth)

        # Check learned model
        # Structure: state -> azimuth -> elevation -> percent
        if state in self._model:
            az_data = self._model[state]
            if az_bucket in az_data:
                elev_data = az_data[az_bucket]
                if elev_bucket in elev_data:
                    return elev_data[elev_bucket]

        # Fallback Defaults
        if state == RECOMMENDATION_MAXIMIZE_SOLAR:
            return 100.0 # Open screens (Max Solar)
        elif state == RECOMMENDATION_INSULATE:
            return 0.0 # Close screens (Insulate)
        elif state == RECOMMENDATION_MITIGATE_SOLAR:
            return 0.0 # Close screens (Block Sun)

        return default_percent

    def learn_correction_percent(self, state: str, elevation: float, azimuth: float, actual_percent: float, cloud_cover: float = 0.0):
        """Learn user preference for the given condition."""
        if state == "none" or elevation <= 0:
            return

        # Cloud Cover Constraint: Only learn when sky is clear enough (< 20%) to ensure
        # the user's action is actually responding to the sun.
        if cloud_cover >= 20.0:
            _LOGGER.debug(f"Solar Optimizer: Learning skipped due to cloud cover {cloud_cover}% (>= 20%).")
            return

        elev_bucket = self._get_elevation_bucket(elevation)
        az_bucket = self._get_azimuth_bucket(azimuth)

        if state not in self._model:
            self._model[state] = {}

        if az_bucket not in self._model[state]:
            self._model[state][az_bucket] = {}

        current_prediction = self._model[state][az_bucket].get(elev_bucket)

        if current_prediction is None:
            # First observation
            self._model[state][az_bucket][elev_bucket] = float(actual_percent)
            _LOGGER.info(f"Solar Optimizer [New]: State={state} Az={az_bucket} Elev={elev_bucket} -> {actual_percent}%")
        else:
            # EMA Update
            new_prediction = current_prediction + self._learning_rate * (actual_percent - current_prediction)
            self._model[state][az_bucket][elev_bucket] = round(new_prediction, 1)
            _LOGGER.debug(f"Solar Optimizer [Update]: State={state} Az={az_bucket} Elev={elev_bucket} -> {new_prediction:.1f}% (was {current_prediction:.1f}%)")

    def get_data(self) -> dict:
        """Get data for persistence."""
        return {
            "model": self._model
        }

    def set_data(self, data: dict):
        """Restore data from persistence with migration support."""
        if not data:
            return

        loaded_model = data.get("model", {})
        if not loaded_model:
            return

        migrated_model = {}
        migrated_count = 0

        for state, state_data in loaded_model.items():
            if not isinstance(state_data, dict):
                continue

            # Detect Format
            # New format: Values are dicts (azimuth -> {elev -> percent})
            # Old format: Values are floats (elev -> percent)

            if not state_data:
                migrated_model[state] = {}
                continue

            # Check first item to determine format
            first_key = next(iter(state_data))
            first_val = state_data[first_key]

            if isinstance(first_val, (int, float)):
                # Legacy Format Detected!
                _LOGGER.info(f"Solar Optimizer: Migrating legacy data for state '{state}' to azimuth-aware structure.")
                migrated_count += 1

                migrated_model[state] = {}
                # Replicate this elevation map across all 12 azimuth buckets
                for az in range(0, 360, 30):
                    az_bucket = str(az)
                    # Deep copy the elevation map
                    migrated_model[state][az_bucket] = state_data.copy()
            else:
                # Assume New Format
                migrated_model[state] = state_data

        self._model = migrated_model
        if migrated_count > 0:
            _LOGGER.info(f"Solar Optimizer: Successfully migrated {migrated_count} states to azimuth buckets.")
