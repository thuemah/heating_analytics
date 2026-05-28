"""Shared test helpers for coordinator mocking (#775)."""


def mock_calculate_saturation(net_demand, solar_potential, val):
    """Mirror of ``SolarCalculator.calculate_saturation`` for MagicMock setups (#1008).

    Tests that mock the solar calculator now need a working stub for
    ``calculate_saturation`` because the two 4D inline saturation paths
    in ``learning.py`` route through it.  This helper reproduces the
    production heating / cooling / OFF dispatch on a string ``val``
    (numeric temperature dispatch is not exercised by current tests).
    """
    from custom_components.heating_analytics.const import (
        MODE_HEATING, MODE_COOLING, MODE_GUEST_HEATING, MODE_GUEST_COOLING,
        MODE_OFF,
    )
    if val in (MODE_HEATING, MODE_GUEST_HEATING):
        limit = max(0.0, net_demand)
        applied = min(solar_potential, limit)
        return (
            round(applied, 3),
            round(solar_potential - applied, 3),
            round(max(0.0, net_demand - applied), 3),
        )
    if val in (MODE_COOLING, MODE_GUEST_COOLING):
        return (round(solar_potential, 3), 0.0, round(net_demand + solar_potential, 3))
    if val == MODE_OFF:
        return (0.0, 0.0, 0.0)
    return (0.0, 0.0, round(net_demand, 3))


def stratified_coeff(s: float = 0.0, e: float = 0.0, w: float = 0.0,
                     *, cooling_s: float | None = None,
                     cooling_e: float | None = None,
                     cooling_w: float | None = None) -> dict:
    """Build a v4-shape mode-stratified solar coefficient dict (#868).

    Default: heating regime takes ``(s, e, w)``; cooling regime is zeros.
    Pass ``cooling_*`` to set cooling regime values explicitly.  Useful in
    tests that need to construct a known coefficient state.
    """
    return {
        "heating": {"s": s, "e": e, "w": w},
        "cooling": {
            "s": cooling_s if cooling_s is not None else 0.0,
            "e": cooling_e if cooling_e is not None else 0.0,
            "w": cooling_w if cooling_w is not None else 0.0,
        },
    }


class ModelProxy:
    """Lightweight proxy for the coordinator.model property in tests.

    Delegates attribute reads to the coordinator's private fields:
    ``coordinator.model.correlation_data`` → ``coordinator._correlation_data``.
    """

    def __init__(self, coordinator):
        object.__setattr__(self, "_coord", coordinator)

    def __getattr__(self, name):
        return getattr(self._coord, f"_{name}")


class CoordinatorModelMixin:
    """Mixin that adds the ``model`` property to test MockCoordinators.

    Delegates ``coordinator.model.X`` to ``coordinator._X``, mirroring
    the real coordinator's lazy ModelState proxy.
    """

    # Default screen correction for tests (100% = screens fully open)
    solar_correction_percent: float = 100.0

    @property
    def model(self):
        return ModelProxy(self)
