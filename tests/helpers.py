"""Shared test helpers for coordinator mocking (#775)."""


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

    @property
    def model(self):
        return ModelProxy(self)
