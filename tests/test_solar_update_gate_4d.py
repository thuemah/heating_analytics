"""Tests for the parameterised ``_update_unit_solar_coefficient`` write-gate.

Commit 5 of #954 generalises the gate so the 4D shadow learner can
write ``{s, e, w, diffuse}`` through the same non-negativity gate as
the 3D learners.  The ``components`` keyword defaults to the 3-tuple
for backward compatibility with all five live 3D paths.

Invariants under test:
- Default (3D) call writes ``{s, e, w, learned: True}`` with non-neg
  clamp.  No diffuse key leaks in.
- 4D call writes ``{s, e, w, diffuse, learned: True}`` with non-neg
  clamp applied uniformly to diffuse.
- Negative components (any of the four) are clamped to 0.
"""
from custom_components.heating_analytics.learning import LearningManager


def test_3d_write_unchanged():
    """Default ``components=("s","e","w")`` matches pre-#954 behaviour:
    writes three components plus the ``learned`` flag; no diffuse key
    on the regime dict.
    """
    mgr = LearningManager()
    storage: dict = {}
    mgr._update_unit_solar_coefficient(
        entity_id="sensor.heater1",
        value={"s": 0.40, "e": 0.20, "w": 0.10},
        solar_coefficients_per_unit=storage,
        regime="heating",
    )
    entry = storage["sensor.heater1"]
    assert entry["heating"] == {
        "s": 0.40, "e": 0.20, "w": 0.10, "learned": True,
    }
    # Cooling regime initialised as zero-vector (pre-#954 behaviour).
    assert entry["cooling"] == {"s": 0.0, "e": 0.0, "w": 0.0}
    # Diffuse must NOT appear on the regime when the caller did not
    # request it — leaking would corrupt 3D readers that key on the
    # canonical {s, e, w, learned} shape.
    assert "diffuse" not in entry["heating"]


def test_3d_write_clamps_negative_components():
    """Non-negativity (invariant #4) for the 3D call."""
    mgr = LearningManager()
    storage: dict = {}
    mgr._update_unit_solar_coefficient(
        entity_id="sensor.h",
        value={"s": -0.5, "e": 0.2, "w": -0.01},
        solar_coefficients_per_unit=storage,
        regime="heating",
    )
    entry = storage["sensor.h"]
    assert entry["heating"]["s"] == 0.0
    assert entry["heating"]["e"] == 0.20
    assert entry["heating"]["w"] == 0.0
    assert entry["heating"]["learned"] is True


def test_4d_write_writes_diffuse():
    """``components=("s","e","w","diffuse")`` writes all four components
    plus the ``learned`` flag.  Diffuse is rounded and stored alongside
    the cardinal directions.
    """
    mgr = LearningManager()
    storage: dict = {}
    mgr._update_unit_solar_coefficient(
        entity_id="sensor.heater1",
        value={"s": 0.40, "e": 0.20, "w": 0.10, "diffuse": 0.05},
        solar_coefficients_per_unit=storage,
        regime="heating",
        components=("s", "e", "w", "diffuse"),
    )
    entry = storage["sensor.heater1"]
    assert entry["heating"] == {
        "s": 0.40, "e": 0.20, "w": 0.10, "diffuse": 0.05, "learned": True,
    }


def test_4d_clamps_negative_diffuse():
    """Non-negativity (invariant #4) extends to the diffuse component
    once it goes through the same gate.
    """
    mgr = LearningManager()
    storage: dict = {}
    mgr._update_unit_solar_coefficient(
        entity_id="sensor.h",
        value={"s": 0.30, "e": 0.10, "w": 0.00, "diffuse": -0.1},
        solar_coefficients_per_unit=storage,
        regime="heating",
        components=("s", "e", "w", "diffuse"),
    )
    entry = storage["sensor.h"]
    assert entry["heating"]["diffuse"] == 0.0
    # Other components unaffected.
    assert entry["heating"]["s"] == 0.30
    assert entry["heating"]["learned"] is True
