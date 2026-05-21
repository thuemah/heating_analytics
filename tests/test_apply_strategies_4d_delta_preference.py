"""#968 — ``apply_strategies_to_global_model`` prefers the 4D delta
regardless of ``experimental_4d_primary``.

Aggregation paths that consume historical ``hourly_log`` entries use the
best available signal per hour: 4D when the entry carries it, 3D
otherwise.  The live read-path (``process_learning``) stays flag-gated;
this consumer does not.
"""
from __future__ import annotations

from pathlib import Path


LEARNING_SRC = Path("custom_components/heating_analytics/learning.py").read_text()


def _find_block_for_apply_strategies() -> str:
    """Slice the source around the ``apply_strategies_to_global_model``
    method so substring checks don't accidentally trip on unrelated
    references to ``experimental_4d_primary`` elsewhere in learning.py
    (e.g. the live ``process_learning`` path)."""
    start = LEARNING_SRC.index("def apply_strategies_to_global_model(")
    # Heuristic: read a generous window forward; the per-entry delta read
    # lives near the top of the iteration loop ~150 lines in.
    return LEARNING_SRC[start:start + 12_000]


def _strip_comments(src: str) -> str:
    """Strip full-line and inline ``#`` comments so documentation
    references to ``experimental_4d_primary`` (explaining the de-gating)
    don't trip the substring checks; only executable code is inspected."""
    out = []
    for line in src.splitlines():
        if line.lstrip().startswith("#"):
            continue
        if "#" in line:
            line = line.split("#", 1)[0]
        out.append(line)
    return "\n".join(out)


def test_apply_strategies_prefers_4d_delta_when_present_no_flag_required():
    """The per-entry delta read inside ``apply_strategies_to_global_model``
    must use the 4D field with a 3D fallback, with no
    ``experimental_4d_primary`` gate."""
    block = _strip_comments(_find_block_for_apply_strategies())
    assert "solar_normalization_delta_4d" in block, (
        "apply_strategies_to_global_model must reference the 4D delta field"
    )

    # Locate the per-entry delta read and verify it's the dict-get fallback
    # pattern, not a flag-gated conditional.
    idx = block.index("solar_normalization_delta_4d")
    window = block[max(0, idx - 400):idx + 400]
    assert "experimental_4d_primary" not in window, (
        "apply_strategies_to_global_model: per-entry delta read must no "
        "longer be gated by experimental_4d_primary (#968)."
    )
    # The new pattern is a nested .get fallback.
    assert "log_entry.get(" in window
    assert "solar_normalization_delta\"" in window or \
           "solar_normalization_delta'," in window or \
           "solar_normalization_delta\"," in window, (
        "expected 3D fallback inside the .get() call"
    )
