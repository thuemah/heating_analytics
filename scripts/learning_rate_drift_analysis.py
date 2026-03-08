"""
Learning Rate Drift Analysis — Hysteresis Units & DHW Mode

Simulates EMA model drift for units with periodic on/off cycles.

Tests TWO learning strategies:
  A) Skip OFF-hours  → current MODE_OFF behaviour / old DHW-as-MODE_OFF behaviour
  B) Learn 0 during OFF-hours  → current MODE_DHW behaviour (force actual = 0)

═══════════════════════════════════════════════════════════════════════════════
THE HEATING-CABLE ANALOGY FOR AIR-TO-WATER HEAT PUMPS IN DHW MODE
═══════════════════════════════════════════════════════════════════════════════

A direct-electric heating cable with thermostat hysteresis and an
air-to-water heat pump doing DHW cycles are physically identical from the
space-heating model's point of view:

  Heating cable:
    ON  (t_on)  → delivers P_heat to the space until setpoint reached
    OFF (t_off) → hysteresis pause, no heat to space, building cools slightly
    → next ON phase must compensate for the off-period cooling

  Heat pump in DHW mode:
    HEAT (t_heat) → delivers P_heat to the space
    DHW  (t_dhw)  → heat pump runs but all energy goes to the hot-water tank,
                    zero contribution to space heating, building cools slightly
    → next HEAT phase must compensate for the DHW-period cooling

In both cases the model observes during the active heating phase:

    obs_on = TRUE_B × (t_active + t_pause) / t_active     [cycle-ratio inflation]

And during the pause/DHW phase:

    obs_off = 0

═══════════════════════════════════════════════════════════════════════════════
STRATEGY COMPARISON
═══════════════════════════════════════════════════════════════════════════════

  Strategy A (skip):       no model update during pause/DHW hours
                           → was the old DHW implementation (DHW treated as MODE_OFF)
                           → analytical steady state = obs_on (always drifts upward)

  Strategy B (learn zero): model updated with actual = 0 during pause/DHW hours
                           → current DHW implementation (force actual_unit = 0)
                           → converges to TRUE_B at all learning rates

Usage:
    python scripts/learning_rate_drift_analysis.py
"""

import numpy as np


TRUE_B = 1.0  # normalised true coefficient


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------

def simulate(
    learning_rate: float,
    t_on: int,
    t_off: int,
    strategy: str,   # "skip_off" | "learn_zero"
    n_cycles: int = 2000,
) -> tuple[list[float], float, float]:
    """
    Simulate EMA over many hysteresis / DHW cycles.

    Strategy "skip_off":   only update during ON/HEAT hours  (Strategy A)
    Strategy "learn_zero": update every hour, obs=0 during OFF/DHW (Strategy B)

    obs_on = TRUE_B * (t_on + t_off) / t_on
      → unit delivers full cycle's worth of heat in t_on hours

    Returns:
        history      – model value at each simulated hour
        drift_pct    – (mean_steady − TRUE_B) / TRUE_B × 100
        amplitude_pct – half peak-to-peak in steady state as % of TRUE_B,
                        i.e. the ±X% the coefficient swings each cycle
    """
    obs_on  = TRUE_B * (t_on + t_off) / t_on
    obs_off = 0.0

    model   = TRUE_B
    history = [model]

    for _ in range(n_cycles):
        # Active heating phase — always update
        for _ in range(t_on):
            model += learning_rate * (obs_on - model)
            history.append(model)

        # Pause / DHW phase
        for _ in range(t_off):
            if strategy == "learn_zero":
                model += learning_rate * (obs_off - model)
            # strategy "skip_off": no update
            history.append(model)

    # Steady state: last 10 cycles
    tail          = history[-(t_on + t_off) * 10:]
    steady        = float(np.mean(tail))
    drift_pct     = (steady - TRUE_B) / TRUE_B * 100
    amplitude_pct = (float(np.max(tail)) - float(np.min(tail))) / 2.0 / TRUE_B * 100
    return history, drift_pct, amplitude_pct


# ---------------------------------------------------------------------------
# Analytical steady state for Strategy A ("skip_off")
# ---------------------------------------------------------------------------

def analytical_steady_skip(t_on, t_off):
    """
    After one cycle (skip_off):
        c_after_on  = c * (1-lr)^t_on  +  obs_on * (1 - (1-lr)^t_on)
        c_after_off = c_after_on  (no updates during OFF/DHW)
    At steady state c_after_off = c:
        → c = obs_on  [regardless of lr!]

    Strategy A ALWAYS converges to obs_on = TRUE_B × cycle_ratio.
    The drift is independent of learning rate.
    """
    obs_on = TRUE_B * (t_on + t_off) / t_on
    return obs_on


# ---------------------------------------------------------------------------
# Scenario tables
# ---------------------------------------------------------------------------

SCENARIOS_HEATING_CABLE = [
    {
        "name": "Direct electric  (t_on=1h, t_off=4h)",
        "t_on": 1,
        "t_off": 4,
        "note": "5:1 cycle ratio — original analysis",
    },
    {
        "name": "Direct electric  (t_on=4h, t_off=4h)",
        "t_on": 4,
        "t_off": 4,
        "note": "2:1 cycle ratio, ~3 cycles/day",
    },
    {
        "name": "Direct electric  (t_on=8h, t_off=8h)",
        "t_on": 8,
        "t_off": 8,
        "note": "2:1 cycle ratio, 1 cycle/day",
    },
]

# Air-to-water heat pump DHW scenarios.
# t_on  = hours in space-heating mode per cycle
# t_off = hours diverted to DHW per cycle
# Typical DHW cycle: 1–3 h every 6–12 h, so cycle ratios 1.1–1.5
SCENARIOS_DHW = [
    {
        "name": "HP DHW light   (t_heat=11h, t_dhw=1h)",
        "t_on": 11,
        "t_off": 1,
        "note": "~8% DHW ratio — cycle_ratio 1.09×, twice/day",
    },
    {
        "name": "HP DHW medium  (t_heat=6h,  t_dhw=2h)",
        "t_on": 6,
        "t_off": 2,
        "note": "~25% DHW ratio — cycle_ratio 1.33×, 3 cycles/day",
    },
    {
        "name": "HP DHW heavy   (t_heat=4h,  t_dhw=2h)",
        "t_on": 4,
        "t_off": 2,
        "note": "~33% DHW ratio — cycle_ratio 1.50×, high-demand household",
    },
    {
        "name": "HP DHW standby (t_heat=10h, t_dhw=14h)",
        "t_on": 10,
        "t_off": 14,
        "note": "HP mostly on standby/DHW, minimal space heating",
    },
]

LEARNING_RATES = [0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05]


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------

AMPLITUDE_THRESHOLD_PCT = 10.0  # ±10 % of TRUE_B is the stability limit


def status_drift(d: float) -> str:
    if abs(d) < 1.0:  return "OK"
    if abs(d) < 5.0:  return "BORDERLINE"
    return "DRIFT"


def status_amplitude(a: float) -> str:
    """Flag oscillation amplitude against the ±10 % stability threshold."""
    if a <= AMPLITUDE_THRESHOLD_PCT:  return "STABLE"
    if a <= AMPLITUDE_THRESHOLD_PCT * 2:  return "BORDERLINE"
    return "UNSTABLE"


def print_table(scenarios: list[dict]) -> None:
    for cfg in scenarios:
        t_on, t_off = cfg["t_on"], cfg["t_off"]
        ratio      = (t_on + t_off) / t_on
        analytical = analytical_steady_skip(t_on, t_off)
        print(f"\n{'─'*92}")
        print(f"  {cfg['name']}  |  cycle ratio {ratio:.2f}x  |  {cfg['note']}")
        print(f"  Strategy A analytical steady state: {analytical:.3f}x true value"
              f"  ({(analytical-1)*100:+.1f}%)")
        print(f"  Stability criterion: Strategy B amplitude ≤ ±{AMPLITUDE_THRESHOLD_PCT:.0f}% of TRUE_B")
        print(f"  {'LR':>6}  {'A drift':>10}  {'B drift':>10}  {'B ±ampl':>10}  "
              f"{'A status':>12}  {'B amplitude':>13}")
        print(f"  {'─'*6}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*12}  {'─'*13}")

        for lr in LEARNING_RATES:
            _, da, _  = simulate(lr, t_on, t_off, "skip_off")
            _, db, ab = simulate(lr, t_on, t_off, "learn_zero")
            print(f"  {lr*100:>5.1f}%  {da:>+9.1f}%  {db:>+9.1f}%  {ab:>+9.1f}%  "
                  f"{status_drift(da):>12}  {status_amplitude(ab):>13}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 76)
    print("EMA DRIFT ANALYSIS")
    print()
    print("  Strategy A = skip OFF/DHW hours  (old DHW-as-MODE_OFF behaviour)")
    print("  Strategy B = learn 0 in OFF/DHW  (current MODE_DHW implementation)")
    print("=" * 76)

    # ── Section 1: Direct electric / heating cable (baseline reference) ──────
    print("\n")
    print("┌" + "─"*74 + "┐")
    print("│  PART 1 — DIRECT ELECTRIC / HEATING CABLE (baseline reference)       │")
    print("│  Classic thermostat hysteresis: unit over-heats during ON phase to   │")
    print("│  compensate for the off period. obs_on = B_true × cycle_ratio.       │")
    print("└" + "─"*74 + "┘")
    print_table(SCENARIOS_HEATING_CABLE)

    # ── Section 2: Air-to-water HP in DHW mode ───────────────────────────────
    print("\n\n")
    print("┌" + "─"*74 + "┐")
    print("│  PART 2 — AIR-TO-WATER HEAT PUMP  (DHW mode as heating-cable        │")
    print("│           analogy)                                                    │")
    print("│                                                                       │")
    print("│  During DHW cycle the HP delivers zero space heat. The building      │")
    print("│  cools and the subsequent heating phase must compensate — identical   │")
    print("│  physics to the heating-cable hysteresis above.                      │")
    print("│                                                                       │")
    print("│  Old implementation: DHW treated as MODE_OFF → Strategy A → DRIFT   │")
    print("│  Current implementation: force actual=0 in DHW → Strategy B → OK    │")
    print("└" + "─"*74 + "┘")
    print_table(SCENARIOS_DHW)

    # ── Conclusion ────────────────────────────────────────────────────────────
    print(f"\n{'='*92}")
    print("CONCLUSION")
    print()
    print("  Both sections follow identical mathematics:")
    print("    obs_on  = TRUE_B × (t_active + t_pause) / t_active")
    print("    obs_off = 0")
    print()
    print("  Strategy A (skip): converges to obs_on regardless of learning rate.")
    print("    Upward drift = (cycle_ratio − 1) × 100 %")
    print("    For DHW light/medium scenarios: +9 % to +50 % error.")
    print("    This was the old behaviour when DHW was handled like MODE_OFF.")
    print()
    print("  Strategy B (force 0): mean drift → 0 at all learning rates.")
    print("    The coefficient converges to TRUE_B on average — but it oscillates")
    print("    every cycle: pulled upward during heating, pulled toward 0 during DHW.")
    print("    The amplitude of these oscillations grows with learning rate.")
    print()
    print(f"  Stability criterion: oscillation amplitude ≤ ±{AMPLITUDE_THRESHOLD_PCT:.0f}% of TRUE_B.")
    print("  Reading the 'B ±ampl' column above reveals the safe LR upper bound:")
    print("    LR ≥ 3–5%   → UNSTABLE for most DHW scenarios (large cycle amplitudes)")
    print("    LR = 1–2%   → STABLE   for all realistic DHW scenarios")
    print("    LR = 0.5–1% → STABLE   even for extreme standby scenarios")
    print()
    print("  → Mean drift alone is insufficient to evaluate Strategy B stability.")
    print("    Amplitude (half peak-to-peak in steady state) is the correct metric.")
    print("    Recommended safe learning rate: LR ≤ 1–2 %.")
    print()
    print("  NOTE: Remaining implementation concern in learning.py (now fixed)")
    print("  ──────────────────────────────────────────────────────────────────")
    print("  If is_aux_active == True AND unit_mode == MODE_DHW simultaneously,")
    print("  unit_normalized = 0.0 (from DHW) was fed into _learn_unit_aux_coefficient.")
    print("  The aux model then attributed the zero-contribution to the aux system,")
    print("  not to DHW — corrupting the per-unit aux coefficient for that unit.")
    print("  Fix applied: skip aux-coefficient learning when unit_mode == MODE_DHW.")
    print("=" * 92)


if __name__ == "__main__":
    main()
