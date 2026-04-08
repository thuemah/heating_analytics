# Heating Analytics Design Documentation

> **Note:** This document outlines the architectural design and logical models of the Heating Analytics integration. It is intended for developers and system architects to understand the *why* and *how* behind the code.

## 1. Architectural Overview

The integration follows a **Coordinator-Manager** pattern to handle complexity and ensure separation of concerns. The system operates on a rigorous thermodynamic model rather than simple linear regression.

```mermaid
graph TD
    Sensors[HA Sensors] --> Coordinator
    Weather[Weather Providers] --> Coordinator

    Coordinator[HeatingDataCoordinator] --> Collector[ObservationCollector]
    Collector -- "HourlyObservation" --> Learning[LearningManager]
    Coordinator -- "ModelState" --> Statistics[StatisticsManager]
    Coordinator -- "ModelState" --> Forecast[ForecastManager]
    Coordinator --> Solar[SolarCalculator]
    Coordinator --> Storage[StorageManager]

    Statistics --> Output[Sensor State]
    Forecast --> Output

    subgraph "Core Logic"
    Statistics -- "Thermodynamics" --> Model[Global Energy Model]
    Learning -- "Hourly EMA" --> Model
    Learning -- "Daily Strategies" --> Model
    Solar -- "Corrections" --> Model
    Strategies -- "Daily Learning" --> Model
    end
```

### Data Contracts (`observation.py`)

The observation → learning pipeline communicates through explicit typed contracts:

*   **`HourlyObservation`** (frozen dataclass): Immutable snapshot of one completed hour — weather averages, energy totals, per-unit breakdowns, aux/solar state. Produced by `ObservationCollector` at each hour boundary.
*   **`ModelState`** (dataclass): Reference-based view of the learned model — correlation tables, aux/solar coefficients, learning buffers. Consumers read via `coordinator.model`; only the learning layer mutates the underlying dicts.
*   **`LearningConfig`** (frozen dataclass): Per-hour learning settings — rates, eligibility flags, service dependencies.
*   **`LearningStrategy`** (protocol): Per-unit strategy for daily learning with two implementations: `DirectMeter` (actual per-hour kWh) and `WeightedSmear` (COP-corrected synthetic baseline from Track C). Both implementations perform mode filtering (#789): units in `MODES_EXCLUDED_FROM_GLOBAL_LEARNING` (OFF, DHW, Guest) return `None` for that hour. Cooling participates in the global model with correct solar normalization (#801).
*   **`CopParams`** (TypedDict in `thermodynamics.py`): MPC COP model parameters — `eta_carnot` and `lwt` (required), `f_defrost`, `defrost_temp_threshold`, `defrost_rh_threshold` (optional with defaults).

### Components

1.  **HeatingDataCoordinator (`coordinator.py`):**
    *   **Role:** The Central Nervous System.
    *   **Duty:** Orchestrates the 1-minute update loop, dispatches data to managers, and exposes learned model state via the `coordinator.model` property (backed by `ModelState` with live references).
    *   **Key Behavior:** Handles "Smart Merging" of data gaps (e.g., if HA restarts) using persisted snapshots.

2.  **ObservationCollector (`observation.py`):**
    *   **Role:** The Scribe.
    *   **Duty:** Owns all hour-scoped mutable accumulators (weather, energy, per-unit deltas). Provides `accumulate_weather()`, `accumulate_expected()`, and atomic `reset()`. Eliminates the class of restart bugs caused by forgetting to zero individual fields.

3.  **StatisticsManager (`statistics.py`):**
    *   **Role:** The Physicist.
    *   **Duty:** Calculates thermodynamic power predictions, deviations, and efficiencies.
    *   **Key Innovation:** **Regime-Aware Prediction** (see below). It treats "Cold" days (physics-dominated) differently from "Mild" days (noise-dominated).
    *   **Model Access:** Reads learned state via `self.coordinator.model.correlation_data`, not direct private fields.

4.  **ForecastManager (`forecast.py`):**
    *   **Role:** The Oracle.
    *   **Duty:** Manages future predictions and "What If" scenarios.
    *   **Key Innovation:** **Shadow Forecasting**. It tracks the accuracy of multiple weather sources (Primary vs. Secondary) *simultaneously*, allowing users to validate a new weather provider without switching the active control logic.

5.  **LearningManager (`learning.py`):**
    *   **Role:** The Student.
    *   **Duty:** Updates the energy model based on observed performance.
    *   **Interface:** `process_learning(obs: HourlyObservation, model: ModelState, config: LearningConfig)` — three typed objects instead of ~30 loose parameters.
    *   **Key Behavior:** Uses **Exponential Moving Average (EMA)** to gently adapt the model over time, preventing single-day anomalies from corrupting long-term stats. It also enforces **Purity Guards** (ignoring data during Guest Mode or mixed heating/cooling usage).

6.  **SolarCalculator (`solar.py`):**
    *   **Role:** The Astronomer.
    *   **Duty:** Calculates theoretical solar potential and learns the **Solar Coefficient** (how well the house captures that potential).

7.  **StorageManager (`storage.py`):**
    *   **Role:** The Librarian.
    *   **Duty:** Handles JSON persistence, schema migrations (e.g., HDD -> TDD), and backfilling missing metadata. Canonical write-path for all model state.
    *   **Data Lifecycle:** Manages the promotion of high-resolution `hourly_log` data into aggregated `daily_history` summaries. This ensures that while granular data (logs) is rotated (90 days), the statistical backbone (history) persists indefinitely (2 years).

---

## 2. Core Logic & Models

### A. Thermal Degree Days (TDD)
Unlike traditional systems that split Heating Degree Days (HDD) and Cooling Degree Days (CDD), this system uses a unified **Thermal Degree Day** metric.

*   **Formula:** `TDD = |Balance_Point - Effective_Temperature| / 24`
*   **Why:** A well-insulated house behaves symmetrically: deviation from the balance point requires energy, whether it's heating or cooling. This allows a single continuous model curve.

### B. Inertia & Effective Temperature
Buildings do not react instantly to outside air temperature. The system calculates an **Effective Temperature** using a causal exponential-decay kernel — physically equivalent to the steady-state response of a first-order RC thermal circuit. This value is exposed via the **Thermal State Sensor** for external automations.

*   **Model:** Each hour's outdoor temperature is weighted by `e^(-t/τ)`, where `t` is how many hours ago it was and `τ` (tau) is the user-configured time constant. The most recent hour always carries the highest weight; influence decays monotonically going back in time, never peaking in the middle. The kernel window is `5×τ` hours, which captures ≥ 99 % of the total weight.

*   **Default τ = 4 hours:** Suitable for a typical well-insulated residential building. The effective temperature is a blend of roughly the last 20 hours, with the past 4 hours dominating.

*   **Configurable Profiles (τ values):**
    *   **Fast (τ = 2h):** High responsiveness. Good for poorly insulated structures or lightweight construction.
    *   **Slow (τ = 12h):** Long thermal memory (~60 h window). Good for high thermal mass (concrete, passive house).

*   **Design Rule:** `NEVER` use future data (forecasts) to calculate inertia for the past. This prevents data leakage in the learning model.

### C. Regime-Aware Prediction
The system changes its prediction strategy based on the severity of the weather ("Regimes").

1.  **Cold Regime (TDD > 4.0):**
    *   **Physics Dominates:** Heat loss is linear and predictable.
    *   **Strategy:** Skips neighbor averaging entirely and forces **Ratio Extrapolation** (`New_Energy = Neighbor_Energy * (New_TDD / Neighbor_TDD)`). This is unconditional in Cold regime — the model always scales thermodynamically.
    *   **Constraint:** Strict guard: source delta-T must exceed 1.0 to prevent extrapolation from noisy near-balance data.

2.  **Mild Regime (TDD <= 4.0):**
    *   **Noise Dominates:** Solar, wind, and internal gains (cooking, people) have a huge relative impact.
    *   **Strategy:** Three-step fallback chain: (1) **Nearest Neighbor Averaging** (+/- 1°C, same wind bucket), (2) **Wind Fallback** (same temp, relaxed wind bucket: extreme → high → normal), (3) **Thermodynamic Scaling** as final fallback (guard: source delta-T > 0.5). Neighbor averaging is prioritized because small TDD differences are physically irrelevant but mathematically noisy.

#### Typical Day Normalization
When calculating "Typical Daily Consumption" (Median) for a given temperature, the system applies **Normalization Logic** to ensure all days are comparable.
*   **Old Approach:** Filter out days with Auxiliary usage (discarding data).
*   **New Approach (Normalization):** `Typical = Actual_kWh + Aux_Impact_kWh`.
*   **Why:** This reconstructs what the primary heating system *would have consumed* if the fireplace/heater hadn't been used. This significantly increases the sample size available for statistics, especially in cold weather where auxiliary heat is common.

### D. Solar Modeling (The Kelvin Twist)
Standard solar integration is difficult because "1000W of sun" doesn't mean "1000W of heat" inside. The system employs a sophisticated 3-Zone Geometric Model known as the **Kelvin Twist**:

1.  **Geometric Zones (scalar recommendation path):**
    *   **Direct Zone (+/- 45°):** Full solar gain. The sun is shining directly into the windows.
    *   **Glancing Zone (+/- 90°):** Partial gain. The sun is hitting at an oblique angle, reducing penetration.
    *   **Backside Zone (> 90°):** Zero gain. The sun is behind the building (shadow side).

2.  **Solar Saturation Logic:**
    *   **Thermodynamic Cap:** Solar gain cannot exceed the base heating demand of the house.
    *   **Prevention:** This prevents the model from calculating negative net consumption on sunny winter days, ensuring that "Free Heat" is capped at "Total Heat Needed".

3.  **Decoupled Learning (Normalization):**
    *   The Base Model continues to learn during sunny periods by using **Normalized Energy** (Actual + Estimated Solar).
    *   **Solar Coefficient Learning:** If `Solar_Factor > 0.1`, the system specifically trains the solar coefficients to improve future normalization.

4.  **2D Solar Vector Model:**
    *   The sun's position is decomposed into a **South** (`S`) and **East** (`E`) component using pure geometry — no manual azimuth input required.
    *   Each unit learns a matching 2D coefficient vector `(Coeff_S, Coeff_E)` that captures its effective window orientation empirically.
    *   `Unit_Solar_Impact = Coeff_S × Potential_S + Coeff_E × Potential_E`  (screen transmittance is implicit in the learned coefficient)
    *   Coefficients are learned against the **potential** (pre-screen) solar vector. Because the learning target (`base − actual`) inherently includes the screen effect, the coefficient converges to `physical_window_coeff × average_screen_transmittance`. This is a deliberate trade-off: using the potential vector keeps the learning signal strong even when screens are closed (avoiding vector-magnitude stalls), while the implicit transmittance coupling is slowly tracked by NLMS. Windowless rooms correctly converge to near-zero coefficients regardless of screen state.
    *   Learning uses **Normalized LMS (NLMS)** instead of standard LMS gradient descent. NLMS divides the step by the input power (`solar_s² + solar_e² + ε`), making convergence rate independent of the solar vector magnitude. This prevents gain-dependent instability where high-solar units oscillate while low-solar units converge. The regularization constant `ε = 0.05` naturally shrinks the east coefficient when the east signal is weak (typical at high latitudes where the sun tracks a narrow arc).
    *   During cold-start (no learned coefficients yet), an initial scalar default is decomposed along a 180° (south) assumption until real data replaces it.

5.  **Air Mass Transmittance:**
    *   Solar attenuation at low sun angles is handled by a physics-based Air Mass Transmittance model (`intensity = 0.7 ** (1 / sin(elevation))`), replacing arbitrary elevation cutoff zones.

6.  **Cloud Coverage Model (Kasten & Czeplak 1980):**
    *   `cloud_factor = 1.0 − 0.75 × (cloud_coverage / 100)^3.4`
    *   The original Kasten & Czeplak formula `G/G_clear = 1 − 0.75 × (N/8)^3.4` was calibrated against ground-observed oktas, not satellite-derived percentages. HA weather APIs report model/satellite cloud area fractions (0–100%) which are a related but different quantity. The exponent 3.4 produces a nearly constant bias multiplier (~1.01×) across all cloud levels when applied to API percentages, which is the critical property: a constant bias is cleanly absorbed by the per-unit coefficient, leaving the coefficient to represent window physics rather than cloud-model compensation. A lower exponent (e.g. 1.5) creates a cloud-level-dependent bias (2–39%) that forces the coefficient to track weather patterns instead of building properties.

7.  **Screen Transmittance Floor:**
    *   When solar screens are closed, a minimum transmittance floor (20%) is enforced. This captures real physics beyond window transmittance: solar radiation heats exterior walls, roof surfaces, and the ground around the building, transferring heat inward through conduction regardless of screen position. The floor keeps the prediction path responsive to solar conditions even at 0% correction.

8.  **Solar Thermal Battery:**
    *   Instead of an instantaneous subtraction, solar gain is modeled as charging an exponential decay accumulator (`_solar_battery_state`). The absorbed heat releases gradually into the building mass over the following hours, dampening instant compensation loops and matching real-world thermal inertia.
    *   Default decay rate is 0.75 (25% loss per hour, half-life ~2.4 hours), a middle ground between lightweight (furniture, interior surfaces) and heavier (concrete, floor heating) construction. Light-frame buildings may benefit from a faster decay (~0.60), while heavy masonry may need a slower one (~0.88). Per-installation calibration via `diagnose_solar` is recommended once 2+ weeks of data is available.
    *   The `diagnose_solar` service includes a calibration sweep that tests decay rates from 0.50 to 0.95 against post-sunset residuals and recommends the optimal value. The `apply_battery_decay` parameter persists the recommendation to `entry.data` without requiring config flow changes.

### E. Auxiliary Heating (Dynamic Coefficients)
For hybrid systems (Heat Pump + Fireplace/Heater), the system does not use a separate "Fireplace Mode" curve. Instead, it learns an **Auxiliary Coefficient**.

*   `Expected_Energy = Base_Model - (Aux_Coefficient * Aux_Duration)`
*   The system learns *how much energy* the auxiliary source saves per minute. This replaces old logic that required a "20% duration threshold" to trigger specific curves.

#### The Kelvin Protocol: Global Authority & Reconciliation
To resolve discrepancies between the Global Model (Top-Down) and the sum of Unit Models (Bottom-Up), the system applies the **Kelvin Protocol Reconciliation**:

1.  **Global Authority:** The Global Model's predicted Aux Reduction is the absolute truth.
2.  **Proportional Scaling:** If the sum of affected unit reductions differs from the Global target (e.g., due to exclusions or model drift), the unit reductions are scaled proportionally so their sum exactly equals the Global target.
3.  **Orphaned Savings:** If global savings cannot be attributed to specific units (e.g., all affected units are off, or list is empty), the savings are tracked as `_accumulated_orphaned_aux`. This ensures that Global Savings are never lost, even if unit attribution is impossible.

> **Note on Reported Savings:** The `accumulated_aux_impact_kwh` sensor strictly reports the **Global Model** value (Total Reduction). The `HeatingPotentialSavingsSensor` breaks this down into `allocated` (Unit Specific) and `unassigned` (Orphaned/Global-Only) savings.

#### Auxiliary Cooldown / Decay Protocol
When auxiliary heating turns off, the building retains heat (thermal mass), causing the heating system to remain idle longer than physics would dictate based on outside temperature. Learning during this "residual heat" period would corrupt the base model (making the house seem magically efficient).

To prevent this, the system implements a **Cooldown State Machine**:

1.  **Trigger:** `auxiliary_heating_active` transitions `True` -> `False`.
2.  **Action:** Learning is **LOCKED** for all units in `aux_affected_entities`.
    *   *Note:* Units NOT in this list continue learning normally (Dual-Track Learning).
3.  **Exit Conditions:**
    *   **Time-out:** `COOLDOWN_MAX_HOURS` (6h) elapsed.
    *   **Convergence:** `Actual_Consumption / Expected_Base > COOLDOWN_CONVERGENCE_THRESHOLD` (92%). This means the heating system has "woken up" and is behaving normally again. Waiting for 95% was unnecessarily conservative for radiant heat sources that linger well beyond the active burn period.

#### Auxiliary Conservation Strategy
When a heating unit is removed from the 'Aux Affected Entities' list (or replaced), its learned auxiliary coefficient (kW reduction) is not lost. The **Conservation Strategy** (`async_migrate_aux_coefficients`) redistributes this coefficient proportionally to the remaining affected units. This preserves the global energy balance—the house doesn't stop saving energy just because you reconfigured a sensor.

### F. Wind Modeling (Bucket Hierarchy)
Wind has a non-linear effect on heating. The system segments wind conditions into three discrete buckets to learn distinct behaviors:

1.  **Normal Wind:** < 5.5 m/s (Light breeze).
2.  **High Wind:** 5.5 - 10.8 m/s (Strong breeze).
3.  **Extreme Wind:** > 10.8 m/s (Storm conditions).

**Fallback Hierarchy:**
To prevent dangerous under-estimation during storms (where data might be sparse), the system enforces a hierarchy:
*   **Direct Fallback:** If *Extreme* data is missing, it looks down to *High*, then *Normal*.
*   **Extrapolation:** When extrapolating from a neighbor temperature (thermodynamic scaling), the system uses the best available wind bucket for that neighbor, prioritizing the requested bucket but accepting *harsher* conditions if necessary to avoid returning zero.

### G. Thermal Mass Correction
The daily energy consumption sent to the `LearningManager` is not always a pure reflection of heat loss — some energy may have been used to heat the building mass itself, or the mass may have released stored heat. Without correcting for this, the U-model would learn incorrectly on days with indoor temperature fluctuations.

To solve this, the system calculates a correction once per day before the U-update:

`delta_kwh = thermal_mass_kwh_per_degree * (T_indoor_end - T_indoor_start)`
`adjusted_consumption = measured_consumption - delta_kwh`

- **Rising Indoor Temp:** Heat was bound in the mass. We add it back to the consumption.
- **Falling Indoor Temp:** Mass provided "free" heat. We subtract it.

The user configures `thermal_mass_kwh_per_degree` (electric kWh/°C) using a rule of thumb based on area and an assumed average COP. Because we only have raw electrical kWh and no dynamic COP curve, a fixed factor is the most pragmatic solution. The error is non-systematic (some days over, some under) and averages out in the long-term U-estimate. Setting this to `0.0` disables the correction.

### H. Daily Learning Mode

The primary learning system (Track A) updates the detailed `temp_key × wind_bucket` correlation table every hour. However, hourly learning is unsuitable for some homes:

- **High thermal mass:** Concrete/stone buildings store heat across many hours. An hourly observation window captures too much lag noise for reliable learning.
- **High minimum modulation:** Heat pumps with a high floor on output power cycle on and off at the minimum rate, making any single hour an unreliable sample.

For these cases, the user can enable **Daily Learning Mode** (`daily_learning_mode = True` in config) to activate **Track B** as the exclusive update path.

**When active:**
- Track A is completely blocked from writing to `_correlation_data`. Track B owns the correlation table exclusively.
- Learning fires once per day at midnight, after `_process_daily_data()` has assembled the full day's statistics.
- **Completeness Guard:** Requires at least **22 of 24 hours** to be present in the day log before any update is applied (raised from 20 in v1.2.7).
- **Diagnostic Output:** A dedicated sensor (`heating_analytics_daily_learning`) exposes the learned U-Coefficient (kWh/TDD/day) as its state. This allows for long-term tracking of the building's overall thermal health independent of specific temperature buckets.

**Inputs:**
1. Total measured consumption for the day (`daily_stats["kwh"]`)
2. Total TDD for the day (`daily_stats["tdd"]`)
3. ΔT_indoor (T_indoor at midnight − T_indoor at the previous midnight) — **optional**, only applied when `indoor_temp_sensor` is configured and the readings are available at both midnight boundaries.

**Process:**
If the indoor temperature correction is active, measured consumption is adjusted before use:
`q_adjusted = daily_kwh − (thermal_mass_kWh_per_°C × ΔT_indoor)`

The system then calculates an observed average hourly consumption and writes it into the bucket for the day's (temp, wind) pair using the same EMA formula as Track A, at the user's configured `learning_rate`:
- Cold start: `bucket = q_adjusted / 24`
- Subsequent: `bucket = bucket + learning_rate × (q_adjusted/24 − bucket)`

In parallel, a single global **U-Coefficient** (kWh_el / TDD / day) is updated via EMA at `DEFAULT_DAILY_LEARNING_RATE` (a lower, fixed rate) for long-term drift tracking. This value is exposed by the **Daily Learning diagnostic sensor** (`heating_analytics_daily_learning`) together with `last_midnight_indoor_temp`, `thermal_mass_kwh_per_degree`, `indoor_temp_sensor`, and `learning_rate` as attributes.

**Purity Guards:** Days with Guest Mode, mixed-mode auxiliary usage (20–80%), or active Cooldown are excluded from Track B learning, identical to Track A's guards.

**Indoor temperature sensor:** The sensor is always optional, even in Daily Learning Mode. Users who do not load-shift but still benefit from daily aggregation (due to high thermal mass or minimum modulation) can enable the mode without providing an indoor sensor — thermal mass correction is simply skipped.

**The Data Resolution Trade-off (Diurnal Collapse):**

While Track B resolves the load-shifting and thermal mass phase-shift problems of Track A, it introduces a severe penalty to learning rate across the temperature spectrum.

Track A exploits the natural diurnal temperature swing: a day ranging from 4°C at night to 14°C at noon will populate up to 10 distinct temperature buckets in a single 24-hour period, delivering a broad, well-sampled slice of the heating curve every day.

Track B mathematically collapses this entire swing into a single daily average (e.g., 9°C), yielding one data point per day regardless of the day's thermal range. To populate the bucket for deep-winter conditions (e.g., −10°C), the model must wait for a calendar day whose 24-hour mean reaches −10°C — an event that may occur only a handful of times per season, compared to the many individual overnight hours that reach that temperature under Track A.

In practice this means:
- **Track A:** The heating curve is broadly characterised within 2–4 weeks of varied weather.
- **Track B:** Full characterisation of the heating curve may require an entire heating season.

Track B should therefore be strictly reserved for setups where the underlying building physics — very high concrete/stone thermal mass or high minimum modulation — render hourly observations thermodynamically invalid as individual learning samples. For systems with MPC-based load shifting, **Track C** (see Section I) is the recommended path: it retains Track A's hourly resolution while completely decoupling the model from the MPC's economic scheduling.

### I. Thermodynamic Baseline Engine (Track C — "The Digital Twin")

Track C is a fundamentally different learning paradigm. Where Track A observes the electricity meter and Track B aggregates the same meter into daily totals, Track C **ignores the electricity meter entirely** and instead consumes real thermal production data from an external Model Predictive Control (MPC) integration — specifically `heatpump_mpc`. This makes Track C a **full-fidelity alternative to Track A** that retains hourly resolution while being completely immune to MPC load-shifting.

#### The Problem Track C Solves

When a heat pump is controlled by an MPC to shift consumption to off-peak hours, the electrical meter no longer reflects the building's real-time heat loss. Without correction, a dangerous feedback loop emerges:

1.  MPC shifts 100 % of the day's heating to 03:00 (cheapest electricity).
2.  Track A sees a massive electricity spike at 03:00 and learns "the house loses enormous heat at 03:00".
3.  The next day, the model demands even more heating at 03:00. The model collapses.

Track B mitigates this by collapsing the day to a single data point, but at the cost of severe **Diurnal Collapse** (see Section H): one temperature bucket per day, full heating-season characterisation time. Track C solves the feedback loop without sacrificing hourly resolution.

#### Data Contract with `heatpump_mpc`

Track C depends on a rolling 24–48 hour buffer exposed by the `heatpump_mpc` integration via the `heatpump_mpc.get_sh_hourly` service. Each hourly record has the following structure:

```json
{
  "datetime": "2026-03-29T13:00:00+01:00",
  "kwh_th_sh": 1.87,
  "kwh_el_sh": 0.55,
  "mode": "sh"
}
```

| Field | Description |
|---|---|
| `kwh_th_sh` | Actual thermal energy (kWh) delivered for space heating. Set to `0.0` during DHW hours. |
| `kwh_el_sh` | Electrical energy (kWh) consumed during space-heating windows only. |
| `mode` | `"sh"` (space heating), `"dhw"` (domestic hot water), or `"off"`. |

**Why COP is not stored in the buffer:** The per-hour COP is computed at midnight from the MPC's learned Carnot model parameters (`η_Carnot`, leaving water temperature, defrost penalty) applied to each hour's actual outdoor conditions. This is more accurate than a time-weighted average COP from the buffer, which would be mathematically incorrect for hours with variable compressor load. A secondary `heatpump_mpc.get_cop_params` service provides the model parameters alongside `get_sh_hourly`.

**DHW Mode:** When a given hour is dominated by DHW production, `mode = "dhw"` and `kwh_th_sh = 0.0`. This is the physical truth — zero space heat was delivered. The model does not learn from DHW hours (preventing false "super-insulation" artefacts), but the building's heat loss during those hours is still accounted for in the smearing step.

#### The Midnight Sync

Track C performs its core calculation once per day, after midnight:

1.  **Collect Production:** Fetch the 24-hour thermal buffer from `heatpump_mpc.get_sh_hourly` and COP model parameters from `heatpump_mpc.get_cop_params`. Sum total `kwh_th_sh` and `kwh_el_sh` (SH-mode hours only).
2.  **Calculate Fallback Daily COP:** `daily_avg_cop = sum(kwh_th_sh) / sum(kwh_el_sh)`. Used only when per-hour COP parameters are unavailable. Fallback to `1.0` if the pump was off all day.
3.  **Calculate Theoretical Loss Weights:** For each of the 24 hours, compute a weather-based loss weight from Delta-T, wind bucket multiplier (1.0 / 1.3 / 1.6), and solar factor (including Solar Battery decay residual). These weights express *when* the building needed heat, based purely on physics.
4.  **Smear Thermal Load:** Distribute the total delivered thermal energy proportionally to the loss weights: `smeared_kwh_th[h] = total_kwh_th × (weight[h] / sum_of_weights)`. DHW hours receive their proportional share — the envelope still lost heat to the environment even when the pump was making hot water.
5.  **Convert to Synthetic Electrical Baseline via Per-Hour COP:** Each smeared thermal hour is divided by that hour's actual COP: `synthetic_kwh_el[h] = smeared_kwh_th[h] / COP(T_outdoor[h], RH[h])`. The COP is computed from the MPC's learned parameters: `COP = max(1.0, min(10.0, η_Carnot × T_hot_K / (T_hot_K − T_cold_K) × defrost_factor))`, where `defrost_factor < 1.0` during icing conditions (cold + humid). The COP cap at 10.0 prevents physically unrealistic values when outdoor temperature approaches the leaving water temperature. After per-hour COP division, the 24-hour synthetic total is **renormalized** to match actual metered electrical consumption: `synthetic_kwh_el[h] *= total_kwh_el / sum(synthetic_kwh_el)`. This preserves the per-hour *shape* (cold hours cost more, warm hours cost less) while anchoring the daily sum to metered reality. When MPC COP parameters are unavailable, falls back to daily average COP (where COP cancels mathematically, giving pure weather-weighted redistribution of actual electrical).

#### Why Track C Is a Full Alternative to Track A

The 24 synthetic hourly baselines produced in step 5 carry the same information density as Track A's raw meter observations:

*   Each hour maps to a specific outdoor temperature and wind condition → populates a distinct `temp_key × wind_bucket` cell in the correlation table.
*   A day ranging from 4 °C at night to 14 °C at noon populates up to 10 temperature buckets — identical to Track A.
*   Full heating-curve characterisation requires **2–4 weeks** of varied weather, the same as Track A.

Compare this to Track B, which collapses the same day into a single average temperature (e.g. 9 °C) and yields one data point — requiring an entire heating season for full characterisation.

The critical difference from Track A is the **data source**: Track A reads the electricity meter directly (and is therefore vulnerable to the MPC feedback loop). Track C reads thermally verified production data from `heatpump_mpc` and reconstructs a physically correct electrical baseline that is completely independent of when the MPC chose to run the compressor.

#### Graceful Degradation

If the `heatpump_mpc` service is unavailable at midnight (e.g. integration restart, network issue), Track C falls back to Track B's thermal mass correction for that day. This ensures learning never stalls, at the cost of one day of reduced resolution.

#### Multi-Unit Installations and Per-Unit Learning

In installations with additional heating units (panel heaters, heated cables, secondary heat pumps), Track C combines two data sources to build the global model:

*   **MPC-controlled unit:** Contributes its *synthetic* electrical baseline (weather-smeared via `ThermodynamicEngine`). Its meter data is corrupted by load-shifting and is never used for learning.
*   **Non-MPC units:** Contribute their *actual* hourly electrical consumption from the meter. Their data is ground truth — no load-shifting to correct — regardless of unit type (resistive or variable-COP heat pump). The model learns the resulting `kWh_el` implicitly, including COP variation with temperature.

The per-hour total written to the global model is simply `synthetic_el[h] + actual_el[h]` — both in pure `kWh_el`, both representing the same hour, from different sources appropriate to each unit's data quality.

This is implemented via the **per-unit learning strategy pattern** (`LearningStrategy` protocol in `observation.py`):

*   **`DirectMeter`:** Assigned to non-MPC units. Returns actual per-hour kWh from `hourly_log["unit_breakdown"]`.
*   **`WeightedSmear`:** Assigned to the MPC-managed unit. Returns `track_c_distribution[hour]["synthetic_kwh_el"]` — the COP-corrected synthetic baseline.

The midnight loop in `_apply_strategies_to_global_model()` iterates all unit strategies uniformly: each strategy contributes a kWh value per hour, the sum is written to the correlation table via the standard buffer → jump-start → EMA pipeline. `build_strategies()` auto-assigns strategies from config — no manual per-unit configuration needed.

**Configuration:** When Track C is enabled, the user selects which energy sensor is MPC-managed (`mpc_managed_sensor`). This sensor is excluded from per-unit Track A learning (its meter data is time-shifted). The MPC sensor intentionally has no per-unit model — its forecast share is derived via subtraction (see Forecast Isolation below), not from a dedicated per-unit correlation table.

*   **Per-unit models (Track A):** Continue to learn hourly via the standard buffer → jump-start → EMA pipeline for all *non-MPC* units. The MPC unit's per-unit model is bootstrapped from the global model but does not receive Track A updates (its meter data is corrupted by load-shifting).

#### Forecast Isolation for MPC

The `get_forecast` service accepts an optional `isolate_sensor` parameter. When set, each hourly prediction is transformed from building-total to unit-isolated demand:

```
forecast_for_mpc[h] = max(0, global[h] - Σ per_unit[h] for non-MPC units)
```

This subtraction-based approach minimises error propagation: the bulk of the prediction rests on the robust global model, while per-unit uncertainty is limited to the marginal contribution from secondary sources. The MPC solver receives `kWh_el` that it converts to thermal demand via its own COP — correct precisely because the panel heaters' electrical contribution has already been subtracted.

**Edge case:** When `Σ per_unit_other > global` (transition season, low total demand), the result is clamped to zero — the MPC unit stands idle and secondary units cover all demand.

### J. Thermodynamic Reconstruction
When reconstructing historical data (e.g., for model comparison), the system prioritizes **Hourly Vectors** (see Section 5). If vectors are missing (legacy data), it performs **Thermodynamic Reconstruction**:

*   It recovers the **Effective Temperature** from the stored TDD value.
*   `Temp_Effective = Balance_Point - (TDD * 24)`
*   This ensures that even scalar historical data respects the thermodynamic conditions under which it was recorded, allowing for accurate re-simulation even if original logs are lost.

---

## 3. The Forecasting Engine

### A. Blended Forecast Strategy
To maximize both short-term precision and long-term stability, the system implements a **Blended Forecast Strategy** that combines two weather providers.

1.  **Primary Provider:** (e.g., Local Sensor or Met.no) Used for the immediate future (Days 1 to X). Provides high-resolution, local accuracy.
2.  **Secondary Provider:** (e.g., Open-Meteo) Used for the long-term (Days X+1 to 14). Provides stable, consistent trends where local sensors might drift or fail.
3.  **Crossover Point:** Configurable via `Forecast Crossover Day`.

### B. Provenance Tracking (Accuracy Attribution)
When mixing weather sources, it is critical to know *who* is responsible for a prediction error. The system logs **Provenance Metadata** for every hourly prediction:

*   **`primary_entity` & `secondary_entity`:** The entity IDs of the providers active at the time of prediction.
*   **`crossover_day`:** The crossover configuration used at prediction time.
*   **`source`:** Which specific provider was used for *that specific hour* (Primary or Secondary).

**Why?** When calculating accuracy statistics, the system filters history by `primary_entity` / `secondary_entity` and only includes entries where the recorded entity matches the *currently configured* entity. This prevents "Cross-Contamination"—e.g., if you switch from AccuWeather to Open-Meteo, the system will not attribute AccuWeather's past errors to Open-Meteo.

### C. Shadow Forecasting & Per-Source Accuracy Metrics
The system ingests forecasts from *both* sources simultaneously, regardless of which one is active for the blended output:

1.  **Active:** The Blended Forecast described above (Primary for days 1–X, Secondary for days X+1–7).
2.  **Shadow:** Energy predictions are generated for *both* Primary and Secondary across the full horizon in parallel.

At the end of each day, both predictions are compared against actual consumption and stored in forecast history with full provenance metadata. This enables independent, contamination-free accuracy statistics per source.

**Exposed attribute:** `forecast_accuracy_by_source` (on the Weather Plan Today sensor) reports the following metrics for each source (`primary` / `secondary`):

#### Energy Accuracy (kWh/day)

| Field | Definition |
|---|---|
| `mae_7d` | Mean Absolute Error over the last 7 days. Computed as the average of \|net daily error\| per day, where the daily error is the signed sum of hourly energy errors (kWh). Measures day-level accuracy. |
| `mape_7d` | MAE expressed as a percentage of actual daily consumption. `(Σ\|daily_net_error\| / Σactual_kWh) × 100`. Scale-independent accuracy indicator. |
| `mae_30d` | Same as `mae_7d` but over a 30-day rolling window. Provides a stable long-term baseline. |
| `mape_30d` | Same as `mape_7d` over 30 days. |

The daily net error is the *algebraic sum* of signed hourly errors within a day before taking absolute value. This deliberately ignores intra-day timing errors (e.g., a forecast that is +2 kWh in the morning and −2 kWh in the afternoon cancels to zero), which would otherwise penalize forecasters for temperature curves shifting a few hours, not for actual energy misjudgment.

#### Temperature Accuracy (°C, net daily deviation)

| Field | Definition |
|---|---|
| `weather_mae_7d` | Mean of \|net daily temperature error\| over 7 days. The daily temperature error is the signed sum of hourly (forecast_temp − actual_temp) differences for that day. Positive = forecaster ran too warm; negative = too cold. |
| `weather_bias_7d` | Mean of the *signed* net daily temperature error over 7 days. A persistent non-zero bias indicates a systematic warm or cold offset in the weather source for this location. |
| `weather_mae_30d` | Same as `weather_mae_7d` over 30 days. |
| `weather_bias_30d` | Same as `weather_bias_7d` over 30 days. |

**Role in blend logic:** `weather_mae` and `weather_bias` are the primary inputs used internally to rank and select the better weather source before the crossover decision is applied. They are temperature-domain metrics (°C), not energy-domain metrics, and are therefore more sensitive to systematic forecast drift than `mae_7d`. They are exposed in the attribute for transparency and diagnostics, not as end-user KPIs.

### D. Hybrid Projection ("The Funnel")
The "Energy Forecast Today" sensor does not just show the morning forecast. It implements a **Hybrid Projection** that converges to reality.

*   `Forecast_Today = (Actual_kWh_So_Far) + (Predicted_kWh_Remaining)`
*   **At 00:00:** 100% Prediction.
*   **At 12:00:** 50% Fact, 50% Prediction.
*   **At 23:59:** 100% Fact.
*   **Why:** This eliminates the frustration of seeing a "10kWh" forecast at 8 PM when you've already burned 15kWh.

This **Hybrid Approach** is also used for multi-day period comparisons (`compare_periods`):
*   **Past Days:** Historical Logs (`daily_history`).
*   **Today:** Hybrid (`energy_today` + `calculate_future_energy`).
*   **Future Days:** Pure Forecast (`get_future_day_prediction`).

### E. Reference vs. Live (Budget vs. Reality)
*   **Reference Forecast ("The Budget"):** Frozen at 00:00 (Midnight Snapshot).
    *   **Goal:** Stability. This is the plan we agreed on at the start of the day.
    *   **Use Case:** Baseline for deviations. If weather changes, the deviation shows *why* consumption changed.
    *   **Source:** `source='reference'` in `ForecastManager`.

*   **Live Forecast ("Thermodynamic Projection"):** Updated hourly.
    *   **Goal:** Reality. "Where are we actually heading?"
    *   **Composition:** `Actuals So Far` + `Live Forecast for Remaining Hours`.
    *   **Use Case:** `thermodynamic_projection_kwh` attribute. This is what you will likely pay at the end of the day.
    *   **Source:** `source='live'` in `ForecastManager`.

---

## 4. Learning & Data Flow

### The Learning Loop (Hourly — Track A)
At the top of every hour, `ObservationCollector` freezes the accumulated sensor readings into an immutable `HourlyObservation`. The coordinator then:
1.  **Validates:** Is data complete? Is the "Guest Switch" off? Is it mixed-mode (20–80% aux)?
2.  **Normalizes:** Removes estimated Solar and Auxiliary impact from the actual consumption to find the "Pure Thermal Base".
3.  **Updates:** Passes `HourlyObservation`, `ModelState`, and `LearningConfig` to `LearningManager.process_learning()`, which feeds the Pure Base into the model using **EMA (Exponential Moving Average)**.
    *   `New_Model_Value = (Old_Value * (1 - Rate)) + (New_Observation * Rate)`
    *   **Rate:** Typically 0.01 (1%). This makes the model "sticky" and resistant to outliers.

### The Learning Loop (Daily — Track B / Track C)
At midnight, `_process_daily_data()` assembles the full day's statistics and applies the appropriate learning path:
*   **Track B:** Writes a single `q_adjusted / 24` value to one `(avg_temp, avg_wind)` bucket — flattened daily learning for high thermal mass buildings.
*   **Track C:** Iterates per-unit `LearningStrategy` objects via `_apply_strategies_to_global_model()`. Each unit's strategy produces a kWh contribution per hour (24 buckets), and the sum is written to the correlation table. `retrain_from_history` uses the same dispatch.

### Jump-Start Mechanism
To prevent slow convergence on new installations:
1.  **Buffering:** The system buffers the first 4 samples for any new Temperature/Wind bucket. (For Solar Coefficients, observations are pooled globally across all temperatures rather than partitioned by bucket to prevent stalling).
2.  **Injection:** Once the buffer is full, it calculates the average and "Jump Starts" the model value directly to this average, bypassing the slow EMA warmup. For solar coefficients, a 2×2 least-squares fit is used to initialise both south and east components from the buffered observations.
3.  **Damping:** Initial jump-start estimates for solar coefficients are mathematically dampened by 25% (`COLD_START_SOLAR_DAMPING = 0.75`) to guard against early-learning noise inflating outliers.
4.  **Post-jump-start:** Solar coefficients transition to NLMS adaptive learning. Base model and auxiliary coefficients continue with standard EMA.
5.  **Result:** Useable predictions appear within hours, not weeks.

### DHW Mode — Air-to-Water Heat Pump Hysteresis

Air-to-water heat pumps periodically switch from space-heating to producing domestic hot water (DHW). During a DHW cycle the compressor runs at full power, but all energy is diverted to the hot-water tank — zero heat is delivered to the space. The building cools slightly, and the subsequent space-heating phase must compensate for the lost time.

This is physically identical to a thermostat-hysteresis cycle on a direct-electric heating cable:

| Phase | Heating cable | Air-to-water HP |
|---|---|---|
| Active | Delivers P_heat until thermostat setpoint | Space-heating |
| Pause | Hysteresis cutoff, no heat to space | DHW cycle, no space heat |

In both cases the space-heating model observes an inflated reading during the active phase:

```
obs_active = TRUE_B × (t_active + t_pause) / t_active   [cycle-ratio inflation]
obs_pause  = 0
```

#### Strategy A — Skip (old `MODE_OFF` behaviour, incorrect for DHW)

If DHW hours are simply skipped (no model update), the EMA converges to `obs_active`, not `TRUE_B`. This produces permanent upward drift proportional to the DHW duty cycle:

| DHW scenario | Duty cycle | Coefficient drift |
|---|---|---|
| Light (1 h DHW / 11 h heat) | 8% | +9% |
| Medium (2 h DHW / 6 h heat) | 25% | +33% |
| Heavy (2 h DHW / 4 h heat) | 33% | +50% |
| Standby (14 h DHW / 10 h heat) | 58% | +140% |

#### Strategy B — Observe Zero (`MODE_DHW`, current behaviour, correct)

Setting a unit to `MODE_DHW` causes the model to update with `actual = 0` during DHW hours — the physically correct observation (zero space-heat contribution). The EMA converges to `TRUE_B` at all learning rates. No drift.

**Sampling Precision:** Starting in v1.2.7, the exclusion logic is applied during a 2-minute accumulation tick (evaluating the state 30 times per hour). This ensures that mid-hour mode switches (e.g., a 10-minute shower at 14:45) are tracked with high temporal resolution, preventing DHW energy from bleeding into the hourly heating total.

#### Oscillation Amplitude and Learning Rate

Strategy B eliminates mean drift but introduces a within-cycle oscillation: the coefficient is pulled toward `obs_active` during heating and toward 0 during DHW. The amplitude grows with learning rate. Analytically simulated steady-state amplitudes (half peak-to-peak as % of TRUE_B, stability criterion ≤ ±10%):

| Scenario | LR 1% | LR 2% | LR 3% |
|---|---|---|---|
| DHW light (8% ratio) | ±0.1% ✓ | ±0.2% ✓ | ±0.3% ✓ |
| DHW medium (25% ratio) | ±0.4% ✓ | ±0.8% ✓ | ±1.2% ✓ |
| DHW heavy (33% ratio) | ±1.0% ✓ | ±2.0% ✓ | ±3.0% ✓ |
| DHW standby (58% ratio) | ±7.0% ✓ | ±14% borderline | ±21% ✗ |

The default global learning rate (1%) remains within the stability threshold even in the most extreme DHW standby scenario. Learning rates above 2–3% are not recommended for installations with high DHW duty cycles.

#### Interaction with Aux Learning

If `is_aux_active` and `unit_mode == MODE_DHW` are simultaneously true, aux-coefficient learning is skipped for that unit. The zero space-heat contribution is caused by the DHW cycle, not the aux system — attributing it to aux would corrupt the per-unit aux coefficient.

### Guest Mode & Purity Guards
The "Guest Mode" is a critical feature for model integrity.
*   **Problem:** Guests crank the heat to 25°C, ruining the "Normal" model for the homeowner (21°C).
*   **Solution:** When Guest Mode is active, the system **stops learning**. It tracks usage for billing/stats ("Guest Impact"), but the underlying thermodynamic model remains untouched.

### Seamless Rolling Efficiency
To avoid the "Midnight Jump" (where efficiency `kWh/TDD` goes to infinity because TDD is 0), the system uses a dynamic window.
*   **Logic:** If `TDD_Today < 0.5`, it borrows data from Yesterday to fill the denominator.
*   **Result:** Efficiency lines are smooth across midnight boundaries.

---

## 5. Persistence & Storage Strategy

To protect the user's hardware (specifically SD cards on Raspberry Pi), the integration implements a **Low-Frequency Write Strategy**:

1.  **Buffer First:** High-frequency sensor data is aggregated in memory (RAM).
2.  **Hourly Flush:** Data is written to disk (`.storage/heating_analytics`) only **once per hour**, or immediately upon system shutdown/restart.
3.  **Crash Recovery:** In the event of a power loss or ungraceful shutdown (where RAM buffer is lost), the **Gap Handling Logic** uses cumulative sensor counters to mathematically reconstruct the total consumption during the downtime.

### Hourly Vectors (High-Fidelity History)
Previously, the system stored daily averages (Avg Temp, Total kWh). This caused 'aliasing' errors where a cold morning + warm afternoon averaged to a mild day, losing the thermodynamic context of the heating spikes.

The system now stores **Hourly Vectors** in the daily history: arrays of 24 hourly values for Temperature, Wind, TDD, and Actual Load.
*   **Structure:** `{ "temp": [v0..v23], "load": [v0..v23], ... }`
*   **Benefit:** Enables precise 'What-If' re-simulations and accurate model updates even years later, as the specific conditions for every hour are preserved.

---

## 6. Sensor Architecture & Logic

This section details the calculation logic for key sensors, clarifying the distinction between "Gross Thermodynamic Demand" (what the house needs) and "Net Electrical Load" (what the meter sees).

### A. Model Comparison Sensors
The Comparison Sensors (`Day`, `Week`, `Month`) answer the question: *"Are we using more energy than normal?"*

They operate on a **Dual-Layer Architecture**:

1.  **State (Primary): Weather-Normalized Comparison**
    *   **Logic:** `Model(Current_Weather) - Model(Reference_Weather)`
    *   **Purpose:** Isolates weather impact (Climatic Delta).
    *   **Explanation:** Calculates the theoretical energy difference driven purely by weather conditions compared to last year. If positive, the weather is forcing higher consumption. If negative, the weather is helping us save.

2.  **Attributes (Secondary): Real-World Comparison**
    *   **Logic:** `Hybrid_Total_Current - Actual_Total_Last_Year`
    *   **Purpose:** Shows the raw billing impact ("What will I pay?").
    *   **Hybrid Calculation:** `Past_Actuals + Today_Budget(Actual+Forecast) + Future_Forecast`.

### B. Thermal Stress Index (Expected Energy Sensor)
The `Energy Baseline Today` sensor calculates the **Thermal Stress Index**, defined as:
*   **Formula:** `(Gross_Thermodynamic_Forecast / Max_Historical_Load) * 100`
*   **Gross vs. Net:** This calculation uses the **Gross Forecast** (i.e., total heat demand *before* any auxiliary heating reduction).
*   **Why:** If you run a fireplace (Aux) on the coldest day of the year, your *electrical* load might be low, but the *thermal stress* on the building is Extreme. Using Gross Demand prevents the "Mild Weather Fallacy," ensuring the system correctly reports that the house is fighting hard against the cold.

### C. Potential Savings (Auxiliary Sensor)
The `AUX Savings Today` sensor quantifies the benefit of auxiliary heat sources (e.g., fireplace, diesel heater).

1.  **Theoretical Max:** The projected savings for the *entire day*.
    *   `Max_Savings = Global_Model_Base(24h) - Global_Model_Aux(24h)`
    *   This projection combines completed hours (cache) with future hours (forecast).
2.  **Live Status (Power Allocation):**
    *   The sensor attributes break down the current instantaneous power (kW) into:
        *   **Allocated:** Useful heat replacing electrical load.
        *   **Unassigned (Overflow):** Heat generated in excess of the model's demand (clamped).
    *   **Mean Power Calculation:** `(Accumulated_kWh / Minutes_Passed) * 60`. This transforms energy integrators into live power gauges for UI dashboards.

---

## 7. Design Principles (Kelvin Protocol)

These principles guide all development on this integration:

1.  **Thermodynamic Validity:** Never produce a result that violates physics (e.g., negative TDD, COP > 10).
2.  **Temporal Stability:**
    *   No linear projections from early data (e.g., don't multiply 1am usage by 24).
    *   No hard logic jumps based on time of day.
3.  **Fail-Forward:** If a sensor fails, the system falls back to the best available estimate (e.g., Reference Forecast -> Semantic Average -> 0), rather than crashing.
4.  **Metric-Driven Development:** Optimization tasks are prioritized by `Value = (Impact × Urgency) / Effort`.

### A. The Feed-Forward Mandate (Analytics vs. Control)

Heating Analytics is strictly a **Feed-Forward Analysis Engine**. It is architecturally prohibited from performing direct closed-loop control (e.g., writing setpoints or triggering relays).

- **Reason 1 — Feedback Loop Corruption:** If the model controls the heating that produces the data it learns from, the training signal becomes a function of the model's own decisions. This leads to *Model Drift*, where the system learns its own behaviour rather than the building's thermodynamics.
- **Reason 2 — Subjective Comfort:** Control requires subjective trade-offs between cost and comfort. This integration provides the Ground Truth (the physics), allowing the user to build their own "Moral" layer (automations) on top of it.
- **Implementation:** All outputs are provided as diagnostic sensors or service responses (`get_forecast`), intended as inputs for external logic (EMHASS, Node-RED, or standard HA Automations).

---

## 8. Known Limitations

### Thermal Hysteresis After Deep Cold Spells

The model treats identical outdoor temperatures as thermodynamically equivalent, regardless of the thermal history that preceded them. In practice, a house at +2 °C after a week below −10 °C behaves very differently from a house at +2 °C after cooling down from a mild period.

**Root causes:**

-   **Deep thermal mass lag.** Foundations and structural elements have a thermal time constant of days to weeks, not hours. After a deep cold spell they continue acting as a heat sink well after outdoor air has warmed, drawing energy from the heating system invisibly to the 4-hour inertia window.
-   **Heat pump COP history.** During the cold spell the heat pump operated in its least efficient zone (low COP, frequent defrost cycles). This accumulated thermal debt is not captured in any current model variable.
-   **Direction-agnostic inertia model.** The exponential decay kernel (τ = 4 h by default) represents only the shallow, fast thermal mass (indoor air, wall surfaces). The same kernel is used regardless of whether temperatures are rising or falling — warming from cold and cooling from warm are treated as mirror images of the same physical process, which they are not.

**Observed symptom:** After a deep cold spell the model consistently *underestimates* actual heat demand during the recovery phase. Conversely, when temperatures drop from a mild baseline the model tends to *overestimate* demand at the same outdoor temperatures. The asymmetry is most pronounced near +1 °C to +3 °C.

**Why it is not fixed:** A correct remedy would require either a multi-day thermal state variable (e.g., an exponentially smoothed 48–72 h temperature integrator) or asymmetric inertia coefficients conditioned on the direction of thermal travel. Both approaches introduce meaningful implementation complexity, require sufficient historical data to calibrate, and carry a real risk of making predictions worse in the statistically dominant case (normal, non-extreme temperature transitions) in order to improve them in the rare case (transition out of a deep cold spell, which occurs only a handful of times per winter season). The current pragmatic decision is to accept this limitation and document it here rather than risk destabilising the model's everyday accuracy.
