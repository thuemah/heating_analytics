# Heating Analytics for Home Assistant

[![hacs_badge](https://img.shields.io/badge/HACS-Custom-orange.svg)](https://github.com/custom-components/hacs)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/thuemah/heating_analytics/graphs/commit-activity)
[![License](https://img.shields.io/github/license/thuemah/heating_analytics.svg)](https://github.com/thuemah/heating_analytics/blob/main/LICENSE)

## Why Heating Analytics?

**Turn your Home Assistant into a smart energy detective.**

Your heating and cooling systems consume energy every day, but how do you know if they are working efficiently? This integration learns your home's unique thermal behavior and tells you exactly how much energy you *should* be using right now—based on current weather—versus what you're *actually* using.

**Catch energy waste before it catches you:**

- Detect open windows draining heat
- Measure real savings from your fireplace or space heaters
- Spot inefficiencies in real-time
- Get accurate daily forecasts based on weather

Think of it as a fitness tracker for your home's heating system.

---

## How It Works

<img width="1308" height="1252" alt="553138952-d1a3b194-ed9c-447c-bc50-3f16f759d6cc" src="https://github.com/user-attachments/assets/fe90394f-e83d-4b4b-abf3-f78fa712ad0b" />




The integration continuously learns the relationship between outdoor conditions (temperature, wind, sun) and your heating and cooling consumption. Once trained, it predicts what you *should* use and compares it to reality—alerting you to unexpected deviations.

> **Architecture note:** Heating Analytics is a *feed-forward* engine — it produces diagnostics and predictions, but never writes setpoints or controls your heating directly. The outputs are designed to feed external logic (HA Automations, EMHASS, Node-RED) so you retain full control and the model's training data stays uncontaminated by its own decisions.

---

## Quick Start

### Level 1: Basic Installation (5 minutes)

1. **Install via HACS:**
   - Go to HACS -> Integrations -> 3 dots (top right) -> Custom repositories
   - Add URL: `https://github.com/thuemah/heating_analytics`
   - Category: `Integration`
   - Click **Add**, then find "Heating Analytics" in the list and install it.
2. **Restart Home Assistant**
3. **Add Integration:**
   - Go to Settings → Devices & Services → Add Integration
   - Search for "Heating Analytics"
4. **Configure Required Sensors:**
   - **Weather Entity (Required):** Used for wind data and forecasts. (e.g., Met.no, Open-Meteo).
   - **Outdoor Temperature (Recommended):** A local outdoor sensor is preferred for "Reality" (what your house actually feels), but the Weather Entity can be used as a fallback.
   - **Energy Meter (Required):** Must be a stable, cumulative kWh counter (Total Energy), not a power sensor (W).

**That's it!** The system starts learning immediately.

> **Note on Accuracy:**
> The model needs **7-14 days** of varied weather to learn your home's behavior. In the first week, predictions may fluctuate as it calibrates. This is normal—it's learning your unique thermal profile!

## Crucial Prerequisites: Consistency is Key
This integration uses high-resolution machine learning to build its thermodynamic models. The system is incredibly robust and will actually learn to compensate for systematic sensor errors (e.g., an energy meter that always underreports by 5%) – **but it cannot compensate for random chaos.**

**To get accurate predictions and deviation alerts, you need:**
1. **Consistent Energy Data:** Your energy sensors can be local (Zigbee/Z-Wave) or cloud-based APIs, as long as the reporting is *consistent*. 
   *  *The enemy is randomness: APIs that frequently drop offline for hours, or batch-update 12 hours of data at once, will ruin the learning cycle.*
2. **Reliable Outdoor Temperature:** A local outdoor sensor is best, but a stable weather integration (like Met.no, Open-Meteo or OpenWeatherMap) works perfectly fine as long as it reasonably tracks your local microclimate.
3. **Wind Data (Weather Integrations Preferred!):** Surprisingly, local personal weather stations often provide *terrible* wind data due to turbulence from nearby trees and houses. Using your standard Home Assistant weather integration for wind speed and gusts is highly recommended to accurately calculate the "Wind Chill Penalty" on your home.
4. **Patience:** The system needs time to observe your home across different temperatures and wind conditions before the models stabilize and the "Deviations" become accurate. Give it a few days of learning!
---

### Level 2: Tuning for Precision (Optional)

Once you have basic data flowing, enhance accuracy:

**Wind Configuration:**
- Wind data is used automatically from your weather entity (or dedicated sensor if configured)
- The system accounts for wind chill and gusts affecting heat loss
- **Supported Units:** `m/s`, `km/h`, `mph`, `kn` (knots)

**Solar Configuration:**
- Solar gain is always modelled — window orientation is learned automatically, no manual azimuth or area input needed
- Tell the integration which facades (south / east / west) have external screens, and which units they affect
- Adjust `Solar Correction` (0–100) to reflect typical blind/screen usage, or automate it from your screens

**Balance Point:**
- Adjust the temperature where heating kicks in (default: 17°C)
- Run `calibrate_balance_point` to check it against your own daily history (see [Calibrate Balance Point](#calibrate-balance-point))
- Fine-tune learning rate if the model reacts too slowly/quickly

**Calibration services:** once a few weeks of data are in, `calibrate_inertia` suggests the thermal inertia setting and `calibrate_wind_thresholds` the wind bucket boundaries for your house.

**Heat-source type:** the integration works out each unit's heat-source type from its history. If you know it, set it in the advanced settings (see [Heat-Source Type per Unit](#heat-source-type-per-unit)).

---

### Level 3: Advanced Setup (Power Users)

**Historical Data Import:**

If you have past energy/weather data, jump-start the learning process:
- Export data from your old system as CSV
- Use `heating_analytics.import_from_csv` service (details below)
- The model trains instantly on months/years of history

**Multi-Zone Tracking:**
- Configure individual heating units (living room, bedroom, heat pump, floor cables, etc.)
- Track per-room efficiency, solar distribution and individual baselines

**Heat Pump Controller (Track C):**
- Connect `heatpump_mpc` so the model learns the building's heat loss from delivered heat instead of from a meter the controller shifts in time (see Track C below)

**Dashboard & Visualizations:**
- Ready-to-paste Plotly cards are available in the [`tools/`](https://github.com/thuemah/heating_analytics/blob/main/tools/) folder
- [plotly_heat_demand_curve.yaml](https://github.com/thuemah/heating_analytics/blob/main/tools/plotly_heat_demand_curve.yaml) – heat demand curve (temperature vs kWh/day, segmented by wind)
- [plotly_today_breakdown_pie.yaml](https://github.com/thuemah/heating_analytics/blob/main/tools/plotly_today_breakdown_pie.yaml) – donut chart of today's per-unit energy split
- [plotly_week_ahead_forecast.yaml](https://github.com/thuemah/heating_analytics/blob/main/tools/plotly_week_ahead_forecast.yaml) – 7-day energy and temperature forecast
- [heating_forecast_sensor_and_dashboard.yaml](https://github.com/thuemah/heating_analytics/blob/main/tools/heating_forecast_sensor_and_dashboard.yaml) – 48-hour hourly forecast with solar contribution

---

## Weather Integration Recommendations

**Works with any weather provider** (Met.no, AccuWeather, etc.), but the accuracy of predictions depends on the data your weather entity provides.

### Required attributes

| Attribute | Used for | Required? | Fallback if missing |
|-----------|----------|-----------|---------------------|
| `temperature` | Outdoor temperature | Only if no dedicated `outdoor_temp_sensor` is configured | None — produces a warning |
| `wind_speed` | Wind chill penalty | Only if no dedicated wind sensor is configured (advanced option, off by default) | 0.0 (no wind penalty) |

### Optional attributes (strongly recommended)

| Attribute | Used for | Fallback if missing |
|-----------|----------|---------------------|
| `cloud_coverage` | Solar model — determines how much solar energy reaches windows | Maps from weather condition text (e.g. "sunny" → 0%, "partlycloudy" → 50%, "cloudy" → 85%), then falls back to 50% if condition is unknown. This coarse estimate significantly reduces solar model accuracy. |
| `direct_normal_irradiance` + `diffuse_radiation` | Direct and diffuse sunlight for the 4D solar model | The 4D model falls back to an estimate from `cloud_coverage`, where it is worse than the standard 3D model. A local irradiance (GHI) sensor, configured in the integration settings, takes precedence over both. |
| `wind_gust_speed` | Wind gust compensation (higher accuracy wind penalty) | Ignored — only sustained wind speed is used |
| `humidity` | Per-hour COP defrost penalty (Track C) | 50% default — defrost may under/over-trigger |
| `forecast` | 7-day hourly energy forecast | No forecast sensors available |

> **How to check what your weather entity provides:**
> Go to Developer Tools → States, find your weather entity (e.g. `weather.home`), and expand the attributes. Look for `cloud_coverage` (numeric 0–100), `wind_gust_speed`, and `humidity`. If `cloud_coverage` is missing, the solar model falls back to condition-text mapping, which can cause the solar coefficient to be poorly calibrated.

### Recommended: Open-Meteo with cloud coverage

The built-in **Met.no** integration works well out of the box — it provides `cloud_coverage`, `humidity`, `wind_gust_speed`, and all required attributes. However, Met.no's forecast is limited to 6-hour blocks (not hourly), which reduces forecast sensor accuracy.

The **standard Open-Meteo** integration in HA does **not** provide `cloud_coverage` or `humidity` as weather entity attributes. This significantly degrades the solar model. If you prefer Open-Meteo, use this custom version that includes the missing attributes:

https://github.com/thuemah/open_meteo

It adds:
- Numeric **cloud coverage** (0–100%) for accurate solar correction
- **Humidity** for Track C COP defrost penalty
- Precise **wind gust** compensation
- **7-day hourly forecast** (higher resolution than Met.no)

*Open-Meteo is free and requires no API key.*

**Summary:**
| Provider | cloud_coverage | humidity | wind_gust | hourly forecast | Works out of box? |
|----------|---------------|----------|-----------|----------------|-------------------|
| Met.no | Yes | Yes | Yes | No (6h blocks) | Yes |
| Open-Meteo (standard) | **No** | **No** | No | Yes | **Solar model degraded** |
| Open-Meteo (custom) | Yes | Yes | Yes | Yes | Yes |

---

## Key Features

### Machine Learning That Just Works

- **Automatic Learning (Track A):** No manual calibration needed—the system learns your home's thermal characteristics every hour. This is the default and recommended mode for the large majority of installations.
- **Daily Learning Mode (Track B):** For homes where hourly learning is unreliable — typically high thermal mass buildings (concrete, stone) or heat pumps with a high minimum modulation level. The model learns once per day at midnight instead of every hour. **Note:** Building a complete heating curve takes months rather than weeks. Only use this if your building dynamics genuinely make hourly observations unreliable — hourly learning (Track A) is the right choice for the large majority of installations.
- **Thermodynamic Baseline Engine (Track C):** For systems with Model Predictive Control (MPC) load-shifting via `heatpump_mpc`. Track C ignores the electricity meter entirely and instead uses real thermal production data from the MPC to construct a synthetic electrical baseline — each hour's smeared thermal load is divided by that hour's actual COP (computed from the MPC's learned Carnot model, including a defrost penalty for cold+humid conditions). Unlike Track B, Track C retains full hourly resolution — making it a complete alternative to Track A with 2–4 week learning time — while being fully immune to the MPC feedback loop. In multi-unit installations, each unit's learning strategy is auto-assigned: MPC units use the synthetic baseline, non-MPC units contribute actual meter data per hour. See [DESIGN.md](DESIGN.md) for details.
- **Regime-Aware Prediction:**
    - **Cold Regime:** Uses thermodynamic scaling for accurate extrapolation in deep winter.
    - **Mild Regime:** Prioritizes neighbor averaging and wind fallbacks for stability in variable transition seasons.
- **Thermal Inertia:** Learns and predicts on an *effective* temperature — recent outdoor temperatures weighted by how many hours ago they were read, with a configurable time constant (default 4 hours) — because buildings don't react instantly. Learning, forecasting and retraining all use the same effective temperature.
- **Adaptive:** Updates hourly based on actual vs expected consumption

### Thermal Degree Days (TDD)

The integration uses **Thermal Degree Days (TDD)** instead of traditional Heating Degree Days (HDD).
- **Formula:** `abs(Balance Point - Outdoor Temp) / 24`
- **Why?** TDD is a unified metric that handles both **Heating** (below Balance Point) and **Cooling** (above Balance Point) in mixed-mode systems.
- **Benefit:** Provides a continuous efficiency metric (`kWh / TDD`) year-round, regardless of season.
- **Fixed at the balance point it was logged under:** every day in the history records the balance point its degree days were summed at. After you change the balance point, comparisons with older days recalculate their degree days from the stored hourly temperatures instead of mixing the old and the new value.

### Thermal Regime & Primary Driver

Two attributes on **Energy Baseline Today** say what the building is doing and what drives it right now:

- **`thermal_regime`:** `heating`, `cooling`, `mixed` or `idle`, from how today's energy splits between units in heating and in cooling mode. It is weighted by energy, not by the number of units — six idle radiators do not outvote one air conditioner doing all the work. Hot water and switched-off units are left out. The explanations use it to decide which way warm weather and sunshine push consumption; under `mixed` they make no directional claim.
- **`primary_driver`:** the largest cost on top of a calm, clear day at the balance point — `Temp`, `Wind` or `Solar_Deficit` (a sky darker than clear) when heating; `Temp` or `Solar_Load` (sunshine adding cooling load) when cooling — or `None` when none of them is significant.

### Shadow Forecasting (Primary vs Secondary)

For maximum accuracy, the system can track two different weather sources simultaneously:

1.  **Primary Source:** (e.g., Local Sensor) Used for short-term precision.
2.  **Secondary Source:** (e.g., Open-Meteo) Used for long-term planning.

**Shadow Tracking:** Even if you only use the Primary source for control, the system secretly tracks the accuracy of the Secondary source in the background. This lets you compare them (MAE/MAPE metrics) and decide which is better without risking your heating control.

<details>
<summary><b>Blended Forecast Configuration</b></summary>

You can configure a **Primary Weather Entity** and an optional **Secondary Weather Entity**.

- **Blended Forecast:** The system can intelligently switch between providers. It uses the Primary source for immediate days (high precision) and the Secondary source for long-range forecasts (better stability).
- **Forecast Crossover Day:** Determines the switch point. Example: With a crossover of `3`, days 1-3 use Primary, days 4-7 use Secondary.
- **Provenance Tracking:** The system logs exactly which weather provider generated the data for every single hour. This ensures that accuracy statistics (MAE/MAPE) correctly attribute errors to the specific provider active at that time, even in a blended scenario.

</details>

### Hybrid Projection (Forecast Today)

The "Energy Forecast Today" sensor is smarter than a simple weather look-up. It uses a **Hybrid Projection**:
- **Past Hours:** Uses **Actual Consumption** (what you really used).
- **Future Hours:** Uses **Predicted Consumption** (based on forecast).
- **Result:** A realistic end-of-day total that converges to the true value as the day progresses.

### Guest Mode

Hosting a party? The **Guest Mode** ensures temporary occupancy spikes don't corrupt your long-term heating model.
- **Action:** Set a heating unit to `Guest Heating` or `Guest Cooling` mode.
- **Effect:**
    - The unit is **excluded** from the learning algorithm.
    - Its consumption is tracked separately as "Guest Impact".
    - The "Expected Energy" model assumes 0 for this unit, correctly flagging the extra usage as a deviation (Impact).

### Auxiliary Heating Tracking

- Track wood stove, fireplace, or space heater usage
- **Orphaned Savings:** If you select "No Units Affected" (empty list) in the configuration, the system still tracks the *Global* savings from the fireplace, even if it can't attribute them to a specific room.
- **Dynamic Coefficient Learning:** Instead of separate curves, the system learns a **Power Reduction Coefficient** (kW).
  - It learns: "When the fireplace is on at -5°C, the heat pump works 2.5 kW less."
  - This allows the model to adapt instantly to auxiliary heat without relearning the entire temperature curve.
- **Precision Auxiliary Analysis (Kelvin Protocol):** Enhanced "Global Authority" logic ensures that the sum of all heating units matches the global model.
  - **Overflow Energy:** Tracks savings potential that exceeded a unit's base load (e.g., if the fireplace saves 5kW but the heater only uses 2kW, 3kW is "Overflow").

#### Cooldown Protocol (Thermal Decay Protection) 🛡️

When you turn off a fireplace or space heater, the house retains heat for hours. If the system immediately resumed normal learning, it would falsely learn that your heating system is "super efficient" during this period, corrupting your baseline model.

To prevent this, the system enters a **Cooldown State** automatically when Auxiliary Heating turns off.
- **Action:** Learning is strictly locked for all affected units.
- **Duration:** 2 to 6 hours (dynamic).
- **Exit Condition:** The system monitors real-time consumption. Once the affected units use at least 92 % of their expected consumption again (convergence), the lock is released.
- **Benefit:** Ensures your "Normal Heating Model" remains pure and unpolluted by residual heat from the fireplace.

### Air-to-Water Heat Pump Support (DHW Mode)

Air-to-water heat pumps regularly switch to producing domestic hot water (DHW). During this cycle the pump runs at full power but delivers zero heat to the space — the building cools slightly, and the next space-heating phase must make up for the gap.

Without explicit DHW tracking, the learning model sees inflated consumption during heating phases and attributes it to the base coefficient, causing the predicted baseline to drift upward. The effect scales with your DHW duty cycle: a system spending 25% of its time in DHW will drift approximately +33%; a high-demand household at 33% DHW will drift +50%.

**Fix:** Set your heat pump unit to `DHW` mode in the Heating Analytics mode selector whenever the pump is in DHW cycle. The model then correctly observes zero space-heat contribution during those hours, and the base coefficient converges to the true value without drift.

#### Automatic Mode Mapping with Blueprints

The repository includes ready-made blueprints that automate the mode transitions — no manual helper management needed:

- **`blueprints/heat_pump_mode_sync.yaml`** — For heat pumps exposing an operation mode sensor (e.g. `"Heating"` / `"Domestic Hot Water"` / `"Defrost"`). Import this blueprint, select your mode sensor and mode helper, and all transitions are handled automatically. Defrost is transparent: the previous heating or DHW mode is preserved so defrost energy is attributed correctly rather than creating a spurious mode flip.

  [![Import the heat_pump_mode_sync blueprint into your Home Assistant instance.](https://my.home-assistant.io/badges/blueprint_import.svg)](https://my.home-assistant.io/redirect/blueprint_import/?blueprint_url=https%3A%2F%2Fgithub.com%2Fthuemah%2Fheating_analytics%2Fblob%2Fmain%2Fblueprints%2Fheat_pump_mode_sync.yaml)

- **`blueprints/climate_sync.yaml`** — For units exposed as standard HA climate entities (`heat` / `cool` / `off`). Optionally enable the guest mode prefix to track occupancy spikes separately from the main model.

  [![Import the climate_sync blueprint into your Home Assistant instance.](https://my.home-assistant.io/badges/blueprint_import.svg)](https://my.home-assistant.io/redirect/blueprint_import/?blueprint_url=https%3A%2F%2Fgithub.com%2Fthuemah%2Fheating_analytics%2Fblob%2Fmain%2Fblueprints%2Fclimate_sync.yaml)

The buttons open the blueprint import dialog in your own Home Assistant instance with the URL filled in. Without them: **Settings → Automations & Scenes → Blueprints → Import Blueprint**, and paste the blueprint's GitHub URL. See [`blueprints/README.md`](blueprints/README.md) for copying the files by hand.

### Heat-Source Type per Unit

Each unit carries a heat-source type. The integration works it out from the unit's own daily history, once a week after midnight, or on demand with `classify_heat_sources`:

| Type | Evidence |
|------|----------|
| Reversible heat pump | The unit used real energy for heating on cold days **and** for cooling on warm days. A panel heater left in cooling mode does not count: mode time alone is no evidence. |
| Efficiency rises with outdoor temperature | The shape of the unit's heating energy against the cold — an air-source heat pump. |
| Constant efficiency | Direct electric heating *or* a ground-source heat pump. Energy data cannot tell those two apart. |
| Unknown | The default. The unit keeps it until the evidence is clear, including without its coldest days: a heat pump that runs at full capacity on the coldest days can otherwise look like direct electric heating. |

- **Stable:** a type is replaced only when another one is found on two runs in a row.
- **Your setting wins:** set the device type yourself in the advanced settings — one list of energy sensors per type: direct electric, ground source, air-to-water, air-to-air — or with `set_heat_source_type` (`auto` goes back to the worked-out type). Both change the same setting.
- **Heat pump controller:** the unit managed by `heatpump_mpc` is always an air-to-water heat pump.
- **Where it shows:** as attributes on the unit's Mode entity, with a notification when a type is worked out for a unit you have not set yourself. Nothing uses it for predictions yet; `calibrate_balance_point` reports it per unit.

### Smart Solar Tracking & Recommendations (Kelvin Twist)

The system employs a sophisticated 3-zone geometric model ("The Kelvin Twist") to distinguish between direct sunlight, glancing angles, and shadow.

- **Precision Tracking:** Differentiates between what's *possible* (Potential) and what's *actual* (Absorbed), capped by the house's thermodynamic demand (Saturation Logic).
- **Per-Direction Learning:** Each unit learns how much sun it gets through south-, east- and west-facing windows, separately for heating and cooling mode. Coefficients can only be zero or positive — a window can only let heat in.
- **4D Solar Model (optional):** If your weather entity provides direct and diffuse irradiance, or you have a local irradiance (GHI) sensor, a fourth component for diffuse light can be switched on in the reconfigure settings. `diagnose_solar` reports whether your installation is ready for it (`four_d_readiness`). Without real irradiance data the standard 3D model is the better choice, and a repair notice appears if the data disappears while 4D is on.
- **Censoring-Aware Learning:** When the sun covers nearly all of a unit's demand, consumption cannot fall any further, so the hour only shows that the sun helped *at least* that much. A Tobit estimator treats such hours as a lower bound instead of an exact value. It runs in the live learner (on by default, switchable with `set_experimental_tobit_live_learner`) and in `batch_fit_solar`.
- **Obstruction Gates:** `fit_solar_obstruction` suggests sun elevations below which terrain or neighbours, or above which an overhang, block direct sun for a unit's facade. Nothing is written until you accept a suggestion with `apply_obstruction_gate`.
- **Recommendation State:** Actionable advice exposed via attributes (e.g., `recommendation_state` on `sensor.heating_expected_energy_today`).
    - **`maximize_solar`:** Cold + Sunny. Advice: Open blinds/curtains to let free heat in.
    - **`mitigate_solar`:** Hot + Sunny. Advice: Close blinds/curtains to prevent overheating.
    - **`insulate`:** Cold + Dark. Advice: Close blinds/curtains to reduce heat loss through windows.

#### Solar Correction Control (`number.heating_analytics_solar_correction`)

A helper entity (`number.heating_analytics_solar_correction`) allows you to inform the system about your current screen/shading status. This adjusts the calculation of "Actual Solar Impact".

- **`100` (100%):** **Screens Up / Fully Open.** You are allowing 100% of the potential solar energy into the house.
- **`0` (0%):** **Screens Down / Fully Blocked.** You are blocking all solar energy (0% enters).
- **Usage:** Automate this entity based on your smart blinds or set it manually to match your habits. For example, if you close screens halfway, set it to `50`.

### Efficiency Metrics

**Full Spectrum Tracking:** The Efficiency sensor (`kWh/TDD`) works seamlessly in both Winter (Heating) and Summer (Cooling) modes.

- **Seamless Rolling Window:** No midnight jumps or "sawtooth" graphs. The system blends Today's data with Yesterday's to ensure a valid sample window (min 0.5 TDD) is always used for calculation. (See [DESIGN.md](DESIGN.md#8-seamless-rolling-window-efficiency) for details).
- **Yesterday, 7-Day, 30-Day:** Compare efficiency trends over time.

### Advanced Sensors

- **Heating Deviation Sensor:** Compares "Thermodynamic Reality" (Model on Actual Weather) vs "Budget" (Frozen Midnight Forecast). Shows exactly how much you are deviating due to behavior, not weather.
- **Thermodynamic Projection (Reality Check):**
    - **Budget:** The static plan created at midnight (based on the forecast at that time).
    - **Projection:** Where you are actually heading. Calculated as `Actual Usage So Far + Live Forecast for Remaining Day`.
    - **Why:** If the weather changes significantly during the day, the "Budget" becomes obsolete. The "Projection" adapts instantly, giving you a realistic end-of-day total to aim for.
- **Heating Efficiency Sensor:** Real-time `kWh/TDD` metric that handles midnight crossovers seamlessly.
- **Potential Savings Sensor:** Shows daily savings from auxiliary heat. Includes `allocated` (unit-specific) and `unassigned` (global/orphaned) savings breakdown.
- **Thermal State Sensor:** Exposes the thermally weighted temperature used for predictions.
- **Confidence Grading:** Sensors expose confidence metrics (sample counts, standard deviation) so you know exactly how reliable a prediction is.

### High-Fidelity History (Vectors)
The system stores **Hourly Data Vectors** (Temp, Wind, Actual Load) for every day in history. This enables:
- **Precise Re-simulation:** "What if" scenarios use the exact weather/load profile of the past, not just daily averages.
- **Thermodynamic Reconstruction:** Accurate recovery of effective temperatures even from years ago.

> [!NOTE]
> **Design Philosophy: Where are the advanced data points?**
>
> To keep your entity registry clean and uncluttered, Heating Analytics does not create a separate sensor for every metric. Instead, rich secondary data — confidence grades, recommendation states, and hourly data vectors — is exposed as **attributes** on the primary sensors. The main `state` value of each sensor remains a simple scalar, ideal for automations and template math, while the full dataset is available as attributes for template sensors, advanced dashboard cards, and developer tooling.

---

## Entities Created

### Sensors (always created)

| Entity | Unit | Description |
|--------|------|-------------|
| Energy Today | kWh | Total energy consumed today by the configured units |
| Energy Baseline Today | kWh | Model expectation given actual weather. Rich attributes: thermal stress, `thermal_regime`, `primary_driver`, solar/wind detail |
| Efficiency | kWh/TDD | Rolling efficiency with historical averages |
| Weather Plan Today | kWh | Full-day weather-based energy plan (frozen at midnight) |
| Energy Estimate Today | kWh | Best estimate: actuals so far + forecast remainder. Includes confidence level |
| Forecast Details | — | Diagnostic: which forecast source is performing better |
| Deviation Today | % | Actual vs expected deviation with contributor breakdown, and today's heating / cooling split (`accumulated_heating_kwh`, `accumulated_cooling_kwh`) |
| Effective Wind | m/s | Current effective wind with gust factor applied |
| Correlation Data | — | Diagnostic: temperature-vs-energy curves for graphing |
| Last Hour Actual | kWh | Diagnostic: last completed hour's actual consumption |
| Last Hour Expected | kWh | Diagnostic: last completed hour's model prediction |
| Last Hour Deviation | kWh | Diagnostic: last hour deviation with model update details |
| AUX Savings Today | kWh | Estimated energy saved by auxiliary heat (e.g. wood stove) |
| Model Comparison Day/Week/Month | kWh | Current vs same period last year (3 sensors) |
| Week Ahead Forecast | kWh | 7-day energy forecast with daily breakdown, and a range from how far past 7-day plans missed once enough weeks have been scored |
| Period Comparison | — | Diagnostic: result of compare_periods service call |
| Thermal State | °C | Inertia-weighted effective outdoor temperature |
| {Unit} Daily | kWh | Per-unit daily consumption (one per energy sensor) |

### Sensors (conditional)

| Entity | Condition | Description |
|--------|-----------|-------------|
| Daily Learning | `daily_learning_mode` enabled | Learned U-coefficient (kWh/TDD) |
| {Unit} Lifetime | `enable_lifetime_tracking` enabled | Cumulative lifetime energy per unit |

### Controls

| Entity | Type | Default | Description |
|--------|------|---------|-------------|
| Learning Rate | Number | 1.0% | EMA rate for model updates (0.1–10%) |
| Solar Correction | Number | 100% | How much solar reaches the building (0–100%) |
| Learning Enabled | Switch | On | Master on/off for model learning |
| Auxiliary Heating Active | Switch | Off | Signals unmetered heat source is active |
| {Unit} Mode | Select | Heating | Per-unit mode (`heating`, `cooling`, `off`, `dhw`, `guest_heating`, `guest_cooling`). Created for each energy sensor. Shows the unit's heat-source type as attributes. |

---

## Dashboard & Visualizations

Rather than a single monolithic dashboard, the [`tools/`](https://github.com/thuemah/heating_analytics/blob/main/tools/) folder contains ready-to-paste Plotly cards that you combine as needed. Each card references standard integration sensors directly — no fragile pre-built dashboard to maintain.

**Available cards:**

| File | Description |
|------|-------------|
| [plotly_heat_demand_curve.yaml](https://github.com/thuemah/heating_analytics/blob/main/tools/plotly_heat_demand_curve.yaml) | Heat demand curve — temperature vs kWh/day across wind conditions |
| [plotly_today_breakdown_pie.yaml](https://github.com/thuemah/heating_analytics/blob/main/tools/plotly_today_breakdown_pie.yaml) | Donut chart of today's per-unit energy split |
| [plotly_week_ahead_forecast.yaml](https://github.com/thuemah/heating_analytics/blob/main/tools/plotly_week_ahead_forecast.yaml) | 7-day bar chart with temperature and wind overlay |
| [heating_forecast_sensor_and_dashboard.yaml](https://github.com/thuemah/heating_analytics/blob/main/tools/heating_forecast_sensor_and_dashboard.yaml) | 48-hour hourly forecast with solar contribution |
| [mushroom_glance_card.yaml](https://github.com/thuemah/heating_analytics/blob/main/tools/mushroom_glance_card.yaml) | Mushroom glance card for key metrics at a glance |


<img width="682" height="1326" alt="553163441-0666e162-e64f-4b49-b7c9-a82a12223692" src="https://github.com/user-attachments/assets/db555776-54dc-48c3-bee7-6106c4d277b0" />

<img width="677" height="941" alt="553163453-ea956d36-d809-470d-bcf3-3a08aa6d29e3" src="https://github.com/user-attachments/assets/7861ec86-ff6d-48bf-8451-340db2aebf01" />

<img width="611" height="605" alt="557900246-4af1f748-3b0a-4d42-a0c1-3a84ab617c99" src="https://github.com/user-attachments/assets/463156c7-ff85-4824-8cd3-782211ccc727" />




**Required HACS Integrations:**

- [Mushroom Cards](https://github.com/piitaya/lovelace-mushroom)
- [Plotly Graph Card](https://github.com/dbuezas/lovelace-plotly-graph-card)

---

## Configuration Options

### Basic Settings

| Setting | Default | Description |
|---------|---------|-------------|
| **Balance Point** | 17°C | Temperature where heating starts |
| **Learning Rate** | 0.01 | How fast the model adapts (1% per hour) |
| **Wind Gust Factor** | 0.6 | Weight given to wind gusts (60%) |
| **Wind Threshold** | 8.0 m/s | Threshold for 'High Wind' conditions. Too low pushes many hours into the high_wind bucket with insufficient samples. |
| **Extreme Wind Threshold** | 10.8 m/s | Threshold for 'Extreme Wind' conditions. |
| **Thermal Inertia** | 4 hours | Time constant of the effective temperature (1–24 h slider). Low for lightweight structures, high for heavy concrete/stone. See [Thermal Inertia](#thermal-inertia) below. |
| **Hourly Log Retention** | 90 days | How long the detailed hourly log is kept (90 / 180 / 365 days). Retraining and most diagnostics work from this log. |
| **External Screens** | South, east and west | Which facades have external screens that the Solar Correction slider moves. |

### Thermal Mass Correction

When **Daily Learning Mode** is active and an **Indoor Temperature Sensor** is configured, the system applies **Thermal Mass Correction** to the daily energy budget before updating the model.

If the indoor temperature rises over a day, some energy went into heating the building mass itself rather than escaping outside. Without correction, the model would underestimate the building's heat loss coefficient on warming days and overestimate it on cooling days. The correction removes this stored-heat component before learning:

`q_adjusted = daily_kWh − (Thermal Mass Factor × ΔT_indoor)`

**Indoor temperature sensor:** Optional. It is your responsibility to ensure this sensor reflects the actual indoor temperature at midnight (e.g., a centrally placed, non-draft-affected sensor). If the sensor is unavailable or not configured, daily learning still runs — thermal mass correction is simply skipped.

**How to set the Thermal Mass Factor:**
`(Area_m² × 35) / 1000 / avg_COP`

- A typical 250m² house with a heat pump (COP ~3) ≈ `2.9 kWh/°C`
- Lightweight construction (timber frame, thin walls): use a lower value.
- Heavy construction (passive house, concrete, stone): use a higher value.
- Set to `0.0` to disable correction entirely (equivalent to not having a sensor).

### Overnight Load Shift Correction

> [!WARNING]
> **This setting is intended for a narrow group of advanced users.** Enable it only if you actively pre-heat or pre-cool your home across the midnight boundary — for example, by deliberately drawing cheap overnight electricity to store heat in your building structure before a high-tariff morning period. If you do not intentionally load-shift across midnight, leave this **off**. Enabling it incorrectly will cause the model to systematically misattribute stored heat as daytime consumption, progressively biasing the learned heat loss coefficient.

When enabled (requires Daily Learning Mode + Indoor Temperature Sensor), the system applies an additional correction to account for thermal energy loaded into the building mass *before* midnight that would otherwise be counted against the following day's energy budget. The adjustment is symmetric with the standard Thermal Mass Correction but applied across the day boundary:

`q_adjusted = daily_kWh − (Thermal Mass Factor × ΔT_midnight_crossover)`

This setting is UI-only and is not stored in the integration's configuration. It is derived automatically from the presence of a configured indoor temperature sensor on subsequent reconfiguration.

### Thermal Inertia

You can configure how quickly your house reacts to outside temperature changes. The model does not learn against the outdoor temperature right now, but against an *effective* temperature: each recent hour's outdoor temperature weighted by `e^(−t/τ)`, where `t` is how many hours ago it was and `τ` is the Thermal Inertia setting. The newest hour counts most, and the weights reach back `5 × τ` hours (at most a week).

- **1–2 hours:** poorly insulated homes or low thermal mass (e.g., wooden cabin).
- **4 hours (default):** standard insulated homes.
- **12 hours or more:** high thermal mass buildings (e.g., concrete/stone, passive houses) where today's heating depends heavily on yesterday's weather.

`calibrate_inertia` finds the value that explains your consumption best. A gap in the history longer than the setting (Home Assistant down for a while) starts the history afresh.

### Heating Strategy: Which Units to Track?

When configuring the integration, you will be asked to select your heating energy sensors.

**Strategy 1: The Essentials (MVP)**
You can start by just adding your main heating sources (e.g., the central Heat Pump or Electric Boiler). The system works well with just the primary consumers and will provide accurate main-load analysis.

**Strategy 2: The "Energy Detective" (Recommended)**
To unlock the full potential of anomaly detection, you should add **as many heating sources as possible**, even the ones you "think" are small (bathroom floor heating, panel heaters, etc.).

*Why?*
If you have a hidden consumer (like a bathroom floor heater) that isn't tracked, its heat output will still warm the house. The system might misinterpret this extra warmth as "better insulation" or "solar gain," skewing your baseline model. By tracking every watt of heat entering the building envelope, the system can distinguish between "free heat" (sun/people) and "paid heat" (electricity), allowing it to spot genuine deviations and energy leaks with much higher precision.

### Solar Settings

Solar correction is always active. The system exposes a **Solar Correction** number entity (0–100%, default 100%) that represents how much solar gain currently reaches the building (100% = screens fully open, 0% = fully closed). Screen attenuation is applied **per direction** based on three configuration booleans — *External screens on south / east / west facade?* — answered during setup (defaults: all three True). A screened facade ramps from 30 % transmittance at slider 0 % up to 100 % at slider 100 %; an unscreened facade stays at 100 % regardless of the slider. The 30 % is not the screen fabric alone (about 8 %): most "screened" facades also have doors and smaller windows without screens. This preserves the model's estimate of unscreened-window gain when only some facades have screens. Installations without explicit per-direction configuration fall back to a composite floor of 30 %, representing the typical Nordic mix of partly-screened buildings (unmonitored north/utility windows, diffuse radiation, conductive gain through the opaque envelope).

Solar coefficients are learned automatically per unit using a 3D vector model (south, east, and west components), with separate coefficients for heating and cooling mode. Each unit learns how much energy it saves (heating) or consumes additionally (cooling) per unit of raw solar irradiance. The **Screen-affected Units** setting limits the screen effect to the units behind screened windows. The coefficients encode window physics (area, orientation, thermal coupling). For unscreened facades the coefficient converges to pure window physics; for screened facades it absorbs that direction's average transmittance — both cases handled transparently by the same prediction path. Learning uses Normalized LMS (NLMS), which adapts the step size to the solar signal strength: high-solar units and low-solar units converge at the same rate, preventing oscillation in sun-exposed rooms.

A **Solar Thermal Battery** smooths solar impact across hours using exponential decay (default 0.50, half-life about 1 hour). This models heat stored in building mass (concrete, floor slabs) that releases gradually after peak sun. The decay rate can be calibrated per installation via the `diagnose_solar` service.

### Auxiliary Settings

| Setting | Default | Description |
|---------|---------|-------------|
| **Auxiliary Affected Entities** | `All Sensors` | List of heating units that benefit from the auxiliary heat source (e.g. fireplace). |

#### Why Configure Affected Entities?
- **Targeted Protection:** When the fireplace turns off, only the listed units are locked into the "Cooldown Protocol" (2-6 hours).
- **Preserved Learning:** Units *not* on this list (e.g., upstairs bedroom unaffected by downstairs fireplace) continue learning normally during the cooldown period.
- **Global Mode:** If you leave the list empty (`[]`), the system switches to "Global Only" mode. It tracks the *Total Savings* ("Orphaned Savings") for the whole house but won't attribute them to specific rooms.

### Tips

- **Drafty house?** → Increase Wind Gust Factor above the default 0.6 (e.g. 0.8)
- **Model too slow to adapt?** → Increase Learning Rate to 0.02 (2%)
- **Made changes to insulation/heating?** → Temporarily increase Learning Rate to 0.03-0.04 (3-4%) for a few days at stable temperatures to speed up re-learning
- **External screens on some facades only?** → Set the screen facades and the screen-affected units in the settings, so closing the screens does not lower the other windows' solar estimate

### Heat Pump Best Practices ⚠️

**TL;DR:** Night temperature setback often saves little to nothing with heat pumps (0-5%), and may even increase costs due to morning recovery at low COP.

**When setback DOES work:**
- ✅ Resistive heating (10-15% savings)
- ✅ Vacation mode (multi-day absences)

**When to avoid:**
- ❌ Heat pumps with short (<6h) setbacks
- ❌ Well-insulated homes

<details>
<summary><b>Why Night Setback Doesn't Work Well with Heat Pumps</b></summary>

Heat pumps are fundamentally different from traditional resistive heating:

**1. Thermal Mass Effects:**
- **Evening (↓ setpoint 21°C → 18°C):** Building "coasts" on stored heat for 2-4 hours with minimal heating demand.
- **Morning (↑ setpoint 18°C → 21°C):** Heat pump must work hard to reheat thermal mass (walls, floors, furniture).
- The energy "saved" during coasting must be "repaid" during recovery.

**2. COP Efficiency Penalty:**
- Heat pumps are most efficient at **steady, low loads**.
- Morning recovery happens at the **coldest time of day** (lowest COP).
- High power demand during recovery = reduced efficiency.
- Net result: Energy saved during night ≈ Energy wasted during recovery.

**3. Model Learning Complications:**
- Linear temperature scaling doesn't account for thermal mass dynamics.
- Night setback creates non-linear consumption patterns.
- Model may learn incorrect baseline from mixed heating modes.

#### Research-Backed Findings
Studies on heat pump night setback consistently show:
- Savings: typically **0-5%** (sometimes negative).
- Resistive heating savings: **10-15%** (no COP penalty).
- Longer recovery periods = worse results.

#### Recommended Strategy for Heat Pumps

**Best efficiency:**
1. Maintain **constant temperature** 24/7 (e.g., 21°C).
2. Let the heat pump work at steady, low power.
3. Maximize COP by avoiding recovery spikes.
4. Let Heating Analytics learn stable baseline.

**Alternative approach:**
- Lower setpoint by 1-2°C during **entire heating season** instead of nightly.
- Better than cycling: same comfort reduction, better efficiency, stable learning.
</details>

---

## Advanced Features

<details>
<summary><b>Model Comparison (vs Last Year)</b></summary>

Compare current usage to what you *would have used* last year under the same weather:

- **Weather-Normalized:** Uses the learned model applied to last year's conditions
- **ISO Week Alignment:** Week comparisons align Monday-to-Monday for consistent occupancy patterns
- **Solar Backfill:** Estimates solar impact for historical data that predates solar tracking
  - **Cloud Coverage Assumption:** Assumes 50% cloud coverage for historical days where no weather data is available

**Current Day Model:** Uses the "Daily Budget" (Predicted) which combines past actual data + current forecast for stable comparisons.

</details>

<details>
<summary><b>Per-Unit Learning (Multi-Zone)</b></summary>

The integration can learn the unique thermal characteristics of individual heating units (e.g., rooms, floors, or specific heaters). This enables granular analysis and more accurate predictions in homes with multiple heating zones.

**Intelligent Coefficient Learning:**

For each unit, the system learns three separate models:

1.  **Base Consumption Model:** The core model that predicts energy usage based on temperature and wind.
2.  **Solar Coefficient Model:** Learns how much "free heat" a specific zone gets from the sun (see Solar Correction section).
3.  **Auxiliary Coefficient Model:** Learns the power reduction provided by an auxiliary heat source in that zone (see Auxiliary Heating section).

**Buffered Jump-Start Mechanism:**

To ensure stability and prevent slow learning, especially for devices that cycle on and off (like underfloor heating), the system uses a buffering mechanism:

1.  **Buffer Samples:** It collects 4 hourly samples for a given temperature/wind condition before updating the model.
2.  **Jump-Start:** Once the buffer is full, it calculates the average of the samples and uses it to initialize or update the model. This "jump-starts" the learning process, avoiding the "soft start" issue where the model slowly creeps up from zero.
3.  **Persistence:** The learning buffers are saved during restarts, so no data is lost.

</details>

For a detailed description of the learning algorithm, EMA update logic, Purity Guard, and internal data structures, see [DESIGN.md](DESIGN.md).

---

## Services & Automation

The integration provides several services for managing learning data, backups, and system state.

### Replace Sensor Source

**Service:** `heating_analytics.replace_sensor_source`

> [!IMPORTANT]
> This is the **only supported method** for replacing or renaming a heating energy sensor. Removing the old sensor and adding the new one via Reconfigure will permanently delete all learned history for that unit. Always use this service instead.

Use this when a sensor entity ID changes — for example, after a hardware replacement, a meter firmware update that renames the entity, or a manual entity ID change in Home Assistant.

**Parameters:**
- `old_entity_id`: The entity ID currently configured in the integration (must exist in the integration's sensor list)
- `new_entity_id`: The replacement entity ID (must exist in Home Assistant; must not already be configured in the integration)

**What gets migrated:**
- All learned correlation data and temperature/wind bucket history
- Learning buffers (warm-start data)
- Auxiliary and solar coefficients
- Unit operating modes
- Observation counts and hourly delta/expected vectors
- Daily and lifetime individual statistics
- Full hourly log history, and the unit's share of the daily history
- The unit's heat-source type, whether worked out or set by you

**What gets reset:**
- The energy baseline for the replaced sensor. The new sensor will establish its own baseline on the next update cycle, preventing consumption spikes caused by differing cumulative totals between the old and new meter.

The integration reloads automatically after the migration completes.

```yaml
action: heating_analytics.replace_sensor_source
data:
  old_entity_id: sensor.heat_pump_energy_old
  new_entity_id: sensor.heat_pump_energy_new
```

---

### Backup & Restore Learning Data

**Service:** `heating_analytics.backup_data`

Create a complete backup of your learning model, history, and configuration to a JSON file.

**Parameters:**
- `file_path`: Absolute path for backup file (e.g., `/config/heating_backup.json`)

The file must be inside the Home Assistant config directory, or in a directory listed under `allowlist_external_dirs`. The config directory's `.storage` folder is not allowed. The same rule applies to `restore_data`, `export_to_csv` and `import_from_csv`.

**What's Included:**
- Complete learning model (all correlation data)
- Per-unit learning models
- Learning buffers (warm-start data)
- Daily history and hourly logs
- Configuration settings
- Observation counts

**When to Use:**
- Before making major system changes
- Regular backups for safety
- Migrating to a new Home Assistant instance

---

**Service:** `heating_analytics.restore_data`

Restore complete system state from a backup JSON file.

**Parameters:**
- `file_path`: Absolute path to backup JSON file (e.g., `/config/heating_backup.json`)

**Warning:** This overwrites ALL current data including learning models and history!

**Example:**
```yaml
service: heating_analytics.backup_data
data:
  file_path: /config/heating_backup_2025_12.json
```

---

### Reset Learning Data

**Service:** `heating_analytics.reset_learning_data`

Reset the entire learning model to start fresh.

**Parameters:** None

**What Gets Reset:**
- All correlation data (global and per-unit)
- Learning buffers
- Observation counts
- Solar and auxiliary coefficients
- The daily learning U-coefficient

**What's Preserved:**
- Daily history
- Hourly logs
- Configuration settings

**When to Use:**
- After major home renovations (new insulation, windows, etc.)
- When the model has learned incorrect patterns
- Starting a new heating season with different patterns

---

**Service:** `heating_analytics.reset_unit_learning_data`

Reset the learned model for a specific heating unit only.

**Parameters:**
- `entity_id`: The heating unit's energy sensor (e.g., `sensor.kitchen_heater_energy`)

**What Gets Reset:**
- Correlation data for this unit only
- Auxiliary and solar coefficients for this unit
- Observation counts for this unit

**What's Preserved:**
- Learning buffer for this unit (enables faster re-learning!)
- All other units' learning data

**When to Use:**
- After insulation work in one room/zone
- After replacing a heater in one room
- When one unit's predictions are inaccurate but others are fine

**Example:**
```yaml
service: heating_analytics.reset_unit_learning_data
data:
  entity_id: sensor.living_room_heater_energy
```

**Why Use This Instead of Full Reset?**

Unlike `reset_learning_data`, this service preserves the learning buffer. The samples in it are applied on the next cold start to "jump-start" learning, giving you usable predictions sooner. To rebuild the unit from the hourly log instead, use `retrain_unit_from_history` with `reset_first` (below).

---

### Retrain from History

**Service:** `heating_analytics.retrain_from_history`

Retrains the model using the existing hourly log — no CSV needed. Replays logged hours through the same learning rules used during live operation, with your current settings (balance point, thermal inertia, wind thresholds, units and modes). A retrain is never required: the model adapts to new settings on its own as new hours come in. A retrain applies them to the stored history at once.

```yaml
service: heating_analytics.retrain_from_history
data:
  days_back: 30
  dry_run: true
```

**Parameters:**
- `entity_id` (optional): Target instance
- `days_back` (optional, 1–730): Limit to most recent N days. Empty = all available
- `reset_first` (default false): Clear all learned data before retraining. The rebuild then comes from the hourly log alone, which is kept for the configured retention period. Temperatures the log does not cover, often the coldest winter days, lose their learned values and cannot be rebuilt.
- `dry_run` (default false): Run the same retrain on a copy of the model and write nothing. The response lists the temperature buckets that would be added, removed and changed (overall, per 5 °C band and per unit), the solar and auxiliary coefficients before and after, and the predictions of the current and the retrained model over the last 30 days, per day and per temperature band. Those are set against the consumption the model learns from; energy used in guest, off and hot-water mode is shown separately, because the model does not predict it. Use it before any retrain, with or without `reset_first`.
- `experimental_cop_smear` (default false): For daily learning (Track B) with `heatpump_mpc` COP parameters available, spread each day over its hours weighted by the heat pump's per-hour efficiency instead of evenly. No effect on hourly learning or Track C.

**`retrain_unit_from_history`** does the same for one unit's own model (`unit_entity_id`), with `days_back`, `reset_first` and `dry_run`. It leaves the unit's solar and auxiliary coefficients as they are.

---

### Other Reset Services

| Service | Description |
|---------|-------------|
| `reset_forecast_accuracy` | Clears forecast accuracy tracking history. Preserves energy logs. |
| `reset_solar_learning` | Resets solar coefficients for one unit (`unit_entity_id`) or all units in an instance. With `replay_from_history`, refits them from the hourly log straight away. |
| `exit_cooldown` | Force-exits the auxiliary cooldown period, resuming normal learning immediately. |
| `compare_periods` | Compares two historical periods. Returns delta analysis as a response. |

---

### Get Hourly Forecast Plan

**Service:** `heating_analytics.get_forecast`

Retrieve the detailed hourly plan (prediction) from the current hour onwards.

**Parameters:**
- `days` (1–14, default 1): How many days ahead
- `isolate_sensor` (optional): An energy sensor. Each hour then returns that unit's share only, `max(0, whole house − Σ other units)` — the demand a heat pump controller's own unit must cover
- `human_readable` (default false): A condensed summary for people instead of the full payload: a day-level summary, the largest contributors per hour with friendly names, the solar split and the weather source per hour

**Returns:**
A dictionary containing the forecast plan:
- `forecast`: List of hourly objects with:
    - `datetime`: ISO timestamp
    - `kwh`: Expected energy usage
    - `temp` / `inertia_temp`: Outdoor and effective temperature
    - `wind_speed`: Wind speed
    - `solar_kwh`, `solar_offset_kwh`, `solar_load_kwh`: Solar effect, split into heating reduction and cooling addition
    - `aux_impact_kwh`: Estimated savings from auxiliary heating (if active)
    - `unit_breakdown`: Expected energy per unit
    - `source`: Which weather source the hour used

**Example:**
```yaml
action: heating_analytics.get_forecast
data:
  days: 2
response_variable: heating_plan
```

**Use Case:**
- Automations that need to know *exactly* how much energy the house will use in the next few hours.
- Heat pump controllers (`isolate_sensor`) and custom dashboards that need raw prediction data.

---

### Calibrate Balance Point

**Service:** `heating_analytics.calibrate_balance_point`

Checks your configured balance point against the last year of daily history. It fits each day's heating energy against how cold the day was: a flat floor above the balance point and a rising arm below it. It works from whole days, so it can run at any time of year. Only energy used by units in heating mode counts, and days with auxiliary heat, guest mode, many windy hours or missing hours are left out.

- **Sunshine** — the day's own and heat still stored from the day before — lowers the day's balance point. It is fitted, so the result is the balance point of a day without sun.
- **Heat pumps** get more heat out of each kWh the milder it is outside, which bends the arm. That is fitted too; the report shows the balance point for each assumed heat pump curve, and each unit's heat-source type.
- **A new value is suggested only when** fits on alternating weeks agree, the result does not depend on the assumed heat pump curve, no part of the sun or heat pump model sits at its limit, the model fits most days, your current value lies outside the fit's uncertainty, and the change is at least 1 °C.

```yaml
action: heating_analytics.calibrate_balance_point
```

**Suggestion only** — nothing is written. To act on a suggestion, change the balance point in the integration's reconfigure settings. The report also tells you how many recent hours the change would reclassify.

---

### Calibrate Wind Thresholds

**Service:** `heating_analytics.calibrate_wind_thresholds`

> [!NOTE]
> This service analyzes **Track A hourly data only**. It requires sufficient pure (non-aux, non-solar) hourly observations across different wind conditions. Not applicable for Track B or Track C installations that have not accumulated Track A history.

Tests historical data to find the optimal high wind and extreme wind thresholds for your location by reclassifying hours and comparing model error (MAE) across different threshold pairs.

```yaml
action: heating_analytics.calibrate_wind_thresholds
data:
  days: 60
```

**Returns:** Recommended thresholds, per-bucket MAE, hour distribution, data quality assessment, and improvement percentage over current thresholds.

**When to use:** After 30–60 days of Track A data, or if you suspect wind thresholds are causing noisy predictions. A too-low high wind threshold pushes many hours into the `high_wind` bucket with insufficient samples, adding noise to the correlation curve.

---

### Calibrate Thermal Inertia

**Service:** `heating_analytics.calibrate_inertia`

> [!NOTE]
> This service analyzes **Track A hourly data only**. It requires at least 30 days of hourly observations to produce reliable results.

Tests historical data to find the ideal thermal inertia time constant (tau, 1–24 hours) for your building. The primary result uses a causal exponential decay kernel matching the coordinator's RC-circuit model.

```yaml
action: heating_analytics.calibrate_inertia
```

**Returns:** Recommended tau (set the Thermal Inertia slider to it), the fit for each tau value, and a Gaussian sweep for reference. `live_axis_comparison` shows how well the effective temperature learning used up to 1.3.15 and the current one explain your consumption, per outdoor-temperature band.

---

### Diagnose Model

**Service:** `heating_analytics.diagnose_model`

Analyzes the learned correlation model and hourly history for data quality issues. Useful for diagnosing model drift, inverted correlation curves, or validating data after configuration changes.

```yaml
action: heating_analytics.diagnose_model
data:
  days: 30
```

**Returns a diagnostic report with:**

- **Monotonicity check:** Per wind bucket, is the correlation curve falling as temperature drops (physically correct)? Reports any inversions and their magnitude.
- **Bucket population:** Observation count per `temp_key × wind_bucket`. Flags under-sampled buckets (< 4 observations) and wind bucket imbalance.
- **Mode contamination:** Per-day breakdown of hours in OFF, DHW, Guest, and Cooling modes. Useful for validating that mode filtering is working correctly.
- **Solar correlation:** Checks whether solar factor correlates with prediction error — positive correlation suggests solar is not fully captured by the model.
- **Track B diagnostics:** For Track B days — `q_adjusted` vs raw kWh, thermal mass correction applied, daily average temperature, and which bucket was populated.

---

### Diagnose Solar Model

**Service:** `heating_analytics.diagnose_solar`

Analyzes per-unit solar coefficient health and global solar model quality. Useful for identifying mis-calibrated units, validating coefficient convergence, and tuning the solar thermal battery decay rate.

```yaml
action: heating_analytics.diagnose_solar
data:
  days: 30
```

**Returns a diagnostic report with:**

- **Per-unit coefficient analysis:** Current vs implied coefficients (back-calculated from hourly data via 3×3 normal equations), stability across 3 time windows, saturation frequency, and dominant component (south, east, or west).
- **Battery decay health:** Post-sunset residual analysis — detects if the thermal battery decays too fast or too slow. Includes a calibration sweep (0.50–0.95) with the recommended decay rate.
- **Screen correction impact:** Compares prediction error at different screen positions (closed vs open) to detect screen-induced coefficient drift.
- **Temporal bias:** Morning vs afternoon mean prediction delta — reveals timing errors in the cloud model or battery decay.
- **Hour-of-day residual curve:** Per-hour mean error from 6:00 to 18:00.
- **4D readiness (`four_d_readiness`) and irradiance source (`dni_dhi_source_mix`):** Whether your weather data supplies real direct and diffuse sunlight, and whether the 4D model has learned enough to be switched on.

**Battery calibration:** To automatically apply the recommended decay rate:

```yaml
action: heating_analytics.diagnose_solar
data:
  days: 30
  apply_battery_decay: true
```

---

### Heat-Source Type Services

| Service | Description |
|---------|-------------|
| `classify_heat_sources` | Works out each unit's heat-source type now instead of waiting for the weekly run, and returns the evidence per unit. |
| `set_heat_source_type` | Sets a unit's type (`unit_entity_id`, `heat_source_type`), overriding the worked-out one. `auto` goes back to the worked-out type. Changes the same setting as the lists in the advanced settings. |

See [Heat-Source Type per Unit](#heat-source-type-per-unit).

---

### Solar Model Services

For cases the live learners cannot settle on their own. The fitting services take a `unit_entity_id` to limit them to one unit and a `dry_run` to preview the result.

| Service | Description |
|---------|-------------|
| `batch_fit_solar` | Fits each unit's solar coefficients over the last `days_back` days (default 30) in one go, for heating and cooling separately. Useful when the sun peaks on mild afternoons with little demand to measure against. When the unit already has a learned coefficient, the result is blended with it. |
| `batch_fit_solar_4d` | The same for the optional 4D model's coefficients. |
| `apply_implied_coefficient` | Writes the coefficient `diagnose_solar` implies from your data, direction by direction, skipping directions that are not stable across time windows (`force` overrides). Meant for units no learner can fit, such as a unit run by a heat pump controller. |
| `fit_solar_obstruction` | Suggests sun elevations below which (terrain, neighbours) or above which (an overhang) direct sun is blocked, per unit and facade. Suggestions only, and only when they hold in at least two of the 30-, 60- and 90-day windows. |
| `apply_obstruction_gate` | Accepts one suggestion (`unit_entity_id`, `facade`, `side`, `value`), or clears it with `clear`. Below 2–20° for terrain, above 20–60° for an overhang. |
| `calibrate_unit_thresholds` | Recalculates each unit's noise floor for solar learning from its dark hours. Runs by itself at every startup. |

#### Tobit Live Learner

| Service | Description |
|---------|-------------|
| `set_experimental_tobit_live_learner` | Turns the censoring-aware live learner on or off (`enabled`; turning it on needs `confirm: true`). On by default. |
| `set_tobit_live_entities` | Limits it to the listed units; an empty list lets it decide per unit. |
| `reset_tobit_live_state` | Clears its running statistics, for one unit or all. The coefficients themselves are not touched. |

---

## Data Import / Export

<details>
<summary><b>Export to CSV</b></summary>

**Service:** `heating_analytics.export_to_csv`

Export your data for backup or external analysis.

**Parameters:**
- `file_path`: Absolute path (e.g., `/config/heating_daily.csv`)
- `export_type`: `daily` or `hourly`

**Daily Export Format:**
```csv
timestamp,kwh,temp,tdd,wind
2025-11-15,25.29,-2.9,19.9,2.1
2025-11-16,22.57,-1.1,18.1,3.5
```

**Hourly Export Format:**
```csv
timestamp,hour,temp,effective_wind,wind_bucket,actual_kwh,expected_kwh,deviation,deviation_pct,auxiliary_active,solar_factor,solar_impact_kwh
2025-11-17T01:00:00,1,-2.9,2.11,normal,1.14,1.05,0.09,8.5,False,0.12,0.05
```

</details>

<details>
<summary><b>Import from CSV</b></summary>

**Service:** `heating_analytics.import_from_csv`

Jump-start learning by importing historical data.

**Parameters:**
- `file_path`: Absolute path to CSV file (required)
- `update_model`: (Optional, default `true`) Set to `false` to update history without retraining the model
- `column_mapping`: Map your CSV headers to data fields (required)
  - `timestamp`: Column name for timestamp/date (required)
  - `energy`: Column name for energy consumption in kWh (optional)
  - `temperature`: Column name for outdoor temperature (optional)
  - `wind_speed`: Column name for wind speed (optional)
  - `wind_gust`: Column name for wind gust (optional)
  - `cloud_coverage`: Column name for cloud coverage percentage 0-100 (optional)
  - `is_auxiliary`: Column name for boolean flag indicating auxiliary heating (optional). Accepts `1`, `true`, `yes`, `on`.

**Example CSV:**
```csv
Date,Energy_kWh,Avg_Temp,Wind_Avg,Gust_Max,Cloud_Cover_Pct,Fireplace
2023-01-01 00:00,2.3,-5.0,3.5,6.0,45,0
2023-01-01 01:00,2.1,-5.5,3.0,5.5,50,1
```

**Example YAML:**
```yaml
service: heating_analytics.import_from_csv
data:
  file_path: /config/history.csv
  update_model: true
  column_mapping:
    timestamp: Date
    energy: Energy_kWh
    temperature: Avg_Temp
    wind_speed: Wind_Avg
    wind_gust: Gust_Max
    cloud_coverage: Cloud_Cover_Pct
    is_auxiliary: Fireplace
```

**Supported Formats:**
- **Daily:** One row per day (energy = daily total)
- **Hourly:** Multiple rows per day (energy = hourly consumption)
- The system auto-detects and aggregates duplicate dates

</details>

---

## Troubleshooting

**"Predictions are way off!"**
- Wait 7-14 days for initial learning (especially if starting from scratch)
- Check that temperature sensor is accurate
- Verify energy meter is reporting correctly
- Ensure balance point matches your heating system's behavior

**"Efficiency sensor shows 'Unavailable'"**
- Normal during warm weather (low heating demand = insufficient Thermal Degree Days)
- Sensor reappears when heating resumes

**"Solar correction seems wrong"**
- Check the `Solar Correction` slider (0–100) matches your typical blind/screen usage
- Check that weather entity provides cloud coverage
- Consider using Open-Meteo for more accurate solar data

**"A repair notice says the irradiance data has stopped"**
- Your weather provider stopped supplying direct and diffuse sunlight while the 4D solar model is on. The model now runs on an estimate from cloud coverage, where the standard 3D model is better. Switch the 4D model off in the reconfigure settings, or fix the weather source. The notice clears by itself when the data comes back.

**"Model learning too slowly"**
- Increase Learning Rate (e.g., from 0.01 to 0.02)
- Import historical data to jump-start

**"Deviation alerts during normal cycling"**
- This is expected for hysteresis-based systems (thermostats cycle on/off)
- The system filters out most false positives automatically
- Reduce sensitivity if needed

---

## Logging & History

**Hourly Logging:**
- Keeps detailed log with configurable retention (90 / 180 / 365 days, default 90)
- Includes: temperature, wind, humidity, expected vs actual energy, per-unit energy and modes, and which units' meters reported
- Available attributes on `sensor.heating_analytics_last_hour_deviation`:
  - `model_updated_temp_category`: Temperature bucket used for learning
  - `model_value_before/after`: Model prediction before/after update
  - `model_delta`: Change in model value
  - `inertia_temperature`: The effective temperature the hour was learned under

**Daily Logging:**
- Stores aggregated daily stats indefinitely
- Includes: kWh, Avg Temp, TDD, wind conditions, forecast accuracy

---

## License

MIT License - See [LICENSE](LICENSE) file for details.

---

## Contributing

Contributions are welcome! Please submit issues and pull requests on [GitHub](https://github.com/thuemah/heating_analytics).

---

## Credits

Developed by [thuemah](https://github.com/thuemah) for the Home Assistant community.

Inspired by the need to understand and optimize home heating in Nordic climates.
