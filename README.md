# Heating Analytics for Home Assistant

[![hacs_badge](https://img.shields.io/badge/HACS-Custom-orange.svg)](https://github.com/custom-components/hacs)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/thuemah/heating_analytics/graphs/commit-activity)
[![License](https://img.shields.io/github/license/thuemah/heating_analytics.svg)](https://github.com/thuemah/heating_analytics/blob/main/LICENSE)

## Why Heating Analytics?

**Turn your Home Assistant into a smart energy detective.**

Your heating system consumes energy every day, but how do you know if it's working efficiently? This integration learns your home's unique thermal behavior and tells you exactly how much energy you *should* be using right now—based on current weather—versus what you're *actually* using.

**Catch energy waste before it catches you:**

- Detect open windows draining heat
- Measure real savings from your fireplace
- Spot inefficiencies in real-time
- Get accurate daily forecasts based on weather

Think of it as a fitness tracker for your home's heating system.

---

## How It Works

<img width="1308" height="1252" alt="flow4" src="https://github.com/user-attachments/assets/d1a3b194-ed9c-447c-bc50-3f16f759d6cc" />



The integration continuously learns the relationship between outdoor conditions (temperature, wind, solar) and your heating consumption. Once trained, it predicts what you *should* use and compares it to reality—alerting you to unexpected deviations.

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
- Enable "Wind Compensation" if you have a weather station with wind data
- The system accounts for wind chill and gusts affecting heat loss
- **Supported Units:** `m/s`, `km/h`, `mph`, `kn` (knots)

**Solar Configuration:**
- Enter your total window area (m²)
- Set window orientation (azimuth: 0°=North, 90°=East, 180°=South, 270°=West)
- The system calculates "free heat" from sunlight to avoid false efficiency readings

**Balance Point:**
- Adjust the temperature where heating kicks in (default: 17°C)
- Fine-tune learning rate if the model reacts too slowly/quickly

---

### Level 3: Advanced Setup (Power Users)

**Historical Data Import:**

If you have past energy/weather data, jump-start the learning process:
- Export data from your old system as CSV
- Use `heating_analytics.import_from_csv` service (details below)
- The model trains instantly on months/years of history

**Multi-Zone Tracking:**
- Configure individual heating units (living room, bedroom, etc.)
- Track per-room efficiency and solar distribution

**Dashboard & Visualizations:**
- See [dashboard_heating.md](dashboard_heating.md) for ready-made Lovelace YAML
- Includes Plotly charts, efficiency cards, and forecast breakdowns

---

## Weather Integration Recommendations

**Works with any weather provider** (Met.no, AccuWeather, etc.)

**Unlock Enhanced Features with Open-Meteo:**

Pair with the [Open-Meteo integration](https://github.com/thuemah/open_meteo) for:

- Precise **Wind Gust** compensation (not all providers include this)
- **Vertical Solar** calculations for windows (vs. roof-mounted panels)
- **7-day hourly forecast** (most providers only give daily forecasts)
- Hourly cloud coverage data for accurate solar correction

*Open-Meteo is free, requires no API key, and provides detailed hourly forecast data.*

This is optional but highly recommended for maximum accuracy.

---

## Key Features

### Machine Learning That Just Works

- **Automatic Learning:** No manual calibration needed—the system learns your home's thermal characteristics
- **Regime-Aware Prediction:**
    - **Cold Regime:** Uses thermodynamic scaling for accurate extrapolation in deep winter.
    - **Mild Regime:** Prioritizes neighbor averaging and wind fallbacks for stability in variable transition seasons.
- **Thermal Inertia:** Accounts for 4-hour rolling average temperature (buildings don't react instantly)
- **Adaptive:** Updates hourly based on actual vs expected consumption

### Thermal Degree Days (TDD)

The integration uses **Thermal Degree Days (TDD)** instead of traditional Heating Degree Days (HDD).
- **Formula:** `abs(Balance Point - Outdoor Temp) / 24`
- **Why?** TDD is a unified metric that handles both **Heating** (below Balance Point) and **Cooling** (above Balance Point) in mixed-mode systems.
- **Benefit:** Provides a continuous efficiency metric (`kWh / TDD`) year-round, regardless of season.

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
- **Exit Condition:** The system monitors real-time consumption. Once the affected units' usage returns to expected levels (convergence), the lock is released.
- **Benefit:** Ensures your "Normal Heating Model" remains pure and unpolluted by residual heat from the fireplace.

### Smart Solar Tracking & Recommendations (Kelvin Twist)

The system employs a sophisticated 3-zone geometric model ("The Kelvin Twist") to distinguish between direct sunlight, glancing angles, and shadow.

- **Precision Tracking:** Differentiates between what's *possible* (Potential) and what's *actual* (Absorbed), capped by the house's thermodynamic demand (Saturation Logic).
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
- **Heating Efficiency Sensor:** Real-time `kWh/TDD` metric that handles midnight crossovers seamlessly. Includes a sanity check (Error if COP > 10).
- **Potential Savings Sensor:** Shows daily savings from auxiliary heat. Includes `allocated` (unit-specific) and `unassigned` (global/orphaned) savings breakdown.
- **Thermal State Sensor:** Exposes the thermally weighted temperature used for predictions.
- **Confidence Grading:** Sensors expose confidence metrics (sample counts, standard deviation) so you know exactly how reliable a prediction is.

### High-Fidelity History (Vectors)
The system stores **Hourly Data Vectors** (Temp, Wind, Actual Load) for every day in history. This enables:
- **Precise Re-simulation:** "What if" scenarios use the exact weather/load profile of the past, not just daily averages.
- **Thermodynamic Reconstruction:** Accurate recovery of effective temperatures even from years ago.

---

## Dashboard & Visualizations

A complete dashboard configuration is available in [dashboard_heating.md](dashboard_heating.md).

**Includes:**

- Plotly charts (Expected vs Actual)
- Efficiency cards with trends
- Deviation alerts
- Weather & solar impact
- Week-ahead forecast

<img width="626" height="417" alt="tmp" src="https://github.com/user-attachments/assets/7d9c91be-8d21-426d-9ccb-ea6b75df20a8" />


<img width="682" height="1326" alt="week_ahead" src="https://github.com/user-attachments/assets/0666e162-e64f-4b49-b7c9-a82a12223692" />


<img width="677" height="941" alt="model_comparison" src="https://github.com/user-attachments/assets/ea956d36-d809-470d-bcf3-3a08aa6d29e3" />


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
| **Wind Threshold** | 5.5 m/s | Threshold for 'High Wind' conditions. |
| **Extreme Wind Threshold** | 10.8 m/s | Threshold for 'Extreme Wind' conditions. |
| **Thermal Inertia** | Normal | Building thermal mass profile (Fast, Normal, Slow) |

### Thermal Inertia Profiles

You can configure how quickly your house reacts to outside temperature changes.

- **Normal (Default):** 4-hour window (20% current, 80% history). Best for standard insulated homes.
- **Fast:** 2-hour window (50% current, 50% history). Best for poorly insulated homes or low thermal mass (e.g., wooden cabin).
- **Slow:** 12-hour window (Bell curve). Best for high thermal mass buildings (e.g., concrete/stone, passive houses) where today's heating depends heavily on yesterday's weather.

### Heating Strategy: Which Units to Track?

When configuring the integration, you will be asked to select your heating energy sensors.

**Strategy 1: The Essentials (MVP)**
You can start by just adding your main heating sources (e.g., the central Heat Pump or Electric Boiler). The system works well with just the primary consumers and will provide accurate main-load analysis.

**Strategy 2: The "Energy Detective" (Recommended)**
To unlock the full potential of anomaly detection, you should add **as many heating sources as possible**, even the ones you "think" are small (bathroom floor heating, panel heaters, etc.).

*Why?*
If you have a hidden consumer (like a bathroom floor heater) that isn't tracked, its heat output will still warm the house. The system might misinterpret this extra warmth as "better insulation" or "solar gain," skewing your baseline model. By tracking every watt of heat entering the building envelope, the system can distinguish between "free heat" (sun/people) and "paid heat" (electricity), allowing it to spot genuine deviations and energy leaks with much higher precision.

### Solar Settings

| Setting | Default | Description |
|---------|---------|-------------|
| **Solar Enabled** | `false` | Enable solar correction |
| **Window Area** | 10 m² | Total window area facing outside |
| **Azimuth** | 180° | Window orientation (South = 180°) |
| **Initial Heating Coefficient** | 0.15 | Starting solar coefficient for heating, optimized for heat pumps. The model will auto-tune this. |
| **Initial Cooling Coefficient** | 0.17 | Starting solar coefficient for cooling. The model will auto-tune this. |

### Auxiliary Settings

| Setting | Default | Description |
|---------|---------|-------------|
| **Auxiliary Affected Entities** | `All Sensors` | List of heating units that benefit from the auxiliary heat source (e.g. fireplace). |

#### Why Configure Affected Entities?
- **Targeted Protection:** When the fireplace turns off, only the listed units are locked into the "Cooldown Protocol" (2-6 hours).
- **Preserved Learning:** Units *not* on this list (e.g., upstairs bedroom unaffected by downstairs fireplace) continue learning normally during the cooldown period.
- **Global Mode:** If you leave the list empty (`[]`), the system switches to "Global Only" mode. It tracks the *Total Savings* ("Orphaned Savings") for the whole house but won't attribute them to specific rooms.

### Tips

- **Drafty house?** → Increase Wind Gust Factor to 0.5
- **Model too slow to adapt?** → Increase Learning Rate to 0.02 (2%)
- **Made changes to insulation/heating?** → Temporarily increase Learning Rate to 0.03-0.04 (3-4%) for a few days at stable temperatures to speed up re-learning
- **Big south-facing windows?** → Enable solar with accurate area for best results

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

<details>
<summary><b>Learning Model Architecture</b></summary>

**Temperature Buckets:**
- Rounds temperatures to nearest integer (18.3°C → "18")
- Uses **inertia temperature** (4-hour rolling average) to account for thermal lag
- Prevents reactivity to short-term fluctuations

**Hourly Learning Updates:**
- Triggered every hour at the hour boundary
- Calculates average conditions for completed hour
- Updates model using **Exponential Moving Average (EMA):**
  ```
  new_prediction = old_prediction + learning_rate × (actual - old_prediction)
  ```
- Tracks observation counts per (temperature, wind) combo for confidence metrics

**Purity Guard (Learning Quality Control):**
- Only learns from hours that are >95% in one heating mode (normal or auxiliary)
- Mixed hours are skipped to prevent model pollution
- Example: If you used auxiliary heating for 58 minutes and normal heating for 2 minutes in one hour, the system learns from that hour (96.6% purity)
- Example: If you switched modes (e.g. 50/50 split or even 90/10), the hour is skipped to ensure model integrity
- Ensures clean learning data by avoiding hours where energy consumption can't be clearly attributed

**Daily Processing:**
- At midnight, logs daily statistics
- Does **not** retrain model (hourly updates only)
- Captures forecast snapshot for deviation baseline

</details>

<details>
<summary><b>Performance Optimization</b></summary>

Even with years of data, the system remains fast:

- **O(1) Efficiency Calculations:** Uses rolling accumulators instead of re-scanning logs
- **Cached Comparisons:** Model comparisons cache past periods (95% reduction in recalculation)
- **Optimized Deviation Analysis:** Pre-calculates global factors once per update
- **Batched Forecast Baselines:** Week-ahead forecasts pre-fetch historical baselines in a single batch (O(1) lookup vs O(N) iterative search)

</details>

---

## Services & Automation

The integration provides several services for managing learning data, backups, and system state.

### Backup & Restore Learning Data

**Service:** `heating_analytics.backup_data`

Create a complete backup of your learning model, history, and configuration to a JSON file.

**Parameters:**
- `file_path`: Absolute path for backup file (e.g., `/config/heating_backup.json`)

**What's Included:**
- Complete learning model (all correlation data)
- Per-unit learning models
- Learning buffers (cold-start data)
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
- Solar coefficients

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

Unlike `reset_learning_data`, this service preserves the learning buffer. The buffer contains the last 10 samples collected for each temperature/wind condition. When the model resets, these samples are immediately used to "jump-start" learning, giving you accurate predictions within hours instead of days.

---

### Get Hourly Forecast Plan

**Service:** `heating_analytics.get_forecast`

Retrieve the detailed hourly heating plan (prediction) for today.

**Parameters:**
- `type`: `hourly` (Only supported type currently)

**Returns:**
A dictionary containing the forecast plan:
- `forecast`: List of hourly objects with:
    - `datetime`: ISO timestamp
    - `predicted_kwh`: Expected energy usage
    - `temperature`: Outdoor temperature
    - `wind_speed`: Wind speed
    - `aux_impact_kwh`: Estimated savings from auxiliary heating (if active)

**Example:**
```yaml
service: heating_analytics.get_forecast
data:
  type: hourly
response_variable: heating_plan
```

**Use Case:**
- Automations that need to know *exactly* how much energy the house will use in the next few hours.
- Custom dashboards that need raw prediction data.

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
- Verify window area and azimuth settings
- Check that weather entity provides cloud coverage
- Consider using Open-Meteo for more accurate solar data

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
- Keeps detailed log of the last 90 days
- Includes: temperature, wind, expected vs actual energy
- Available attributes on `sensor.heating_analytics_last_hour_deviation`:
  - `model_updated_temp_category`: Temperature bucket used for learning
  - `model_value_before/after`: Model prediction before/after update
  - `model_delta`: Change in model value
  - `inertia_temperature`: 4-hour rolling average temperature

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
