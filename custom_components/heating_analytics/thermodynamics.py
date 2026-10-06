"""
Thermodynamic Engine (Track C)

This module implements the pure thermodynamic modeling for Heating Analytics.
It decouples the physical heat loss of the building from the electrical consumption
of the heat pump, which is necessary when using Model Predictive Control (MPC)
and Load Shifting.

The core philosophy:
1. The heat pump delivers Thermal Energy (kWh) to the house.
2. The house loses Heat (Thermal Energy) to the environment continuously based on
   Temperature (Delta-T), Wind, and Solar radiation.
3. To avoid the "Feedback Loop of Death" where MPC shifting heat production to night
   tricks the model into thinking the house loses more heat at night, we separate
   production from consumption.

The "Midnight Sync" (Track C):
1. Collect exactly how much Thermal Energy (kWh) the heat pump delivered over 24 hours.
2. Calculate the theoretical heat loss for each of those 24 hours based ONLY on weather.
3. "Smear" (distribute) the total delivered Thermal Energy across the 24 hours
   proportionally to the theoretical heat loss.
4. Calculate the Daily Average COP (sum(kwh_th_sh) / sum(kwh_el_sh)).
   We DO NOT use an unweighted average of hourly COPs (The "Average Fallacy"),
   and we DO NOT use the hourly COP to convert the smeared thermal load back to electricity,
   because the heat pump might have been OFF during that hour (COP = undefined).
5. Convert the smeared hourly thermal load back into a "Synthetic Electrical Baseline"
   by dividing it by the Daily Average COP. This baseline represents what the heat
   pump *would* have consumed if it ran continuously to exactly match the heat loss.
"""
from __future__ import annotations

import logging
from typing import TypedDict

_LOGGER = logging.getLogger(__name__)


class MPCData(TypedDict):
    """Data contract for incoming MPC hourly data."""
    datetime: str
    kwh_th_sh: float  # Thermal energy delivered to space heating
    kwh_el_sh: float  # Electrical energy consumed for space heating
    mode: str         # "sh", "dhw", "off"


class WeatherData(TypedDict):
    """Theoretical physical factors for an hour."""
    datetime: str
    delta_t: float    # Indoor Temp - Outdoor Temp
    wind_factor: float # Multiplier for wind
    solar_factor: float # Reduction factor for solar gain (0.0 to 1.0)
    outdoor_temp: float  # Raw outdoor temp (°C) — for COP calculation
    humidity: float  # Relative humidity (%) — for defrost penalty


class _CopParamsRequired(TypedDict):
    """Required COP model parameters."""
    eta_carnot: float  # Second-law efficiency (~0.38–0.46)
    lwt: float  # Leaving water temperature setpoint (°C)


class CopParams(_CopParamsRequired, total=False):
    """MPC COP model parameters from heatpump_mpc.get_cop_params.

    ``eta_carnot`` and ``lwt`` are required; the rest have sensible
    defaults in ``cop_at_conditions()``.
    """
    f_defrost: float  # Defrost penalty multiplier (< 1.0 during icing)
    defrost_temp_threshold: float  # Outdoor temp below which defrost applies (°C)
    defrost_rh_threshold: float  # RH above which defrost applies (%)


class HourlyDistribution(TypedDict):
    """The result of the thermodynamic smearing for a single hour."""
    datetime: str
    mode: str
    theoretical_loss_weight: float
    smeared_kwh_th: float
    synthetic_kwh_el: float


class ThermodynamicEngine:
    """Engine for Track C thermodynamic calculations."""

    _KELVIN_OFFSET = 273.15

    def __init__(self, balance_point: float = 17.0):
        self.balance_point = balance_point

    @staticmethod
    def cop_at_conditions(
        t_outdoor: float,
        humidity: float,
        eta_carnot: float,
        lwt: float,
        f_defrost: float,
        defrost_temp_threshold: float = 7.0,
        defrost_rh_threshold: float = 70.0,
    ) -> float:
        """Compute COP from MPC model parameters for a given hour's conditions.

        Uses the Carnot COP scaled by the MPC's learned second-law efficiency
        (η_Carnot), with a defrost penalty applied when outdoor conditions
        indicate evaporator icing (cold + humid).

        This mirrors heatpump_mpc's HeatPumpModel.get_effective_cop() but
        requires no access to the MPC codebase — only the learned parameters
        returned by ``heatpump_mpc.get_cop_params``.
        """
        t_hot_k = lwt + ThermodynamicEngine._KELVIN_OFFSET
        t_cold_k = t_outdoor + ThermodynamicEngine._KELVIN_OFFSET
        if t_hot_k <= t_cold_k:
            return 1.0  # No lift → resistive equivalent
        cop_carnot = t_hot_k / (t_hot_k - t_cold_k)
        is_icing = t_outdoor < defrost_temp_threshold and humidity > defrost_rh_threshold
        penalty = f_defrost if is_icing else 1.0
        return max(1.0, min(10.0, eta_carnot * cop_carnot * penalty))

    def _calculate_theoretical_loss_weight(self, weather: WeatherData) -> float:
        """
        Calculate a relative 'weight' representing the physical HVAC demand
        for a given hour, based purely on weather data.

        Supports both heating and cooling regimes (#792):
        - delta_t is abs(balance_point - outdoor), always >= 0
        - For heating (outdoor < balance_point): solar reduces demand
        - For cooling (outdoor > balance_point): solar increases demand
        """
        # HVAC demand is proportional to |delta_T| regardless of direction.
        demand = max(0.0, weather["delta_t"])

        # Wind increases heat transfer in both directions
        wind_multiplier = max(1.0, weather.get("wind_factor", 1.0))
        demand *= wind_multiplier

        # Solar effect depends on regime:
        # is_cooling is set by the caller when outdoor > balance_point.
        solar_f = weather.get("solar_factor", 1.0)
        if weather.get("is_cooling", False):
            # Sun adds to cooling load — more sun = more demand
            solar_multiplier = max(0.1, 1.0 + (1.0 - solar_f) * 0.5)
        else:
            # Sun reduces heating load — more sun = less demand
            solar_multiplier = max(0.0, min(1.0, solar_f))
        demand *= solar_multiplier

        return demand

    def calculate_synthetic_baseline(
        self,
        mpc_data_24h: list[MPCData],
        weather_data_24h: list[WeatherData],
        cop_params: CopParams | None = None,
    ) -> list[HourlyDistribution]:
        """
        Perform the "Midnight Sync" calculation for a 24-hour period.

        When ``cop_params`` is provided (from ``heatpump_mpc.get_cop_params``),
        each hour is converted from thermal to electrical using the MPC's
        COP model evaluated at that hour's outdoor temperature and humidity.
        This correctly attributes higher electrical cost to cold hours (low COP)
        and lower cost to warm hours (high COP).

        When ``cop_params`` is None (legacy / fallback), the daily average COP
        is used uniformly — COP cancels and the result is a pure weather-weighted
        redistribution of actual electrical consumption.

        Args:
            mpc_data_24h: 24 hours of actual production data from the heat pump/MPC.
            weather_data_24h: 24 hours of corresponding weather conditions.
            cop_params: Optional MPC COP model parameters for per-hour conversion.

        Returns:
            A list of 24 hourly distributions containing the smeared thermal load
            and the synthetic electrical baseline.
        """
        if len(mpc_data_24h) != len(weather_data_24h):
            _LOGGER.warning("Mismatch between MPC data length and Weather data length.")

        total_kwh_th = 0.0
        total_kwh_el = 0.0

        # 1. Sum up total delivered thermal energy and total electrical energy for Space Heating.
        for data in mpc_data_24h:
            # Even if mode == "dhw", the actual thermal delivered to Space Heating
            # should be 0.0 in the MPC data per contract, but we sum it safely anyway.
            if data["mode"] != "dhw":
                total_kwh_th += max(0.0, data["kwh_th_sh"])
                total_kwh_el += max(0.0, data["kwh_el_sh"])

        # 2. Calculate the Daily Average COP
        # Guard against division by zero (e.g., heat pump was off all day or only ran DHW).
        # Fallback to COP = 1.0 (direct electric heating equivalent) to avoid zero-division downstream.
        if total_kwh_el > 0.0:
            daily_avg_cop = total_kwh_th / total_kwh_el
            # Also guard against zero thermal output (e.g., bad tagging, standby) to prevent division by zero below
            if daily_avg_cop <= 0.0:
                daily_avg_cop = 1.0
        else:
            daily_avg_cop = 1.0

        _LOGGER.debug(f"Midnight Sync: Total Thermal={total_kwh_th:.2f}, Total Elec={total_kwh_el:.2f}, Daily COP={daily_avg_cop:.2f}")

        # 3. Calculate the theoretical loss weights for each hour
        weights = []
        for weather in weather_data_24h:
            weights.append(self._calculate_theoretical_loss_weight(weather))

        total_weight = sum(weights)

        # Safe default weather if completely empty
        default_weather: WeatherData = {
            "datetime": "unknown",
            "delta_t": 10.0,
            "wind_factor": 1.0,
            "solar_factor": 1.0,
            "outdoor_temp": self.balance_point - 10.0,
            "humidity": 50.0,
        }

        # Pre-compute per-hour COP when MPC model parameters are available.
        # This correctly attributes cold hours (low COP) with higher electrical
        # cost than warm hours (high COP).
        use_per_hour_cop = cop_params is not None and "eta_carnot" in cop_params

        hour_weights: list[float] = []
        hour_cops: list[float] = []
        for i, _mpc in enumerate(mpc_data_24h):
            # Match weather to MPC data (assuming they are aligned/sorted)
            if i < len(weather_data_24h):
                weather = weather_data_24h[i]
            elif weather_data_24h:
                weather = weather_data_24h[-1]
            else:
                weather = default_weather
            hour_weights.append(
                weights[i] if i < len(weights) else self._calculate_theoretical_loss_weight(weather)
            )
            # Per-hour COP: uses MPC model at this hour's (T, RH) → correct
            #   attribution of COP variation across the day.
            # Daily avg COP (fallback): COP cancels mathematically → equivalent
            #   to weather-weighted redistribution of actual electrical.
            hour_cops.append(
                self._hour_cop(weather, cop_params) if use_per_hour_cop else daily_avg_cop
            )

        # Renormalize per-hour COP path to preserve daily electrical total.
        # sum(smeared_th_h / COP_h) != total_kwh_el in general (Jensen's
        # inequality).  Scaling preserves the per-hour *shape* while anchoring
        # the 24h total to actual metered consumption.  The daily-avg COP
        # path already conserves energy by construction.
        return self._smear(
            [(mpc["datetime"], mpc["mode"]) for mpc in mpc_data_24h],
            hour_weights,
            total_weight,
            total_kwh_th,
            hour_cops,
            renormalize_to=total_kwh_el if use_per_hour_cop else None,
        )

    # Below this, an hour's stored smeared / synthetic kWh are too coarse
    # (rounded to 3 decimals) to recover its COP from their ratio: at 0.01 kWh
    # each carries up to 5 % rounding error.
    RESMEAR_MIN_RATIO_KWH = 0.01

    def resmear_distribution(
        self,
        stored: list[HourlyDistribution],
        weather_rows: list[WeatherData],
        cop_params: CopParams | None = None,
    ) -> tuple[list[HourlyDistribution], str]:
        """Re-spread a stored day at this engine's settings (#1111).

        A day's distribution is smeared once, at midnight, with the balance
        point, inertia axis, wind thresholds and solar battery decay in force
        then.  Retrain replays history under the current settings, so it
        re-spreads each day with ``weather_rows`` built from them — one row
        per stored hour, in order.  The day's totals are kept: thermal is
        ``Σ smeared_kwh_th`` and electrical ``Σ synthetic_kwh_el``.

        The per-hour COP comes from ``cop_params`` when the day recorded
        them (exact).  Otherwise it is recovered from the stored hours: the
        midnight sync made ``synthetic_h = total_el · (w_h / COP_h) /
        Σ w_k / COP_k``, so ``smeared_h / synthetic_h`` is ``COP_h`` times
        a daily constant, and the constant cancels when the result is
        renormalised to the electrical total.  An hour too small to read
        (or with no weight at midnight, hence no energy) takes the day's
        mean ratio.

        Returns the new distribution and ``"exact_cop"`` or
        ``"recovered_cop"``.
        """
        total_kwh_th = sum(max(0.0, float(d.get("smeared_kwh_th", 0.0) or 0.0)) for d in stored)
        total_kwh_el = sum(max(0.0, float(d.get("synthetic_kwh_el", 0.0) or 0.0)) for d in stored)
        weights = [self._calculate_theoretical_loss_weight(w) for w in weather_rows]
        hours = [(d.get("datetime"), d.get("mode", "sh")) for d in stored]

        if cop_params is not None and "eta_carnot" in cop_params:
            hour_cops = [self._hour_cop(w, cop_params) for w in weather_rows]
            method = "exact_cop"
        else:
            mean_ratio = total_kwh_th / total_kwh_el if total_kwh_el > 0 else 1.0
            if mean_ratio <= 0:
                mean_ratio = 1.0
            hour_cops = []
            for d in stored:
                th = float(d.get("smeared_kwh_th", 0.0) or 0.0)
                el = float(d.get("synthetic_kwh_el", 0.0) or 0.0)
                if th >= self.RESMEAR_MIN_RATIO_KWH and el >= self.RESMEAR_MIN_RATIO_KWH:
                    hour_cops.append(th / el)
                else:
                    hour_cops.append(mean_ratio)
            method = "recovered_cop"

        return self._smear(
            hours,
            weights,
            sum(weights),
            total_kwh_th,
            hour_cops,
            renormalize_to=total_kwh_el if total_kwh_el > 0 else None,
        ), method

    def _hour_cop(self, weather: WeatherData, cop_params: CopParams) -> float:
        """The MPC COP model at one hour's outdoor temperature and humidity."""
        return self.cop_at_conditions(
            t_outdoor=weather.get("outdoor_temp", self.balance_point - weather["delta_t"]),
            humidity=weather.get("humidity", 50.0),
            eta_carnot=cop_params["eta_carnot"],
            lwt=cop_params["lwt"],
            f_defrost=cop_params.get("f_defrost", 0.85),
            defrost_temp_threshold=cop_params.get("defrost_temp_threshold", 7.0),
            defrost_rh_threshold=cop_params.get("defrost_rh_threshold", 70.0),
        )

    @staticmethod
    def _smear(
        hours: list[tuple[str, str]],
        weights: list[float],
        total_weight: float,
        total_kwh_th: float,
        hour_cops: list[float],
        *,
        renormalize_to: float | None,
    ) -> list[HourlyDistribution]:
        """Spread a day's thermal total over its hours and convert to electrical.

        The one smearing implementation: the midnight sync
        (:meth:`calculate_synthetic_baseline`) and the retrain re-smear
        (:meth:`resmear_distribution`) both end here.  ``hours`` are the
        (datetime, mode) of each hour, ``weights`` its theoretical loss
        weight and ``hour_cops`` its COP.  ``renormalize_to`` rescales the
        electrical hours to that daily total (per-hour COP does not conserve
        it by itself).
        """
        distribution: list[HourlyDistribution] = []
        for (dt_str, mode), weight, hour_cop in zip(hours, weights, hour_cops):
            # Smear the thermal load proportionally
            if total_weight > 0:
                smeared_th = total_kwh_th * (weight / total_weight)
            else:
                # If the weather says 0 heat loss for the whole 24h, distribute evenly or set to 0.
                # Usually, total_weight > 0 if there was any heating need.
                smeared_th = total_kwh_th / len(hours) if hours else 0.0

            # NOTE ON DHW:
            # If mode == "dhw", the heat pump was producing hot water, not heating the house.
            # However, the building envelope still lost heat to the environment during this hour!
            # Therefore, this hour STILL receives its proportional share of `smeared_th`.
            # The thermodynamic truth is that the house cooled down, and that lost energy
            # had to be replaced either before or after the DHW run.
            synthetic_el = smeared_th / hour_cop

            distribution.append({
                "datetime": dt_str,
                "mode": mode,
                "theoretical_loss_weight": weight,
                "smeared_kwh_th": round(smeared_th, 3),
                "synthetic_kwh_el": round(synthetic_el, 3),
            })

        if renormalize_to is not None and renormalize_to > 0:
            synthetic_total = sum(d["synthetic_kwh_el"] for d in distribution)
            if synthetic_total > 0:
                scale = renormalize_to / synthetic_total
                for d in distribution:
                    d["synthetic_kwh_el"] = round(d["synthetic_kwh_el"] * scale, 3)

        return distribution
