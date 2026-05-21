"""Focused tests for ``batch_fit_solar_coefficients_4d`` (#954 commit 8).

Strict shadow: writes only to ``solar_coefficients_4d_per_unit``.
Driver assembles 4D samples via the f41ffd8 hour-midpoint pipeline
(``get_approx_sun_pos`` + ``resolve_dni_dhi`` +
``calculate_unit_potential_4d``) and threads them through the new
``_solve_tobit`` MLE.
"""
from __future__ import annotations

import random
from unittest.mock import MagicMock

from custom_components.heating_analytics.const import (
    MODE_HEATING,
    SOLAR_COEFF_CAP,
)
from custom_components.heating_analytics.learning import LearningManager
from custom_components.heating_analytics.solar import SolarCalculator


def _stub_sun_pos(dt_obj):
    """Deterministic sun-position stub for offline tests.

    Astral isn't available in the sandbox.  We don't need real
    ephemerides — we need a function that returns ``elev > 0`` at
    hours 8, 11, 13, 16 and yields enough azimuth diversity that
    the 4D matrix is well-conditioned.  Use a simple smooth hourly
    curve over the day.
    """
    import math
    h = dt_obj.hour + dt_obj.minute / 60.0
    # Solar noon ~12:30 in this stub; elev peaks ~55° (mid-spring NL).
    elev = max(0.0, 55.0 * math.sin(math.pi * (h - 6.0) / 12.0))
    # Azimuth: east=90 at 6h, south=180 at 12h, west=270 at 18h.
    az = 90.0 + 15.0 * (h - 6.0)
    az = max(60.0, min(300.0, az))
    return (elev, az)


def _make_coord(*, correlation_data_per_unit=None):
    coord = MagicMock()
    coord.screen_config = (False, False, False)
    coord._unit_strategies = {}
    coord._correlation_data_per_unit = correlation_data_per_unit or {}
    coord.screen_config_for_entity = MagicMock(
        side_effect=lambda _eid: (False, False, False)
    )
    coord.latitude = 52.0
    coord.longitude = 5.0
    coord.timezone = "UTC"
    solar = SolarCalculator(coord)
    # Patch the sun-position lookup so tests don't depend on astral.
    solar.get_approx_sun_pos = _stub_sun_pos
    coord.solar = solar
    model = MagicMock()
    model.correlation_data_per_unit = coord._correlation_data_per_unit
    coord.model = model
    coord._solar_affected_set = None
    coord.wind_threshold = 8.0
    coord.extreme_wind_threshold = 10.8
    coord._get_wind_bucket = lambda w: (
        "extreme_wind" if w >= coord.extreme_wind_threshold
        else "high_wind" if w >= coord.wind_threshold
        else "normal"
    )
    return coord


def _entry(
    ts: str,
    *,
    sensor_id: str = "sensor.heater1",
    dni: float = 700.0,
    dhi: float = 150.0,
    correction: float = 100.0,
    actual_kwh: float = 1.5,
    mode: str = MODE_HEATING,
    aux_active: bool = False,
    shutdown: bool = False,
    temp: float = 10.0,
    wind_bucket: str = "normal",
):
    return {
        "timestamp": ts,
        "hour": int(ts[11:13]),
        "temp": temp,
        "temp_key": str(int(round(temp))),
        "wind_bucket": wind_bucket,
        "dni": dni,
        "dhi": dhi,
        "ghi": None,
        "cloud_coverage_avg": None,
        "correction_percent": correction,
        "auxiliary_active": aux_active,
        "actual_kwh": actual_kwh,
        "unit_modes": {sensor_id: mode},
        "unit_breakdown": {sensor_id: actual_kwh},
        "solar_dominant_entities": [sensor_id] if shutdown else [],
        "learning_status": "logged",
    }


def _build_log_4d(
    n_days: int,
    true_coeff: dict[str, float],
    *,
    base: float = 2.5,
    sensor_id: str = "sensor.heater1",
    seed: int = 1337,
) -> list[dict]:
    """Generate per-hour log entries across multiple days at noon-ish hours.

    Build 4 daytime hours per day so we get diverse sun azimuth /
    elevation values.  ``actual = base − coeff·potential_4d + ε``.
    """
    entries = []
    rng = random.Random(seed)
    # Use a fixed solar calc with the same lat/lon for offline synth.
    coord = _make_coord()
    solar = coord.solar
    from datetime import datetime, timedelta
    from custom_components.heating_analytics.solar import resolve_dni_dhi

    # Spread across a month to get azimuth diversity
    start = datetime(2026, 4, 1, 0, 0, 0)
    for d in range(n_days):
        for h in (8, 11, 13, 16):
            ts_dt = start + timedelta(days=d, hours=h)
            mid_dt = ts_dt + timedelta(minutes=30)
            elev, az = solar.get_approx_sun_pos(mid_dt)
            if elev <= 5.0:
                continue
            dni, dhi, _ = resolve_dni_dhi(700.0, 150.0, None, None, elev, ts_dt.timetuple().tm_yday)
            pot = solar.calculate_unit_potential_4d(
                sensor_id, dni, dhi, elev, az,
                (False, False, False), 100.0,
            )
            impact = (
                true_coeff.get("s", 0.0) * pot[0]
                + true_coeff.get("e", 0.0) * pot[1]
                + true_coeff.get("w", 0.0) * pot[2]
                + true_coeff.get("diffuse", 0.0) * pot[3]
            )
            eps = rng.gauss(0.0, 0.02)
            actual = max(0.0, base - impact + eps)
            entries.append(_entry(
                ts_dt.strftime("%Y-%m-%dT%H:00:00"),
                sensor_id=sensor_id,
                dni=700.0,
                dhi=150.0,
                actual_kwh=actual,
            ))
    return entries


class TestBatchFit4D:

    def test_batch_fit_4d_writes_when_gates_pass(self):
        """30 days × 4 daytime hours → enough samples, writes 4D coeff."""
        true_c = {"s": 0.0012, "e": 0.0008, "w": 0.0009, "diffuse": 0.0006}
        entries = _build_log_4d(30, true_c, base=2.5)
        coord = _make_coord(
            correlation_data_per_unit={
                "sensor.heater1": {
                    str(t): {"normal": 2.5} for t in range(0, 20)
                }
            }
        )
        coeffs_4d: dict = {}
        lm = LearningManager()
        result = lm.batch_fit_solar_coefficients_4d(
            hourly_log=entries,
            solar_coefficients_4d_per_unit=coeffs_4d,
            energy_sensors=["sensor.heater1"],
            coordinator=coord,
        )
        diag = result["sensor.heater1"]["heating"]
        # Some entries may fall below sun-elev threshold in early/late
        # hours — assert at least 20 uncensored survived the gates.
        assert diag.get("skip_reason") is None, diag.get("skip_reason")
        assert diag["applied"] is True
        learned = coeffs_4d["sensor.heater1"]["heating"]
        assert learned["learned"] is True
        for k in ("s", "e", "w", "diffuse"):
            assert 0.0 <= learned[k] <= SOLAR_COEFF_CAP

    def test_batch_fit_4d_dry_run_no_write(self):
        true_c = {"s": 0.0012, "e": 0.0008, "w": 0.0009, "diffuse": 0.0006}
        entries = _build_log_4d(30, true_c, base=2.5)
        coord = _make_coord(
            correlation_data_per_unit={
                "sensor.heater1": {
                    str(t): {"normal": 2.5} for t in range(0, 20)
                }
            }
        )
        coeffs_4d: dict = {}
        lm = LearningManager()
        result = lm.batch_fit_solar_coefficients_4d(
            hourly_log=entries,
            solar_coefficients_4d_per_unit=coeffs_4d,
            energy_sensors=["sensor.heater1"],
            coordinator=coord,
            dry_run=True,
        )
        diag = result["sensor.heater1"]["heating"]
        assert diag.get("applied") is False
        assert diag.get("dry_run") is True
        # coefficient_after still reports the would-be result
        assert set(diag["coefficient_after"].keys()) >= {
            "s", "e", "w", "diffuse",
        }
        # Live dict untouched.
        assert "sensor.heater1" not in coeffs_4d

    def test_batch_fit_4d_reads_correct_log_keys(self):
        """Collector must read ``ghi_wm2`` / ``cloud_coverage`` (canonical
        names emitted by ``hourly_processor``), not ``ghi`` /
        ``cloud_coverage_avg``.  Regression for Codex review on #954:
        installs without native DNI/DHI fell through to the "none"
        ladder source and every row was dropped as ``no_dni_dhi`` even
        though valid GHI / cloud history existed.

        Build a log carrying ONLY the canonical keys (dni/dhi None) and
        verify (a) drop_counts.no_dni_dhi is small (samples actually
        routed via the GHI → Erbs path), and (b) at least one row made
        it through to the solver (non-zero ``n_uncensored`` in diag).
        """
        true_c = {"s": 0.0012, "e": 0.0008, "w": 0.0009, "diffuse": 0.0006}
        # Synthesise entries with only canonical fields populated.
        from datetime import datetime, timedelta
        from custom_components.heating_analytics.solar import resolve_dni_dhi

        coord = _make_coord(
            correlation_data_per_unit={
                "sensor.heater1": {
                    str(t): {"normal": 2.5} for t in range(0, 20)
                }
            }
        )
        solar = coord.solar
        entries = []
        rng = random.Random(7)
        start = datetime(2026, 4, 1, 0, 0, 0)
        for d in range(30):
            for h in (8, 11, 13, 16):
                ts_dt = start + timedelta(days=d, hours=h)
                mid_dt = ts_dt + timedelta(minutes=30)
                elev, az = solar.get_approx_sun_pos(mid_dt)
                if elev <= 5.0:
                    continue
                # Pre-compute what the GHI-ladder will yield so synthetic
                # actual matches the resolved DNI/DHI.
                dni, dhi, _ = resolve_dni_dhi(
                    None, None, 700.0, 20.0, elev, ts_dt.timetuple().tm_yday,
                )
                pot = solar.calculate_unit_potential_4d(
                    "sensor.heater1", dni or 0.0, dhi or 0.0, elev, az,
                    (False, False, False), 100.0,
                )
                impact = (
                    true_c["s"] * pot[0]
                    + true_c["e"] * pot[1]
                    + true_c["w"] * pot[2]
                    + true_c["diffuse"] * pot[3]
                )
                actual = max(0.0, 2.5 - impact + rng.gauss(0.0, 0.02))
                entry = _entry(
                    ts_dt.strftime("%Y-%m-%dT%H:00:00"),
                    actual_kwh=actual,
                )
                # Strip the legacy / non-canonical keys; populate ONLY
                # what ``hourly_processor`` actually writes.
                entry["dni"] = None
                entry["dhi"] = None
                entry.pop("ghi", None)
                entry.pop("cloud_coverage_avg", None)
                entry["ghi_wm2"] = 700.0
                entry["cloud_coverage"] = 20.0
                entries.append(entry)

        coeffs_4d: dict = {}
        lm = LearningManager()
        result = lm.batch_fit_solar_coefficients_4d(
            hourly_log=entries,
            solar_coefficients_4d_per_unit=coeffs_4d,
            energy_sensors=["sensor.heater1"],
            coordinator=coord,
        )
        diag = result["sensor.heater1"]["heating"]
        # The bug surfaced as "every row dropped → insufficient_uncensored".
        # Post-fix: samples must flow through the GHI ladder.
        drop_counts = diag.get("drop_counts", {})
        # At most a handful of legitimate drops (sun-below-horizon early/late
        # in the day); definitely not every row.
        assert drop_counts.get("no_dni_dhi", 0) < 10, (
            f"GHI ladder bypassed — no_dni_dhi={drop_counts.get('no_dni_dhi')}"
        )
        tobit_diag = diag.get("tobit_diagnostics", {})
        assert tobit_diag.get("n_uncensored", 0) > 20, (
            f"No samples reached the solver; diag={diag}"
        )

    def test_batch_fit_4d_skips_below_min_uncensored(self):
        """Below TOBIT_MIN_UNCENSORED → skip with 'insufficient_uncensored'."""
        true_c = {"s": 0.001, "e": 0.001, "w": 0.001, "diffuse": 0.0005}
        # 2 days → ~8 entries, well below the 20-sample floor
        entries = _build_log_4d(2, true_c, base=2.5)
        coord = _make_coord(
            correlation_data_per_unit={
                "sensor.heater1": {
                    str(t): {"normal": 2.5} for t in range(0, 20)
                }
            }
        )
        coeffs_4d: dict = {}
        lm = LearningManager()
        result = lm.batch_fit_solar_coefficients_4d(
            hourly_log=entries,
            solar_coefficients_4d_per_unit=coeffs_4d,
            energy_sensors=["sensor.heater1"],
            coordinator=coord,
        )
        diag = result["sensor.heater1"]["heating"]
        assert diag["skip_reason"] == "insufficient_uncensored"
        assert "sensor.heater1" not in coeffs_4d
