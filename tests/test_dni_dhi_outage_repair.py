"""Tests for the DNI/DHI outage repair issue (#1070).

Two halves, tested separately:

* ``DiagnosticsEngine._compute_dni_dhi_outage`` — the pure verdict over a
  trailing window of daylight hours.  No HA involvement.
* ``repairs.async_check_dni_dhi_outage`` — the create/delete side-effect
  that maps a verdict onto HA's issue registry.

The seam between them is deliberate: the verdict is the part with the
physics-shaped reasoning (daylight filtering, hysteresis), and it is
testable without standing up a config entry or an issue registry.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from custom_components.heating_analytics.const import (
    REPAIR_DNI_DHI_OUTAGE_MIN_HOURS,
    REPAIR_DNI_DHI_OUTAGE_WINDOW_HOURS,
)
from custom_components.heating_analytics.diagnostics import DiagnosticsEngine


def _entry(source: str | None, *, solar_factor: float = 0.5) -> dict:
    """One hourly_log entry with the two fields the window reads."""
    e = {"timestamp": "2026-07-27T12:00:00", "solar_factor": solar_factor}
    if source is not None:
        e["dni_dhi_source"] = source
    return e


def _engine(log, *, flag_on=True, solar_on=True) -> DiagnosticsEngine:
    coord = MagicMock()
    coord._hourly_log = log
    coord.experimental_4d_primary = flag_on
    coord.solar_enabled = solar_on
    return DiagnosticsEngine(coord)


def _window(source: str, n: int = REPAIR_DNI_DHI_OUTAGE_WINDOW_HOURS) -> list:
    return [_entry(source) for _ in range(n)]


# ---------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------

def test_raises_when_provider_stops_supplying_irradiance():
    """A full window of kasten_synthetic while 4D is live -> raise."""
    result = _engine(_window("kasten_synthetic"))._compute_dni_dhi_outage()
    assert result["verdict"] == "raise"
    assert result["real_source_share"] == 0.0
    assert result["daylight_hours_examined"] == REPAIR_DNI_DHI_OUTAGE_WINDOW_HOURS


def test_clears_when_provider_recovers():
    result = _engine(_window("native"))._compute_dni_dhi_outage()
    assert result["verdict"] == "clear"
    assert result["real_source_share"] == 1.0


def test_erbs_from_ghi_counts_as_a_real_source():
    """A local pyranometer is a real source, same as native DNI/DHI."""
    result = _engine(_window("erbs_from_ghi"))._compute_dni_dhi_outage()
    assert result["verdict"] == "clear"


def test_hysteresis_band_holds_rather_than_flapping():
    """Between the raise and clear bars, neither fires.

    This is the whole point of the asymmetric thresholds: a provider that
    drops the fields intermittently must not create and delete the repair
    on alternating days.  12/24 native is below the 0.5 clear bar and
    above the 0.1 raise bar.
    """
    log = _window("native", 11) + _window("kasten_synthetic", 13)
    result = _engine(log)._compute_dni_dhi_outage()
    assert result["verdict"] == "hold"
    assert 0.10 <= result["real_source_share"] < 0.50


def test_single_stray_real_hour_still_raises():
    """The raise bar is slack for a stray hour, not a statistical claim."""
    log = _window("kasten_synthetic", 23) + _window("native", 1)
    result = _engine(log)._compute_dni_dhi_outage()
    assert result["verdict"] == "raise"


# ---------------------------------------------------------------------
# The two gates
# ---------------------------------------------------------------------

def test_not_applicable_when_4d_is_not_routed():
    result = _engine(
        _window("kasten_synthetic"), flag_on=False
    )._compute_dni_dhi_outage()
    assert result["verdict"] == "not_applicable"


def test_not_applicable_when_solar_is_disabled():
    """Gated on solar_enabled too: with solar off the 4D path never runs.

    Warning about the input quality of a pipeline that is not executing
    is noise.
    """
    result = _engine(
        _window("kasten_synthetic"), solar_on=False
    )._compute_dni_dhi_outage()
    assert result["verdict"] == "not_applicable"


def test_flag_must_be_literally_true():
    """MagicMock coordinators yield truthy auto-attributes for any name.

    Same ``is True`` discipline as the 4D dispatcher in statistics.py.
    """
    coord = MagicMock()
    coord._hourly_log = _window("kasten_synthetic")
    coord.solar_enabled = True
    # experimental_4d_primary left as an auto-created MagicMock attribute
    result = DiagnosticsEngine(coord)._compute_dni_dhi_outage()
    assert result["verdict"] == "not_applicable"


# ---------------------------------------------------------------------
# Window population
# ---------------------------------------------------------------------

def test_partial_window_yields_insufficient_data_not_a_raise():
    """A fresh 4D install must not raise a repair on day one."""
    log = _window("kasten_synthetic", REPAIR_DNI_DHI_OUTAGE_MIN_HOURS - 1)
    result = _engine(log)._compute_dni_dhi_outage()
    assert result["verdict"] == "insufficient_data"
    assert result["real_source_share"] is None


def test_night_hours_do_not_count_toward_the_window():
    """The daylight filter is load-bearing, not hygiene.

    ``derive_dni_dhi_source_label`` never emits ``"no_sun"``, so night
    hours carry ``kasten_synthetic`` on any install with cloud data.  A
    wall-clock window would fire every night on a healthy install.  Here
    a full window of *native* daylight hours is buried under night hours
    that must be skipped rather than counted as an outage.
    """
    night = [_entry("kasten_synthetic", solar_factor=0.0) for _ in range(50)]
    log = _window("native") + night
    result = _engine(log)._compute_dni_dhi_outage()
    assert result["verdict"] == "clear"
    assert result["daylight_hours_examined"] == REPAIR_DNI_DHI_OUTAGE_WINDOW_HOURS


def test_unlabelled_hours_are_skipped_not_counted_either_way():
    """Pre-#1058 entries carry no label and are not evidence.

    Counting them as healthy would mask a real outage; counting them as
    an outage would fire on every freshly-upgraded install.
    """
    log = _window("native", 5) + [_entry(None) for _ in range(40)]
    result = _engine(log)._compute_dni_dhi_outage()
    assert result["verdict"] == "insufficient_data"
    assert result["daylight_hours_examined"] == 5


def test_window_reads_the_most_recent_hours():
    """Walks backwards: an old outage must not mask a current recovery."""
    old_outage = _window("kasten_synthetic", 100)
    recent_recovery = _window("native")
    result = _engine(old_outage + recent_recovery)._compute_dni_dhi_outage()
    assert result["verdict"] == "clear"


# ---------------------------------------------------------------------
# Registry side-effect
# ---------------------------------------------------------------------

def _coord_for_repair(verdict: str):
    coord = MagicMock()
    coord.entry.entry_id = "abc123"
    coord.entry.title = "Heating Analytics"
    coord.evaluate_dni_dhi_outage.return_value = {
        "verdict": verdict,
        "daylight_hours_examined": 24,
        "real_source_hours": 0,
    }
    return coord


@pytest.mark.parametrize("verdict", ["raise"])
def test_raise_verdict_creates_a_fixable_warning(verdict):
    from custom_components.heating_analytics import repairs

    with patch.object(repairs, "ir") as mock_ir:
        repairs.async_check_dni_dhi_outage(MagicMock(), _coord_for_repair(verdict))

    assert mock_ir.async_create_issue.called
    kwargs = mock_ir.async_create_issue.call_args.kwargs
    assert kwargs["is_fixable"] is True
    assert kwargs["translation_key"] == "dni_dhi_outage"
    assert kwargs["data"] == {"entry_id": "abc123"}
    # Every placeholder the translation references must be supplied.
    assert set(kwargs["translation_placeholders"]) >= {"name", "hours", "real_hours"}
    assert not mock_ir.async_delete_issue.called


@pytest.mark.parametrize("verdict", ["clear", "not_applicable"])
def test_recovery_and_disable_both_delete_the_issue(verdict):
    """It must auto-resolve, or it becomes an absorbing state."""
    from custom_components.heating_analytics import repairs

    with patch.object(repairs, "ir") as mock_ir:
        repairs.async_check_dni_dhi_outage(MagicMock(), _coord_for_repair(verdict))

    assert mock_ir.async_delete_issue.called
    assert not mock_ir.async_create_issue.called


@pytest.mark.parametrize("verdict", ["hold", "insufficient_data"])
def test_hold_and_insufficient_do_not_touch_the_registry_at_all(verdict):
    """Neither creates nor deletes — the existing state must persist.

    ``hold`` is the hysteresis band and ``insufficient_data`` means no
    evidence either way; both must leave whatever is registered alone.
    """
    from custom_components.heating_analytics import repairs

    with patch.object(repairs, "ir") as mock_ir:
        repairs.async_check_dni_dhi_outage(MagicMock(), _coord_for_repair(verdict))

    assert not mock_ir.async_create_issue.called
    assert not mock_ir.async_delete_issue.called


def test_registry_failure_never_breaks_the_hour_boundary():
    """Hourly processing owns learning and persistence; it must not die here."""
    from custom_components.heating_analytics import repairs

    coord = _coord_for_repair("raise")
    with patch.object(repairs, "ir") as mock_ir:
        mock_ir.async_create_issue.side_effect = RuntimeError("registry exploded")
        repairs.async_check_dni_dhi_outage(MagicMock(), coord)  # must not raise


class TestFixFlow:
    """The only #1070 code that mutates production config.

    Previously untested, including the scaffolding in conftest that was
    added specifically to test it.
    """

    @staticmethod
    def _flow(entry_id: str, *, entry_exists: bool = True):
        from custom_components.heating_analytics import repairs

        flow = repairs.DniDhiOutageRepairFlow(entry_id)
        flow.hass = MagicMock()
        entry = MagicMock()
        entry.data = {
            "experimental_4d_primary": True,
            "energy_sensors": ["sensor.a"],
        }
        flow.hass.config_entries.async_get_entry = MagicMock(
            return_value=entry if entry_exists else None
        )

        async def _reload(_eid):
            return True

        flow.hass.config_entries.async_reload = MagicMock(side_effect=_reload)
        return flow, entry

    @pytest.mark.asyncio
    async def test_first_call_shows_a_confirmation_form(self):
        """The repair asks; it does not act."""
        flow, entry = self._flow("abc123")
        await flow.async_step_confirm()
        assert flow.async_show_form.called
        assert not flow.hass.config_entries.async_update_entry.called

    @pytest.mark.asyncio
    async def test_confirming_clears_the_flag_and_reloads(self):
        from custom_components.heating_analytics.const import (
            CONF_EXPERIMENTAL_4D_PRIMARY,
        )

        flow, entry = self._flow("abc123")
        await flow.async_step_confirm(user_input={})

        assert flow.hass.config_entries.async_update_entry.called
        written = flow.hass.config_entries.async_update_entry.call_args.kwargs["data"]
        assert written[CONF_EXPERIMENTAL_4D_PRIMARY] is False
        # Other config must survive untouched.
        assert written["energy_sensors"] == ["sensor.a"]
        assert flow.hass.config_entries.async_reload.called
        assert flow.async_create_entry.called

    @pytest.mark.asyncio
    async def test_init_step_routes_to_confirm(self):
        flow, _ = self._flow("abc123")
        await flow.async_step_init()
        assert flow.async_show_form.called

    @pytest.mark.asyncio
    async def test_missing_entry_aborts_rather_than_reporting_success(self):
        """A silent no-op that reads as a fix is worse than a failure.

        HA's RepairsFlowManager deletes the issue on any non-ABORT
        result, so returning ``async_create_entry`` here would make the
        repair vanish while the flag stayed on.
        """
        flow, _ = self._flow("abc123", entry_exists=False)
        result = await flow.async_step_confirm(user_input={})
        assert result["type"] == "abort"
        assert not flow.async_create_entry.called
        assert not flow.hass.config_entries.async_update_entry.called

    @pytest.mark.asyncio
    async def test_empty_entry_id_aborts(self):
        """The latent path: issue data with no entry_id."""
        flow, _ = self._flow("")
        result = await flow.async_step_confirm(user_input={})
        assert result["type"] == "abort"
        assert not flow.async_create_entry.called

    @pytest.mark.asyncio
    async def test_create_fix_flow_extracts_the_entry_id(self):
        from custom_components.heating_analytics import repairs

        flow = await repairs.async_create_fix_flow(
            MagicMock(), "dni_dhi_outage_4d_active_xyz", {"entry_id": "xyz"}
        )
        assert isinstance(flow, repairs.DniDhiOutageRepairFlow)
        assert flow._entry_id == "xyz"

    @pytest.mark.asyncio
    async def test_create_fix_flow_tolerates_missing_data(self):
        """Must not raise — HA calls this from the repairs UI."""
        from custom_components.heating_analytics import repairs

        for data in (None, {}, {"entry_id": None}):
            flow = await repairs.async_create_fix_flow(
                MagicMock(), "dni_dhi_outage_4d_active", data
            )
            assert flow._entry_id == ""


def test_issue_id_is_scoped_per_config_entry():
    """Multi-instance installs must not share one issue.

    A shared issue's fix flow would not know which entry to demote.
    """
    from custom_components.heating_analytics import repairs

    assert repairs._issue_id("a") != repairs._issue_id("b")
    assert "a" in repairs._issue_id("a")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
