"""Home Assistant repair issues for Heating Analytics (#1070).

First and currently only repair: the 4D solar model is routed live while
its irradiance input has gone away.

Why this one earns a repair issue when other misconfigurations do not:
the failure is **silent and self-inflicted by the weather provider**, not
by the user.  An install can be correctly configured for 4D on Monday and
be running the worse model on Tuesday because the provider stopped
publishing ``direct_normal_irradiance`` / ``diffuse_radiation``.  The
input then falls through to the Kasten branch, where 4D is actively worse
than 3D (see CLAUDE.md > Solar Model > 4D shadow learner), and nothing
surfaces it unless the user happens to run ``diagnose_solar`` and read
``dni_dhi_source.verdict``.

**Demotion only.**  3D works everywhere, so switching back has no cliff.
The promotion direction is deliberately absent: an untrained 4D regime
returns the zero-vector, so wrongly routing *to* 4D silently removes
solar from predictions entirely.  That asymmetry is why routing stays a
human decision made in the reconfigure flow, where #1062 surfaces
readiness.

**The repair asks; it does not act.**  Consistent with the #1020
precedent of surfacing suggestions rather than auto-applying them, and
with #1062's decision to warn rather than refuse.

**It must auto-resolve.**  When the provider recovers, the issue is
deleted.  Without that it becomes an absorbing state with no exit —
exactly the problem ``ready_to_enable`` carries in #1062's summary
verdict, and not one to repeat here.  Hysteresis lives in the verdict
(see ``diagnostics._compute_dni_dhi_outage``): the raise and clear bars
differ, so an intermittent provider cannot flap the issue on alternating
days.
"""
from __future__ import annotations

import logging
from typing import Any

import voluptuous as vol
from homeassistant.components.repairs import RepairsFlow
from homeassistant.core import HomeAssistant
from homeassistant.helpers import issue_registry as ir

from .const import (
    CONF_EXPERIMENTAL_4D_PRIMARY,
    DOMAIN,
    REPAIR_ISSUE_DNI_DHI_OUTAGE,
)

_LOGGER = logging.getLogger(__name__)


def _issue_id(entry_id: str) -> str:
    """Per-entry issue ID.

    Scoped to the config entry so a multi-instance install raises (and
    fixes) one issue per instance rather than one shared issue whose fix
    flow would not know which entry to demote.
    """
    return f"{REPAIR_ISSUE_DNI_DHI_OUTAGE}_{entry_id}"


def async_check_dni_dhi_outage(hass: HomeAssistant, coordinator) -> None:
    """Create or delete the outage repair to match the current verdict.

    Called at the hour boundary.  Idempotent in both directions — HA's
    issue registry tolerates re-registering an existing issue and
    deleting a non-existent one — which matters because this verdict is
    recomputed from scratch every hour and re-asserted each time.

    The issue is deliberately NOT ``is_persistent``: it is a statement
    about a trailing window of recent hours, so it should be re-derived
    from evidence after a restart rather than restored from disk.  The
    cost is that a restart clears the card until the next hour boundary
    re-raises it; the benefit is that a stale warning cannot outlive the
    condition it describes.

    Never raises: a repair-registry failure must not take down hourly
    processing, which owns learning and persistence.
    """
    try:
        result = coordinator.evaluate_dni_dhi_outage()
        verdict = result.get("verdict")
        issue_id = _issue_id(coordinator.entry.entry_id)

        if verdict == "raise":
            ir.async_create_issue(
                hass,
                DOMAIN,
                issue_id,
                is_fixable=True,
                # Warning, not error: predictions still run.  They run on
                # the worse model, which is a degradation, not an outage.
                severity=ir.IssueSeverity.WARNING,
                translation_key="dni_dhi_outage",
                translation_placeholders={
                    "name": getattr(coordinator.entry, "title", None) or DOMAIN,
                    "hours": str(result.get("daylight_hours_examined", 0)),
                    "real_hours": str(result.get("real_source_hours", 0)),
                },
                data={"entry_id": coordinator.entry.entry_id},
            )
        elif verdict in ("clear", "not_applicable"):
            # ``hold`` and ``insufficient_data`` deliberately do nothing:
            # the first is the hysteresis band, where the existing state
            # must persist, and the second means we have no evidence
            # either way.  Only a definitive recovery — or 4D no longer
            # being routed at all — removes the issue.
            ir.async_delete_issue(hass, DOMAIN, issue_id)
    except Exception:  # noqa: BLE001 — never break the hour boundary
        _LOGGER.debug("DNI/DHI outage repair check failed", exc_info=True)


class DniDhiOutageRepairFlow(RepairsFlow):
    """Confirm-only fix flow: switch this entry back to the 3D model.

    Subclasses ``RepairsFlow`` rather than ``ConfirmRepairFlow``: the
    latter only dismisses the issue, and this flow has to write the
    config entry and reload it.  ``self.hass``, ``async_show_form`` and
    ``async_create_entry`` all come from the base class.
    """

    def __init__(self, entry_id: str) -> None:
        super().__init__()
        self._entry_id = entry_id

    async def async_step_init(self, user_input: dict[str, Any] | None = None):
        return await self.async_step_confirm()

    async def async_step_confirm(self, user_input: dict[str, Any] | None = None):
        if user_input is None:
            # ``vol.Schema({})``, not ``None``.  An empty schema renders a
            # confirm dialog with a submit button and no fields, which is
            # what HA core's own ``ConfirmRepairFlow`` passes; ``None``
            # means "no schema" and leaves the frontend with nothing to
            # submit.  Not caught by the suite — ``conftest.py`` stubs the
            # whole repairs module, so the form is never rendered.
            return self.async_show_form(
                step_id="confirm", data_schema=vol.Schema({})
            )

        # Abort rather than report success when there is nothing to
        # demote.  Both of these used to fall through to
        # ``async_create_entry``, and HA's RepairsFlowManager deletes the
        # issue on any non-ABORT result — so the user watched the repair
        # disappear while the flag stayed on.  A silent no-op that reads
        # as a successful fix is worse than a visible failure.
        if not self._entry_id:
            # No ``entry_id`` in the issue's stored data.  Should not
            # happen — ``async_check_dni_dhi_outage`` always writes it —
            # but a registry entry written by an older version, or a
            # hand-edited one, would land here.
            _LOGGER.warning(
                "DNI/DHI outage repair confirmed with no entry_id; "
                "cannot switch back to the 3D solar model"
            )
            return self.async_abort(reason="entry_not_found")

        entry = self.hass.config_entries.async_get_entry(self._entry_id)
        if entry is None:
            # Entry removed between the issue firing and the user acting.
            # Nothing to demote, and the issue is stale — drop it, but
            # abort so the result is not reported as a fix.
            ir.async_delete_issue(
                self.hass, DOMAIN, _issue_id(self._entry_id)
            )
            return self.async_abort(reason="entry_not_found")

        new_data = {**entry.data, CONF_EXPERIMENTAL_4D_PRIMARY: False}
        self.hass.config_entries.async_update_entry(entry, data=new_data)
        _LOGGER.info(
            "Repair applied: switched entry %s back to the 3D solar model "
            "after irradiance input became unavailable",
            self._entry_id,
        )
        await self.hass.config_entries.async_reload(self._entry_id)
        # HA's RepairsFlowManager deletes the issue itself on flow
        # completion, so no explicit ``async_delete_issue`` is needed on
        # this path — an earlier revision had one, justified by a
        # "resolved issue stays on screen for up to an hour" concern that
        # does not apply.  The next hour boundary would also clear it via
        # ``not_applicable``; both are belt-and-braces, neither is load-
        # bearing.
        return self.async_create_entry(title="", data={})


async def async_create_fix_flow(
    hass: HomeAssistant,
    issue_id: str,
    data: dict[str, str | int | float | None] | None,
):
    """HA repairs platform entry point.

    A missing ``entry_id`` yields a flow that aborts on confirm rather
    than one that silently succeeds; see ``async_step_confirm``.
    """
    entry_id = (data or {}).get("entry_id")
    return DniDhiOutageRepairFlow(str(entry_id) if entry_id else "")
