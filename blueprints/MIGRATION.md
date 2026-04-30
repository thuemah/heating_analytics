# Migration Guide

Two migration paths are documented here:

1. **v1 → v2**: behaviour changes in the climate_sync blueprint shipped with 1.3.5.
2. **Manual automation → blueprint**: original v1-era guide, still valid for the
   3-line conversion if you're starting from a hand-rolled automation.

---

## v1 → v2: climate_sync behaviour changes (1.3.5)

The v1 climate_sync blueprint mapped any climate state outside `heat`/`cool`
to the mode helper's `off` value.  v2 changes this in three ways:

| Climate State          | v1 (1.3.4 and earlier) | v2 (1.3.5+)        |
|------------------------|------------------------|--------------------|
| `auto`, `heat_cool`    | `off` (incorrect)      | `heating`          |
| `dry`, `fan_only`      | `off` (incorrect)      | *preserve*         |
| `off` (standard mode)  | `off`                  | *preserve*         |
| `off` (guest mode)     | `off`                  | `off` (unchanged)  |

*preserve* = the mode helper is left at its current value.

### Why the change?

**Standard `off` → preserve.**  v1 silently set `mode=off` whenever the
climate entity went to `off`, even when the off was transient (automation
pause, modulation, brief user toggle).  This canceled Heating Analytics
learning for the period and forced a cold-start re-run on the next
on-cycle.  v2 reserves `MODE_OFF` for units the user explicitly disables
— matching the convention already documented in
`heat_pump_mode_sync.yaml`'s comment ("MODE_OFF is reserved for units
that are deliberately disabled by the user").

**Guest `off` → off (unchanged).**  Guest units genuinely idle when the
guest leaves, so off is the correct mode.

**`auto` / `heat_cool` → heating.**  v1 lumped these into the off
default, mislabelling active heating as off.  These are active modes
where the climate device picks heat or cool based on room temperature
— for installations using this integration (predominantly heating-
focused), heating is the right routing.

**`dry` / `fan_only` → preserve.**  Small loads, neither heating nor
cooling.  v1 marked them off; v2 leaves the mode helper untouched and
lets per-unit base learning absorb whatever modest consumption appears.

### What you need to do

**Most users: nothing.**  The new behaviour is strictly more conservative —
it fixes incorrect routing without breaking the heat/cool happy path.
Pull the new blueprint, re-import in the automation editor (if HA caches
old versions), and that's it.

**Users who relied on `climate=off → mode=off` for standard units.**
If you used a Lovelace climate card to manually turn off your heat pump
and expected Heating Analytics to stop tracking, that side-effect is
gone.  Either:
- (Recommended) Set the select helper to `off` directly via the UI when
  you want to stop tracking — explicit user intent, matches the
  reserved-meaning of MODE_OFF.
- (If you have many entities) Add a separate automation that mirrors
  climate=off → mode=off for the specific entities where you want the
  v1 behaviour.

**Users who hit `auto` / `heat_cool` / `dry` / `fan_only` regularly.**
Verify the mode helper now reflects the correct regime.  Heating-
dominant installs should see no change beyond the obvious bug fix.

### Rollback

If v2 breaks something for your installation, the v1 blueprint is
preserved in git history.  Check out `blueprints/climate_sync.yaml`
at commit `caebf73` (or just before this change) and reload the
blueprint.  Please open an issue describing the case so we can
address it in a follow-up.

---

## Manual Automations → Blueprint

This guide helps you convert existing manual climate sync automations to use the blueprint.

## Before (Manual Automation - ~50 lines)

```yaml
- id: heating_analytics_state_sync_kjokken
  alias: "Heating Analytics - Sync Kjøkken State"
  description: >
    Synchronizes the operational state from climate.kjokken to the
    heating_analytics select helper. This is a one-way sync for
    data collection. Uses !!str type casting for explicit 'off' handling.
  mode: queued
  max_exceeded: silent
  trigger:
    - platform: state
      entity_id: climate.kjokken
      to: !!str heat
    - platform: state
      entity_id: climate.kjokken
      to: !!str cool
    - platform: state
      entity_id: climate.kjokken
      to: !!str off
  variables:
    hvac_mode: "{{ trigger.to_state.state }}"
    target_entity: >
      select.heating_analytics_vp_kjokken_energiforbruk_mode
  action:
    - choose:
        - conditions:
            - condition: template
              value_template: "{{ hvac_mode == 'heat' }}"
          sequence:
            - service: select.select_option
              target:
                entity_id: "{{ target_entity }}"
              data:
                option: "heating"
        - conditions:
            - condition: template
              value_template: "{{ hvac_mode == 'cool' }}"
          sequence:
            - service: select.select_option
              target:
                entity_id: "{{ target_entity }}"
              data:
                option: "cooling"
        - conditions:
            - condition: template
              value_template: "{{ hvac_mode == 'off' }}"
          sequence:
            - service: select.select_option
              target:
                entity_id: "{{ target_entity }}"
              data:
                option: "off"
```

## After (Blueprint - 3 lines)

```yaml
- use_blueprint:
    path: heating_analytics/climate_sync.yaml
    input:
      climate_entity: climate.kjokken
      mode_helper: select.heating_analytics_vp_kjokken_energiforbruk_mode
      use_guest_prefix: false
```

## Migration Steps

### 1. Install the Blueprint

Copy `climate_sync.yaml` to your Home Assistant config:
```bash
cp blueprints/climate_sync.yaml <ha_config>/blueprints/automation/heating_analytics/
```

### 2. Identify Your Manual Automations

Look for automations with these patterns:
- Trigger on climate entity state changes
- Map `heat` → `heating`, `cool` → `cooling`, `off` → `off`
- Update a `select.heating_analytics_*_mode` helper

### 3. Convert Each Automation

**For standard tracking (heating/cooling):**
```yaml
# OLD: 50+ lines manual automation
# NEW:
- use_blueprint:
    path: heating_analytics/climate_sync.yaml
    input:
      climate_entity: climate.YOUR_ENTITY
      mode_helper: select.heating_analytics_YOUR_HELPER_mode
      use_guest_prefix: false
```

**For guest mode tracking:**
```yaml
# OLD: Manual automation with guest_ prefix logic
# NEW:
- use_blueprint:
    path: heating_analytics/climate_sync.yaml
    input:
      climate_entity: climate.YOUR_ENTITY
      mode_helper: select.heating_analytics_YOUR_HELPER_mode
      use_guest_prefix: true  # ← Enable guest prefix
```

### 4. Delete Old Automations

Once you've verified the blueprint works:
1. Go to Settings → Automations & Scenes
2. Find the old manual automation
3. Delete it (or disable and monitor for a few days first)

## What If I Have Custom Logic?

**Example: Temperature-based guest mode activation**
```yaml
# Bad kjeller: Only activate if temp > 10°C
```

For cases like this, **keep the manual automation**. The blueprint is for standard cases only.

The blueprint covers ~90% of use cases. Complex logic (temperature checks, time-based rules, etc.)
should remain as manual automations.

## Verification

After migration:
1. Check Developer Tools → States
2. Change your climate entity state manually
3. Verify the mode helper updates correctly
4. Check automation traces in Developer Tools → Automations

## Rollback

If something doesn't work:
1. Re-enable the old manual automation
2. Disable the blueprint instance
3. Report the issue on GitHub

## Summary

| Aspect            | Manual Automation | Blueprint      |
|-------------------|-------------------|----------------|
| Lines per device  | ~50               | 3              |
| Localization fix  | Needs !!str       | Built-in       |
| Maintenance       | Per-device edits  | Central update |
| Custom logic      | Full flexibility  | Limited        |
| Best for          | Edge cases        | Standard cases |
