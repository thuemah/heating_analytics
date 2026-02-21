# Migration Guide: Manual Automations → Blueprint

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
