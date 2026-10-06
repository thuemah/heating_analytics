# Heating Analytics Blueprints

This directory contains Home Assistant automation blueprints for common Heating Analytics tasks.

## Installation

### One-click import (recommended)

Each button opens the blueprint import dialog in your own Home Assistant
instance with the blueprint's URL filled in — no file access needed.

| Blueprint | Import |
|-----------|--------|
| Climate State Sync (`climate_sync.yaml`) | [![Import the climate_sync blueprint into your Home Assistant instance.](https://my.home-assistant.io/badges/blueprint_import.svg)](https://my.home-assistant.io/redirect/blueprint_import/?blueprint_url=https%3A%2F%2Fgithub.com%2Fthuemah%2Fheating_analytics%2Fblob%2Fmain%2Fblueprints%2Fclimate_sync.yaml) |
| Heat Pump Mode Sync (`heat_pump_mode_sync.yaml`) | [![Import the heat_pump_mode_sync blueprint into your Home Assistant instance.](https://my.home-assistant.io/badges/blueprint_import.svg)](https://my.home-assistant.io/redirect/blueprint_import/?blueprint_url=https%3A%2F%2Fgithub.com%2Fthuemah%2Fheating_analytics%2Fblob%2Fmain%2Fblueprints%2Fheat_pump_mode_sync.yaml) |

An imported blueprint remembers where it came from, so Home Assistant can
re-import it when a new version is published (⋮ menu on the blueprint →
**Re-import blueprint**).

The same works by hand: **Settings → Automations & Scenes → Blueprints →
Import Blueprint**, then paste the blueprint's GitHub URL.

### Manual copy

If you prefer, copy the files into your Home Assistant configuration directory:

```bash
# From your Home Assistant config directory:
mkdir -p blueprints/automation/heating_analytics
cp <path-to-repo>/blueprints/*.yaml blueprints/automation/heating_analytics/
```

Or one at a time:
1. Copy `climate_sync.yaml` (or `heat_pump_mode_sync.yaml`) to `<config>/blueprints/automation/heating_analytics/`
2. Restart Home Assistant or reload automations
3. The blueprint will appear in the automation editor under "Blueprints"

## Available Blueprints

### Climate State Sync

**File:** `climate_sync.yaml`

Synchronizes climate entity states (heat/cool/off) to Heating Analytics mode select helpers.

**Use cases:**
- Automatic tracking of heat pumps, air conditioners, or thermostats
- Guest mode tracking (prefix states with `guest_` for separate analytics)
- Simple 3-line configuration per device

**Parameters:**
- `climate_entity` - The climate entity to monitor
- `mode_helper` - The Heating Analytics select helper to update
- `use_guest_prefix` - Enable guest mode tracking (default: false)

**Example usage:**

```yaml
# In your automations.yaml

# Standard tracking (main heating system)
- use_blueprint:
    path: heating_analytics/climate_sync.yaml
    input:
      climate_entity: climate.kjokken
      mode_helper: select.heating_analytics_vp_kjokken_energiforbruk_mode
      use_guest_prefix: false

# Guest mode tracking (secondary/occasional heating)
- use_blueprint:
    path: heating_analytics/climate_sync.yaml
    input:
      climate_entity: climate.guest_room
      mode_helper: select.heating_analytics_guest_room_mode
      use_guest_prefix: true
```

**State mapping (v2):**

| Climate State          | Standard Mode | Guest Mode        |
|------------------------|---------------|-------------------|
| `heat`                 | `heating`     | `guest_heating`   |
| `auto`, `heat_cool`    | `heating`     | `guest_heating`   |
| `cool`                 | `cooling`     | `guest_cooling`   |
| `off`                  | *preserve*    | `off`             |
| `dry`, `fan_only`      | *preserve*    | *preserve*        |
| `unavailable`, `unknown` | *no trigger* | *no trigger*    |

*preserve* = the mode helper is left at its current value.  Standard
units treat off as transient (modulation, automation pause) and don't
auto-disable tracking; users who want explicit off must set the select
manually via the UI.  See `MIGRATION.md` for the rationale and upgrade
notes if you're coming from v1.

**Features:**
- ✅ Handles Norwegian localization issues automatically (no templates that fail on localized states)
- ✅ Queued mode prevents race conditions
- ✅ Asymmetric off-handling: guest off → off, standard off → preserve
- ✅ Active states like `auto` and `heat_cool` correctly route to heating
- ✅ Inert states (`dry`, `fan_only`) leave the mode helper alone

### Heat Pump Mode Sync

**File:** `heat_pump_mode_sync.yaml`

Maps a heat pump's operation-mode sensor (e.g. `Heating` / `Domestic Hot
Water` / `Defrost`) to a Heating Analytics mode select helper.  Defrost is
transparent: the previous heating or DHW mode is kept, so defrost energy is
attributed to the mode it belongs to.  The blueprint's own description lists
its inputs.

## Custom Logic

For advanced use cases (e.g., temperature-based state selection, custom conditions),
you may still need to create manual automations. The blueprint covers ~90% of standard cases.

## Troubleshooting

**Blueprint not appearing in UI:**
- After a one-click import, check **Settings → Automations & Scenes → Blueprints**
- After a manual copy, ensure the file is in `blueprints/automation/heating_analytics/`
- Restart Home Assistant or reload automations from Developer Tools

**States not syncing:**
- Verify the climate entity and mode helper entity IDs are correct
- Check automation trace in Developer Tools → Automations
- Ensure the mode helper has the correct options configured

**Need help?**
Open an issue at: https://github.com/thuemah/heating_analytics/issues
