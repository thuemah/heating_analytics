# Heating Analytics Blueprints

This directory contains Home Assistant automation blueprints for common Heating Analytics tasks.

## Installation

Blueprints must be copied to your Home Assistant configuration directory:

```bash
# From your Home Assistant config directory:
mkdir -p blueprints/automation/heating_analytics
cp <path-to-repo>/blueprints/*.yaml blueprints/automation/heating_analytics/
```

Or manually:
1. Copy `climate_sync.yaml` to `<config>/blueprints/automation/heating_analytics/`
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

**State mapping:**

| Climate State | Standard Mode | Guest Mode        |
|---------------|---------------|-------------------|
| `heat`        | `heating`     | `guest_heating`   |
| `cool`        | `cooling`     | `guest_cooling`   |
| `off`         | `off`         | `off`             |
| `unavailable` | `off`         | `off`             |
| `unknown`     | `off`         | `off`             |

**Features:**
- ✅ Handles Norwegian localization issues automatically (default fallback)
- ✅ Queued mode prevents race conditions
- ✅ Gracefully handles unavailable/unknown states
- ✅ No template conditions that can fail on localized states

## Custom Logic

For advanced use cases (e.g., temperature-based state selection, custom conditions),
you may still need to create manual automations. The blueprint covers ~90% of standard cases.

## Troubleshooting

**Blueprint not appearing in UI:**
- Ensure the file is in `blueprints/automation/heating_analytics/`
- Restart Home Assistant or reload automations from Developer Tools

**States not syncing:**
- Verify the climate entity and mode helper entity IDs are correct
- Check automation trace in Developer Tools → Automations
- Ensure the mode helper has the correct options configured

**Need help?**
Open an issue at: https://github.com/thuemah/heating_analytics/issues
