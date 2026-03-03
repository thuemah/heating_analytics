views:
  - title: Heating Analytics
    path: heating-analytics
    icon: mdi:radiator
    badges: []
    cards:
      - type: custom:mushroom-chips-card
        chips:
          - type: template
            icon: mdi:home
            content: Home
            icon_color: blue
            tap_action:
              action: navigate
              navigation_path: /lovelace/0
          - type: entity
            entity: sensor.heating_analytics_historical_consumption
            icon: mdi:history
            content_info: state
          - type: entity
            entity: sensor.outdoor_temperature
            icon: mdi:thermometer
            content_info: state
          - type: entity
            entity: sensor.heating_analytics_deviation
            icon: mdi:percent
            icon_color: >
              {% set val = states('sensor.heating_analytics_deviation') |
              float(0) %} {% if val > 20 %}
                red
              {% elif val > 10 %}
                orange
              {% elif val < -10 %}
                green
              {% else %}
                blue
              {% endif %}
            content_info: state
          - type: conditional
            conditions:
              - entity: >-
                  binary_sensor.heating_analytics_auxiliary_untracked_heating_active
                state: 'on'
            chip:
              type: entity
              entity: >-
                binary_sensor.heating_analytics_auxiliary_untracked_heating_active
              icon: mdi:fireplace
              icon_color: orange
              content_info: name
          - type: entity
            entity: sensor.wind_speed
            icon: mdi:weather-windy
            content_info: state
          - type: entity
            entity: sensor.heating_analytics_effective_wind
            icon: mdi:windsock
            content_info: state
            icon_color: >
              {% set eff = states('sensor.heating_analytics_effective_wind') |
              float(0) %} {% set threshold =
              states('number.heating_analytics_wind_threshold') | float(4.2) %}
              {% set extreme =
              states('number.heating_analytics_extreme_wind_threshold') |
              float(6.4) %} {% if eff > extreme %}red{% elif eff > threshold
              %}orange{% else %}blue{% endif %}
          - type: entity
            entity: binary_sensor.heating_analytics_efficiency_good
            icon_color: green
      - type: vertical-stack
        cards:
          - type: custom:mushroom-template-card
            primary: 'Heating: 🎯 Forecast | ✅ Actual | 📊 Deviation'
            secondary: >
              {% set actual_today =
              states('sensor.heating_analytics_historical_consumption') |
              float(0) %} {% set temp = states('sensor.outdoor_temperature') |
              float(0) %} {% set forecast =
              states('sensor.heating_analytics_forecast_today') | float(0) %} {%
              set predicted =
              states('sensor.heating_analytics_predicted_consumption') |
              float(0) %} {% set deviation =
              states('sensor.heating_analytics_deviation') | float(0) %}

              {% set aux_active =
              is_state('binary_sensor.heating_analytics_auxiliary_untracked_heating_active',
              'on') %} {% set current_wind =
              states('sensor.heating_analytics_effective_wind') | float(0) %} {%
              set wind_threshold =
              states('number.heating_analytics_wind_threshold') | float(4.2) %}
              {% set extreme_threshold =
              states('number.heating_analytics_extreme_wind_threshold') |
              float(6.4) %}

              {% if aux_active %}
                {% set category = '🔥' %}
              {% elif current_wind > extreme_threshold %}
                {% set category = '🌪️' %}
              {% elif current_wind > wind_threshold %}
                {% set category = '💨' %}
              {% else %}
                {% set category = '' %}
              {% endif %}

              {% if forecast > 0 %}
                {% if deviation > 15 %}⚠️{% elif deviation > 8 %}📈{% elif deviation < -8 %}📉{% else %}✅{% endif %} 🎯 ~{{ forecast | round(1) }} kWh | ✅ {{ actual_today | round(1) }} kWh | 📊 {{ deviation | round(1) }}% | 🌡️ {{ temp | round(1) }}°C {{ category }}
              {% else %}
                🔄 Calculating...
              {% endif %}
            icon: >
              {% set deviation = states('sensor.heating_analytics_deviation') |
              float(0) %} {% if deviation > 20 %}
                mdi:alert-circle
              {% elif deviation > 10 %}
                mdi:arrow-up-circle
              {% elif deviation < -10 %}
                mdi:arrow-down-circle
              {% else %}
                mdi:circle-outline
              {% endif %}
            tap_action:
              action: more-info
            color: >
              {% set deviation = states('sensor.heating_analytics_deviation') |
              float(0) %} {% if deviation > 20 %}
                red
              {% elif deviation > 10 %}
                orange
              {% elif deviation < -10 %}
                green
              {% else %}
                blue
              {% endif %}
            features_position: bottom
          - type: custom:mushroom-template-card
            primary: Deviation from Expected
            secondary: >
              {% set deviation = states('sensor.heating_analytics_deviation') |
              float(0) %} {% set forecast =
              states('sensor.heating_analytics_forecast_today') | float(0) %} {%
              set predicted =
              states('sensor.heating_analytics_predicted_consumption') |
              float(0) %} {% if deviation > 0 %}
                +{{ deviation }}% | 🎯 ~{{ forecast | round(1) }} kWh (Expected {{ predicted | round(1) }} kWh)
              {% elif deviation < 0 %}
                {{ deviation }}% | 🎯 ~{{ forecast | round(1) }} kWh (Expected {{ predicted | round(1) }} kWh)
              {% else %}
                On Track | 🎯 ~{{ forecast | round(1) }} kWh
              {% endif %}
            icon: >
              {% set deviation = states('sensor.heating_analytics_deviation') |
              float(0) %} {% if deviation > 20 %}
                mdi:alert-circle
              {% elif deviation > 10 %}
                mdi:alert
              {% elif deviation < -10 %}
                mdi:check-circle
              {% else %}
                mdi:minus-circle
              {% endif %}
            tap_action:
              action: more-info
            color: >
              {% set deviation = states('sensor.heating_analytics_deviation') |
              float(0) %} {% if deviation > 20 %}
                red
              {% elif deviation > 10 %}
                orange
              {% elif deviation < -10 %}
                green
              {% else %}
                blue
              {% endif %}
            features_position: bottom
      - type: grid
        columns: 2
        square: false
        cards:
          - type: custom:mushroom-entity-card
            entity: sensor.heating_analytics_forecast_today
            primary_info: name
            secondary_info: state
            icon: mdi:crystal-ball
          - type: custom:mushroom-entity-card
            entity: sensor.heating_analytics_energy_consumption_today
            primary_info: name
            secondary_info: state
            icon: mdi:counter
          - type: custom:mushroom-entity-card
            entity: sensor.heating_analytics_expected_consumption_so_far_today
            primary_info: name
            secondary_info: state
            icon: mdi:chart-bell-curve
          - type: custom:mushroom-entity-card
            entity: sensor.heating_analytics_potential_savings
            primary_info: name
            secondary_info: state
            icon: mdi:piggy-bank
          - type: custom:mushroom-entity-card
            entity: sensor.heating_analytics_efficiency
            primary_info: name
            secondary_info: state
            icon: mdi:home-thermometer
          - type: custom:mushroom-entity-card
            entity: sensor.heating_analytics_thermal_degree_days
            primary_info: name
            secondary_info: state
            icon: mdi:temperature-celsius
      - type: entities
        title: Environment & Weather
        show_header_toggle: false
        entities:
          - entity: sensor.outdoor_temperature
          - entity: sensor.wind_speed
          - entity: sensor.wind_gust
          - entity: sensor.heating_analytics_effective_wind
      - type: entities
        title: Temperature Analysis (Actual vs Last Year)
        show_header_toggle: false
        entities:
          - entity: sensor.heating_analytics_temp_actual_today
          - entity: sensor.heating_analytics_temp_forecast_today
          - entity: sensor.heating_analytics_temp_last_year_day
          - type: divider
          - entity: sensor.heating_analytics_temp_actual_week
          - entity: sensor.heating_analytics_temp_last_year_week
          - type: divider
          - entity: sensor.heating_analytics_temp_actual_month
          - entity: sensor.heating_analytics_temp_last_year_month
      - type: entities
        title: Last Hour Stats
        show_header_toggle: false
        entities:
          - entity: sensor.heating_analytics_last_hour_actual
          - entity: sensor.heating_analytics_last_hour_expected
          - entity: sensor.heating_analytics_last_hour_deviation
      - type: vertical-stack
        cards:
          - type: conditional
            conditions:
              - entity: sensor.heating_analytics_sensor_heating_unit_1_energy_daily
                state_not: "unavailable"
              - entity: sensor.heating_analytics_sensor_heating_unit_1_energy_daily
                state_not: "unknown"
            card:
              type: custom:mushroom-entity-card
              entity: sensor.heating_analytics_sensor_heating_unit_1_energy_daily
          - type: conditional
            conditions:
              - entity: sensor.heating_analytics_sensor_heating_unit_2_energy_daily
                state_not: "unavailable"
              - entity: sensor.heating_analytics_sensor_heating_unit_2_energy_daily
                state_not: "unknown"
            card:
              type: custom:mushroom-entity-card
              entity: sensor.heating_analytics_sensor_heating_unit_2_energy_daily
          - type: conditional
            conditions:
              - entity: sensor.heating_analytics_sensor_heating_unit_3_energy_daily
                state_not: "unavailable"
              - entity: sensor.heating_analytics_sensor_heating_unit_3_energy_daily
                state_not: "unknown"
            card:
              type: custom:mushroom-entity-card
              entity: sensor.heating_analytics_sensor_heating_unit_3_energy_daily
          - type: conditional
            conditions:
              - entity: sensor.heating_analytics_sensor_heating_unit_4_energy_daily
                state_not: "unavailable"
              - entity: sensor.heating_analytics_sensor_heating_unit_4_energy_daily
                state_not: "unknown"
            card:
              type: custom:mushroom-entity-card
              entity: sensor.heating_analytics_sensor_heating_unit_4_energy_daily
          - type: conditional
            conditions:
              - entity: sensor.heating_analytics_sensor_heating_unit_5_energy_daily
                state_not: "unavailable"
              - entity: sensor.heating_analytics_sensor_heating_unit_5_energy_daily
                state_not: "unknown"
            card:
              type: custom:mushroom-entity-card
              entity: sensor.heating_analytics_sensor_heating_unit_5_energy_daily
          - type: conditional
            conditions:
              - entity: sensor.heating_analytics_sensor_heating_unit_6_energy_daily
                state_not: "unavailable"
              - entity: sensor.heating_analytics_sensor_heating_unit_6_energy_daily
                state_not: "unknown"
            card:
              type: custom:mushroom-entity-card
              entity: sensor.heating_analytics_sensor_heating_unit_6_energy_daily
          - type: conditional
            conditions:
              - entity: sensor.heating_analytics_sensor_heating_unit_7_energy_daily
                state_not: "unavailable"
              - entity: sensor.heating_analytics_sensor_heating_unit_7_energy_daily
                state_not: "unknown"
            card:
              type: custom:mushroom-entity-card
              entity: sensor.heating_analytics_sensor_heating_unit_7_energy_daily
          - type: conditional
            conditions:
              - entity: sensor.heating_analytics_sensor_heating_unit_8_energy_daily
                state_not: "unavailable"
              - entity: sensor.heating_analytics_sensor_heating_unit_8_energy_daily
                state_not: "unknown"
            card:
              type: custom:mushroom-entity-card
              entity: sensor.heating_analytics_sensor_heating_unit_8_energy_daily
          - type: conditional
            conditions:
              - entity: sensor.heating_analytics_sensor_heating_unit_9_energy_daily
                state_not: "unavailable"
              - entity: sensor.heating_analytics_sensor_heating_unit_9_energy_daily
                state_not: "unknown"
            card:
              type: custom:mushroom-entity-card
              entity: sensor.heating_analytics_sensor_heating_unit_9_energy_daily
          - type: conditional
            conditions:
              - entity: sensor.heating_analytics_sensor_heating_unit_10_energy_daily
                state_not: "unavailable"
              - entity: sensor.heating_analytics_sensor_heating_unit_10_energy_daily
                state_not: "unknown"
            card:
              type: custom:mushroom-entity-card
              entity: sensor.heating_analytics_sensor_heating_unit_10_energy_daily
          - type: conditional
            conditions:
              - entity: sensor.heating_analytics_sensor_heating_unit_11_energy_daily
                state_not: "unavailable"
              - entity: sensor.heating_analytics_sensor_heating_unit_11_energy_daily
                state_not: "unknown"
            card:
              type: custom:mushroom-entity-card
              entity: sensor.heating_analytics_sensor_heating_unit_11_energy_daily
          - type: conditional
            conditions:
              - entity: sensor.heating_analytics_sensor_heating_unit_12_energy_daily
                state_not: "unavailable"
              - entity: sensor.heating_analytics_sensor_heating_unit_12_energy_daily
                state_not: "unknown"
            card:
              type: custom:mushroom-entity-card
              entity: sensor.heating_analytics_sensor_heating_unit_12_energy_daily
      - type: entities
        title: Configuration
        show_header_toggle: false
        entities:
          - entity: switch.heating_analytics_learning_enabled
          - entity: number.heating_analytics_learning_rate
          - entity: number.heating_analytics_wind_threshold
          - entity: number.heating_analytics_extreme_wind_threshold
          - entity: number.heating_analytics_wind_gust_factor
          - entity: switch.heating_analytics_auxiliary_untracked_heating_active
      - type: markdown
        content: >
          **Learning:** Automatically updates based on historical data.

          **Learning Rate:** Weight of new vs. old data (5% default).

          **Wind Threshold:** Hours with wind above threshold → high_wind/extreme_wind
          category (m/s).

          **Wind Gust Factor:** Weighting of wind gusts in effective wind (0.6
          default).

          **Effective Wind:** MAX(avg, gust × factor) - used for
          categorization.

          **Auxiliary Heating:** Detects extra heating not measured by the system.

          **Categories:** normal, high_wind, extreme_wind, with_auxiliary
      - type: custom:plotly-graph
        refresh_interval: 300
        raw_plotly_config: true
        config:
          displayModeBar: false
        layout:
          showlegend: true
          legend:
            orientation: h
            'y': 1.1
          xaxis:
            title: Temperature (°C)
            range:
              - -20
              - 28
            dtick: 5
            gridcolor: rgba(128, 128, 128, 0.2)
            zeroline: true
            zerolinecolor: rgba(255, 255, 255, 0.3)
          yaxis:
            title: Heat Demand (kWh/day)
            gridcolor: rgba(128, 128, 128, 0.2)
            zeroline: true
            zerolinecolor: rgba(255, 255, 255, 0.3)
          paper_bgcolor: rgba(0,0,0,0)
          plot_bgcolor: rgba(0,0,0,0)
          font:
            color: var(--primary-text-color)
          margin:
            l: 60
            r: 20
            t: 40
            b: 60
          hovermode: x unified
        entities:
          - entity: sensor.heating_analytics_correlation_data
            name: Normal
            mode: lines
            line:
              color: '#2196F3'
              width: 3
              shape: spline
            x: |
              $fn ({ hass }) => {
                const data = hass.states['sensor.heating_analytics_correlation_data'].attributes;
                const SKIP_KEYS = ['normal_x', 'normal_y', 'high_wind_x', 'high_wind_y', 'extreme_wind_x', 'extreme_wind_y', 'auxiliary_heating_x', 'auxiliary_heating_y', 'friendly_name', 'icon'];
                let temps = [];
                Object.keys(data).forEach(key => {
                  if (SKIP_KEYS.includes(key)) return;
                  const temp = parseFloat(key);
                  if (!isNaN(temp) && data[key]?.normal !== undefined) {
                    temps.push(temp);
                  }
                });
                return temps.sort((a, b) => a - b);
              }
            'y': |
              $fn ({ hass }) => {
                const data = hass.states['sensor.heating_analytics_correlation_data'].attributes;
                const SKIP_KEYS = ['normal_x', 'normal_y', 'high_wind_x', 'high_wind_y', 'extreme_wind_x', 'extreme_wind_y', 'auxiliary_heating_x', 'auxiliary_heating_y', 'friendly_name', 'icon'];
                let values = [];
                Object.keys(data).forEach(key => {
                  if (SKIP_KEYS.includes(key)) return;
                  const temp = parseFloat(key);
                  if (!isNaN(temp) && data[key]?.normal !== undefined) {
                    values.push({ temp: temp, kwh: parseFloat(data[key].normal) * 24 });
                  }
                });
                return values.sort((a, b) => a.temp - b.temp).map(v => v.kwh);
              }
          - entity: sensor.heating_analytics_correlation_data
            name: High Wind
            mode: lines
            line:
              color: '#FF9800'
              width: 3
              shape: spline
              dash: dot
            x: |
              $fn ({ hass }) => {
                const data = hass.states['sensor.heating_analytics_correlation_data'].attributes;
                const SKIP_KEYS = ['normal_x', 'normal_y', 'high_wind_x', 'high_wind_y', 'extreme_wind_x', 'extreme_wind_y', 'auxiliary_heating_x', 'auxiliary_heating_y', 'friendly_name', 'icon'];
                let temps = [];
                Object.keys(data).forEach(key => {
                  if (SKIP_KEYS.includes(key)) return;
                  const temp = parseFloat(key);
                  if (!isNaN(temp) && data[key]?.high_wind !== undefined) {
                    temps.push(temp);
                  }
                });
                return temps.sort((a, b) => a - b);
              }
            'y': |
              $fn ({ hass }) => {
                const data = hass.states['sensor.heating_analytics_correlation_data'].attributes;
                const SKIP_KEYS = ['normal_x', 'normal_y', 'high_wind_x', 'high_wind_y', 'extreme_wind_x', 'extreme_wind_y', 'auxiliary_heating_x', 'auxiliary_heating_y', 'friendly_name', 'icon'];
                let values = [];
                Object.keys(data).forEach(key => {
                  if (SKIP_KEYS.includes(key)) return;
                  const temp = parseFloat(key);
                  if (!isNaN(temp) && data[key]?.high_wind !== undefined) {
                    values.push({ temp: temp, kwh: parseFloat(data[key].high_wind) * 24 });
                  }
                });
                return values.sort((a, b) => a.temp - b.temp).map(v => v.kwh);
              }
          - entity: sensor.heating_analytics_correlation_data
            name: Extreme Wind
            mode: lines
            line:
              color: '#F44336'
              width: 3
              shape: spline
              dash: dash
            x: |
              $fn ({ hass }) => {
                const data = hass.states['sensor.heating_analytics_correlation_data'].attributes;
                const SKIP_KEYS = ['normal_x', 'normal_y', 'high_wind_x', 'high_wind_y', 'extreme_wind_x', 'extreme_wind_y', 'auxiliary_heating_x', 'auxiliary_heating_y', 'friendly_name', 'icon'];
                let temps = [];
                Object.keys(data).forEach(key => {
                  if (SKIP_KEYS.includes(key)) return;
                  const temp = parseFloat(key);
                  if (!isNaN(temp) && data[key]?.extreme_wind !== undefined && data[key]?.extreme_wind !== null) {
                    temps.push(temp);
                  }
                });
                return temps.sort((a, b) => a - b);
              }
            'y': |
              $fn ({ hass }) => {
                const data = hass.states['sensor.heating_analytics_correlation_data'].attributes;
                const SKIP_KEYS = ['normal_x', 'normal_y', 'high_wind_x', 'high_wind_y', 'extreme_wind_x', 'extreme_wind_y', 'auxiliary_heating_x', 'auxiliary_heating_y', 'friendly_name', 'icon'];
                let values = [];
                Object.keys(data).forEach(key => {
                  if (SKIP_KEYS.includes(key)) return;
                  const temp = parseFloat(key);
                  if (!isNaN(temp) && data[key]?.extreme_wind !== undefined && data[key]?.extreme_wind !== null) {
                    values.push({ temp: temp, kwh: parseFloat(data[key].extreme_wind) * 24 });
                  }
                });
                return values.sort((a, b) => a.temp - b.temp).map(v => v.kwh);
              }
          - entity: sensor.heating_analytics_correlation_data
            name: With Auxiliary Heating
            mode: lines
            line:
              color: '#4CAF50'
              width: 3
              shape: spline
              dash: longdash
            x: |
              $fn ({ hass }) => {
                const data = hass.states['sensor.heating_analytics_correlation_data'].attributes;
                const SKIP_KEYS = ['normal_x', 'normal_y', 'high_wind_x', 'high_wind_y', 'extreme_wind_x', 'extreme_wind_y', 'auxiliary_heating_x', 'auxiliary_heating_y', 'friendly_name', 'icon'];
                let temps = [];
                Object.keys(data).forEach(key => {
                  if (SKIP_KEYS.includes(key)) return;
                  const temp = parseFloat(key);
                  if (!isNaN(temp) && data[key]?.with_auxiliary_heating !== undefined) {
                    temps.push(temp);
                  }
                });
                return temps.sort((a, b) => a - b);
              }
            'y': |
              $fn ({ hass }) => {
                const data = hass.states['sensor.heating_analytics_correlation_data'].attributes;
                const SKIP_KEYS = ['normal_x', 'normal_y', 'high_wind_x', 'high_wind_y', 'extreme_wind_x', 'extreme_wind_y', 'auxiliary_heating_x', 'auxiliary_heating_y', 'friendly_name', 'icon'];
                let values = [];
                Object.keys(data).forEach(key => {
                  if (SKIP_KEYS.includes(key)) return;
                  const temp = parseFloat(key);
                  if (!isNaN(temp) && data[key]?.with_auxiliary_heating !== undefined) {
                    values.push({ temp: temp, kwh: parseFloat(data[key].with_auxiliary_heating) * 24 });
                  }
                });
                return values.sort((a, b) => a.temp - b.temp).map(v => v.kwh);
              }
      - type: custom:plotly-graph
        title: 📈 Hourly Consumption - Actual vs Expected (Trend)
        hours_to_show: 24
        refresh_interval: 300
        entities:
          - entity: sensor.heating_analytics_last_hour_actual
            name: Actual
            mode: lines
            line:
              color: '#F44336'
              width: 3
              shape: spline
            fill: tozeroy
            fillcolor: rgba(244, 67, 54, 0.1)
          - entity: sensor.heating_analytics_last_hour_expected
            name: Expected
            mode: lines
            connectgaps: true
            line:
              color: '#2196F3'
              width: 3
              shape: spline
              dash: dash
        layout:
          height: 300
          showlegend: true
          legend:
            orientation: h
            'y': 1.15
          xaxis:
            title: Time
            showgrid: true
            gridcolor: rgba(128, 128, 128, 0.2)
            rangeselector:
              buttons:
                - count: 1
                  label: 1d
                  step: day
                  stepmode: backward
                - count: 2
                  label: 2d
                  step: day
                  stepmode: backward
                - count: 1
                  label: 1w
                  step: week
                  stepmode: backward
                - count: 2
                  label: 2w
                  step: week
                  stepmode: backward
                - count: 4
                  label: 4w
                  step: week
                  stepmode: backward
                - step: all
                  label: All
              x: 0
              'y': 1.05
            rangeslider:
              visible: false
          yaxis:
            title: Consumption (kWh)
            showgrid: true
            gridcolor: rgba(128, 128, 128, 0.2)
          hovermode: x unified
          plot_bgcolor: rgba(0, 0, 0, 0)
          paper_bgcolor: rgba(0, 0, 0, 0)
