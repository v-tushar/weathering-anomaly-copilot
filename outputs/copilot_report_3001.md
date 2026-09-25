# Copilot report: simulated run 3001 (scripted)
_Synthetic data; fictional X-100 manual. Advisory output only._

Injected faults (ground truth): seal_leak @ 2026-08-04 03:54:00, control_fault @ 2026-08-12 05:54:00, sensor_spike @ 2026-08-14 08:36:00, lamp_degradation @ 2026-08-17 07:54:00

## Alert: `rh_pct` at 2026-08-04 06:18:00
**Status:** ok  |  steps: 3  |  tools: get_alert_context, search_manual  |  0.0s  |  tokens: 0
**Likely cause:** humidity_seal_leak (medium confidence)

Alert on rh_pct. Rule-based classification: humidity_seal_leak.

**Evidence**
- rh_pct peak |z| during alert was 19.93
- rh_pct median during alert 71.43 vs baseline 50.44

**Recommended actions**
- Follow manual section HUM-01.

**Test validity:** Check whether exposure conditions left tolerance (GEN-03).
**Sources:** HUM-01, LAMP-03

## Alert: `chamber_air_c` at 2026-08-12 06:06:00
**Status:** ok  |  steps: 3  |  tools: get_alert_context, search_manual  |  0.0s  |  tokens: 0
**Likely cause:** temperature_control_fault (medium confidence)

Alert on chamber_air_c. Rule-based classification: temperature_control_fault.

**Evidence**
- chamber_air_c peak |z| during alert was 14.69
- chamber_air_c median during alert 49.35 vs baseline 46.97

**Recommended actions**
- Follow manual section TEMP-01.

**Test validity:** Check whether exposure conditions left tolerance (GEN-03).
**Sources:** TEMP-01, GEN-01

## Alert: `black_panel_c` at 2026-08-12 06:12:00
**Status:** ok  |  steps: 3  |  tools: get_alert_context, search_manual  |  0.0s  |  tokens: 0
**Likely cause:** temperature_control_fault (medium confidence)

Alert on black_panel_c. Rule-based classification: temperature_control_fault.

**Evidence**
- black_panel_c peak |z| during alert was 14.61
- black_panel_c median during alert 71.7 vs baseline 69.91

**Recommended actions**
- Follow manual section TEMP-01.

**Test validity:** Check whether exposure conditions left tolerance (GEN-03).
**Sources:** TEMP-01, GEN-01

## Alert: `irradiance` at 2026-08-14 08:36:00
**Status:** ok  |  steps: 3  |  tools: get_alert_context, search_manual  |  0.0s  |  tokens: 0
**Likely cause:** irradiance_sensor_fault (medium confidence)

Alert on irradiance. Rule-based classification: irradiance_sensor_fault.

**Evidence**
- irradiance peak |z| during alert was 139.89
- irradiance median during alert 0.55 vs baseline 0.549

**Recommended actions**
- Follow manual section IRR-01.

**Test validity:** Check whether exposure conditions left tolerance (GEN-03).
**Sources:** IRR-01, GEN-01

## Alert: `lamp_power_pct` at 2026-08-17 10:36:00
**Status:** ok  |  steps: 3  |  tools: get_alert_context, search_manual  |  0.0s  |  tokens: 0
**Likely cause:** lamp_degradation (medium confidence)

Alert on lamp_power_pct. Rule-based classification: lamp_degradation.

**Evidence**
- lamp_power_pct peak |z| during alert was 150.59
- lamp_power_pct median during alert 95.3 vs baseline 63.29

**Recommended actions**
- Follow manual section LAMP-01.

**Test validity:** Check whether exposure conditions left tolerance (GEN-03).
**Sources:** LAMP-01, FN-01

## Alert: `irradiance` at 2026-08-25 11:36:00
**Status:** ok  |  steps: 3  |  tools: get_alert_context, search_manual  |  0.0s  |  tokens: 0
**Likely cause:** lamp_degradation (medium confidence)

Alert on irradiance. Rule-based classification: lamp_degradation.

**Evidence**
- irradiance peak |z| during alert was 30.28
- irradiance median during alert 0.482 vs baseline 0.549

**Recommended actions**
- Follow manual section LAMP-01.

**Test validity:** Check whether exposure conditions left tolerance (GEN-03).
**Sources:** LAMP-01, FN-01
