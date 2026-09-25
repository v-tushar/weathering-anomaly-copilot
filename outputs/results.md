## Detection (per fault type)

| detector         | fault            |   detection_rate |   median_ttd_h |   p90_ttd_h |   detection_rate_noisy_sensors |
|:-----------------|:-----------------|-----------------:|---------------:|------------:|-------------------------------:|
| static_threshold | control_fault    |             1    |           0.3  |        1.2  |                           1    |
| static_threshold | lamp_degradation |             1    |         221.2  |      259.32 |                           1    |
| static_threshold | seal_leak        |             1    |           3.95 |        5.13 |                           1    |
| static_threshold | sensor_spike     |             0.74 |           0.1  |        0.3  |                           0.74 |
| residual_z       | control_fault    |             1    |           0.2  |        0.2  |                           1    |
| residual_z       | lamp_degradation |             1    |           3.2  |        3.81 |                           1    |
| residual_z       | seal_leak        |             1    |           2.4  |        3.01 |                           1    |
| residual_z       | sensor_spike     |             1    |           0.1  |        0.1  |                           1    |
| iforest_cusum    | control_fault    |             1    |           0.2  |        0.3  |                           1    |
| iforest_cusum    | lamp_degradation |             1    |           3.2  |        3.81 |                           1    |
| iforest_cusum    | seal_leak        |             0.08 |          10.75 |       11.74 |                           1    |
| iforest_cusum    | sensor_spike     |             0.56 |           0    |        0.2  |                           0.86 |

## False alarms per instrument-day

| detector         |   fa_per_day_clean_runs |   fa_per_day_faulty_runs |   fa_per_day_clean_runs_noisy |   fa_per_day_faulty_runs_noisy |
|:-----------------|------------------------:|-------------------------:|------------------------------:|-------------------------------:|
| static_threshold |                       0 |                    0     |                         0     |                          0     |
| residual_z       |                       0 |                    0     |                         0.11  |                          0.051 |
| iforest_cusum    |                       0 |                    0.006 |                         2.912 |                          1.505 |

Test set: 50 faulty runs, 20 clean runs (600 clean instrument-days). Zero false alarms in 600 days -> 95% upper bound ~0.005/day (rule of three). 'noisy' = all sensor noise x1.5 vs. calibration (distribution shift).
