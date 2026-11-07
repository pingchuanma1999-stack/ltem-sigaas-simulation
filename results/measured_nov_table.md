# November 3-6 scans: amplitude metric

Scale: -25 V, X = -2.5 um = 1.37 uV (mean of two Nov 3 single scans) == 0.52 (Oct 19) -> 1.00 (the +25 V, X = +5 um reference) == 2.63 uV, +-19 %.

| file | date | bias | X (um) | kind | raw RMS (uV) | noise diff (uV) | noise STD/sqrt3 (uV) | corrected RMS (uV) | relative | note |
|---|---|---|---|---|---|---|---|---|---|---|
| 22h45m12s | 2026-11-03 | -25 | -2.5 | single | 1.41 | 0.43 |  | 1.34 +- 0.08 | 0.51 +- 0.03 | before moving the tip |
| 23h05m15s | 2026-11-03 | -25 | -2.5 | single | 1.47 | 0.45 |  | 1.39 +- 0.08 | 0.53 +- 0.03 | after moving the tip to the middle |
| 23h27m21s | 2026-11-03 | -25 | +5 | single | 0.84 | 0.42 |  | 0.72 +- 0.08 | 0.27 +- 0.03 | scan 2 of 3 (scan 1 noisy and not used) |
| 23h27m21s | 2026-11-03 | -25 | +5 | single | 0.61 | 0.44 |  | 0.43 +- 0.09 | 0.16 +- 0.04 | scan 3 of 3 |
| 00h28m32s | 2026-11-04 | +25 | -5 | mean3 | 0.31 | 0.23 | 0.32 | 0.00 +- 0.16 | 0.00 +- 0.06 |  |
| 01h18m30s | 2026-11-04 | +25 | -2.5 | mean3 | 0.28 | 0.26 | 0.29 | 0.00 +- 0.14 | 0.00 +- 0.05 |  |
| 02h07m42s | 2026-11-04 | +25 | -7.5 | mean3 | 0.42 | 0.28 | 0.35 | 0.23 +- 0.09 | 0.09 +- 0.03 |  |
| 01h48m18s | 2026-11-06 | +25 | -7.5 | mean3 | 0.62 | 0.43 | 0.49 | 0.39 +- 0.11 | 0.15 +- 0.04 |  |
| 04h25m30s | 2026-11-06 | +25 | -5 | mean3 | 0.49 | 0.35 | 0.48 | 0.09 +- 0.24 | 0.04 +- 0.09 |  |

Waveform correlation in 82.78-83.26 mm (pure noise: |r| < 0.31 at 95 % for 40 points):

- -25 V X=-2.5: before vs after moving the tip: r = +0.68
- -25 V X=+5: scan 2 vs scan 3: r = -0.00
- +25 V X=-5: Nov 4 vs Nov 6: r = -0.24
- +25 V X=-7.5: Nov 4 vs Nov 6: r = -0.12
- +25 V X=-5 vs X=-2.5 (Nov 4): r = +0.02
- +25 V X=-5 vs the -25 V X=-2.5 template: r = -0.08
- same X=-2.5: +25 V vs -25 V: r = -0.35
- mirror pair: +25 V X=-5 vs -25 V X=+5 (scan 2): r = +0.22

Combined (relative to +25 V, X = +5 um; error includes the 20 % scale uncertainty):

- +25 V, X = -7.5 um: 0.11 +- 0.08
- +25 V, X = -5 um: 0.01 +- 0.08
- +25 V, X = -2.5 um: 0.00 +- 0.08
- -25 V, X = +5 um: 0.23 +- 0.10
