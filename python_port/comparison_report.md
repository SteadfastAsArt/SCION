# SCION Python Port vs Octave Baseline — Comparison Report

- Octave source: `/home/laz/proj/SCION/python_port/scion_octave_baseline.mat`
- Python source: `/home/laz/proj/SCION/python_port/scion_python_baseline.npz`
- Comparison time grid: -1000 Ma to 0 Ma, 1001 points (1 Myr resolution)
- Fields compared: **54**

## Final-state side-by-side (21 reservoirs + isotopes + GAST + iceline)

| Field | Octave final | Python final | Abs diff | Rel diff |
|---|---:|---:|---:|---:|
| P | 3.2829e+15 | 3.28293e+15 | +3.329e+10 | +1.014e-05 |
| O | 3.62841e+19 | 3.62847e+19 | +6.504e+14 | +1.792e-05 |
| A | 3.24744e+18 | 3.24779e+18 | +3.460e+14 | +1.065e-04 |
| S | 4.88581e+19 | 4.88594e+19 | +1.232e+15 | +2.523e-05 |
| G | 1.24573e+21 | 1.24573e+21 | -2.748e+13 | -2.206e-08 |
| C | 4.93686e+21 | 4.93686e+21 | -2.425e+15 | -4.913e-07 |
| PYR | 1.59308e+20 | 1.59308e+20 | -7.145e+14 | -4.485e-06 |
| GYP | 1.68234e+20 | 1.68233e+20 | -5.179e+14 | -3.079e-06 |
| N | 4.54018e+16 | 4.54025e+16 | +7.219e+11 | +1.590e-05 |
| OSr | 1.26385e+17 | 1.26384e+17 | -3.055e+11 | -2.417e-06 |
| SSr | 5.39375e+18 | 5.39373e+18 | -1.640e+13 | -3.040e-06 |
| d13c_A | 0.908668 | 0.908687 | +1.947e-05 | +2.143e-05 |
| delta_mccb | 0.908668 | 0.908687 | +1.947e-05 | +2.143e-05 |
| d34s_S | 20.933 | 20.9329 | -1.016e-04 | -4.852e-06 |
| delta_G | -29.8408 | -29.8408 | +3.351e-06 | +1.123e-07 |
| delta_C | 0.968487 | 0.968489 | +1.522e-06 | +1.571e-06 |
| delta_PYR | -7.02416 | -7.02419 | -3.260e-05 | -4.641e-06 |
| delta_GYP | 19.2842 | 19.2842 | -6.345e-05 | -3.290e-06 |
| delta_OSr | 0.709374 | 0.709374 | -6.011e-08 | -8.474e-08 |
| GAST (vs tempC) | 14.4302 | 14.4309 | +7.715e-04 | +5.346e-05 |
| iceline (vs iceline) | 46.662 | 46.662 | +0.000e+00 | +0.000e+00 |

## Classification summary

- excellent (max_rel < 1e-3): **50**
- good (max_rel < 1e-2): **2**
- fair (max_rel < 5e-2): **2**
- poor (max_rel >= 5e-2): **0**

## All compared fields (sorted: poor first, then by max_rel_err desc)

| Field (py) | Octave key | MAE | RMSE | Max abs err | Max rel err | Class |
|---|---|---:|---:|---:|---:|---|
| iceline | iceline | 1.3635e-02 | 1.4677e-01 | 2.8925e+00 | 3.2139e-02 | fair |
| BAS_AREA | BAS_AREA | 1.7148e-04 | 1.2907e-03 | 2.8794e-02 | 1.3456e-02 | fair |
| basw | basw | 3.4207e+08 | 2.4557e+09 | 4.9663e+10 | 7.3834e-03 | good |
| granw | granw | 3.5922e+08 | 2.4562e+09 | 4.9627e+10 | 5.3281e-03 | good |
| erosion_tot | erosion_tot | 1.6628e+05 | 1.0375e+06 | 2.1656e+07 | 7.3872e-04 | excellent |
| mocb | mocb | 1.0454e+07 | 3.7303e+07 | 9.5542e+08 | 2.4923e-04 | excellent |
| d13c_A | d13c_A | 8.8018e-06 | 2.3002e-05 | 3.5390e-04 | 1.9334e-04 | excellent |
| delta_mccb | delta_mccb | 8.8018e-06 | 2.3002e-05 | 3.5390e-04 | 1.9334e-04 | excellent |
| mgsb | mgsb | 1.4894e+06 | 1.0390e+07 | 2.7474e+08 | 1.7090e-04 | excellent |
| N | N | 8.4294e+10 | 3.2119e+11 | 8.7620e+12 | 1.6266e-04 | excellent |
| mpsb | mpsb | 2.7525e+06 | 8.3198e+06 | 1.8256e+08 | 1.5715e-04 | excellent |
| locb | locb | 1.0097e+07 | 3.0377e+07 | 4.7246e+08 | 1.0844e-04 | excellent |
| W | W | 2.4255e-07 | 3.9465e-06 | 1.0150e-04 | 1.0150e-04 | excellent |
| nfix | nfix | 3.1090e+09 | 9.7211e+09 | 1.5469e+11 | 9.6304e-05 | excellent |
| GRAN_AREA | GRAN_AREA | 2.6589e-06 | 1.6151e-05 | 2.3301e-04 | 9.6154e-05 | excellent |
| ocdeg | ocdeg | 1.1849e+07 | 2.1565e+07 | 1.9361e+08 | 9.1365e-05 | excellent |
| ccdeg | ccdeg | 1.4081e+08 | 2.4060e+08 | 1.8677e+09 | 9.0688e-05 | excellent |
| DEGASS | DEGASS | 1.1803e-05 | 2.0235e-05 | 1.5873e-04 | 9.0653e-05 | excellent |
| denit | denit | 3.0263e+09 | 8.8425e+09 | 1.5434e+11 | 9.0019e-05 | excellent |
| mccb | mccb | 5.7159e+07 | 1.3709e+08 | 2.0733e+09 | 7.8455e-05 | excellent |
| P | P | 1.1934e+10 | 3.1870e+10 | 6.0345e+11 | 7.7468e-05 | excellent |
| phosw | phosw | 1.3889e+05 | 3.4071e+05 | 4.7995e+06 | 7.7312e-05 | excellent |
| ANOX | ANOX | 1.4643e-06 | 4.0559e-06 | 7.6265e-05 | 7.6703e-05 | excellent |
| silw | silw | 3.5369e+07 | 8.9112e+07 | 1.1403e+09 | 7.6118e-05 | excellent |
| oxidw | oxidw | 8.1318e+06 | 2.1417e+07 | 4.2441e+08 | 7.2991e-05 | excellent |
| sfw | sfw | 4.0366e+07 | 6.9807e+07 | 5.3949e+08 | 6.6272e-05 | excellent |
| carbw | carbw | 2.3319e+07 | 5.7296e+07 | 9.3299e+08 | 6.5589e-05 | excellent |
| pyrw | pyrw | 2.0164e+06 | 4.9054e+06 | 6.8600e+07 | 6.5494e-05 | excellent |
| gypw | gypw | 2.3967e+06 | 5.7779e+06 | 9.4697e+07 | 6.0799e-05 | excellent |
| mrO2 | mrO2 | 1.5076e-07 | 5.2199e-07 | 1.3147e-05 | 5.4050e-05 | excellent |
| GAST | tempC | 3.1700e-05 | 9.8884e-05 | 1.6084e-03 | 4.6964e-05 | excellent |
| VEG | VEG | 1.5237e-06 | 5.3078e-06 | 7.5014e-05 | 4.4188e-05 | excellent |
| RO2 | RO2 | 7.5172e-07 | 2.2372e-06 | 5.3091e-05 | 4.3905e-05 | excellent |
| O | O | 2.7814e+13 | 8.2777e+13 | 1.9643e+15 | 4.3905e-05 | excellent |
| EVO | EVO | 1.2063e-07 | 1.7106e-06 | 3.7234e-05 | 3.7234e-05 | excellent |
| d34s_S | d34s_S | 1.0850e-05 | 5.6724e-05 | 1.2006e-03 | 3.4303e-05 | excellent |
| SAT_tropical | SAT_tropical | 2.5025e-05 | 8.1044e-05 | 1.2194e-03 | 3.2345e-05 | excellent |
| SAT_equator | SAT_equator | 2.3946e-05 | 7.3018e-05 | 9.9684e-04 | 2.5569e-05 | excellent |
| S | S | 2.3878e+13 | 5.5798e+13 | 1.2325e+15 | 2.3760e-05 | excellent |
| A | A | 2.4558e+13 | 4.7036e+13 | 6.9781e+14 | 1.9085e-05 | excellent |
| OSr | OSr | 2.9216e+10 | 7.3952e+10 | 1.4793e+12 | 1.1705e-05 | excellent |
| RCO2 | RCO2 | 8.3395e-05 | 1.7697e-04 | 1.3861e-03 | 1.0570e-05 | excellent |
| delta_PYR | delta_PYR | 1.8987e-06 | 2.6698e-06 | 3.2598e-05 | 4.4979e-06 | excellent |
| delta_C | delta_C | 3.3404e-07 | 5.3774e-07 | 7.3951e-06 | 3.6976e-06 | excellent |
| PYR | PYR | 2.1766e+13 | 3.8262e+13 | 7.1454e+14 | 3.3544e-06 | excellent |
| SSr | SSr | 7.2542e+11 | 1.1005e+12 | 1.6397e+13 | 3.0400e-06 | excellent |
| delta_GYP | delta_GYP | 1.6398e-06 | 3.0022e-06 | 6.3451e-05 | 2.7011e-06 | excellent |
| GYP | GYP | 3.1180e+13 | 5.0080e+13 | 5.1794e+14 | 2.6159e-06 | excellent |
| delta_OSr | delta_OSr | 1.6276e-08 | 7.0558e-08 | 1.1212e-06 | 1.5803e-06 | excellent |
| G | G | 8.4521e+13 | 1.1301e+14 | 1.7733e+15 | 1.4235e-06 | excellent |
| C | C | 6.7140e+13 | 1.3250e+14 | 2.4255e+15 | 4.7993e-07 | excellent |
| delta_G | delta_G | 1.6053e-06 | 1.9441e-06 | 6.7570e-06 | 2.0172e-07 | excellent |
| gypdeg | gypdeg | 0.0000e+00 | 0.0000e+00 | 0.0000e+00 | 0.0000e+00 | excellent |
| pyrdeg | pyrdeg | 0.0000e+00 | 0.0000e+00 | 0.0000e+00 | 0.0000e+00 | excellent |

## Concerning fields (classification != excellent/good)

| Field | Class | Max rel err | Likely explanation |
|---|---|---:|---|
| iceline | fair | 3.214e-02 | Localized transient mismatch (only 2 grid points near worst diff) |
| BAS_AREA | fair | 1.346e-02 | Localized transient mismatch (only 1 grid points near worst diff) |

## Metadata

- Octave native points: 13388 (13387 unique)
- Python native points: 7853 (7853 unique)
- Comparison grid size: 1001
- Plot files: /home/laz/proj/SCION/python_port/comparison_plots/group_NN.png (14 figures)
- Total wall-clock for compare.py: **11.23 s**
