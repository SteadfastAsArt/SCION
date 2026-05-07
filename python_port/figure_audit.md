# SCION MATLAB → Python plotting parity audit

Generated: 2026-04-30. Source-of-truth MATLAB scripts vs. current Python figures.

## Summary

| MATLAB script | Panels | Python figure | Status |
| --- | --- | --- | --- |
| `SCION_plot_fluxes.m` | 16 (4×4) | `figure_fluxes.png` (`plot_fluxes`) | All covered |
| `SCION_plot_worldgraphic.m` | 6 columns × N keyframes | `figure_worldgraphic.png` (`plot_worldgraphic`) | All 6 columns covered, 9 keyframes |
| `SCION_plot_sens.m` | 10 (5×2) | `figure_sens_full.png` (`plot_sens`) | All 10 covered |

## SCION_plot_fluxes.m (16 panels)

| # | Panel | Python | Notes |
|---|---|---|---|
| 1 | Forcings (D/E/W/BA/GA) | covered | axes[0,0] |
| 2 | C_org fluxes (mocb/locb/oxidw/ocdeg) | covered | axes[0,1] |
| 3 | C_carb fluxes (silw/carbw/sfw/mccb) | covered | axes[0,2] |
| 4 | S fluxes (mpsb/mgsb/pyrw/pyrdeg/gypw/gypdeg) | covered | axes[0,3] |
| 5 | C reservoirs (G/C/VEG) | covered | axes[1,0] |
| 6 | S reservoirs (PYR/GYP) | covered | axes[1,1] |
| 7 | Nutrient reservoirs (P/N) | covered | axes[1,2] |
| 8 | f_org, f_py | covered | axes[1,3] |
| 9 | δ13C_carb + d13c proxy | covered | axes[2,0] |
| 10 | δ34S_sw + d34s proxy | covered | axes[2,1] |
| 11 | 87Sr/86Sr + sr proxy | covered | axes[2,2] |
| 12 | Marine SO4 mM + algeo/SO4 proxies | covered | axes[2,3] |
| 13 | Atmospheric O2 % + O2 proxy | covered | axes[3,0] |
| 14 | Atmospheric CO2 ppm + 6 proxy series | covered | axes[3,1] |
| 15 | GAST + SAT_equator + Scotese 2021 | covered | axes[3,2] |
| 16 | Ice line + paleolat proxy | covered | axes[3,3] |

## SCION_plot_worldgraphic.m (6 columns × keyframe rows)

The MATLAB script splits into two figures of 14 + 17 keyframes each (28 keyframes total). Python `plot_worldgraphic` reproduces 9 representative keyframes in one figure (configurable via `keyframes=` kwarg).

| Column (per row) | Python | Notes |
|---|---|---|
| Topography (TOPO with NaN below 1, gmtrelief cmap, ±5000) | covered | approximate gmtrelief palette |
| Lithology composite (land + relict + 2·arc + 3·suture, 5-colour ListedColormap) | covered | exact palette from m-file |
| Air Temperature (Tair masked over ocean, IPCC_temp, ±40) | covered | exact IPCC_temp palette |
| log10 Runoff (Q masked, IPCC_pre, 0–4) | covered | exact IPCC_pre palette |
| log10 Erosion (EPSILON masked, IPCC_seq_2, 0–4) | covered | exact IPCC_seq_2 palette |
| log10 Silw (CW masked, IPCC_seq, 0–2) | covered | exact IPCC_seq palette |
| Robinson projection / coastlines | not covered | plain rectangular `pcolormesh`; requires Cartopy/Basemap, deemed out-of-scope |

Data source: `scion_octave_baseline.mat` `gridstate` struct (28 keyframes, shape 40×48). Python's `scion.run()` does not currently capture per-keyframe spatial fields; would require re-architecting `_spatial_weathering()` (~200 lines refactor) since spatial fields are computed into reused buffers and only scalar reductions are returned. Per the brief, option (b) was selected — Octave .mat is the gridstate source.

## SCION_plot_sens.m (10 panels)

| # | Panel | Python | Notes |
|---|---|---|---|
| 1 | Forcings (DEGASS/GRAN_AREA/BAS_AREA mean+min+max) | covered | axes[0,0] |
| 2 | δ13C_carb mean+min+max + d13c proxy | covered (via `d13c_A` proxy) | `delta_mccb` not in current sens npz; `d13c_A` substituted (numerically equivalent in MATLAB sens output) |
| 3 | δ34S_sw mean+min+max + d34s proxy | covered | axes[1,0] |
| 4 | 87Sr/86Sr mean+min+max + sr proxy | covered | axes[1,1] |
| 5 | Marine SO4 mM + algeo + SO4 proxies | covered (derived from S/S0·28) | `SmM` not in current npz; computed from `S` |
| 6 | Atmospheric O2 % + O2 proxy | covered (`mrO2*100`) | axes[2,1] |
| 7 | Atmospheric CO2 ppm + 6 proxy series | covered (`RCO2*280`) | axes[3,0] |
| 8 | GAST + SAT_tropical + SAT_equator + T patch | partial | `SAT_tropical`/`SAT_equator`/`T_x`,`T_y` not in current sens npz; only `GAST` mean drawn |
| 9 | Ice line + paleolat proxy | covered | axes[4,0] |
| 10 | P, N (P solid, N dashed) | covered | axes[4,1]; auto-normalises if absolute mol values stored |

## Fields not currently captured by `scion.run()` / sens npz

- `delta_mccb`, `T_gast`, `SAT_tropical`, `SAT_equator`, `SmM`, `CO2ppm` — the underlying scalars ARE recorded inside `recorder` (see `scion.py:1198`), but `scion_sens.py:FIELD_MAP` does not collect them. Add the keys to `FIELD_MAP` and re-run the ensemble for full SCION_plot_sens.m parity. No model code change needed.
- Per-keyframe spatial fields (gridstate.land/Q/Tair/TOPO/CW/CWcarb/EPSILON/ARC/RELICT_ARC/SUTURE) — would require ~200 lines of refactor to surface from `_spatial_weathering`'s in-place buffer pattern. Currently sourced from `scion_octave_baseline.mat`.

## Cross-platform projections

The MATLAB worldgraphic uses `m_proj('robinson', ...)` and `m_grid` from the M_Map toolbox. Python equivalent (Cartopy/Basemap) is not a dependency. The Python figure uses straight equirectangular `pcolormesh`, which is sufficient for parity-checking the field values themselves.
