# SCION Python Port API Audit

Audit of `python_port/` against the source-of-truth MATLAB top-level entries
(`SCION_initialise.m`, `SCION_sens.m`, `SCION_run_tuning.m`,
`SCION_tuning_function.m`).

Symbols: matched, ⚠ partial, ✗ missing.

## Top-level entry points

| MATLAB                                | Python                                                                | Status   | Note |
|---|---|---|---|
| `SCION_initialise(0)` (full det. run) | `scion.run(runcontrol=0)`                                             | matched  | Bit-exact reproduction of prior Python baseline. |
| `SCION_initialise(-1)` (skip world)   | `scion.run(runcontrol=-1)`                                            | matched* | Now sets `out['skip_worldgraphic'] = True`; numerics identical to 0 (matches MATLAB behaviour: -1 only changes plotting). |
| `SCION_initialise(-2)` (steady state) | `scion.run(runcontrol=-2)`                                            | matched  | Now skips ODE integration, evaluates RHS once at `t=0`, returns scalar diagnostic dict + `gridstate` of length 1. Wall < 1 s. |
| `SCION_initialise(N>=1)` (sens mode)  | `scion.run(runcontrol=N, sens_params={'r1':...})`                     | matched  | sens_params explicit (no implicit MATLAB-rand); `runcontrol>=1` now disables gridstate capture (mirrors MATLAB `sensanal==0` guard). |
| `SCION_sens.m` (100-run ensemble)     | `scion_sens.run_ensemble(n_runs=100, seed=None)`                      | matched  | `seed` kwarg added; per-run RNG = `default_rng(run_index + seed_offset)`. |
| `SCION_run_tuning.m`                  | `scion_tune.tune(...)`                                                | matched  | Nelder-Mead vs MATLAB patternsearch (algorithmic substitution; same cost function). |
| `SCION_tuning_function(params)`       | `scion_tune.cost(params)`                                             | matched  | Now exposed as a top-level callable returning the same scalar. |
| `pars` struct (introspectable)        | `scion.build_pars()` and `out['pars']`                                | matched  | Public alias added; `run()` return now includes `'pars'`. |

## Output schema (`run()` return + npz save)

| MATLAB `state` field        | Python out key | Status |
|---|---|---|
| 21 reservoirs (P,O,A,...)   | `P,O,A,S,G,C,PYR,GYP,TEMP,CAL,N,G_iso,...,SSr_iso` | matched |
| Diagnostic fluxes (mocb,..) | `mocb,locb,mccb,silw,basw,granw,carbw,oxidw,phosw,pyrw,gypw,ocdeg,ccdeg,sfw,pyrdeg,gypdeg,mpsb,mgsb,monb,nfix,denit` | matched |
| Climate / forcings          | `RCO2,RO2,mrO2,VEG,ANOX,iceline,GAST,SAT_tropical,SAT_equator,DEGASS,W,EVO,BAS_AREA,GRAN_AREA,erosion_tot` | matched |
| Isotopes                    | `d13c_A,delta_mccb,d34s_S,delta_G,delta_C,delta_PYR,delta_GYP,delta_OSr` | matched |
| Time vector                 | `t` (years), `time_myr` (Myr)                  | matched |
| `gridstate` (40×48×28 maps) | `out['gridstate']` dict + `gridstate__*` keys in npz | matched (newly added) |
| `pars` struct               | `out['pars']` (dict)                            | matched (newly added) |
| `forcings` struct           | not exposed                                     | ⚠ partial — internal `_load_forcings()` is private; not surfaced through `run()`. Low-priority: callers can recompute by re-loading from disk. |
| `sensparams` struct         | `sens_params` is the *input*; not echoed back  | ⚠ partial — caller already owns the input dict, so this is intentional. |

## Implemented in this audit

1. **`scion.build_pars()`** — public alias of `_build_pars` (scion.py).
2. **`out['pars']`** — full `pars` dict surfaced from `run()`.
3. **`runcontrol = -2` steady-state path** — single RHS evaluation at `t=0`, no
   integration, returns reservoir scalars + diagnostics + 1-stamp gridstate.
4. **`runcontrol = -1` flag** — `out['skip_worldgraphic'] = True`; numerics
   identical to `runcontrol=0` (this matches MATLAB: line 430 just guards
   `SCION_plot_worldgraphic` with `if runcontrol > -1`).
5. **`gridstate` capture** — pure-Python keystamp tracker inside `_make_rhs`.
   When `t_geol` crosses the next INTERPSTACK keytime (or hits 0), the numpy
   per-cell kernel is invoked once to populate per-cell `Q`, `Tair`, `EPSILON`,
   `CW_per_km2`; `land`/`TOPO`/`ARC`/`RELICT_ARC`/`SUTURE` are sliced from
   INTERPSTACK at the past keyframe; `CWcarb` is computed as
   `k_carb_scale * Q_past * GRID_AREA` with NaNs zeroed (mirrors MATLAB
   SCION_equations.m lines 343–349). Stacked into `(40, 48, 28)` arrays.
   The numba-accelerated reductions path is left intact for the integrator
   loop — gridstate is only computed at the 28 stamp boundaries (~28 calls
   per run), so no perf regression.
6. **`scion_tune.cost(params, forcings_dir='forcings')`** — top-level standalone
   cost evaluator returning the same scalar the optimizer uses internally.
7. **`scion_sens.run_ensemble(..., seed=None)`** — `seed` kwarg added;
   per-run RNG is `default_rng(run_index + seed_offset)`. Default keeps prior
   deterministic behaviour (`seed_offset = 0`).

## Left as-is (and why)

- **`forcings` struct on `run()` return**: the dict is loaded from disk at
  every `run()` call; callers who want it can call the private `_load_forcings`
  or re-load directly. Surfacing the multi-MB forcings struct on every return
  would bloat saved npz files unnecessarily.
- **MATLAB `patternsearch` → SciPy `Nelder-Mead`**: SciPy has no patternsearch;
  Nelder-Mead with bounds is the closest derivative-free analogue available.
  The cost function is identical — only the search algorithm differs.
- **`sensparams` echo on return**: the input dict is owned by the caller, so
  there is no information loss from omitting it.

## Self-verification

| Check | Result |
|---|---|
| `scion.run()` numerically identical to prior baseline | All checked fields max_rel_err = 0.0e+00 (bit-exact). |
| `compare.py` against Octave baseline | **excellent=50, good=2, fair=2, poor=0** (matches target 50/2/2/0). |
| `scion.run(runcontrol=-2)` wall-clock | 0.56 s (target < 1 s). |
| `scion.build_pars()` key count | 55 (>30 required). |
| `out['gridstate']` shape | `(40, 48, 28)` for all 10 spatial fields; `time_myr` length 28. |
| `gridstate` vs Octave baseline | bit-exact for `time_myr/land/TOPO/ARC/RELICT_ARC/SUTURE`; ≤3.5e-3 max_rel for `Q/Tair/EPSILON/CW/CWcarb` (boundary-crossing time differs slightly between BDF and ode15s, expected). |
| `scion_tune.cost([0.45,1,1.1,1,0.1,0.05,3])` | returns 1.715e-01 in 5.5 s, identical to in-`tune()` evaluation. |
| `np.load('scion_python_baseline.npz')` | loads cleanly, 81 keys including all `gridstate__*`. |
