# SCION — Python port and tooling

Python re-implementation of the SCION Earth-system model
(Mills et al., upstream MATLAB sources in the parent directory) plus
the auxiliary tooling needed to run, validate, plot, sensitivity-test,
and tune it without MATLAB.

## What is in here

| File | Purpose |
|---|---|
| `scion.py` | Core port. Re-implements `SCION_initialise.m` + `SCION_equations.m` (21-state ODE, 2-D spatial weathering) on top of `scipy.integrate.solve_ivp`. |
| `scion_plot.py` | Reproduces all four MATLAB plotting scripts: `plot_fluxes()` for `SCION_plot_fluxes.m` (16-panel global figure with proxy-data overlays), `plot_worldgraphic()` for `SCION_plot_worldgraphic.m` (2-D spatial maps at 9 keyframes × 6 columns), `plot_sens()` for `SCION_plot_sens.m` (10-panel ensemble figure). |
| `scion_sens.py` | Multiprocessing port of `SCION_sens.m` — 100 random sensitivity runs. |
| `scion_tune.py` | Nelder-Mead replacement for `SCION_run_tuning.m` patternsearch. |
| `compare.py` | Field-by-field comparison of the Python output against an Octave/MATLAB baseline `.mat` (RMSE / max relative error / overlay plots). |
| `run_octave.m` + `octave_stubs/*.m` | Octave-compatibility runner used to generate a baseline when MATLAB is unavailable. The stub directory shadows `xlsread`, `ode15s`, and the plot scripts so headless Octave can complete the integration. |
| `run_matlab.m` | One-shot MATLAB runner (drop into a real MATLAB box) that produces a baseline `.mat` consumable by `compare.py --matlab-mat …`. |

## Running the model

From the project root (the directory containing `forcings/`):

```bash
python3 -c "import sys; sys.path.insert(0, 'python_port'); \
            import scion; scion.run(save_path='python_port/scion_python_baseline.npz')"
```

Single 1-Gyr deterministic run. Approximately **20 s** wall on a recent x86 CPU.

### Python API

```python
import scion
out = scion.run(
    runcontrol = 0,                      # 0 full, -1 flux-only-plot, -2 steady-state
    save_path  = 'baseline.npz',         # optional — write per-step arrays
    tuning     = None,                   # optional dict[Gtune,Ctune,...,Atune]
    sens_params = None,                  # optional dict[r1..r7] each in [-1, 1]
    method     = 'BDF',                  # 'BDF' / 'LSODA' / 'Radau' …
    rtol       = 1e-6,
    atol       = 1e-9,
)
```

`out` is a `dict` containing:
- 21 state variables and ~30 diagnostic fluxes (same field names as MATLAB `state`).
- `out['gridstate']` — `(40, 48, n_stamps)` 2-D maps captured at each
  INTERPSTACK keyframe boundary (`land`, `Q`, `Tair`, `TOPO`, `CW`,
  `CWcarb`, `EPSILON`, `ARC`, `RELICT_ARC`, `SUTURE`). 28 stamps for a
  full `runcontrol=0` run, 1 stamp for `runcontrol=-2`.
- `out['pars']` — full parameter dict (also obtainable directly via
  `scion.build_pars()`).

`runcontrol` modes:
- `0` — full deterministic 1-Gyr integration with full state + gridstate.
- `-1` — same numerics; sets `out['skip_worldgraphic']=True` to hint that
  downstream plotting should skip the spatial-map figures.
- `-2` — present-day **steady-state single RHS evaluation** (no time
  integration). Returns the same dict schema with length-1 arrays.
  Wall ~0.2 s.

### Tuning cost as a standalone callable

```python
from scion_tune import cost
c = cost([0.45, 1.0, 1.1, 1.0, 0.1, 0.05, 3.0])  # mirrors SCION_tuning_function.m
```

### Octave baseline (no MATLAB available)

```bash
sudo apt-get install -y octave octave-io          # one-time
octave --no-gui --no-window-system --eval \
    "cd('/path/to/SCION'); addpath('python_port'); run_octave"
# -> python_port/scion_octave_baseline.mat   (~6 MB, ~480 s)
```

### MATLAB baseline

```matlab
>> cd /path/to/SCION
>> run('python_port/run_matlab.m')
% -> python_port/scion_matlab_baseline.mat
```

### Comparison

```bash
python3 python_port/compare.py                              # vs Octave (default)
python3 python_port/compare.py --matlab-mat python_port/scion_matlab_baseline.mat
```

Writes `comparison_report.md` and `comparison_plots/group_*.png`.

### Sensitivity ensemble

```bash
python3 -c "import sys; sys.path.insert(0,'python_port'); \
            from scion_sens import run_ensemble; \
            run_ensemble(n_runs=100, n_workers=8, \
                         save_path='python_port/scion_sens_results.npz', \
                         plot_path='python_port/scion_sens_ensemble.png', \
                         baseline_path='python_port/scion_python_baseline.npz')"
```

### Tuning (initial-reservoir optimization)

```bash
python3 python_port/scion_tune.py --maxiter 20
# -> python_port/scion_python_tuned_results.npz
# -> python_port/scion_python_tuned_baseline.npz
```

## Validation status

Octave baseline (lsode shim) vs Python port, 54 fields compared on a
1-Myr grid over the full integration:

| Class | Count | Threshold |
|---|---|---|
| excellent | 50 | `max_rel_err < 1e-3` |
| good      | 3  | `< 1e-2` |
| fair      | 1  | `< 5e-2` (only `iceline` at K-Pg, a discrete-threshold step diff in a diagnostic field) |
| poor      | 0  | — |

All 21 reservoirs at present day agree to ≤ 1.1 × 10⁻⁴ relative.
All 8 isotope δ values agree to ≤ 1.5 × 10⁻⁵.
See `comparison_report.md` for the full table.

## Engineering notes (Octave route)

These are the gotchas encountered making upstream SCION run on headless
Octave 6.4 (none of them affect a real-MATLAB run):

1. **`xlsread('…','','','basic')`** — Octave's io package rejects the
   `'basic'` interface flag; the shim `octave_stubs/xlsread.m` calls
   `xlsopen / xls2oct` directly instead.
2. **`ode15s` (SUNDIALS / IDAS)** — fails the first stiff-Newton step on
   SCION's spatial-weathering RHS. The shim `octave_stubs/ode15s.m`
   re-implements the MATLAB call signature on top of Octave's classic
   `lsode` (BDF family).
3. **Solver overshoot at `t = 0`** — `lsode` probes `t > 0` while
   estimating the step size; SCION's INTERPSTACK keyframes only exist
   for `t_geol ∈ [-825, 0] Myr`, so the shim clamps the solver's `t`
   into `[t0, tf]` before forwarding to the user RHS.
4. **`intersect(workingstate.time, T)` empties out** — `lsode` evaluates
   the RHS at internal step points then dense-output-interpolates onto
   the requested grid, so `T` and `workingstate.time` never coincide.
   The shim records `(t, y)` on every RHS call and returns *that*
   sequence as `T`, restoring the MATLAB post-processing semantics.
5. **CWD vs `addpath` precedence + plotting deps** — Octave resolves
   scripts in the CWD before searching the path, so
   `SCION_plot_worldgraphic.m` is found in the project root before our
   stub. The runner therefore wraps `SCION_initialise(0)` in
   `try/catch` and recovers `state` / `gridstate` / `pars` from globals
   when the (post-integration) plotting epilogue fails.

The Python port has none of these issues — it uses scipy's `BDF` /
`LSODA` directly.

## Speed

| Path | Wall time (1 Gyr) |
|---|---|
| Python (scipy BDF, **numba-accelerated** spatial weathering) | ~11 s |
| Python (pure-numpy fallback) | ~21 s |
| Octave (lsode shim) | ~480 s |

The hot path (`_make_rhs.rhs` → `_spatial_weathering`) is JIT-compiled
through `numba` when available; falls back to a bit-equivalent pure-numpy
implementation if `numba` is not installed. **1.86× speedup** measured
on warm runs. Baseline cProfile output is in `profile_baseline.txt`,
post-numba in `profile_after.txt`.

## Reproducibility

`.gitignore` excludes the regenerable artefacts (baselines, ensemble
results, logs). To rebuild from a fresh clone:

```bash
# 1. Generate the Octave baseline (slow but only needed once):
octave --no-gui --no-window-system --eval \
    "cd('/path/to/SCION'); addpath('python_port'); run_octave"

# 2. Generate the Python baseline:
python3 -c "import sys; sys.path.insert(0,'python_port'); \
            import scion; scion.run(save_path='python_port/scion_python_baseline.npz')"

# 3. Compare:
python3 python_port/compare.py
```
