"""SCION 100-run Monte-Carlo sensitivity ensemble (Python port of SCION_sens.m)."""
from __future__ import annotations
import os
import sys
import time as _time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

warnings.filterwarnings('ignore', category=RuntimeWarning, message='overflow encountered')
warnings.filterwarnings('ignore', category=RuntimeWarning)


# Fields requested in the deliverable. Map dict-key -> scion.run output key.
# scion.run uses 't' for time; we expose it as 'time'.
FIELD_MAP = {
    'time': 't', 'time_myr': 'time_myr',
    'A': 'A', 'O': 'O', 'P': 'P', 'S': 'S',
    'G': 'G', 'C': 'C', 'PYR': 'PYR', 'GYP': 'GYP', 'N': 'N',
    'OSr': 'OSr', 'SSr': 'SSr',
    'RCO2': 'RCO2', 'mrO2': 'mrO2',
    'GAST': 'GAST', 'ANOX': 'ANOX', 'iceline': 'iceline',
    'BAS_AREA': 'BAS_AREA', 'GRAN_AREA': 'GRAN_AREA', 'DEGASS': 'DEGASS',
    'mocb': 'mocb', 'locb': 'locb', 'silw': 'silw',
    'carbw': 'carbw', 'oxidw': 'oxidw',
    'd13c_A': 'd13c_A', 'd34s_S': 'd34s_S', 'delta_OSr': 'delta_OSr',
    'VEG': 'VEG',
}

T_GRID = np.linspace(-1e9, 0.0, 1001)


def _worker(args) -> dict | None:
    """Run a single SCION integration with seeded random sens params.

    args: tuple (run_index, seed_offset). Per-run RNG seed = run_index + seed_offset.
    Returns dict of (T_grid,) interpolated arrays plus 'run_index', or None
    if the integration failed or ANOX has NaN.
    """
    run_index, seed_offset = args
    # Late import inside child so each worker re-imports scion fresh.
    sys.path.insert(0, os.path.join(os.path.dirname(__file__) or '.', '.'))
    import scion  # type: ignore

    rng = np.random.default_rng(run_index + seed_offset)
    rs = rng.uniform(-1.0, 1.0, size=7)
    sens_params = {f'r{i+1}': float(rs[i]) for i in range(7)}

    try:
        out = scion.run(sens_params=sens_params)
    except RuntimeError:
        return None
    except Exception:
        return None

    # Source time grid for this run
    src_t = np.asarray(out['t'], dtype=float)
    if src_t.size < 2:
        return None

    interp_out: dict = {'run_index': run_index}
    for out_key, src_key in FIELD_MAP.items():
        if src_key not in out:
            interp_out[out_key] = np.full(T_GRID.shape, np.nan)
            continue
        y = np.asarray(out[src_key], dtype=float)
        if y.shape != src_t.shape:
            interp_out[out_key] = np.full(T_GRID.shape, np.nan)
            continue
        # np.interp requires increasing xp; src_t already increasing (-1e9 -> 0).
        # Out-of-range queries get edge values; mark them NaN to mimic interp1 default.
        yi = np.interp(T_GRID, src_t, y)
        # Mask anything outside source range
        mask = (T_GRID < src_t[0]) | (T_GRID > src_t[-1])
        if mask.any():
            yi = yi.astype(float)
            yi[mask] = np.nan
        interp_out[out_key] = yi

    # Drop if ANOX has any NaN inside the in-range region
    anox = interp_out.get('ANOX')
    if anox is None or np.any(np.isnan(anox)):
        return None
    return interp_out


def run_ensemble(n_runs: int = 100,
                 n_workers: int | None = None,
                 save_path: str = 'python_port/scion_sens_results.npz',
                 plot_path: str = 'scion_sens_ensemble.png',
                 baseline_path: str = 'python_port/scion_python_baseline.npz',
                 seed: int | None = None) -> dict:
    """Run a parallel Monte-Carlo SCION sensitivity ensemble and save results.

    Seeding semantics: each worker uses np.random.default_rng(run_index + seed_offset).
    seed=None (default) -> seed_offset=0 -> per-run seed = run_index (legacy
    deterministic behaviour, identical to prior versions). Pass seed=K to shift
    all per-run seeds by K for an independent reproducible ensemble. MATLAB's
    SCION_sens.m uses MATLAB's implicit RNG state and is non-reproducible by
    default; this Python port is reproducible by construction.
    """
    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 2) // 2)
    seed_offset = 0 if seed is None else int(seed)

    print(f'[scion_sens] launching {n_runs} runs across {n_workers} workers '
          f'(seed_offset={seed_offset})')
    wall0 = _time.time()

    completed: list[dict] = []
    n_failed = 0

    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(_worker, (i, seed_offset)): i for i in range(n_runs)}
        done = 0
        for fut in as_completed(futures):
            done += 1
            res = fut.result()
            if res is None:
                n_failed += 1
                print(f'  [{done}/{n_runs}] run {futures[fut]} DROPPED')
            else:
                completed.append(res)
                print(f'  [{done}/{n_runs}] run {futures[fut]} ok')

    wall = _time.time() - wall0
    n_kept = len(completed)
    print(f'[scion_sens] done: kept={n_kept} dropped={n_failed} wall={wall:.1f}s')

    # Sort kept runs by run_index for deterministic column order
    completed.sort(key=lambda d: d['run_index'])

    # Stack into (T, N_kept) matrices
    out: dict = {}
    for key in FIELD_MAP.keys():
        if n_kept == 0:
            out[key] = np.zeros((T_GRID.size, 0))
        else:
            out[key] = np.stack([d[key] for d in completed], axis=1)

    out['run_indices'] = np.array([d['run_index'] for d in completed], dtype=int)
    out['n_kept'] = np.array([n_kept])
    out['n_dropped'] = np.array([n_failed])
    out['wall_seconds'] = np.array([wall])

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        np.savez_compressed(save_path, **out)
        print(f'[scion_sens] saved {save_path}')

    if plot_path is not None and n_kept > 0:
        _plot_ensemble(out, baseline_path=baseline_path, plot_path=plot_path)

    return out


def _plot_ensemble(out: dict, baseline_path: str, plot_path: str) -> None:
    """Quick 4-panel ensemble plot: CO2 ppm, GAST, mrO2, d13c_A."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    t_myr = T_GRID * 1e-6  # x axis in Myr

    # Derive series for plotting
    def stats(M):
        med = np.nanmedian(M, axis=1)
        sd = np.nanstd(M, axis=1)
        return med, sd

    co2_ppm = out['RCO2'] * 280.0
    gast = out['GAST']
    mro2 = out['mrO2']
    d13c = out['d13c_A']

    panels = [
        ('CO2 (ppm)', co2_ppm, 'log'),
        ('GAST (K)', gast, 'linear'),
        ('O2 mixing ratio', mro2, 'linear'),
        ('delta 13C (per mil)', d13c, 'linear'),
    ]

    baseline = None
    if baseline_path and os.path.exists(baseline_path):
        baseline = np.load(baseline_path)

    base_lookup = {
        'CO2 (ppm)': lambda b: (b['time_myr'], b['RCO2'] * 280.0),
        'GAST (K)': lambda b: (b['time_myr'], b['GAST']),
        'O2 mixing ratio': lambda b: (b['time_myr'], b['mrO2']),
        'delta 13C (per mil)': lambda b: (b['time_myr'], b['d13c_A']),
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for ax, (title, M, yscale) in zip(axes.flat, panels):
        med, sd = stats(M)
        ax.fill_between(t_myr, med - sd, med + sd, alpha=0.3, color='C0',
                        label='ensemble +/-1 sigma')
        ax.plot(t_myr, med, color='C0', lw=1.5, label='ensemble median')
        if baseline is not None:
            bx, by = base_lookup[title](baseline)
            ax.plot(bx, by, color='k', lw=1.2, ls='--',
                    label='deterministic baseline')
        ax.set_title(title)
        ax.set_yscale(yscale)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='best')
    for ax in axes[1]:
        ax.set_xlabel('Time (Myr)')
    fig.suptitle(f'SCION sensitivity ensemble (N={out["n_kept"][0]} kept, '
                 f'{out["n_dropped"][0]} dropped)')
    fig.tight_layout()
    fig.savefig(plot_path, dpi=130)
    plt.close(fig)
    print(f'[scion_sens] saved {plot_path}')


if __name__ == '__main__':
    run_ensemble()
