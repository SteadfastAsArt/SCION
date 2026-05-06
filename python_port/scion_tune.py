"""SCION tuning — Nelder-Mead optimization of starting reservoir multipliers."""
from __future__ import annotations

import argparse
import os
import time
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)

import numpy as np
from scipy.optimize import minimize

import scion


PARAM_NAMES = ['Gtune', 'Ctune', 'PYRtune', 'GYPtune', 'Otune', 'Stune', 'Atune']
RESERVOIR_KEYS = ['G', 'C', 'PYR', 'GYP', 'O', 'S', 'A']
RESERVOIR_X0_KEYS = ['G0', 'C0', 'PYR0', 'GYP0', 'O0', 'S0', 'A0']
DEFAULT_INITIAL = np.array([0.45, 1.0, 1.1, 1.0, 0.1, 0.05, 3.0])


def _params_to_dict(x: np.ndarray) -> dict:
    return {name: float(x[i]) for i, name in enumerate(PARAM_NAMES)}


def _final_relatives(result: dict, p: dict) -> np.ndarray:
    """Return [G_rel, C_rel, PYR_rel, GYP_rel, O_rel, S_rel, A_rel] at t=end."""
    rels = np.empty(7)
    for i, (key, x0_key) in enumerate(zip(RESERVOIR_KEYS, RESERVOIR_X0_KEYS)):
        rels[i] = result[key][-1] / p[x0_key]
    return rels


def tune(initial_guess: np.ndarray | None = None,
         save_path: str = 'scion_python_tuned_results.npz',
         baseline_save_path: str = 'scion_python_tuned_baseline.npz',
         maxiter: int = 50,
         xatol: float = 1e-3,
         fatol: float = 1e-6,
         forcings_dir: str = 'forcings') -> dict:
    """Run Nelder-Mead optimization over the seven tuning multipliers.

    Returns dict with best_params, best_cost, history, n_evals.
    """
    if initial_guess is None:
        initial_guess = DEFAULT_INITIAL.copy()
    initial_guess = np.asarray(initial_guess, dtype=float)
    assert initial_guess.shape == (7,), 'initial_guess must be length 7'

    # Bounds: [0.05*x0, 3*x0] per MATLAB SCION_run_tuning.m line 31.
    lo = 0.05 * initial_guess
    hi = 3.0 * initial_guess
    bounds = list(zip(lo, hi))

    # Reference X0 values for relative-final calc.
    p_ref = scion._build_pars()

    history: list = []
    eval_counter = {'n': 0}
    wall_start = time.time()

    def cost_fn(x: np.ndarray) -> float:
        eval_counter['n'] += 1
        it = eval_counter['n']
        params = _params_to_dict(x)
        try:
            # LSODA is more forgiving than BDF for unbalanced starting reservoirs
            # that the optimizer explores; deterministic baseline still uses BDF.
            result = scion.run(tuning=params, forcings_dir=forcings_dir,
                               method='LSODA', rtol=1e-4, atol=1e-7)
            rels = _final_relatives(result, p_ref)
            cost = float(np.sum((rels - 1.0) ** 2))
        except Exception as exc:
            print(f'[iter {it:>3}] FAILED: {exc!r} -> cost=1e6')
            rels = np.full(7, np.nan)
            cost = 1e6
        elapsed = time.time() - wall_start
        history.append({
            'iteration': it,
            'params': x.copy(),
            'cost': cost,
            'final_relatives': rels.copy(),
            'elapsed': elapsed,
        })
        rel_str = '  '.join(f'{v:.3f}' for v in rels)
        par_str = '  '.join(f'{v:.4f}' for v in x)
        print(f'[iter {it:>3}] t={elapsed:6.1f}s  cost={cost:.6e}')
        print(f'            params:    {par_str}')
        print(f'            final_rel: {rel_str}  (G C PYR GYP O S A)')
        return cost

    print('=== SCION tuning (Nelder-Mead with bounds) ===')
    print(f'initial guess: {initial_guess}')
    print(f'bounds:        lo={lo}')
    print(f'               hi={hi}')
    print(f'maxiter={maxiter}  xatol={xatol}  fatol={fatol}')
    print()

    res = minimize(
        cost_fn,
        initial_guess,
        method='Nelder-Mead',
        bounds=bounds,
        options={
            'maxiter': maxiter,
            'xatol': xatol,
            'fatol': fatol,
            'disp': True,
            'adaptive': True,
        },
    )

    wall_total = time.time() - wall_start
    best_params = _params_to_dict(res.x)
    best_cost = float(res.fun)

    print()
    print('=== optimization complete ===')
    print(f'  wall:        {wall_total:.1f} s')
    print(f'  n_evals:     {eval_counter["n"]}')
    print(f'  best cost:   {best_cost:.6e}')
    print(f'  best params: {best_params}')
    print(f'  scipy msg:   {res.message}')

    # Persist optimization history
    if save_path is not None:
        hist_iter = np.array([h['iteration'] for h in history], dtype=int)
        hist_params = np.array([h['params'] for h in history])
        hist_cost = np.array([h['cost'] for h in history])
        hist_relfin = np.array([h['final_relatives'] for h in history])
        hist_elapsed = np.array([h['elapsed'] for h in history])
        np.savez_compressed(
            save_path,
            best_params=res.x,
            best_cost=np.array([best_cost]),
            param_names=np.array(PARAM_NAMES),
            history_iter=hist_iter,
            history_params=hist_params,
            history_cost=hist_cost,
            history_final_relatives=hist_relfin,
            history_elapsed=hist_elapsed,
            n_evals=np.array([eval_counter['n']]),
            wall_seconds=np.array([wall_total]),
        )
        print(f'  saved history -> {save_path}')

    # Run deterministic SCION with best params and persist that npz too.
    if baseline_save_path is not None:
        print()
        print(f'=== running deterministic SCION with best params -> {baseline_save_path} ===')
        scion.run(tuning=best_params,
                  save_path=baseline_save_path,
                  forcings_dir=forcings_dir)
        print(f'  saved baseline -> {baseline_save_path}')

    return {
        'best_params': best_params,
        'best_cost': best_cost,
        'history': history,
        'n_evals': eval_counter['n'],
        'wall_seconds': wall_total,
        'scipy_result': res,
    }


def _cli() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--maxiter', type=int, default=50)
    ap.add_argument('--save-path', default='scion_python_tuned_results.npz')
    ap.add_argument('--baseline-save-path', default='scion_python_tuned_baseline.npz')
    ap.add_argument('--forcings-dir', default='forcings')
    args = ap.parse_args()
    tune(maxiter=args.maxiter,
         save_path=args.save_path,
         baseline_save_path=args.baseline_save_path,
         forcings_dir=args.forcings_dir)


if __name__ == '__main__':
    _cli()
