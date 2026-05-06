"""Compare a reference SCION baseline (Octave or MATLAB) vs the Python port."""

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

plt.style.use("default")

T_START = time.time()

HERE = Path("/home/laz/proj/SCION/python_port")

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--ref-mat",
                    help="Reference baseline .mat (Octave or real MATLAB). "
                         "Default: scion_octave_baseline.mat",
                    default=str(HERE / "scion_octave_baseline.mat"))
parser.add_argument("--matlab-mat",
                    help="Shortcut for --ref-mat with --ref-label MATLAB so "
                         "plots and the report say 'MATLAB' instead of 'Octave'.",
                    default=None)
parser.add_argument("--py-npz",
                    help="Python baseline .npz. Default: scion_python_baseline.npz",
                    default=str(HERE / "scion_python_baseline.npz"))
parser.add_argument("--ref-label",
                    help="Label for the reference dataset in plots/report.",
                    default="Octave")
parser.add_argument("--plot-dir", default=str(HERE / "comparison_plots"))
parser.add_argument("--report",   default=str(HERE / "comparison_report.md"))
args = parser.parse_args()

if args.matlab_mat is not None:
    args.ref_mat = args.matlab_mat
    if args.ref_label == "Octave":
        args.ref_label = "MATLAB"
    if args.plot_dir == str(HERE / "comparison_plots"):
        args.plot_dir = str(HERE / "comparison_plots_matlab")
    if args.report == str(HERE / "comparison_report.md"):
        args.report = str(HERE / "comparison_report_matlab.md")

OCT_PATH = Path(args.ref_mat)        # name kept for minimal diff downstream
PY_PATH  = Path(args.py_npz)
PLOT_DIR = Path(args.plot_dir)
REPORT_PATH = Path(args.report)
REF_LABEL = args.ref_label
PLOT_DIR.mkdir(exist_ok=True)

# Load Octave baseline
oct_raw = loadmat(str(OCT_PATH), squeeze_me=True, struct_as_record=False)
oct_state = oct_raw["state"]
oct_t = np.asarray(oct_state.time_myr, dtype=float)
# Octave time_myr can have duplicates from MATLAB ode output; dedup by unique
_, uniq_idx = np.unique(oct_t, return_index=True)
uniq_idx.sort()
oct_t_u = oct_t[uniq_idx]

# Load Python baseline
py_data = np.load(str(PY_PATH), allow_pickle=True)
py_t = np.asarray(py_data["time_myr"], dtype=float)
_, py_uniq_idx = np.unique(py_t, return_index=True)
py_uniq_idx.sort()
py_t_u = py_t[py_uniq_idx]

# Field aliasing: Python uses TEMP/GAST instead of tempC; treat them as equivalents.
ALIASES = {
    # python_key: octave_key
    "GAST": "tempC",  # global average surface temperature in Celsius
}

# Build common-field set (Octave field name space)
oct_fields = set([f for f in dir(oct_state) if not f.startswith("_") and isinstance(getattr(oct_state, f), np.ndarray) and getattr(oct_state, f).ndim == 1 and getattr(oct_state, f).shape[0] == oct_t.shape[0]])
py_fields = set([k for k in py_data.keys() if py_data[k].ndim == 1 and py_data[k].shape[0] == py_t.shape[0]])

# Build mapping py_key -> oct_key for fields we will compare
field_map = {}
for f in oct_fields & py_fields:
    if f == "time_myr":
        continue
    field_map[f] = f
for py_key, oct_key in ALIASES.items():
    if py_key in py_fields and oct_key in oct_fields:
        field_map[py_key] = oct_key

# Common time grid
GRID = np.linspace(-1000.0, 0.0, 1001)
EPS = 1e-30


def interp_field(t_u, vals, uniq_idx):
    v_u = np.asarray(vals, dtype=float)[uniq_idx]
    # Mark NaN/inf samples; np.interp would propagate them silently.
    bad = ~np.isfinite(v_u)
    if bad.any():
        # Use a sentinel grid that interpolates only over finite samples and
        # leaves a mask where the source had no finite value nearby.
        finite_t = t_u[~bad]
        finite_v = v_u[~bad]
        if finite_t.size == 0:
            return np.full_like(GRID, np.nan)
        out = np.interp(GRID, finite_t, finite_v)
        # Mask any grid point that falls inside an all-NaN gap longer than 5 Myr.
        # Cheaper: mask any grid point closer to a NaN-source-time than to a finite one.
        nan_t = t_u[bad]
        # For each grid point, if nearest source-time is NaN, mask it.
        # Use simple distance on sorted arrays via searchsorted.
        idx_nan = np.searchsorted(nan_t, GRID)
        idx_nan = np.clip(idx_nan, 1, len(nan_t) - 1) if len(nan_t) > 1 else np.zeros_like(GRID, dtype=int)
        # Distance to nearest NaN
        if len(nan_t) > 0:
            left = np.clip(np.searchsorted(nan_t, GRID) - 1, 0, len(nan_t) - 1)
            right = np.clip(np.searchsorted(nan_t, GRID), 0, len(nan_t) - 1)
            d_nan = np.minimum(np.abs(GRID - nan_t[left]), np.abs(GRID - nan_t[right]))
        else:
            d_nan = np.full_like(GRID, np.inf)
        left_f = np.clip(np.searchsorted(finite_t, GRID) - 1, 0, len(finite_t) - 1)
        right_f = np.clip(np.searchsorted(finite_t, GRID), 0, len(finite_t) - 1)
        d_fin = np.minimum(np.abs(GRID - finite_t[left_f]), np.abs(GRID - finite_t[right_f]))
        mask_bad = d_nan < d_fin
        out[mask_bad] = np.nan
        return out
    return np.interp(GRID, t_u, v_u)


def metrics(oct_v, py_v):
    # Compare only where both are finite; NaN gaps in either signal are
    # treated as "no reference data" rather than zero (otherwise NaN regions
    # in the Octave baseline would dominate the metric).
    valid = np.isfinite(oct_v) & np.isfinite(py_v)
    if not valid.any():
        return float("nan"), float("nan"), float("nan"), float("nan"), 0
    diff = py_v[valid] - oct_v[valid]
    abs_d = np.abs(diff)
    mae = float(np.mean(abs_d))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    max_abs = float(np.max(abs_d))
    oct_finite = oct_v[valid]
    scale = max(float(np.max(np.abs(oct_finite))), float(np.std(oct_finite)), EPS)
    max_rel = float(max_abs / scale)
    return mae, rmse, max_abs, max_rel, int(valid.sum())


def classify(max_rel):
    if max_rel < 1e-3:
        return "excellent"
    if max_rel < 1e-2:
        return "good"
    if max_rel < 5e-2:
        return "fair"
    return "poor"


# Compute metrics for every mapped field
results = []
resampled = {}  # py_key -> (oct_v, py_v)
for py_key, oct_key in sorted(field_map.items()):
    oct_raw_vals = np.asarray(getattr(oct_state, oct_key), dtype=float)
    py_raw_vals = np.asarray(py_data[py_key], dtype=float)
    oct_v = interp_field(oct_t_u, oct_raw_vals, uniq_idx)
    py_v = interp_field(py_t_u, py_raw_vals, py_uniq_idx)
    mae, rmse, mab, mre, n_valid = metrics(oct_v, py_v)
    cls = classify(mre) if np.isfinite(mre) else "poor"
    results.append({
        "py_key": py_key,
        "oct_key": oct_key,
        "mae": mae,
        "rmse": rmse,
        "max_abs_err": mab,
        "max_rel_err": mre,
        "classification": cls,
        "n_valid": n_valid,
        "n_total": GRID.size,
    })
    resampled[py_key] = (oct_v, py_v)

# Sort: poor first, then by max_rel desc
ORDER = {"poor": 0, "fair": 1, "good": 2, "excellent": 3}
results_sorted = sorted(results, key=lambda r: (ORDER[r["classification"]], -r["max_rel_err"]))

# Plotting: 4 fields per figure (2x2), top=overlay, bottom=abs diff
def plot_groups(results_list):
    n = len(results_list)
    per_fig = 4
    n_figs = (n + per_fig - 1) // per_fig
    UNITS = {
        "tempC": "C", "GAST": "C", "iceline": "deg lat", "RCO2": "x PAL",
        "RO2": "x PAL", "ANOX": "frac", "VEG": "frac",
        "DEGASS": "rel", "EVO": "rel", "BAS_AREA": "rel", "GRAN_AREA": "rel", "W": "rel",
        "SAT_tropical": "C", "SAT_equator": "C",
    }
    # Group results in given order
    for fig_i in range(n_figs):
        chunk = results_list[fig_i * per_fig:(fig_i + 1) * per_fig]
        names = [r["py_key"] for r in chunk]
        fig, axes = plt.subplots(4, 2, figsize=(12, 12))
        # axes shape (4,2); we want 2x2 panels each containing top+bottom -> use 4 rows x 2 cols
        for j, r in enumerate(chunk):
            row_top = (j // 2) * 2
            col = j % 2
            ax_top = axes[row_top, col]
            ax_bot = axes[row_top + 1, col]
            oct_v, py_v = resampled[r["py_key"]]
            ax_top.plot(GRID, oct_v, color="tab:blue", linestyle="-", label=REF_LABEL, linewidth=1.2)
            ax_top.plot(GRID, py_v, color="tab:orange", linestyle="--", label="Python", linewidth=1.2)
            unit = UNITS.get(r["py_key"], "")
            ylab = f"{r['py_key']}" + (f" ({unit})" if unit else "")
            ax_top.set_ylabel(ylab, fontsize=9)
            ax_top.set_title(f"{r['py_key']} [{r['classification']}, max_rel={r['max_rel_err']:.2e}]", fontsize=9)
            ax_top.legend(fontsize=7, loc="best")
            ax_top.grid(True, alpha=0.3)
            ax_bot.plot(GRID, np.abs(py_v - oct_v), color="tab:red", linewidth=0.9)
            ax_bot.set_ylabel("|py - oct|", fontsize=9)
            ax_bot.set_xlabel("Time (Ma)", fontsize=9)
            ax_bot.grid(True, alpha=0.3)
        # Hide unused axes
        for j in range(len(chunk), per_fig):
            row_top = (j // 2) * 2
            col = j % 2
            axes[row_top, col].axis("off")
            axes[row_top + 1, col].axis("off")
        fig.suptitle(" | ".join(names), fontsize=11)
        fig.subplots_adjust(hspace=0.5, wspace=0.3, top=0.94)
        out = PLOT_DIR / f"group_{fig_i + 1:02d}.png"
        fig.savefig(str(out), dpi=300)
        plt.close(fig)


plot_groups(results_sorted)

# Final-state table: 21 reservoirs + GAST + iceline.
RESERVOIRS = ["P", "O", "A", "S", "G", "C", "PYR", "GYP", "N", "OSr", "SSr"]
ISO_FIELDS = ["d13c_A", "delta_mccb", "d34s_S", "delta_G", "delta_C", "delta_PYR", "delta_GYP", "delta_OSr"]
DIAG_FIELDS = ["GAST", "iceline"]

final_rows = []

def fetch_final(py_key, oct_key):
    if oct_key in oct_fields:
        oct_arr = np.asarray(getattr(oct_state, oct_key), dtype=float)
        oct_arr = oct_arr[uniq_idx]
        oct_final = float(oct_arr[-1])
    else:
        oct_final = float("nan")
    if py_key in py_data.files:
        py_arr = np.asarray(py_data[py_key], dtype=float)[py_uniq_idx]
        py_final = float(py_arr[-1])
    else:
        py_final = float("nan")
    return oct_final, py_final


for f in RESERVOIRS + ISO_FIELDS:
    oct_f, py_f = fetch_final(f, f)
    abs_d = py_f - oct_f
    rel_d = abs_d / max(abs(oct_f), EPS)
    final_rows.append((f, oct_f, py_f, abs_d, rel_d))

# GAST: Python GAST <-> Octave tempC
for py_key in DIAG_FIELDS:
    oct_key = ALIASES.get(py_key, py_key)
    oct_f, py_f = fetch_final(py_key, oct_key)
    abs_d = py_f - oct_f
    rel_d = abs_d / max(abs(oct_f), EPS)
    final_rows.append((f"{py_key} (vs {oct_key})", oct_f, py_f, abs_d, rel_d))

# Concerning fields
concerning = [r for r in results_sorted if r["classification"] not in ("excellent", "good")]


def explain(r):
    oct_v, py_v = resampled[r["py_key"]]
    valid = np.isfinite(oct_v) & np.isfinite(py_v)
    n_nan_oct = int(np.sum(~np.isfinite(oct_v)))
    n_nan_py = int(np.sum(~np.isfinite(py_v)))
    if not valid.any():
        return "No overlapping finite samples"
    oct_f = oct_v[valid]
    py_f = py_v[valid]
    if n_nan_oct > GRID.size * 0.1:
        return f"{REF_LABEL} NaN over {n_nan_oct}/{GRID.size} grid points; compared on remaining region only"
    if n_nan_py > GRID.size * 0.1:
        return f"Python NaN over {n_nan_py}/{GRID.size} grid points"
    if np.max(np.abs(oct_f)) < 1e-12:
        return f"{REF_LABEL} values ~ 0 throughout integration"
    mag_oct = np.log10(max(np.max(np.abs(oct_f)), EPS))
    mag_py = np.log10(max(np.max(np.abs(py_f)), EPS))
    if abs(mag_oct - mag_py) > 1.5:
        return f"Scales differ by ~{abs(mag_oct - mag_py):.1f} orders of magnitude"
    if np.std(oct_f) < 1e-12:
        return f"{REF_LABEL} value is essentially constant; small Python deviation amplified"
    # Identify if it's transient localized
    diff = np.abs(py_f - oct_f)
    threshold = 0.5 * np.max(diff)
    n_above = int(np.sum(diff > threshold))
    if n_above < 5:
        return f"Localized transient mismatch (only {n_above} grid points near worst diff)"
    if r["max_rel_err"] > 0.5:
        return "Large divergence — likely a port bug or different forcing/units"
    return f"Trajectory diverges from {REF_LABEL} but in same scale"


# Build report
total_wall = time.time() - T_START

lines = []
lines.append(f"# SCION Python Port vs {REF_LABEL} Baseline — Comparison Report")
lines.append("")
lines.append(f"- {REF_LABEL} source: `{OCT_PATH}`")
lines.append(f"- Python source: `{PY_PATH}`")
lines.append(f"- Comparison time grid: -1000 Ma to 0 Ma, {GRID.size} points (1 Myr resolution)")
lines.append(f"- Fields compared: **{len(results)}**")
lines.append("")
lines.append("## Final-state side-by-side (21 reservoirs + isotopes + GAST + iceline)")
lines.append("")
lines.append(f"| Field | {REF_LABEL} final | Python final | Abs diff | Rel diff |")
lines.append("|---|---:|---:|---:|---:|")
for name, o, p, ad, rd in final_rows:
    lines.append(f"| {name} | {o:.6g} | {p:.6g} | {ad:+.3e} | {rd:+.3e} |")
lines.append("")

# Classification summary
counts = {"excellent": 0, "good": 0, "fair": 0, "poor": 0}
for r in results:
    counts[r["classification"]] += 1
lines.append("## Classification summary")
lines.append("")
lines.append(f"- excellent (max_rel < 1e-3): **{counts['excellent']}**")
lines.append(f"- good (max_rel < 1e-2): **{counts['good']}**")
lines.append(f"- fair (max_rel < 5e-2): **{counts['fair']}**")
lines.append(f"- poor (max_rel >= 5e-2): **{counts['poor']}**")
lines.append("")

lines.append("## All compared fields (sorted: poor first, then by max_rel_err desc)")
lines.append("")
lines.append(f"| Field (py) | {REF_LABEL} key | MAE | RMSE | Max abs err | Max rel err | Class |")
lines.append("|---|---|---:|---:|---:|---:|---|")
for r in results_sorted:
    lines.append(
        f"| {r['py_key']} | {r['oct_key']} | {r['mae']:.4e} | {r['rmse']:.4e} | "
        f"{r['max_abs_err']:.4e} | {r['max_rel_err']:.4e} | {r['classification']} |"
    )
lines.append("")

lines.append("## Concerning fields (classification != excellent/good)")
lines.append("")
if not concerning:
    lines.append("_None — every field is `excellent` or `good`._")
else:
    lines.append("| Field | Class | Max rel err | Likely explanation |")
    lines.append("|---|---|---:|---|")
    for r in concerning:
        lines.append(f"| {r['py_key']} | {r['classification']} | {r['max_rel_err']:.3e} | {explain(r)} |")
lines.append("")

lines.append("## Metadata")
lines.append("")
lines.append(f"- {REF_LABEL} native points: {oct_t.size} ({oct_t_u.size} unique)")
lines.append(f"- Python native points: {py_t.size} ({py_t_u.size} unique)")
lines.append(f"- Comparison grid size: {GRID.size}")
lines.append(f"- Plot files: {PLOT_DIR}/group_NN.png ({(len(results) + 3) // 4} figures)")
lines.append(f"- Total wall-clock for compare.py: **{total_wall:.2f} s**")
lines.append("")

REPORT_PATH.write_text("\n".join(lines))

# Console summary
print(f"Wrote {REPORT_PATH}")
print(f"Wrote {(len(results) + 3) // 4} plot files in {PLOT_DIR}")
print(f"Counts: {counts}")
print(f"Wall: {total_wall:.2f}s")
