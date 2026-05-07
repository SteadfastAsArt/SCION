"""Reproduce SCION_plot_fluxes.m 4x4, SCION_plot_worldgraphic.m, and SCION_plot_sens.m figures."""

from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from scipy.io import loadmat


# MATLAB IPCC palettes from SCION_plot_fluxes.m / SCION_plot_worldgraphic.m
IPCC_pre = np.array([
    [223, 194, 125], [246, 232, 195], [245, 245, 245], [199, 234, 229],
    [128, 205, 193], [53, 151, 143], [1, 102, 94], [0, 60, 48],
]) / 255.0

IPCC_temp = np.flipud(np.array([
    [103, 0, 31], [178, 24, 43], [214, 96, 77], [244, 165, 130],
    [253, 219, 199], [247, 247, 247], [209, 229, 240], [146, 197, 222],
    [67, 147, 195], [33, 102, 172], [5, 48, 97],
])) / 255.0

IPCC_seq = np.array([
    [255, 255, 204], [161, 218, 180], [65, 182, 196],
    [44, 127, 184], [37, 52, 148],
]) / 255.0

IPCC_seq_2 = np.array([
    [237, 248, 251], [179, 205, 227], [140, 150, 198],
    [136, 86, 167], [129, 15, 124],
]) / 255.0

CMAP_PRE = LinearSegmentedColormap.from_list('IPCC_pre', IPCC_pre, N=256)
CMAP_TEMP = LinearSegmentedColormap.from_list('IPCC_temp', IPCC_temp, N=256)
CMAP_SEQ = LinearSegmentedColormap.from_list('IPCC_seq', IPCC_seq, N=256)
CMAP_SEQ2 = LinearSegmentedColormap.from_list('IPCC_seq_2', IPCC_seq_2, N=256)
CMAP_LITHO = ListedColormap([
    (1.00, 1.00, 1.00),
    (0.50, 0.50, 0.50),
    (0.00, 0.45, 0.74),
    (0.85, 0.33, 0.10),
    (0.49, 0.18, 0.56),
])
CMAP_TOPO = LinearSegmentedColormap.from_list('gmtrelief', [
    (0.00, '#3b2c1f'), (0.25, '#7a5c3b'), (0.45, '#bfa37a'),
    (0.50, '#ffffff'), (0.55, '#a8d8b9'), (0.75, '#3b8aa8'),
    (1.00, '#0b2545'),
], N=256)

# Reservoir / parameter normalization constants from SCION_initialise.m / pars
PARS_DEFAULTS = dict(
    G0=1.25e21,
    C0=5.0e21,
    P0=3.1e15,
    N0=4.35e16,
    S0=4.0e19,
    PYR0=1.8e20,
    GYP0=2.0e20,
    O0=3.7e19,
    A0=3.193e18,
    OSr0=1.2e17,
    SSr0=5.0e18,
    whenstart=-1000.0,  # in Ma
    whenend=0.0,
)

# Proxy color palette (from SCION_plot_fluxes.m lines 67-72)
PC1 = np.array([65, 195, 199]) / 255.0
PC2 = np.array([73, 167, 187]) / 255.0
PC3 = np.array([82, 144, 170]) / 255.0
PC4 = np.array([88, 119, 149]) / 255.0
PC5 = np.array([89, 96, 125]) / 255.0
PC6 = np.array([82, 56, 100]) / 255.0


def _load_python(npz_path):
    """Load Python npz baseline; return dict-like Bunch with field access."""
    d = np.load(npz_path, allow_pickle=True)
    out = {k: d[k] for k in d.files}
    # alias tempC <- GAST (MATLAB script uses state.tempC for panel 15)
    if "tempC" not in out and "GAST" in out:
        out["tempC"] = out["GAST"]
    return out


def _load_octave(mat_path):
    """Load Octave baseline; return state object with attribute access and pars dict."""
    m = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    state = m["state"]
    pars = m["pars"]
    pars_dict = {f: getattr(pars, f) for f in pars._fieldnames}
    return state, pars_dict


def _load_geochem(mat_path):
    m = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    return {k: m[k] for k in m if not k.startswith("__")}


def _xrange(pars):
    return [pars["whenstart"] / 1.0 if abs(pars["whenstart"]) <= 2000 else pars["whenstart"] / 1e6,
            pars["whenend"] / 1.0 if abs(pars["whenend"]) <= 2000 else pars["whenend"] / 1e6]


def plot_fluxes(python_npz_path, octave_mat_path=None, geochem_mat_path=None,
                scotese_mat_path=None, save_path=None):
    """Reproduce SCION_plot_fluxes.m 4x4 panel figure.

    Solid colored lines = Python; dashed black overlay = Octave (if given).
    """
    py = _load_python(python_npz_path)

    # Resolve pars from octave if available, else defaults
    if octave_mat_path is not None:
        oct_state, oct_pars = _load_octave(octave_mat_path)
        # Octave pars store whenstart in years; convert to Ma for xlim
        pars = dict(PARS_DEFAULTS)
        for k, v in oct_pars.items():
            pars[k] = v
        pars["whenstart"] = oct_pars["whenstart"] / 1e6
        pars["whenend"] = oct_pars["whenend"] / 1e6
    else:
        oct_state = None
        pars = dict(PARS_DEFAULTS)

    geochem = _load_geochem(geochem_mat_path) if geochem_mat_path else {}
    scotese = _load_geochem(scotese_mat_path) if scotese_mat_path else {}

    xlim = (pars["whenstart"], pars["whenend"])
    t_py = py["time_myr"]
    t_oc = oct_state.time_myr if oct_state is not None else None

    fig, axes = plt.subplots(4, 4, figsize=(16, 12), facecolor=(1, 0.98, 0.95))
    fig.subplots_adjust(left=0.05, right=0.98, top=0.96, bottom=0.05,
                        wspace=0.32, hspace=0.42)

    def plot_pair(ax, py_y, oc_y, color, label=None):
        ax.plot(t_py, py_y, color=color, linewidth=1.2, label=label)
        if oc_y is not None and t_oc is not None:
            ax.plot(t_oc, oc_y, color="black", linestyle="--", linewidth=0.9, alpha=0.7)

    # ---- Panel 1: Forcings ----
    ax = axes[0, 0]
    ax.set_xlim(xlim); ax.set_ylim(0, 2.5)
    ax.set_xlabel("Time (Ma)"); ax.set_ylabel("Relative forcing")
    plot_pair(ax, py["DEGASS"], getattr(oct_state, "DEGASS", None) if oct_state else None, "r")
    plot_pair(ax, py["BAS_AREA"], getattr(oct_state, "BAS_AREA", None) if oct_state else None, "k")
    plot_pair(ax, py["EVO"], getattr(oct_state, "EVO", None) if oct_state else None, "g")
    plot_pair(ax, py["W"], getattr(oct_state, "W", None) if oct_state else None, "b")
    plot_pair(ax, py["GRAN_AREA"], getattr(oct_state, "GRAN_AREA", None) if oct_state else None, (0.8, 0.8, 0.8))
    ax.text(-590, 2.4, "D", color="r")
    ax.text(-590, 2.2, "E", color="g")
    ax.text(-590, 2.0, "W", color="b")
    ax.text(-590, 1.6, "BA", color="k")
    ax.text(-590, 1.4, "GA", color=(0.8, 0.8, 0.8))
    ax.set_title("Forcings"); ax.grid(False)

    # ---- Panel 2: Corg fluxes ----
    ax = axes[0, 1]
    ax.set_xlim(xlim)
    ax.set_xlabel("Time (Ma)"); ax.set_ylabel("Flux (mol/yr)")
    plot_pair(ax, py["mocb"], getattr(oct_state, "mocb", None) if oct_state else None, "b")
    plot_pair(ax, py["locb"], getattr(oct_state, "locb", None) if oct_state else None, "g")
    plot_pair(ax, py["oxidw"], getattr(oct_state, "oxidw", None) if oct_state else None, "r")
    plot_pair(ax, py["ocdeg"], getattr(oct_state, "ocdeg", None) if oct_state else None, "k")
    ax.text(-590, 5e12, "mocb", color="b")
    ax.text(-590, 4e12, "locb", color="g")
    ax.text(-590, 3e12, "oxidw", color="r")
    ax.text(-590, 2e12, "ocdeg", color="k")
    ax.set_title(r"$\mathrm{C_{org}}$ fluxes")

    # ---- Panel 3: Ccarb fluxes ----
    ax = axes[0, 2]
    ax.set_xlim(xlim)
    ax.set_xlabel("Time (Ma)"); ax.set_ylabel("Flux (mol/yr)")
    plot_pair(ax, py["silw"], getattr(oct_state, "silw", None) if oct_state else None, "r")
    plot_pair(ax, py["carbw"], getattr(oct_state, "carbw", None) if oct_state else None, "c")
    plot_pair(ax, py["sfw"], getattr(oct_state, "sfw", None) if oct_state else None, "b")
    plot_pair(ax, py["mccb"], getattr(oct_state, "mccb", None) if oct_state else None, "k")
    ax.text(-590, 28e12, "silw", color="r")
    ax.text(-590, 24e12, "carbw", color="c")
    ax.text(-590, 20e12, "sfw", color="b")
    ax.text(-590, 16e12, "mccb", color="k")
    ax.set_title(r"$\mathrm{C_{carb}}$ fluxes")

    # ---- Panel 4: S fluxes ----
    ax = axes[0, 3]
    ax.set_xlim(xlim)
    ax.set_xlabel("Time (Ma)"); ax.set_ylabel("Fluxes (mol/yr)")
    plot_pair(ax, py["mpsb"], getattr(oct_state, "mpsb", None) if oct_state else None, "k")
    plot_pair(ax, py["mgsb"], getattr(oct_state, "mgsb", None) if oct_state else None, "c")
    plot_pair(ax, py["pyrw"], getattr(oct_state, "pyrw", None) if oct_state else None, "r")
    plot_pair(ax, py["pyrdeg"], getattr(oct_state, "pyrdeg", None) if oct_state else None, "m")
    plot_pair(ax, py["gypw"], getattr(oct_state, "gypw", None) if oct_state else None, "b")
    plot_pair(ax, py["gypdeg"], getattr(oct_state, "gypdeg", None) if oct_state else None, "g")
    ax.text(-590, 1.9e12, "mpsb", color="k")
    ax.text(-590, 1.7e12, "mgsb", color="c")
    ax.text(-590, 1.5e12, "pyrw", color="r")
    ax.text(-590, 1.2e12, "pyrdeg", color="m")
    ax.text(-590, 1.0e12, "gypw", color="b")
    ax.text(-590, 0.8e12, "gypdeg", color="g")
    ax.set_title("S fluxes")

    # ---- Panel 5: C reservoirs ----
    ax = axes[1, 0]
    ax.set_xlim(xlim)
    ax.set_xlabel("Time (Ma)"); ax.set_ylabel("Relative size")
    plot_pair(ax, py["G"] / pars["G0"], (oct_state.G / pars["G0"]) if oct_state else None, "k")
    plot_pair(ax, py["C"] / pars["C0"], (oct_state.C / pars["C0"]) if oct_state else None, "c")
    ax.plot(t_py, py["VEG"], "g--", linewidth=1.2)
    if oct_state is not None:
        ax.plot(t_oc, oct_state.VEG, color="black", linestyle=":", linewidth=0.9, alpha=0.7)
    ax.text(-590, 1.5, "VEG", color="g")
    ax.text(-590, 1.25, "G", color="k")
    ax.text(-590, 1.0, "C", color="c")
    ax.set_title("C reservoirs")

    # ---- Panel 6: S reservoirs ----
    ax = axes[1, 1]
    ax.set_xlim(xlim)
    ax.set_xlabel("Time (Ma)"); ax.set_ylabel("Relative size")
    plot_pair(ax, py["PYR"] / pars["PYR0"], (oct_state.PYR / pars["PYR0"]) if oct_state else None, "k")
    plot_pair(ax, py["GYP"] / pars["GYP0"], (oct_state.GYP / pars["GYP0"]) if oct_state else None, "c")
    ax.text(-590, 1.0, "PYR", color="k")
    ax.text(-590, 0.9, "GYP", color="c")
    ax.set_title("S reservoirs")

    # ---- Panel 7: Nutrient reservoirs ----
    ax = axes[1, 2]
    ax.set_xlim(xlim); ax.set_ylim(0, 3)
    ax.set_xlabel("Time (Ma)"); ax.set_ylabel("Relative size")
    plot_pair(ax, py["P"] / pars["P0"], (oct_state.P / pars["P0"]) if oct_state else None, "b")
    plot_pair(ax, py["N"] / pars["N0"], (oct_state.N / pars["N0"]) if oct_state else None, "g")
    ax.text(-590, 1.5, "P", color="b")
    ax.text(-590, 1.0, "N", color="g")
    ax.set_title("Nutrient reservoirs")

    # ---- Panel 8: f_org and f_py ----
    ax = axes[1, 3]
    ax.set_xlim(xlim)
    ax.set_xlabel("Time (Ma)"); ax.set_ylabel(r"$f_{org}, f_{py}$")
    forg_py = py["mocb"] / (py["mocb"] + py["mccb"])
    fpy_py = py["mpsb"] / (py["mpsb"] + py["mgsb"])
    plot_pair(ax, forg_py,
              (oct_state.mocb / (oct_state.mocb + oct_state.mccb)) if oct_state else None, "k")
    plot_pair(ax, fpy_py,
              (oct_state.mpsb / (oct_state.mpsb + oct_state.mgsb)) if oct_state else None, "m")
    ax.set_title(r"$f_{org}$, $f_{py}$")

    # ---- Panel 9: d13C ----
    ax = axes[2, 0]
    ax.set_xlim(xlim)
    ax.set_xlabel("Time (Ma)"); ax.set_ylabel(r"$\delta^{13}C_{carb}$")
    if "d13c_x" in geochem:
        ax.plot(geochem["d13c_x"], geochem["d13c_y"], ".", color=PC2, markersize=2, alpha=0.5)
    plot_pair(ax, py["delta_mccb"], getattr(oct_state, "delta_mccb", None) if oct_state else None, "k")
    ax.set_title(r"$\delta^{13}C$ record")

    # ---- Panel 10: d34S ----
    ax = axes[2, 1]
    ax.set_xlim(xlim)
    ax.set_xlabel("Time (Ma)"); ax.set_ylabel(r"$\delta^{34}S_{sw}$")
    if "d34s_x" in geochem:
        ax.plot(geochem["d34s_x"], geochem["d34s_y"], ".", color=PC2, markersize=2, alpha=0.5)
    plot_pair(ax, py["d34s_S"], getattr(oct_state, "d34s_S", None) if oct_state else None, "k")
    ax.set_title(r"$\delta^{34}S$ record")

    # ---- Panel 11: 87Sr/86Sr ----
    ax = axes[2, 2]
    ax.set_xlim(xlim); ax.set_ylim(0.706, 0.710)
    ax.set_xlabel("Time (Ma)"); ax.set_ylabel(r"$^{87}Sr/^{86}Sr$ seawater")
    if "sr_x" in geochem:
        ax.plot(geochem["sr_x"], geochem["sr_y"], color=PC2, linewidth=1.0, alpha=0.7)
    plot_pair(ax, py["delta_OSr"], getattr(oct_state, "delta_OSr", None) if oct_state else None, "k")
    ax.set_title("Ocean 87Sr/86Sr")

    # ---- Panel 12: SO4 ----
    ax = axes[2, 3]
    ax.set_xlim(xlim)
    ax.set_xlabel("Time (Ma)"); ax.set_ylabel(r"Marine $SO_4$ (mM)")
    if "sconc_max_x" in geochem:
        ax.plot(geochem["sconc_max_x"], geochem["sconc_max_y"], color=PC1, linewidth=0.8)
        ax.plot(geochem["sconc_min_x"], geochem["sconc_min_y"], color=PC1, linewidth=0.8)
        ax.plot(geochem["sconc_mid_x"], geochem["sconc_mid_y"], color=PC2, linewidth=0.8)
    if "SO4_x" in geochem:
        SO4_x = geochem["SO4_x"]; SO4_y = geochem["SO4_y"]
        for u in range(0, len(SO4_x) - 1, 2):
            ax.plot([SO4_x[u], SO4_x[u]], [SO4_y[u], SO4_y[u + 1]], color=PC3, linewidth=0.8)
    plot_pair(ax, (py["S"] / pars["S0"]) * 28.0,
              ((oct_state.S / pars["S0"]) * 28.0) if oct_state else None, "k")
    ax.set_title("SO4")

    # ---- Panel 13: O2 (%) ----
    ax = axes[3, 0]
    ax.set_xlim(xlim)
    ax.set_xlabel("Time (Ma)"); ax.set_ylabel(r"Atmospheric $O_2$ (%)")
    if "O2_x" in geochem:
        O2_x = geochem["O2_x"]; O2_y = geochem["O2_y"]
        for u in range(0, len(O2_x) - 1, 2):
            ax.plot([O2_x[u], O2_x[u]], [O2_y[u], O2_y[u + 1]], color=PC2, linewidth=0.8)
    plot_pair(ax, py["mrO2"] * 100.0,
              (oct_state.mrO2 * 100.0) if oct_state else None, "k")
    ax.set_title(r"$O_2$ (%)")

    # ---- Panel 14: CO2 ppm ----
    ax = axes[3, 1]
    ax.set_xlim(xlim); ax.set_ylim(100, 10000); ax.set_yscale("log")
    ax.set_xlabel("Time (Ma)"); ax.set_ylabel(r"Atmospheric $CO_2$ (ppm)")
    proxy_specs = [
        ("paleosol_age", "paleosol_co2", PC1),
        ("alkenone_age", "alkenone_co2", PC2),
        ("boron_age", "boron_co2", PC3),
        ("stomata_age", "stomata_co2", PC4),
        ("liverwort_age", "liverwort_co2", PC5),
        ("phytane_age", "phytane_co2", PC6),
    ]
    for ax_key, co2_key, color in proxy_specs:
        if ax_key in geochem:
            ax.plot(geochem[ax_key], geochem[co2_key], ".", color=color, markersize=3)
    plot_pair(ax, py["RCO2"] * 280.0,
              (oct_state.RCO2 * 280.0) if oct_state else None, "k")
    ax.set_title(r"$CO_2$ ppm")

    # ---- Panel 15: GAST ----
    ax = axes[3, 2]
    ax.set_xlim(xlim); ax.set_ylim(5, 40)
    ax.set_xlabel("Time (Ma)"); ax.set_ylabel("GAST (C)")
    if "Scotese_2021_age" in scotese:
        ax.plot(scotese["Scotese_2021_age"], scotese["Scotese_2021_GAT"],
                color=PC1, linewidth=1.0)
    plot_pair(ax, py["tempC"], getattr(oct_state, "tempC", None) if oct_state else None, "k")
    plot_pair(ax, py["SAT_equator"], getattr(oct_state, "SAT_equator", None) if oct_state else None, "r")
    ax.set_title("GAST / SAT_equator")

    # ---- Panel 16: ice line ----
    ax = axes[3, 3]
    ax.set_xlim(xlim); ax.set_ylim(0, 90)
    ax.set_xlabel("Time (Ma)"); ax.set_ylabel("Ice line")
    if "paleolat_x" in geochem:
        ax.plot(geochem["paleolat_x"], geochem["paleolat_y"], color=PC1, linewidth=1.0)
    plot_pair(ax, py["iceline"], getattr(oct_state, "iceline", None) if oct_state else None, "k")
    ax.set_title("Ice line")

    if save_path is not None:
        fig.savefig(save_path, dpi=150, facecolor=fig.get_facecolor())
    return fig


def _load_gridstate(source):
    """Return gridstate dict with arrays time_myr, land, Q, Tair, TOPO, CW, CWcarb,
    EPSILON, ARC, RELICT_ARC, SUTURE. Source: path to Octave .mat OR dict already in memory."""
    if isinstance(source, dict):
        return source
    m = loadmat(str(source), squeeze_me=True, struct_as_record=False)
    gs = m["gridstate"]
    fields = ["time_myr", "land", "Q", "Tair", "TOPO", "CW", "CWcarb",
              "EPSILON", "ARC", "RELICT_ARC", "SUTURE"]
    return {f: np.asarray(getattr(gs, f)) for f in fields}


def _load_interpstack(path):
    m = loadmat(str(path), squeeze_me=True, struct_as_record=False)
    isk = m["INTERPSTACK"]
    return dict(lat=np.asarray(isk.lat), lon=np.asarray(isk.lon),
                time=np.asarray(isk.time))


def plot_worldgraphic(gridstate_source, interpstack_path, save_path=None,
                      keyframes=None):
    """Reproduce SCION_plot_worldgraphic.m: per-keyframe spatial maps in one tall figure.

    gridstate_source: path to .mat with 'gridstate' OR dict with the gridstate arrays.
    Columns (left -> right): Topography, Lithology, Air Temp, log Runoff,
    log Erosion, log Silw. Rows: keyframes (oldest at top, present at bottom).
    """
    gs = _load_gridstate(gridstate_source)
    isk = _load_interpstack(interpstack_path)
    times_all = gs["time_myr"]
    n_stamps = times_all.size

    if keyframes is None:
        # pick ~9 keyframes spanning whatever range gridstate covers
        n_pick = min(9, n_stamps)
        idxs = np.linspace(0, n_stamps - 1, n_pick).round().astype(int)
    else:
        idxs = np.asarray(keyframes, dtype=int)

    lon = isk["lon"] - 180.0
    lat = isk["lat"]

    n_rows = len(idxs)
    n_cols = 6
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.4, n_rows * 1.5),
                             facecolor=(1, 0.98, 0.95))
    axes = np.atleast_2d(axes)

    col_titles = ["Topography (m)", "Lithology", "Air Temp (C)",
                  "Runoff (log mm/yr)", "Erosion (log t/km2/yr)", "Silw (log t/km2/yr)"]

    # Display oldest first (top row) -> present (bottom row): reverse so latest at bottom
    ordered = list(idxs)
    if times_all[ordered[0]] > times_all[ordered[-1]]:
        ordered = ordered[::-1]

    for row, gi in enumerate(ordered):
        land = gs["land"][:, :, gi]
        topo = gs["TOPO"][:, :, gi].astype(float)
        topo_masked = np.where(topo < 1, np.nan, topo)
        Tair = np.where(land == 0, np.nan, gs["Tair"][:, :, gi])
        Q = np.where(land == 0, np.nan, gs["Q"][:, :, gi])
        EPS = np.where(land == 0, np.nan, gs["EPSILON"][:, :, gi])
        CW = np.where(land == 0, np.nan, gs["CW"][:, :, gi])

        # Lithology composite (1=land, +1 relict, +2 arc, +3 suture)
        relict = np.where(gs["RELICT_ARC"][:, :, gi] > 0, 1.0, 0.0)
        arc = np.where(gs["ARC"][:, :, gi] > 0, 1.0, 0.0)
        suture = np.where(gs["SUTURE"][:, :, gi] > 0, 1.0, 0.0)
        litho = land + relict + 2.0 * arc + 3.0 * suture

        kwargs = dict(shading="auto")
        ax = axes[row, 0]
        ax.pcolormesh(lon, lat, topo_masked, cmap=CMAP_TOPO, vmin=-5000, vmax=5000, **kwargs)
        ax.set_facecolor((0.85, 0.92, 0.98))
        ax.text(0.02, 0.85, f"{int(round(times_all[gi]))} Ma",
                transform=ax.transAxes, fontsize=8, fontweight="bold",
                bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=1))

        ax = axes[row, 1]
        ax.pcolormesh(lon, lat, litho, cmap=CMAP_LITHO, vmin=0, vmax=4, **kwargs)

        ax = axes[row, 2]
        ax.pcolormesh(lon, lat, Tair, cmap=CMAP_TEMP, vmin=-40, vmax=40, **kwargs)

        ax = axes[row, 3]
        with np.errstate(divide="ignore", invalid="ignore"):
            ax.pcolormesh(lon, lat, np.log10(Q), cmap=CMAP_PRE, vmin=0, vmax=4, **kwargs)

        ax = axes[row, 4]
        with np.errstate(divide="ignore", invalid="ignore"):
            ax.pcolormesh(lon, lat, np.log10(EPS), cmap=CMAP_SEQ2, vmin=0, vmax=4, **kwargs)

        ax = axes[row, 5]
        with np.errstate(divide="ignore", invalid="ignore"):
            ax.pcolormesh(lon, lat, np.log10(CW), cmap=CMAP_SEQ, vmin=0, vmax=2, **kwargs)

        for c in range(n_cols):
            axes[row, c].set_xticks([]); axes[row, c].set_yticks([])
            axes[row, c].set_xlim(lon.min(), lon.max())
            axes[row, c].set_ylim(lat.min(), lat.max())

    for c, t in enumerate(col_titles):
        axes[0, c].set_title(t, fontsize=9)

    fig.subplots_adjust(left=0.01, right=0.99, top=0.96, bottom=0.02,
                        wspace=0.05, hspace=0.05)
    if save_path is not None:
        fig.savefig(save_path, dpi=140, facecolor=fig.get_facecolor())
    return fig


def plot_sens(npz_path, geochem_mat_path=None, save_path=None):
    """Reproduce SCION_plot_sens.m 5x2 panel figure from Python sensitivity .npz."""
    d = np.load(npz_path, allow_pickle=True)
    geochem = _load_geochem(geochem_mat_path) if geochem_mat_path else {}

    t = np.asarray(d["time_myr"])
    if t.ndim == 2:
        t = t[:, 0]

    c_mean = np.array([255, 132, 34]) / 255.0
    c_range = np.array([255, 225, 192]) / 255.0

    def stats(M):
        return (np.nanmean(M, axis=1),
                np.nanmin(M, axis=1),
                np.nanmax(M, axis=1))

    def envelope(ax, key, factor=1.0):
        M = np.asarray(d[key]) * factor
        mu, lo, hi = stats(M)
        ax.plot(t, mu, color=c_mean, linewidth=1.0)
        ax.plot(t, lo, color=c_range, linewidth=0.5)
        ax.plot(t, hi, color=c_range, linewidth=0.5)

    fig, axes = plt.subplots(5, 2, figsize=(14, 16),
                             facecolor=(0.80, 0.80, 0.70))
    fig.subplots_adjust(left=0.06, right=0.98, top=0.97, bottom=0.04,
                        wspace=0.22, hspace=0.42)

    # 1. Forcings
    ax = axes[0, 0]
    for key in ("DEGASS", "GRAN_AREA", "BAS_AREA"):
        envelope(ax, key)
    ax.set_xlabel("Time (Ma)"); ax.set_ylabel("Forcings"); ax.grid(False)

    # 2. delta 13C (Python sens lacks delta_mccb so use d13c_A as proxy)
    ax = axes[0, 1]
    if "d13c_x" in geochem:
        ax.plot(geochem["d13c_x"], geochem["d13c_y"], ".", color=PC2,
                markersize=2, alpha=0.5)
    key = "delta_mccb" if "delta_mccb" in d.files else "d13c_A"
    envelope(ax, key)
    ax.set_xlabel("Time (Ma)"); ax.set_ylabel(r"$\delta^{13}C_{carb}$")

    # 3. delta 34S
    ax = axes[1, 0]
    if "d34s_x" in geochem:
        ax.plot(geochem["d34s_x"], geochem["d34s_y"], ".", color=PC2,
                markersize=2, alpha=0.5)
    envelope(ax, "d34s_S")
    ax.set_xlabel("Time (Ma)"); ax.set_ylabel(r"$\delta^{34}S_{sw}$")

    # 4. 87Sr/86Sr
    ax = axes[1, 1]
    if "sr_x" in geochem:
        ax.plot(geochem["sr_x"], geochem["sr_y"], color=PC2, linewidth=1.0,
                alpha=0.7)
    envelope(ax, "delta_OSr")
    ax.set_ylim(0.706, 0.710)
    ax.set_xlabel("Time (Ma)"); ax.set_ylabel(r"$^{87}Sr/^{86}Sr$ seawater")

    # 5. SO4 (mM)
    ax = axes[2, 0]
    if "sconc_max_x" in geochem:
        ax.plot(geochem["sconc_max_x"], geochem["sconc_max_y"], color=PC1, linewidth=0.8)
        ax.plot(geochem["sconc_min_x"], geochem["sconc_min_y"], color=PC1, linewidth=0.8)
        ax.plot(geochem["sconc_mid_x"], geochem["sconc_mid_y"], color=PC2, linewidth=0.8)
    if "SO4_x" in geochem:
        SO4_x = geochem["SO4_x"]; SO4_y = geochem["SO4_y"]
        for u in range(0, len(SO4_x) - 1, 2):
            ax.plot([SO4_x[u], SO4_x[u]], [SO4_y[u], SO4_y[u + 1]],
                    color=PC3, linewidth=0.8)
    if "SmM" in d.files:
        envelope(ax, "SmM")
    else:
        S0 = PARS_DEFAULTS["S0"]
        M = (np.asarray(d["S"]) / S0) * 28.0
        mu, lo, hi = (np.nanmean(M, 1), np.nanmin(M, 1), np.nanmax(M, 1))
        ax.plot(t, mu, color=c_mean, linewidth=1.0)
        ax.plot(t, lo, color=c_range, linewidth=0.5)
        ax.plot(t, hi, color=c_range, linewidth=0.5)
    ax.set_xlabel("Time (Ma)"); ax.set_ylabel(r"Marine $SO_4$ (mM)")

    # 6. O2 (%)
    ax = axes[2, 1]
    if "O2_x" in geochem:
        O2_x = geochem["O2_x"]; O2_y = geochem["O2_y"]
        for u in range(0, len(O2_x) - 1, 2):
            ax.plot([O2_x[u], O2_x[u]], [O2_y[u], O2_y[u + 1]],
                    color=PC2, linewidth=0.8)
    envelope(ax, "mrO2", factor=100.0)
    ax.set_xlabel("Time (Ma)"); ax.set_ylabel(r"Atmospheric $O_2$ (%)")

    # 7. CO2 ppm
    ax = axes[3, 0]
    proxy_specs = [
        ("paleosol_age", "paleosol_co2", PC1),
        ("alkenone_age", "alkenone_co2", PC2),
        ("boron_age", "boron_co2", PC3),
        ("stomata_age", "stomata_co2", PC4),
        ("liverwort_age", "liverwort_co2", PC5),
        ("phytane_age", "phytane_co2", PC6),
    ]
    for ax_key, co2_key, color in proxy_specs:
        if ax_key in geochem:
            ax.plot(geochem[ax_key], geochem[co2_key], ".", color=color, markersize=3)
    if "CO2ppm" in d.files:
        envelope(ax, "CO2ppm")
    else:
        envelope(ax, "RCO2", factor=280.0)
    ax.set_yscale("log"); ax.set_ylim(100, 10000)
    ax.set_xlabel("Time (Ma)"); ax.set_ylabel(r"Atmospheric $CO_2$ (ppm)")

    # 8. GAST + tropical + equator
    ax = axes[3, 1]
    if "T_x" in geochem and "T_y" in geochem:
        ax.fill(np.atleast_1d(geochem["T_x"]).ravel(),
                np.atleast_1d(geochem["T_y"]).ravel(),
                color=PC2, alpha=0.4, edgecolor="none")
    gast_key = "T_gast" if "T_gast" in d.files else "GAST"
    envelope(ax, gast_key)
    if "SAT_tropical" in d.files:
        M = np.asarray(d["SAT_tropical"])
        mu = np.nanmean(M, 1)
        ax.plot(t, mu, color=c_mean, linewidth=1.0, linestyle=":")
    if "SAT_equator" in d.files:
        M = np.asarray(d["SAT_equator"])
        mu = np.nanmean(M, 1)
        ax.plot(t, mu, color=c_mean, linewidth=1.0, linestyle=":")
    ax.set_ylim(5, 40)
    ax.set_xlabel("Time (Ma)"); ax.set_ylabel("GAST (C)")

    # 9. Ice line
    ax = axes[4, 0]
    if "paleolat_x" in geochem:
        ax.plot(geochem["paleolat_x"], geochem["paleolat_y"], color=PC2, linewidth=1.0)
    envelope(ax, "iceline")
    ax.set_xlabel("Time (Ma)"); ax.set_ylabel("Ice line")

    # 10. P and N
    ax = axes[4, 1]
    P0 = PARS_DEFAULTS["P0"]; N0 = PARS_DEFAULTS["N0"]
    Pn = np.asarray(d["P"]) / (P0 if np.nanmax(d["P"]) > 100 else 1.0)
    Nn = np.asarray(d["N"]) / (N0 if np.nanmax(d["N"]) > 100 else 1.0)
    muP = np.nanmean(Pn, 1); loP = np.nanmin(Pn, 1); hiP = np.nanmax(Pn, 1)
    muN = np.nanmean(Nn, 1); loN = np.nanmin(Nn, 1); hiN = np.nanmax(Nn, 1)
    ax.plot(t, muP, color=c_mean, linewidth=1.0)
    ax.plot(t, loP, color=c_range, linewidth=0.5)
    ax.plot(t, hiP, color=c_range, linewidth=0.5)
    ax.plot(t, muN, color=c_mean, linewidth=1.0, linestyle="--")
    ax.plot(t, loN, color=c_range, linewidth=0.5, linestyle="--")
    ax.plot(t, hiN, color=c_range, linewidth=0.5, linestyle="--")
    ax.set_xlabel("Time (Ma)"); ax.set_ylabel("P (-), N (--)")

    n_kept = int(np.asarray(d["n_kept"]).item()) if "n_kept" in d.files else \
        np.asarray(d[next(iter(d.files))]).shape[1]
    fig.suptitle(f"SCION sensitivity ensemble (N={n_kept})", fontsize=12)

    if save_path is not None:
        fig.savefig(save_path, dpi=140, facecolor=fig.get_facecolor())
    return fig


def main():
    here = Path(__file__).resolve().parent
    repo = here.parent
    npz = here / "scion_python_baseline.npz"
    octave = here / "scion_octave_baseline.mat"
    geochem = repo / "data" / "geochem_data_2020.mat"
    scotese = repo / "data" / "Scotese_GAT_2021.mat"
    interpstack = repo / "forcings" / "INTERPSTACK_1Ga_weatherable_areas.mat"
    sens_npz = here / "scion_sens_results.npz"

    fig = plot_fluxes(
        python_npz_path=str(npz),
        octave_mat_path=str(octave) if octave.exists() else None,
        geochem_mat_path=str(geochem) if geochem.exists() else None,
        scotese_mat_path=str(scotese) if scotese.exists() else None,
        save_path=str(here / "figure_fluxes.png"),
    )
    plt.close(fig)
    print(f"wrote {here / 'figure_fluxes.png'}")

    if octave.exists() and interpstack.exists():
        fig = plot_worldgraphic(
            gridstate_source=str(octave),
            interpstack_path=str(interpstack),
            save_path=str(here / "figure_worldgraphic.png"),
        )
        plt.close(fig)
        print(f"wrote {here / 'figure_worldgraphic.png'}")

    if sens_npz.exists():
        fig = plot_sens(
            npz_path=str(sens_npz),
            geochem_mat_path=str(geochem) if geochem.exists() else None,
            save_path=str(here / "figure_sens_full.png"),
        )
        plt.close(fig)
        print(f"wrote {here / 'figure_sens_full.png'}")


if __name__ == "__main__":
    main()
