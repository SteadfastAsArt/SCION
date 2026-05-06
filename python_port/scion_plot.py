"""Reproduce SCION_plot_fluxes.m 4x4 panel figure from Python (and optional Octave) baselines."""

from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from scipy.io import loadmat

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


def main():
    here = Path(__file__).resolve().parent
    repo = here.parent
    npz = here / "scion_python_baseline.npz"
    octave = here / "scion_octave_baseline.mat"
    geochem = repo / "data" / "geochem_data_2020.mat"
    scotese = repo / "data" / "Scotese_GAT_2021.mat"
    out = here / "figure_fluxes.png"

    fig = plot_fluxes(
        python_npz_path=str(npz),
        octave_mat_path=str(octave) if octave.exists() else None,
        geochem_mat_path=str(geochem) if geochem.exists() else None,
        scotese_mat_path=str(scotese) if scotese.exists() else None,
        save_path=str(out),
    )
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
