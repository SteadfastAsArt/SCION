"""SCION Earth-system model — Python port of SCION_initialise.m + SCION_equations.m."""
from __future__ import annotations
import os
import time as _time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.io import loadmat

# ---------------------------------------------------------------------------
# Numba acceleration (with pure-numpy fallback)
# ---------------------------------------------------------------------------
try:
    from numba import njit
    _HAS_NUMBA = True
except Exception:  # pragma: no cover
    _HAS_NUMBA = False

    def njit(*args, **kwargs):  # type: ignore
        # Decorator factory fallback: returns a no-op decorator.
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        def _wrap(fn):
            return fn
        return _wrap


if _HAS_NUMBA:
    @njit(cache=True, fastmath=False)
    def _interp1_nb(x, y, xq):
        """Linear 1-D interpolation matching numpy.interp behaviour for ascending x."""
        n = x.shape[0]
        if xq <= x[0]:
            return y[0]
        if xq >= x[n - 1]:
            return y[n - 1]
        lo = 0
        hi = n - 1
        while hi - lo > 1:
            mid = (hi + lo) >> 1
            if x[mid] <= xq:
                lo = mid
            else:
                hi = mid
        x0 = x[lo]; x1 = x[hi]
        y0 = y[lo]; y1 = y[hi]
        if x1 == x0:
            return y0
        return y0 + (y1 - y0) * (xq - x0) / (x1 - x0)
else:
    def _interp1_nb(x, y, xq):  # pragma: no cover
        return float(np.interp(xq, x, y))


@njit(cache=True, fastmath=False)
def _spatial_weathering_full(
    t_geol, CO2ppm,
    IS_time, IS_CO2, IS_lat,
    IS_runoff, IS_Tair, IS_land,
    IS_slope, IS_arc, IS_relict_arc, IS_suture, IS_gridarea,
    rel_contrib,
    ARCfactor, SUTUREfactor,
    k_erosion, Xm, Kw, kw, Ea, z, sigplus1, T0, R, Tcrit,
    k_carb_scale, k_basw, k_granw, silw_scale,
):
    """Full spatial weathering done inside numba (fastest path).

    Reductions are linear accumulations; this can introduce associativity
    differences vs numpy at the ~1e-15 relative level per element which the
    BDF solver may amplify into ~1e-6 trajectory differences over 1Gyr.
    For bit-exact (numpy-order) reductions use _spatial_weathering_split.
    """
    nT = IS_time.shape[0]
    nC = IS_CO2.shape[0]
    nlat = IS_runoff.shape[0]
    nlon = IS_runoff.shape[1]

    # ---- time keyframe selection ----
    has_past = False
    key_past_time = 0.0
    for i in range(nT):
        v = IS_time[i] - t_geol
        if v <= 0.0:
            if (not has_past) or IS_time[i] > key_past_time:
                key_past_time = IS_time[i]
                has_past = True
    has_future = False
    key_future_time = 0.0
    for i in range(nT):
        v = IS_time[i] - t_geol
        if v >= 0.0:
            if (not has_future) or IS_time[i] < key_future_time:
                key_future_time = IS_time[i]
                has_future = True
    if not has_past:
        key_past_time = key_future_time
    if not has_future:
        key_future_time = key_past_time
    key_past_index = 0
    for i in range(nT):
        if IS_time[i] == key_past_time:
            key_past_index = i
            break
    key_future_index = 0
    for i in range(nT):
        if IS_time[i] == key_future_time:
            key_future_index = i
            break
    dist_to_past = abs(key_past_time - t_geol)
    dist_to_future = abs(key_future_time - t_geol)
    denom = dist_to_past + dist_to_future
    if denom == 0.0:
        contribution_past = 1.0
        contribution_future = 0.0
    else:
        contribution_past = dist_to_future / denom
        contribution_future = dist_to_past / denom

    # ---- CO2 keyframes ----
    has_upper = False
    key_upper_CO2 = 0.0
    for i in range(nC):
        d = IS_CO2[i] - CO2ppm
        if d >= 0.0:
            if (not has_upper) or IS_CO2[i] < key_upper_CO2:
                key_upper_CO2 = IS_CO2[i]
                has_upper = True
    has_lower = False
    key_lower_CO2 = 0.0
    for i in range(nC):
        d = IS_CO2[i] - CO2ppm
        if d <= 0.0:
            if (not has_lower) or IS_CO2[i] > key_lower_CO2:
                key_lower_CO2 = IS_CO2[i]
                has_lower = True
    if not has_upper:
        key_upper_CO2 = key_lower_CO2
    if not has_lower:
        key_lower_CO2 = key_upper_CO2
    key_upper_CO2_index = 0
    for i in range(nC):
        if IS_CO2[i] == key_upper_CO2:
            key_upper_CO2_index = i
            break
    key_lower_CO2_index = 0
    for i in range(nC):
        if IS_CO2[i] == key_lower_CO2:
            key_lower_CO2_index = i
            break
    dist_to_upper = abs(key_upper_CO2 - CO2ppm)
    dist_to_lower = abs(key_lower_CO2 - CO2ppm)
    denom2 = dist_to_upper + dist_to_lower
    if denom2 == 0.0:
        contribution_lower = 1.0
        contribution_upper = 0.0
    else:
        contribution_upper = dist_to_lower / denom2
        contribution_lower = dist_to_upper / denom2

    # ---- spatial accumulation loop ----
    erosion_tot_past = 0.0
    erosion_tot_future = 0.0
    CW_sum_past = 0.0
    CW_sum_future = 0.0
    CWcarb_sum_past = 0.0
    CWcarb_sum_future = 0.0
    GAST_past_acc = 0.0
    GAST_future_acc = 0.0
    SATtrop_past_acc = 0.0
    SATtrop_future_acc = 0.0
    SATeq_past_acc = 0.0
    SATeq_future_acc = 0.0
    ngrid = nlat * nlon
    nlat_eq_count = 2 * nlon
    nlat_trop_count = 12 * nlon
    inv_T0R = Ea / (R * T0)

    iceline_past = 90.0
    iceline_future = 90.0

    for i in range(nlat):
        latband_past_acc = 0.0
        latband_future_acc = 0.0
        for j in range(nlon):
            ru_p = (contribution_upper * IS_runoff[i, j, key_upper_CO2_index, key_past_index]
                    + contribution_lower * IS_runoff[i, j, key_lower_CO2_index, key_past_index])
            ru_f = (contribution_upper * IS_runoff[i, j, key_upper_CO2_index, key_future_index]
                    + contribution_lower * IS_runoff[i, j, key_lower_CO2_index, key_future_index])
            ta_p = (contribution_upper * IS_Tair[i, j, key_upper_CO2_index, key_past_index]
                    + contribution_lower * IS_Tair[i, j, key_lower_CO2_index, key_past_index])
            ta_f = (contribution_upper * IS_Tair[i, j, key_upper_CO2_index, key_future_index]
                    + contribution_lower * IS_Tair[i, j, key_lower_CO2_index, key_future_index])

            slope_p = IS_slope[i, j, key_past_index]
            slope_f = IS_slope[i, j, key_future_index]
            arc_p = IS_arc[i, j, key_past_index]
            arc_f = IS_arc[i, j, key_future_index]
            rarc_p = IS_relict_arc[i, j, key_past_index]
            rarc_f = IS_relict_arc[i, j, key_future_index]
            sut_p = IS_suture[i, j, key_past_index]
            sut_f = IS_suture[i, j, key_future_index]
            land_p = IS_land[i, j, key_past_index]
            land_f = IS_land[i, j, key_future_index]
            ga = IS_gridarea[i, j]
            rc = rel_contrib[i, j]

            if ru_p < 0.0:
                q_p = 0.0
            else:
                q_p = ru_p
            if ru_f < 0.0:
                q_f = 0.0
            else:
                q_f = ru_f
            tk_p = ta_p + 273.0
            tk_f = ta_f + 273.0

            tair_p_eff = ta_p if ta_p > 2.0 else 2.0
            tair_f_eff = ta_f if ta_f > 2.0 else 2.0

            eps_p = ((k_erosion * (q_p ** 0.31)) * slope_p) * tair_p_eff
            eps_f = ((k_erosion * (q_f ** 0.31)) * slope_f) * tair_f_eff
            eps_box_p = eps_p * ga * 1e6
            eps_box_f = eps_f * ga * 1e6
            if eps_box_p == eps_box_p:
                erosion_tot_past += eps_box_p
            if eps_box_f == eps_box_f:
                erosion_tot_future += eps_box_f

            R_T_p = np.exp(inv_T0R - (Ea / (R * tk_p)))
            R_T_f = np.exp(inv_T0R - (Ea / (R * tk_f)))
            R_Q_p = 1.0 - np.exp(-kw * q_p)
            R_Q_f = 1.0 - np.exp(-kw * q_f)
            if eps_p == 0.0:
                R_reg_p = np.inf
            else:
                R_reg_p = ((z / eps_p) ** sigplus1) / sigplus1
            if eps_f == 0.0:
                R_reg_f = np.inf
            else:
                R_reg_f = ((z / eps_f) ** sigplus1) / sigplus1
            cw_per_p = 1e6 * eps_p * Xm * (1.0 - np.exp(-Kw * R_Q_p * R_T_p * R_reg_p))
            cw_per_f = 1e6 * eps_f * Xm * (1.0 - np.exp(-Kw * R_Q_f * R_T_f * R_reg_f))

            arc_plus_p = (ARCfactor - 1.0) * arc_p
            arc_plus_f = (ARCfactor - 1.0) * arc_f
            rarc_plus_p = (ARCfactor - 1.0) * rarc_p
            rarc_plus_f = (ARCfactor - 1.0) * rarc_f
            sut_plus_p = (SUTUREfactor - 1.0) * sut_p
            sut_plus_f = (SUTUREfactor - 1.0) * sut_f
            cw_per_p = cw_per_p * (1.0 + arc_plus_p + rarc_plus_p + sut_plus_p)
            cw_per_f = cw_per_f * (1.0 + arc_plus_f + rarc_plus_f + sut_plus_f)
            cw_p = cw_per_p * ga
            cw_f = cw_per_f * ga
            if cw_p == cw_p:
                CW_sum_past += cw_p
            if cw_f == cw_f:
                CW_sum_future += cw_f

            cwcarb_per_p = k_carb_scale * q_p
            cwcarb_per_f = k_carb_scale * q_f
            cwcarb_p = cwcarb_per_p * ga
            cwcarb_f = cwcarb_per_f * ga
            if cwcarb_p == cwcarb_p:
                CWcarb_sum_past += cwcarb_p
            if cwcarb_f == cwcarb_f:
                CWcarb_sum_future += cwcarb_f

            GAST_past_acc += ta_p * rc
            GAST_future_acc += ta_f * rc
            if i >= 14 and i < 26:
                SATtrop_past_acc += ta_p * rc * 0.67
                SATtrop_future_acc += ta_f * rc * 0.67
            if i >= 19 and i < 21:
                SATeq_past_acc += ta_p
                SATeq_future_acc += ta_f

            if ta_p < Tcrit:
                latband_past_acc += land_p
            if ta_f < Tcrit:
                latband_future_acc += land_f

        # latband ice line: collapse by lat row
        if latband_past_acc > 0.0:
            v = abs(IS_lat[i])
            if v < iceline_past:
                iceline_past = v
        if latband_future_acc > 0.0:
            v = abs(IS_lat[i])
            if v < iceline_future:
                iceline_future = v

    erosion_tot = erosion_tot_past * contribution_past + erosion_tot_future * contribution_future
    CW_tot = CW_sum_past * contribution_past + CW_sum_future * contribution_future
    silw_spatial = CW_tot * ((k_basw + k_granw) / silw_scale)
    carbw_spatial = CWcarb_sum_past * contribution_past + CWcarb_sum_future * contribution_future

    GAST = GAST_past_acc / ngrid * contribution_past + GAST_future_acc / ngrid * contribution_future
    SAT_tropical = SATtrop_past_acc / nlat_trop_count * contribution_past + SATtrop_future_acc / nlat_trop_count * contribution_future
    SAT_equator = SATeq_past_acc / nlat_eq_count * contribution_past + SATeq_future_acc / nlat_eq_count * contribution_future
    iceline = iceline_past * contribution_past + iceline_future * contribution_future

    return (silw_spatial, carbw_spatial, GAST, SAT_tropical, SAT_equator,
            iceline, erosion_tot, contribution_past, contribution_future,
            key_past_index, key_future_index)


@njit(cache=True, fastmath=False)
def _spatial_weathering_kernel(
    t_geol, CO2ppm,
    IS_time, IS_CO2,
    IS_runoff, IS_Tair,
    IS_slope, IS_arc, IS_relict_arc, IS_suture,
    ARCfactor, SUTUREfactor,
    k_erosion, Xm, Kw, kw, Ea, z, sigplus1, T0, R,
    out_eps_p, out_eps_f,
    out_cw_per_p, out_cw_per_f,
    out_q_p, out_q_f,
    out_Tair_p, out_Tair_f,
):
    """Compute keyframes + per-cell weathering fields. Reductions are done
    in numpy (afterwards) so that summation order matches the original code.

    Returns (contribution_past, contribution_future, key_past_index, key_future_index).
    """
    nT = IS_time.shape[0]
    nC = IS_CO2.shape[0]
    nlat = IS_runoff.shape[0]
    nlon = IS_runoff.shape[1]

    # ---- time keyframe selection ----
    has_past = False
    key_past_time = 0.0
    for i in range(nT):
        v = IS_time[i] - t_geol
        if v <= 0.0:
            if (not has_past) or IS_time[i] > key_past_time:
                key_past_time = IS_time[i]
                has_past = True
    has_future = False
    key_future_time = 0.0
    for i in range(nT):
        v = IS_time[i] - t_geol
        if v >= 0.0:
            if (not has_future) or IS_time[i] < key_future_time:
                key_future_time = IS_time[i]
                has_future = True
    if not has_past:
        key_past_time = key_future_time
    if not has_future:
        key_future_time = key_past_time
    key_past_index = 0
    for i in range(nT):
        if IS_time[i] == key_past_time:
            key_past_index = i
            break
    key_future_index = 0
    for i in range(nT):
        if IS_time[i] == key_future_time:
            key_future_index = i
            break
    dist_to_past = abs(key_past_time - t_geol)
    dist_to_future = abs(key_future_time - t_geol)
    denom = dist_to_past + dist_to_future
    if denom == 0.0:
        contribution_past = 1.0
        contribution_future = 0.0
    else:
        contribution_past = dist_to_future / denom
        contribution_future = dist_to_past / denom

    # ---- CO2 keyframes ----
    has_upper = False
    key_upper_CO2 = 0.0
    for i in range(nC):
        d = IS_CO2[i] - CO2ppm
        if d >= 0.0:
            if (not has_upper) or IS_CO2[i] < key_upper_CO2:
                key_upper_CO2 = IS_CO2[i]
                has_upper = True
    has_lower = False
    key_lower_CO2 = 0.0
    for i in range(nC):
        d = IS_CO2[i] - CO2ppm
        if d <= 0.0:
            if (not has_lower) or IS_CO2[i] > key_lower_CO2:
                key_lower_CO2 = IS_CO2[i]
                has_lower = True
    if not has_upper:
        key_upper_CO2 = key_lower_CO2
    if not has_lower:
        key_lower_CO2 = key_upper_CO2
    key_upper_CO2_index = 0
    for i in range(nC):
        if IS_CO2[i] == key_upper_CO2:
            key_upper_CO2_index = i
            break
    key_lower_CO2_index = 0
    for i in range(nC):
        if IS_CO2[i] == key_lower_CO2:
            key_lower_CO2_index = i
            break
    dist_to_upper = abs(key_upper_CO2 - CO2ppm)
    dist_to_lower = abs(key_lower_CO2 - CO2ppm)
    denom2 = dist_to_upper + dist_to_lower
    if denom2 == 0.0:
        contribution_lower = 1.0
        contribution_upper = 0.0
    else:
        contribution_upper = dist_to_lower / denom2
        contribution_lower = dist_to_upper / denom2

    # ---- per-cell field computation ----
    # Mirrors the exact arithmetic of the original numpy code, in the same
    # left-to-right order, so that the only difference is the final reduction.
    inv_T0R = Ea / (R * T0)
    for i in range(nlat):
        for j in range(nlon):
            # interpolated runoff/Tair (same expr as numpy)
            ru_p = (contribution_upper * IS_runoff[i, j, key_upper_CO2_index, key_past_index]
                    + contribution_lower * IS_runoff[i, j, key_lower_CO2_index, key_past_index])
            ru_f = (contribution_upper * IS_runoff[i, j, key_upper_CO2_index, key_future_index]
                    + contribution_lower * IS_runoff[i, j, key_lower_CO2_index, key_future_index])
            ta_p = (contribution_upper * IS_Tair[i, j, key_upper_CO2_index, key_past_index]
                    + contribution_lower * IS_Tair[i, j, key_lower_CO2_index, key_past_index])
            ta_f = (contribution_upper * IS_Tair[i, j, key_upper_CO2_index, key_future_index]
                    + contribution_lower * IS_Tair[i, j, key_lower_CO2_index, key_future_index])

            slope_p = IS_slope[i, j, key_past_index]
            slope_f = IS_slope[i, j, key_future_index]
            arc_p = IS_arc[i, j, key_past_index]
            arc_f = IS_arc[i, j, key_future_index]
            rarc_p = IS_relict_arc[i, j, key_past_index]
            rarc_f = IS_relict_arc[i, j, key_future_index]
            sut_p = IS_suture[i, j, key_past_index]
            sut_f = IS_suture[i, j, key_future_index]

            # Q: numpy `np.where(RUNOFF<0, 0, RUNOFF)` keeps NaN.
            if ru_p < 0.0:
                q_p = 0.0
            else:
                q_p = ru_p
            if ru_f < 0.0:
                q_f = 0.0
            else:
                q_f = ru_f

            tk_p = ta_p + 273.0
            tk_f = ta_f + 273.0

            tair_p_eff = ta_p if ta_p > 2.0 else 2.0
            tair_f_eff = ta_f if ta_f > 2.0 else 2.0

            # EPSILON = ((k_erosion * Q^0.31) * slope) * max(Tair, 2)
            eps_p = ((k_erosion * (q_p ** 0.31)) * slope_p) * tair_p_eff
            eps_f = ((k_erosion * (q_f ** 0.31)) * slope_f) * tair_f_eff

            # Maffre weathering, mirroring numpy associativity exactly.
            # Note: Ea/(R*T0) - Ea/(R*T) and z/EPS, both follow numpy ordering.
            R_T_p = np.exp(inv_T0R - (Ea / (R * tk_p)))
            R_T_f = np.exp(inv_T0R - (Ea / (R * tk_f)))
            R_Q_p = 1.0 - np.exp(-kw * q_p)
            R_Q_f = 1.0 - np.exp(-kw * q_f)
            # R_reg = ((z / EPS) ** sigplus1) / sigplus1; numpy with EPS=0 gives
            # +inf for positive z. Numba raises on integer/0; we explicitly emit
            # +inf to match numpy semantics. eps_p NaN propagates naturally.
            if eps_p == 0.0:
                R_reg_p = np.inf
            else:
                R_reg_p = ((z / eps_p) ** sigplus1) / sigplus1
            if eps_f == 0.0:
                R_reg_f = np.inf
            else:
                R_reg_f = ((z / eps_f) ** sigplus1) / sigplus1
            # 1 - exp(-Kw * R_Q * R_T * R_reg) where R_reg may be +inf and R_Q
            # may be 0 -> 0*inf = NaN in numpy too; handled identically here.
            cw_per_p = 1e6 * eps_p * Xm * (1.0 - np.exp(-Kw * R_Q_p * R_T_p * R_reg_p))
            cw_per_f = 1e6 * eps_f * Xm * (1.0 - np.exp(-Kw * R_Q_f * R_T_f * R_reg_f))

            # ARC/SUTURE multipliers (exact numpy order)
            arc_plus_p = (ARCfactor - 1.0) * arc_p
            arc_plus_f = (ARCfactor - 1.0) * arc_f
            rarc_plus_p = (ARCfactor - 1.0) * rarc_p
            rarc_plus_f = (ARCfactor - 1.0) * rarc_f
            sut_plus_p = (SUTUREfactor - 1.0) * sut_p
            sut_plus_f = (SUTUREfactor - 1.0) * sut_f
            cw_per_p = cw_per_p * (1.0 + arc_plus_p + rarc_plus_p + sut_plus_p)
            cw_per_f = cw_per_f * (1.0 + arc_plus_f + rarc_plus_f + sut_plus_f)

            out_eps_p[i, j] = eps_p
            out_eps_f[i, j] = eps_f
            out_cw_per_p[i, j] = cw_per_p
            out_cw_per_f[i, j] = cw_per_f
            out_q_p[i, j] = q_p
            out_q_f[i, j] = q_f
            out_Tair_p[i, j] = ta_p
            out_Tair_f[i, j] = ta_f

    return (contribution_past, contribution_future, key_past_index, key_future_index)


def _spatial_weathering_kernel_numpy(
    t_geol, CO2ppm,
    IS_time, IS_CO2,
    IS_runoff, IS_Tair,
    IS_slope, IS_arc, IS_relict_arc, IS_suture,
    ARCfactor, SUTUREfactor,
    k_erosion, Xm, Kw, kw, Ea, z, sigplus1, T0, R,
    out_eps_p, out_eps_f, out_cw_per_p, out_cw_per_f,
    out_q_p, out_q_f, out_Tair_p, out_Tair_f,
):
    """Pure-numpy fallback. Same outputs and ordering as the numba kernel."""
    # ---- time keyframe selection ----
    diff_time = IS_time - t_geol
    past_mask = diff_time <= 0
    future_mask = diff_time >= 0
    if past_mask.any():
        key_past_time = IS_time[past_mask].max()
    else:
        key_past_time = IS_time[future_mask].min()
    if future_mask.any():
        key_future_time = IS_time[future_mask].min()
    else:
        key_future_time = key_past_time
    key_past_index = int(np.where(IS_time == key_past_time)[0][0])
    key_future_index = int(np.where(IS_time == key_future_time)[0][0])
    dist_to_past = abs(key_past_time - t_geol)
    dist_to_future = abs(key_future_time - t_geol)
    if dist_to_past + dist_to_future == 0:
        contribution_past = 1.0; contribution_future = 0.0
    else:
        contribution_past = dist_to_future / (dist_to_past + dist_to_future)
        contribution_future = dist_to_past / (dist_to_past + dist_to_future)

    # ---- CO2 keyframes ----
    upper_mask = (IS_CO2 - CO2ppm) >= 0
    lower_mask = (IS_CO2 - CO2ppm) <= 0
    if upper_mask.any():
        key_upper_CO2 = IS_CO2[upper_mask].min()
    else:
        key_upper_CO2 = IS_CO2[lower_mask].max()
    if lower_mask.any():
        key_lower_CO2 = IS_CO2[lower_mask].max()
    else:
        key_lower_CO2 = key_upper_CO2
    key_upper_CO2_index = int(np.where(IS_CO2 == key_upper_CO2)[0][0])
    key_lower_CO2_index = int(np.where(IS_CO2 == key_lower_CO2)[0][0])
    dist_to_upper = abs(key_upper_CO2 - CO2ppm)
    dist_to_lower = abs(key_lower_CO2 - CO2ppm)
    if dist_to_upper + dist_to_lower == 0:
        contribution_lower = 1.0; contribution_upper = 0.0
    else:
        contribution_upper = dist_to_lower / (dist_to_upper + dist_to_lower)
        contribution_lower = dist_to_upper / (dist_to_upper + dist_to_lower)

    RUNOFF_past = (contribution_upper * IS_runoff[:, :, key_upper_CO2_index, key_past_index]
                   + contribution_lower * IS_runoff[:, :, key_lower_CO2_index, key_past_index])
    RUNOFF_future = (contribution_upper * IS_runoff[:, :, key_upper_CO2_index, key_future_index]
                     + contribution_lower * IS_runoff[:, :, key_lower_CO2_index, key_future_index])
    Tair_past = (contribution_upper * IS_Tair[:, :, key_upper_CO2_index, key_past_index]
                 + contribution_lower * IS_Tair[:, :, key_lower_CO2_index, key_past_index])
    Tair_future = (contribution_upper * IS_Tair[:, :, key_upper_CO2_index, key_future_index]
                   + contribution_lower * IS_Tair[:, :, key_lower_CO2_index, key_future_index])
    tslope_past = IS_slope[:, :, key_past_index]
    tslope_future = IS_slope[:, :, key_future_index]
    ARC_past = IS_arc[:, :, key_past_index]
    ARC_future = IS_arc[:, :, key_future_index]
    RELICT_ARC_past = IS_relict_arc[:, :, key_past_index]
    RELICT_ARC_future = IS_relict_arc[:, :, key_future_index]
    SUTURE_past = IS_suture[:, :, key_past_index]
    SUTURE_future = IS_suture[:, :, key_future_index]

    Q_past = np.where(RUNOFF_past < 0, 0.0, RUNOFF_past)
    Q_future = np.where(RUNOFF_future < 0, 0.0, RUNOFF_future)

    EPSILON_past = k_erosion * (Q_past ** 0.31) * tslope_past * np.maximum(Tair_past, 2.0)
    EPSILON_future = k_erosion * (Q_future ** 0.31) * tslope_future * np.maximum(Tair_future, 2.0)

    with np.errstate(divide='ignore', invalid='ignore'):
        T_past = Tair_past + 273.0
        T_future = Tair_future + 273.0
        R_T_past = np.exp((Ea / (R * T0)) - (Ea / (R * T_past)))
        R_T_future = np.exp((Ea / (R * T0)) - (Ea / (R * T_future)))
        R_Q_past = 1 - np.exp(-kw * Q_past)
        R_Q_future = 1 - np.exp(-kw * Q_future)
        R_reg_past = ((z / EPSILON_past) ** sigplus1) / sigplus1
        R_reg_future = ((z / EPSILON_future) ** sigplus1) / sigplus1
        CW_per_km2_past = 1e6 * EPSILON_past * Xm * (1 - np.exp(-Kw * R_Q_past * R_T_past * R_reg_past))
        CW_per_km2_future = 1e6 * EPSILON_future * Xm * (1 - np.exp(-Kw * R_Q_future * R_T_future * R_reg_future))

    ARC_plusmap_past = (ARCfactor - 1) * ARC_past
    ARC_plusmap_future = (ARCfactor - 1) * ARC_future
    RELICT_ARC_plusmap_past = (ARCfactor - 1) * RELICT_ARC_past
    RELICT_ARC_plusmap_future = (ARCfactor - 1) * RELICT_ARC_future
    SUTURE_plusmap_past = (SUTUREfactor - 1) * SUTURE_past
    SUTURE_plusmap_future = (SUTUREfactor - 1) * SUTURE_future
    CW_per_km2_past = CW_per_km2_past * (1 + ARC_plusmap_past + RELICT_ARC_plusmap_past + SUTURE_plusmap_past)
    CW_per_km2_future = CW_per_km2_future * (1 + ARC_plusmap_future + RELICT_ARC_plusmap_future + SUTURE_plusmap_future)

    out_eps_p[:] = EPSILON_past
    out_eps_f[:] = EPSILON_future
    out_cw_per_p[:] = CW_per_km2_past
    out_cw_per_f[:] = CW_per_km2_future
    out_q_p[:] = Q_past
    out_q_f[:] = Q_future
    out_Tair_p[:] = Tair_past
    out_Tair_f[:] = Tair_future

    return (contribution_past, contribution_future, key_past_index, key_future_index)


def _spatial_weathering(
    t_geol, CO2ppm,
    IS_time, IS_CO2, IS_lat,
    IS_runoff, IS_Tair, IS_topo, IS_land,
    IS_slope, IS_arc, IS_relict_arc, IS_suture, IS_gridarea,
    rel_contrib,
    ARCfactor, SUTUREfactor,
    k_erosion, Xm, Kw, kw, Ea, z, sigplus1, T0, R, Tcrit,
    k_carb_scale, k_basw, k_granw, silw_scale,
    _bufs=None,
):
    """Spatial weathering, GAST, ice-line, and key-frame selection.

    Dispatches to the numba 'full' kernel when available (fastest), or the
    numpy reference implementation otherwise.

    Returns (silw_spatial, carbw_spatial, GAST, SAT_tropical, SAT_equator,
             iceline, erosion_tot, contribution_past, contribution_future,
             key_past_index, key_future_index).
    """
    if _HAS_NUMBA:
        return _spatial_weathering_full(
            t_geol, CO2ppm,
            IS_time, IS_CO2, IS_lat,
            IS_runoff, IS_Tair, IS_land,
            IS_slope, IS_arc, IS_relict_arc, IS_suture, IS_gridarea,
            rel_contrib,
            ARCfactor, SUTUREfactor,
            k_erosion, Xm, Kw, kw, Ea, z, sigplus1, T0, R, Tcrit,
            k_carb_scale, k_basw, k_granw, silw_scale,
        )
    # numba fallback path: numpy kernel + numpy reductions
    # Allocate or reuse buffers (one set per closure)
    if _bufs is None:
        nlat = IS_runoff.shape[0]
        nlon = IS_runoff.shape[1]
        _bufs = {
            'eps_p': np.empty((nlat, nlon)), 'eps_f': np.empty((nlat, nlon)),
            'cw_per_p': np.empty((nlat, nlon)), 'cw_per_f': np.empty((nlat, nlon)),
            'q_p': np.empty((nlat, nlon)), 'q_f': np.empty((nlat, nlon)),
            'tair_p': np.empty((nlat, nlon)), 'tair_f': np.empty((nlat, nlon)),
        }
    eps_p = _bufs['eps_p']; eps_f = _bufs['eps_f']
    cw_per_p = _bufs['cw_per_p']; cw_per_f = _bufs['cw_per_f']
    q_p = _bufs['q_p']; q_f = _bufs['q_f']
    Tair_past = _bufs['tair_p']; Tair_future = _bufs['tair_f']

    (contribution_past, contribution_future,
     key_past_index, key_future_index) = _spatial_weathering_kernel_numpy(
        t_geol, CO2ppm,
        IS_time, IS_CO2,
        IS_runoff, IS_Tair,
        IS_slope, IS_arc, IS_relict_arc, IS_suture,
        ARCfactor, SUTUREfactor,
        k_erosion, Xm, Kw, kw, Ea, z, sigplus1, T0, R,
        eps_p, eps_f, cw_per_p, cw_per_f, q_p, q_f, Tair_past, Tair_future,
    )

    GRID_AREA_km2 = IS_gridarea
    land_past = IS_land[:, :, key_past_index]
    land_future = IS_land[:, :, key_future_index]

    # erosion totals (numpy nansum, matching original code)
    EPS_per_box_past = eps_p * GRID_AREA_km2 * 1e6
    EPS_per_box_future = eps_f * GRID_AREA_km2 * 1e6
    erosion_tot_past = np.nansum(EPS_per_box_past)
    erosion_tot_future = np.nansum(EPS_per_box_future)
    erosion_tot = erosion_tot_past * contribution_past + erosion_tot_future * contribution_future

    # CW totals
    CW_past = cw_per_p * GRID_AREA_km2
    CW_future = cw_per_f * GRID_AREA_km2
    CW_past = np.where(np.isnan(CW_past), 0.0, CW_past)
    CW_future = np.where(np.isnan(CW_future), 0.0, CW_future)
    CW_sum_past = CW_past.sum()
    CW_sum_future = CW_future.sum()
    CW_tot = CW_sum_past * contribution_past + CW_sum_future * contribution_future
    silw_spatial = CW_tot * ((k_basw + k_granw) / silw_scale)

    # carbonate weathering totals
    CWcarb_per_km2_past = k_carb_scale * q_p
    CWcarb_per_km2_future = k_carb_scale * q_f
    CWcarb_past = CWcarb_per_km2_past * GRID_AREA_km2
    CWcarb_future = CWcarb_per_km2_future * GRID_AREA_km2
    CWcarb_past = np.where(np.isnan(CWcarb_past), 0.0, CWcarb_past)
    CWcarb_future = np.where(np.isnan(CWcarb_future), 0.0, CWcarb_future)
    CWcarb_sum_past = CWcarb_past.sum()
    CWcarb_sum_future = CWcarb_future.sum()
    carbw_spatial = CWcarb_sum_past * contribution_past + CWcarb_sum_future * contribution_future

    # GAST + SAT
    GAST = (np.mean(Tair_past * rel_contrib) * contribution_past
            + np.mean(Tair_future * rel_contrib) * contribution_future)
    SAT_tropical = (np.mean(Tair_past[14:26, :] * rel_contrib[14:26, :] * 0.67) * contribution_past
                    + np.mean(Tair_future[14:26, :] * rel_contrib[14:26, :] * 0.67) * contribution_future)
    SAT_equator = (np.mean(Tair_past[19:21, :]) * contribution_past
                   + np.mean(Tair_future[19:21, :]) * contribution_future)

    # ice line (numpy)
    Tair_past_ice = np.where(Tair_past < Tcrit, 1.0, 0.0) * land_past
    Tair_future_ice = np.where(Tair_future < Tcrit, 1.0, 0.0) * land_future
    latbands_past = Tair_past_ice.sum(axis=1)
    latbands_future = Tair_future_ice.sum(axis=1)
    latbands_past = np.where(latbands_past > 0, 1.0, 0.0)
    latbands_future = np.where(latbands_future > 0, 1.0, 0.0)
    latresults_past = IS_lat * latbands_past
    latresults_future = IS_lat * latbands_future
    latresults_past = np.where(latresults_past == 0, 90.0, latresults_past)
    latresults_future = np.where(latresults_future == 0, 90.0, latresults_future)
    iceline_past = np.min(np.abs(latresults_past))
    iceline_future = np.min(np.abs(latresults_future))
    iceline = iceline_past * contribution_past + iceline_future * contribution_future

    return (silw_spatial, carbw_spatial, GAST, SAT_tropical, SAT_equator,
            iceline, erosion_tot, contribution_past, contribution_future,
            key_past_index, key_future_index)


# ---------------------------------------------------------------------------
# Parameters & forcings loading
# ---------------------------------------------------------------------------

@dataclass
class Pars:
    # populated below
    pass


def _build_pars() -> dict:
    p: dict = {}
    p['k_reductant_input'] = 0.4e12
    p['k_locb'] = 2.5e12
    p['k_mocb'] = 2.5e12
    p['k_ocdeg'] = 1.25e12
    p['k_ccdeg'] = 12e12
    p['k_carbw'] = 8e12
    p['k_sfw'] = 1.75e12
    p['k_mccb'] = p['k_carbw'] + p['k_ccdeg'] - p['k_sfw']
    p['k_silw'] = p['k_mccb'] - p['k_carbw']
    p['basfrac'] = 0.3
    p['k_granw'] = p['k_silw'] * (1 - p['basfrac'])
    p['k_basw'] = p['k_silw'] * p['basfrac']
    p['k_mpsb'] = 0.7e12
    p['k_mgsb'] = 1e12
    p['k_pyrw'] = 7e11
    p['k_gypw'] = 1e12
    p['k_pyrdeg'] = 0.0
    p['k_gypdeg'] = 0.0
    p['k_capb'] = 2e10
    p['k_fepb'] = 1e10
    p['k_mopb'] = 1e10
    p['k_phosw'] = 4.25e10
    p['k_landfrac'] = 0.0588
    p['k_nfix'] = 8.67e12
    p['k_denit'] = 4.3e12
    p['k_oxidw'] = p['k_mocb'] + p['k_locb'] - p['k_ocdeg'] - p['k_reductant_input']
    p['k_Sr_sedw'] = 17e9
    p['k_Sr_mantle'] = 7.3e9
    p['k_Sr_silw'] = 13e9
    p['k_Sr_granw'] = p['k_Sr_silw'] * (1 - p['basfrac'])
    p['k_Sr_basw'] = p['k_Sr_silw'] * p['basfrac']
    p['total_Sr_removal'] = p['k_Sr_granw'] + p['k_Sr_basw'] + p['k_Sr_sedw'] + p['k_Sr_mantle']
    p['k_Sr_sfw'] = p['total_Sr_removal'] * (p['k_sfw'] / (p['k_sfw'] + p['k_mccb']))
    p['k_Sr_sedb'] = p['total_Sr_removal'] * (p['k_mccb'] / (p['k_sfw'] + p['k_mccb']))
    p['k_Sr_metam'] = 13e9
    p['k_oxfrac'] = 0.9975
    p['Pconc0'] = 2.2
    p['Nconc0'] = 30.9
    p['newp0'] = 117 * min(p['Nconc0'] / 16.0, p['Pconc0'])
    p['copsek16'] = 3.762
    p['a'] = 0.5
    p['b'] = 2
    p['kfire'] = 3
    # reservoirs (mol)
    p['P0'] = 3.1e15
    p['O0'] = 3.7e19
    p['A0'] = 3.193e18
    p['G0'] = 1.25e21
    p['C0'] = 5e21
    p['PYR0'] = 1.8e20
    p['GYP0'] = 2e20
    p['S0'] = 4e19
    p['CAL0'] = 1.397e19
    p['N0'] = 4.35e16
    p['OSr0'] = 1.2e17
    p['SSr0'] = 5e18
    return p


def _load_forcings(forcings_dir: str) -> dict:
    f: dict = {}
    interp = loadmat(os.path.join(forcings_dir, 'INTERPSTACK_1Ga_weatherable_areas.mat'),
                     squeeze_me=True, struct_as_record=False)['INTERPSTACK']
    f['INTERPSTACK'] = interp
    copse = loadmat(os.path.join(forcings_dir, 'COPSE_forcings.mat'),
                    squeeze_me=True, struct_as_record=False)['forcings']
    f['t'] = np.asarray(copse.t, dtype=float)
    f['E'] = np.asarray(copse.E, dtype=float)
    f['W'] = np.asarray(copse.W, dtype=float)

    gr = pd.read_excel(os.path.join(forcings_dir, 'GR_BA.xlsx'), header=None).to_numpy()
    # First row is header text; drop non-numeric rows
    mask = np.array([np.issubdtype(type(x), np.number) or
                     (isinstance(x, str) is False) for x in gr[:, 0]])
    # safer: convert
    grnum = []
    for row in gr:
        try:
            a = float(row[0]); b = float(row[1])
            grnum.append([a, b])
        except Exception:
            pass
    grnum = np.array(grnum, dtype=float)
    # MATLAB does *1e6 then /1e6 inside RHS, so xlsx col1 already in Myr units
    f['GR_BA'] = grnum  # col1 in Myr already

    ga = pd.read_excel(os.path.join(forcings_dir, 'GA_revised.xlsx'), header=None).to_numpy()
    ganum = []
    for row in ga:
        try:
            a = float(row[0]); b = float(row[1])
            ganum.append([a, b])
        except Exception:
            pass
    ganum = np.array(ganum, dtype=float)
    f['newGA'] = ganum

    dforce = loadmat(os.path.join(forcings_dir, 'D_force_Merdith25_Mills25.mat'),
                     squeeze_me=True, struct_as_record=False)
    f['D_force_x'] = np.asarray(dforce['D_force_x'], dtype=float)
    f['D_force_min'] = np.asarray(dforce['D_force_min'], dtype=float)
    f['D_force_max'] = np.asarray(dforce['D_force_max'], dtype=float)
    f['D_force_mid'] = np.asarray(dforce['D_force_mid'], dtype=float)

    sl = loadmat(os.path.join(forcings_dir, 'shoreline.mat'),
                 squeeze_me=True, struct_as_record=False)
    f['shoreline_time'] = np.asarray(sl['shoreline_time'], dtype=float)
    f['shoreline_relative'] = np.asarray(sl['shoreline_relative'], dtype=float)
    return f


def _interp1(x, y, xq):
    """1-D interpolation matching MATLAB interp1qr (linear, x must be sorted ascending)."""
    return float(np.interp(xq, x, y))


# ---------------------------------------------------------------------------
# Build initial state vector
# ---------------------------------------------------------------------------

def _build_y0(p: dict, tuning: dict | None = None) -> np.ndarray:
    # tuning dict keys: Gtune, Ctune, PYRtune, GYPtune, Otune, Stune, Atune
    # When provided, REPLACES the default outputs vector (mirrors MATLAB
    # SCION_initialise lines 306-314: pars.gstart = pars.G0 * abs(tuning.Gtune)).
    if tuning is not None:
        outputs = np.array([
            abs(tuning['Gtune']), abs(tuning['Ctune']),
            abs(tuning['PYRtune']), abs(tuning['GYPtune']),
            abs(tuning['Otune']), abs(tuning['Stune']),
            abs(tuning['Atune']),
        ])
    else:
        outputs = np.array([0.6, 1.0, 0.98, 0.99, 0.07, 0.05, 2.5])
    gstart = p['G0'] * outputs[0]
    cstart = p['C0'] * outputs[1]
    pyrstart = p['PYR0'] * outputs[2]
    gypstart = p['GYP0'] * outputs[3]
    ostart = p['O0'] * outputs[4]
    sstart = p['S0'] * outputs[5]
    astart = p['A0'] * outputs[6]
    pstart = p['P0']
    tempstart = 288.0
    CAL_start = p['CAL0']
    N_start = p['N0']
    OSr_start = p['OSr0']
    SSr_start = p['SSr0']
    delta_A_start = 0.0
    delta_S_start = 35.0
    delta_G_start = -27.0
    delta_C_start = -2.0
    delta_PYR_start = -5.0
    delta_GYP_start = 20.0
    delta_OSr_start = 0.708
    delta_SSr_start = 0.708
    y0 = np.zeros(21)
    y0[0] = pstart
    y0[1] = ostart
    y0[2] = astart
    y0[3] = sstart
    y0[4] = gstart
    y0[5] = cstart
    y0[6] = pyrstart
    y0[7] = gypstart
    y0[8] = tempstart
    y0[9] = CAL_start
    y0[10] = N_start
    y0[11] = gstart * delta_G_start
    y0[12] = cstart * delta_C_start
    y0[13] = pyrstart * delta_PYR_start
    y0[14] = gypstart * delta_GYP_start
    y0[15] = astart * delta_A_start
    y0[16] = sstart * delta_S_start
    y0[17] = OSr_start
    y0[18] = OSr_start * delta_OSr_start
    y0[19] = SSr_start
    y0[20] = SSr_start * delta_SSr_start
    return y0


# ---------------------------------------------------------------------------
# RHS
# ---------------------------------------------------------------------------

def _make_rhs(p: dict, F: dict, rel_contrib: np.ndarray, recorder: list,
              sens_params: dict | None = None):
    """Build closure for the ODE RHS plus per-call diagnostic recording.

    sens_params: dict with keys r1..r7 each in [-1,1], mirroring SCION_equations.m
    sensitivity perturbations to DEGASS, BAS_AREA, GRAN_AREA, PREPLANT, capdelS,
    capdelC_land, capdelC_marine. None -> deterministic run.
    """
    interp = F['INTERPSTACK']
    IS_time = np.ascontiguousarray(interp.time, dtype=float)
    IS_CO2 = np.ascontiguousarray(interp.CO2, dtype=float)
    IS_lat = np.ascontiguousarray(interp.lat, dtype=float)
    IS_runoff = np.ascontiguousarray(interp.runoff, dtype=float)
    IS_Tair = np.ascontiguousarray(interp.Tair, dtype=float)
    IS_topo = np.ascontiguousarray(interp.topo, dtype=float)
    IS_land = np.ascontiguousarray(interp.land, dtype=float)
    IS_slope = np.ascontiguousarray(interp.slope, dtype=float)
    IS_arc = np.ascontiguousarray(interp.arc, dtype=float)
    IS_relict_arc = np.ascontiguousarray(interp.relict_arc, dtype=float)
    IS_suture = np.ascontiguousarray(interp.suture, dtype=float)
    IS_gridarea = np.ascontiguousarray(interp.gridarea, dtype=float)
    rel_contrib_c = np.ascontiguousarray(rel_contrib, dtype=float)

    F_t = np.ascontiguousarray(F['t'], dtype=float)
    F_E = np.ascontiguousarray(F['E'], dtype=float)
    F_W = np.ascontiguousarray(F['W'], dtype=float)
    GR_BA = F['GR_BA']; newGA = F['newGA']
    GR_BA_x = np.ascontiguousarray(GR_BA[:, 0], dtype=float)
    GR_BA_y = np.ascontiguousarray(GR_BA[:, 1], dtype=float)
    newGA_x = np.ascontiguousarray(newGA[:, 0], dtype=float)
    newGA_y = np.ascontiguousarray(newGA[:, 1], dtype=float)
    Dx = np.ascontiguousarray(F['D_force_x'], dtype=float)
    Dmid = np.ascontiguousarray(F['D_force_mid'], dtype=float)
    Dmin = np.ascontiguousarray(F['D_force_min'], dtype=float)
    Dmax = np.ascontiguousarray(F['D_force_max'], dtype=float)
    sl_t = np.ascontiguousarray(F['shoreline_time'], dtype=float)
    sl_r = np.ascontiguousarray(F['shoreline_relative'], dtype=float)

    # bioturbation table
    biot_x = np.array([-1000.0, -525.0, -520.0, 0.0])
    biot_y = np.array([0.0, 0.0, 1.0, 1.0])
    cb_x = np.array([0.0, 1.0])
    cb_y = np.array([1.2, 1.0])

    # Pre-allocated buffers reused by every rhs call (avoid per-call allocation).
    _nlat = IS_runoff.shape[0]; _nlon = IS_runoff.shape[1]
    _spatial_bufs = {
        'eps_p': np.empty((_nlat, _nlon)), 'eps_f': np.empty((_nlat, _nlon)),
        'cw_per_p': np.empty((_nlat, _nlon)), 'cw_per_f': np.empty((_nlat, _nlon)),
        'q_p': np.empty((_nlat, _nlon)), 'q_f': np.empty((_nlat, _nlon)),
        'tair_p': np.empty((_nlat, _nlon)), 'tair_f': np.empty((_nlat, _nlon)),
    }

    A0 = p['A0']; O0 = p['O0']; G0 = p['G0']; C0 = p['C0']
    PYR0 = p['PYR0']; GYP0 = p['GYP0']; S0 = p['S0']; P0 = p['P0']; N0 = p['N0']
    OSr0 = p['OSr0']; SSr0 = p['SSr0']
    silw_scale = 6.5e8
    k_carb_scale = 200.0
    ARCfactor = 7.0
    SUTUREfactor = 20.0
    k_erosion = 3.3e-3
    Xm = 0.1; Kw = 6e-5; kw = 1e-3; Ea = 20.0; z = 10.0
    sigplus1 = 0.9; T0 = 286.0; R = 8.31e-3
    Tcrit = -10.0
    # PREPLANT, capdelS, capdelC_land, capdelC_marine are now per-call so that
    # sens_params can override them; baseline values applied inside rhs().
    sp = sens_params
    P_half = 183.6; P_min = 10.0
    k_anox = 12.0; k_u = 0.5
    CNsea = 37.5
    CPbiot = 250.0; CPlam = 1000.0
    pfrac_silw = 0.8; pfrac_carbw = 0.14; pfrac_oxidw = 0.06
    atfrac0 = 0.01614

    def rhs(t, y):
        P = y[0]; O = y[1]; A = y[2]; S = y[3]; G = y[4]; C = y[5]
        PYR = y[6]; GYP = y[7]
        N = y[10]
        OSr = y[17]; SSr = y[19]

        t_geol = t * 1e-6

        delta_G = y[11]/y[4]
        delta_C = y[12]/y[5]
        delta_PYR = y[13]/y[6]
        delta_GYP = y[14]/y[7]

        atfrac = atfrac0 * (A/A0)
        RCO2 = (A/A0) * (atfrac/atfrac0)
        CO2atm = RCO2 * 280e-6
        CO2ppm = RCO2 * 280.0
        mrO2 = (O/O0) / ((O/O0) + p['copsek16'])
        RO2 = O/O0

        # forcings
        E_reloaded = _interp1_nb(F_t, F_E, t_geol)
        W_reloaded = _interp1_nb(F_t, F_W, t_geol)
        GR_BA_v = _interp1_nb(GR_BA_x, GR_BA_y, t_geol)
        newGA_v = _interp1_nb(newGA_x, newGA_y, t_geol)
        D_mid = _interp1_nb(Dx, Dmid, t_geol)
        DEGASS = D_mid
        W = W_reloaded
        EVO = E_reloaded
        BAS_AREA = GR_BA_v
        GRAN_AREA = newGA_v
        PREPLANT = 1.0/7.0
        capdelS = 27.0
        capdelC_land = 27.0
        capdelC_marine = 35.0
        if sp is not None:
            r1 = sp['r1']; r2 = sp['r2']; r3 = sp['r3']; r4 = sp['r4']
            r5 = sp['r5']; r6 = sp['r6']; r7 = sp['r7']
            if r1 > 0:
                D_max = _interp1_nb(Dx, Dmax, t_geol)
                DEGASS = (1.0 - r1) * DEGASS + r1 * D_max
            else:
                D_min = _interp1_nb(Dx, Dmin, t_geol)
                DEGASS = (1.0 + r1) * DEGASS - r1 * D_min
            BAS_AREA = BAS_AREA * (1.0 + 0.2 * r2)
            GRAN_AREA = GRAN_AREA * (1.0 + 0.2 * r3)
            PREPLANT = 1.0 / (4.0 + 3.0 * r4)
            capdelS = 30.0 + 10.0 * r5
            capdelC_land = 25.0 + 5.0 * r6
            capdelC_marine = 30.0 + 5.0 * r7
        SHORELINE = _interp1_nb(sl_t, sl_r, t_geol)
        f_biot = _interp1_nb(biot_x, biot_y, t_geol)
        CB = _interp1_nb(cb_x, cb_y, f_biot)

        # ---- spatial weathering block (numba-accelerated) ----
        (silw_spatial, carbw_spatial, GAST, SAT_tropical, SAT_equator,
         iceline, erosion_tot, contribution_past, contribution_future,
         key_past_index, key_future_index) = _spatial_weathering(
            t_geol, CO2ppm,
            IS_time, IS_CO2, IS_lat,
            IS_runoff, IS_Tair, IS_topo, IS_land,
            IS_slope, IS_arc, IS_relict_arc, IS_suture, IS_gridarea,
            rel_contrib_c,
            ARCfactor, SUTUREfactor,
            k_erosion, Xm, Kw, kw, Ea, z, sigplus1, T0, R, Tcrit,
            k_carb_scale, p['k_basw'], p['k_granw'], silw_scale,
            _spatial_bufs,
        )

        # ---- Global vars ----
        V_T = 1.0 - ((GAST - 25.0) / 25.0)**2
        P_atm = CO2atm * 1e6
        V_co2 = (P_atm - P_min) / (P_half + P_atm - P_min)
        V_o2 = 1.5 - 0.5 * (O/O0)
        V_npp = 2 * EVO * V_T * V_o2 * V_co2
        ignit = min(max(48 * mrO2 - 9.08, 0.0), 5.0)
        firef = p['kfire'] / (p['kfire'] - 1 + ignit)
        VEG = V_npp * firef

        Tsurf = GAST + 273.0
        TEMP_gast = Tsurf
        V = VEG
        f_biota = (1 - min(V*W, 1.0)) * PREPLANT * (RCO2**0.5) + (V*W)

        denom = p['basfrac']*BAS_AREA + (1 - p['basfrac'])*GRAN_AREA
        basw = silw_spatial * (p['basfrac']*BAS_AREA / denom)
        granw = silw_spatial * ((1 - p['basfrac'])*GRAN_AREA / denom)
        basw = basw * f_biota
        granw = granw * f_biota
        carbw = carbw_spatial * f_biota
        silw = basw + granw
        carbw_relative = carbw / p['k_carbw']

        oxidw = p['k_oxidw'] * carbw_relative * (G/G0) * ((O/O0)**p['a'])
        pyrw = p['k_pyrw'] * carbw_relative * (PYR/PYR0)
        gypw = p['k_gypw'] * (GYP/GYP0) * carbw_relative

        f_T_sfw = np.exp(0.0608 * (Tsurf - 288.0))
        sfw = p['k_sfw'] * f_T_sfw * DEGASS

        ocdeg = p['k_ocdeg'] * DEGASS * (G/G0)
        ccdeg = p['k_ccdeg'] * DEGASS * (C/C0)
        pyrdeg = p['k_pyrdeg'] * (PYR/PYR0) * DEGASS
        gypdeg = p['k_gypdeg'] * (GYP/GYP0) * DEGASS

        mgsb = p['k_mgsb'] * (S/S0) * (1.0/SHORELINE)
        mccb = carbw + silw

        phosw = p['k_phosw'] * (pfrac_silw*(silw/p['k_silw'])
                                 + pfrac_carbw*(carbw/p['k_carbw'])
                                 + pfrac_oxidw*(oxidw/p['k_oxidw']))
        pland = p['k_landfrac'] * VEG * phosw
        pland0 = p['k_landfrac'] * p['k_phosw']
        psea = phosw - pland

        Pconc = (P/P0) * 2.2
        Nconc = (N/N0) * 30.9
        newp = 117 * min(Nconc/16.0, Pconc)

        mocb = p['k_mocb'] * ((newp/p['newp0'])**p['b']) * CB
        locb = p['k_locb'] * (pland/pland0) * 1.0  # CPLAND = 1
        fox = 1.0 / (O/O0)
        mpsb = p['k_mpsb'] * (S/S0) * fox * (mocb/p['k_mocb'])

        ANOX = 1.0 / (1.0 + np.exp(-k_anox * (k_u * (newp/p['newp0']) - (O/O0))))
        monb = mocb / CNsea
        mopb = mocb * ((f_biot/CPbiot) + ((1 - f_biot)/CPlam))
        capb = p['k_capb'] * (mocb/p['k_mocb'])
        fepb = (p['k_fepb']/p['k_oxfrac']) * (1 - ANOX) * (P/P0)

        if (N/16.0) < P:
            nfix = p['k_nfix'] * (((P - (N/16.0)) / (P0 - (N0/16.0)))**2)
        else:
            nfix = 0.0
        denit = p['k_denit'] * (1.0 + (ANOX/(1 - p['k_oxfrac']))) * (N/N0)

        reductant_input = p['k_reductant_input'] * DEGASS

        # isotopes
        d13c_A = y[15] / y[2]
        d34s_S = y[16] / y[3]
        delta_locb = d13c_A - capdelC_land
        delta_mocb = d13c_A - capdelC_marine
        delta_mccb = d13c_A
        delta_mpsb = d34s_S - capdelS

        # ---- Sr ----
        Sr_granw = p['k_Sr_granw'] * (granw/p['k_granw'])
        Sr_basw = p['k_Sr_basw'] * (basw/p['k_basw'])
        Sr_sedw = p['k_Sr_sedw'] * (carbw/p['k_carbw']) * (SSr/SSr0)
        Sr_mantle = p['k_Sr_mantle'] * DEGASS
        Sr_sfw = p['k_Sr_sfw'] * (sfw/p['k_sfw']) * (OSr/OSr0)
        Sr_metam = p['k_Sr_metam'] * DEGASS * (SSr/SSr0)
        Sr_sedb = p['k_Sr_sedb'] * (mccb/p['k_mccb']) * (OSr/OSr0)

        delta_OSr = y[18] / y[17]
        delta_SSr = y[20] / y[19]

        RbSr_bas = 0.1; RbSr_gran = 0.26; RbSr_mantle = 0.066; RbSr_carbonate = 0.5
        dSr0 = 0.69898; lam = 1.4e-11
        tforwards = 4.5e9 + t
        dSr_bas = dSr0 + RbSr_bas * (1 - np.exp(-lam*tforwards))
        dSr_gran = dSr0 + RbSr_gran * (1 - np.exp(-lam*tforwards))
        dSr_mantle = dSr0 + RbSr_mantle * (1 - np.exp(-lam*tforwards))

        # ---- dy ----
        dy = np.zeros(21)
        dy[0] = psea - mopb - capb - fepb
        dy[1] = locb + mocb - oxidw - ocdeg + 2*(mpsb - pyrw - pyrdeg) - reductant_input
        dy[2] = -locb - mocb + oxidw + ocdeg + ccdeg + carbw - mccb - sfw + reductant_input
        dy[3] = gypw + pyrw - mgsb - mpsb + gypdeg + pyrdeg
        dy[4] = locb + mocb - oxidw - ocdeg
        dy[5] = mccb + sfw - carbw - ccdeg
        dy[6] = mpsb - pyrw - pyrdeg
        dy[7] = mgsb - gypw - gypdeg
        # dy[8] TEMP unused
        # dy[9] CAL unused
        dy[10] = nfix - denit - monb
        dy[11] = locb*delta_locb + mocb*delta_mocb - oxidw*delta_G - ocdeg*delta_G
        dy[12] = mccb*delta_mccb + sfw*delta_mccb - carbw*delta_C - ccdeg*delta_C
        dy[13] = mpsb*delta_mpsb - pyrw*delta_PYR - pyrdeg*delta_PYR
        dy[14] = mgsb*d34s_S - gypw*delta_GYP - gypdeg*delta_GYP
        dy[15] = (-locb*delta_locb - mocb*delta_mocb + oxidw*delta_G + ocdeg*delta_G
                  + ccdeg*delta_C + carbw*delta_C - mccb*delta_mccb - sfw*delta_mccb
                  + reductant_input*-5)
        dy[16] = (gypw*delta_GYP + pyrw*delta_PYR - mgsb*d34s_S - mpsb*delta_mpsb
                  + gypdeg*delta_GYP + pyrdeg*delta_PYR)
        dy[17] = Sr_granw + Sr_basw + Sr_sedw + Sr_mantle - Sr_sedb - Sr_sfw
        dy[18] = (Sr_granw*dSr_gran + Sr_basw*dSr_bas + Sr_sedw*delta_SSr
                  + Sr_mantle*dSr_mantle - Sr_sedb*delta_OSr - Sr_sfw*delta_OSr)
        dy[19] = Sr_sedb - Sr_sedw - Sr_metam
        dy[20] = (Sr_sedb*delta_OSr - Sr_sedw*delta_SSr - Sr_metam*delta_SSr
                  + SSr*lam*RbSr_carbonate*np.exp(lam*tforwards))

        # record
        recorder.append({
            't': t, 'y': y.copy(),
            'mocb': mocb, 'locb': locb, 'mccb': mccb,
            'silw': silw, 'basw': basw, 'granw': granw, 'carbw': carbw,
            'oxidw': oxidw, 'phosw': phosw, 'pyrw': pyrw, 'gypw': gypw,
            'ocdeg': ocdeg, 'ccdeg': ccdeg, 'sfw': sfw,
            'pyrdeg': pyrdeg, 'gypdeg': gypdeg,
            'mpsb': mpsb, 'mgsb': mgsb, 'monb': monb, 'nfix': nfix, 'denit': denit,
            'RCO2': RCO2, 'RO2': RO2, 'mrO2': mrO2,
            'VEG': VEG, 'ANOX': ANOX, 'iceline': iceline,
            'GAST': GAST, 'SAT_tropical': SAT_tropical, 'SAT_equator': SAT_equator,
            'DEGASS': DEGASS, 'W': W, 'EVO': EVO,
            'BAS_AREA': BAS_AREA, 'GRAN_AREA': GRAN_AREA,
            'erosion_tot': erosion_tot,
            'd13c_A': d13c_A, 'delta_mccb': delta_mccb, 'd34s_S': d34s_S,
            'delta_G': delta_G, 'delta_C': delta_C,
            'delta_PYR': delta_PYR, 'delta_GYP': delta_GYP,
            'delta_OSr': delta_OSr,
        })
        return dy

    return rhs


# ---------------------------------------------------------------------------
# run()
# ---------------------------------------------------------------------------

def run(runcontrol: int = 0, save_path: str | None = None,
        forcings_dir: str = 'forcings',
        tuning: dict | None = None,
        sens_params: dict | None = None,
        method: str = 'BDF',
        rtol: float = 1e-6,
        atol: float = 1e-9) -> dict:
    """Single run; returns a dict of state and diagnostic time series.

    tuning: optional dict with keys Gtune, Ctune, PYRtune, GYPtune, Otune,
        Stune, Atune used to override the default starting reservoir multipliers
        (mirrors MATLAB tuning global).
    sens_params: optional dict with keys r1..r7 in [-1, 1] applying the seven
        MATLAB sensitivity perturbations to DEGASS, BAS_AREA, GRAN_AREA,
        PREPLANT, capdelS, capdelC_land, capdelC_marine.
    """
    p = _build_pars()
    F = _load_forcings(forcings_dir)
    interp = F['INTERPSTACK']
    lat = np.asarray(interp.lat, dtype=float)
    lat_areas = np.cos(np.deg2rad(lat))
    lat_weights = lat_areas / np.mean(lat_areas)
    rel_contrib = np.tile(lat_weights[:, None], (1, 48))

    y0 = _build_y0(p, tuning=tuning)
    recorder: list = []
    rhs = _make_rhs(p, F, rel_contrib, recorder, sens_params=sens_params)

    t_start = -1e9
    t_end = 0.0

    wall0 = _time.time()
    sol = solve_ivp(rhs, (t_start, t_end), y0,
                    method=method, max_step=5e5,
                    rtol=rtol, atol=atol)
    wall = _time.time() - wall0
    if not sol.success:
        raise RuntimeError(f'solve_ivp failed: {sol.message}')

    # Match recorder entries to accepted timesteps (mirror MATLAB intersect approach).
    rec_t = np.array([r['t'] for r in recorder])
    # For each accepted t, find the LAST recorder entry whose t equals it (most recent record).
    accepted_t = sol.t
    idxs = []
    for tt in accepted_t:
        matches = np.where(rec_t == tt)[0]
        if len(matches) == 0:
            # fall back to nearest
            idxs.append(int(np.argmin(np.abs(rec_t - tt))))
        else:
            idxs.append(int(matches[-1]))
    idxs = np.array(idxs)

    out: dict[str, Any] = {}
    out['t'] = accepted_t
    out['time_myr'] = accepted_t * 1e-6
    Y = sol.y  # (21, N)
    out['P'] = Y[0]; out['O'] = Y[1]; out['A'] = Y[2]; out['S'] = Y[3]
    out['G'] = Y[4]; out['C'] = Y[5]; out['PYR'] = Y[6]; out['GYP'] = Y[7]
    out['TEMP'] = Y[8]; out['CAL'] = Y[9]; out['N'] = Y[10]
    out['G_iso'] = Y[11]; out['C_iso'] = Y[12]
    out['PYR_iso'] = Y[13]; out['GYP_iso'] = Y[14]
    out['A_iso'] = Y[15]; out['S_iso'] = Y[16]
    out['OSr'] = Y[17]; out['OSr_iso'] = Y[18]
    out['SSr'] = Y[19]; out['SSr_iso'] = Y[20]

    diag_keys = ['mocb','locb','mccb','silw','basw','granw','carbw','oxidw',
                 'phosw','pyrw','gypw','ocdeg','ccdeg','sfw','pyrdeg','gypdeg',
                 'mpsb','mgsb','monb','nfix','denit','RCO2','RO2','mrO2',
                 'VEG','ANOX','iceline','GAST','SAT_tropical','SAT_equator',
                 'DEGASS','W','EVO','BAS_AREA','GRAN_AREA','erosion_tot',
                 'd13c_A','delta_mccb','d34s_S','delta_G','delta_C',
                 'delta_PYR','delta_GYP','delta_OSr']
    for k in diag_keys:
        out[k] = np.array([recorder[i][k] for i in idxs])

    out['n_steps_total'] = len(recorder)
    out['n_accepted'] = len(accepted_t)
    out['wall_seconds'] = wall

    if save_path is not None:
        save_dict = {k: v for k, v in out.items() if isinstance(v, np.ndarray)}
        save_dict['n_steps_total'] = np.array([out['n_steps_total']])
        save_dict['n_accepted'] = np.array([out['n_accepted']])
        save_dict['wall_seconds'] = np.array([out['wall_seconds']])
        np.savez_compressed(save_path, **save_dict)

    return out
