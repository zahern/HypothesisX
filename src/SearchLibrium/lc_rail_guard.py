"""Rail ("parameter pinned at a bound") detection for latent-class fits.

A rail is a parameter pressed against a hard bound or a coefficient whose
standard error is numerically degenerate. Either makes the estimate unusable
even when the optimizer reports success, so railed fits must not be treated as
usable solutions.
"""
from __future__ import annotations

import numpy as np


def rail_flags(values, ses=None, bounds=None, tol=1e-3, se_degenerate=1e5):
    """Flag bound-pinned parameters and degenerate standard errors.

    Parameters
    ----------
    values, ses : dict name -> float. ``ses`` is optional.
    bounds : dict name -> (lo, hi); None disables a side. A key also matches
        every parameter whose name starts with it.
    tol : absolute tolerance for "sitting on the bound".
    se_degenerate : |SE| at/above this is a rail.
    """
    rail = []
    warnings = []
    ses = {str(k): v for k, v in (ses or {}).items()}
    bounds = dict(bounds or {})
    for name, val in (values or {}).items():
        name = str(name)
        try:
            v = float(val)
        except Exception:
            continue
        if not np.isfinite(v):
            continue
        lo = hi = None
        for key, bd in bounds.items():
            if name == str(key) or name.startswith(str(key)):
                if isinstance(bd, (tuple, list)) and len(bd) == 2:
                    lo, hi = bd
                break
        if lo is not None and np.isfinite(lo) and v <= float(lo) + tol:
            rail.append(f"{name} at/below lower bound {float(lo):g} "
                        f"(value={v:.4g})")
        if hi is not None and np.isfinite(hi) and v >= float(hi) - tol:
            rail.append(f"{name} at/above upper bound {float(hi):g} "
                        f"(value={v:.4g})")
        s = ses.get(name)
        if s is not None:
            try:
                s = float(s)
            except Exception:
                s = None
        if s is not None and np.isfinite(s):
            if s >= se_degenerate:
                rail.append(f"{name} has degenerate SE {s:.1e} "
                            f"(flat/unidentified direction)")
            elif s <= 0.0:
                warnings.append(f"{name} SE=0 (parameter locked/fixed)")
    return {"rail": rail, "railed": bool(rail), "warnings": warnings}


def lc_result_rail(names, betas, ses, shares, membership_coefs=None,
                   bounds=15.0, share_floor=0.05, method="lc-em",
                   tol=1e-3, se_degenerate=1e5):
    """Rail scan for ``estimate_dest_choice_lc`` style result arrays."""
    rail = []
    warnings = []
    names = [str(n) for n in names]
    for c, b in enumerate(betas):
        try:
            arr = np.asarray(b, dtype=float).ravel()
        except Exception:
            continue
        se_arr = None
        if ses is not None and c < len(ses):
            try:
                candidate = np.asarray(ses[c], dtype=float).ravel()
                if candidate.size == arr.size:
                    se_arr = candidate
            except Exception:
                se_arr = None
        for j, nm in enumerate(names):
            if j >= arr.size:
                break
            v = float(arr[j])
            if np.isfinite(v) and bounds is not None:
                bound = float(bounds)
                if v >= bound - tol:
                    rail.append(f"class {c + 1} {nm} at/above beta bound "
                                f"{bound:g} (value={v:.4g})")
                elif v <= -bound + tol:
                    rail.append(f"class {c + 1} {nm} at/below beta bound "
                                f"{-bound:g} (value={v:.4g})")
            if se_arr is not None:
                s = float(se_arr[j])
                if np.isfinite(s):
                    if s >= se_degenerate:
                        rail.append(f"class {c + 1} {nm} has degenerate SE "
                                    f"{s:.1e} (flat/unidentified direction)")
                    elif s <= 0.0:
                        warnings.append(f"class {c + 1} {nm} SE=0 "
                                        f"(parameter locked/fixed)")
    if method == "lc-em":
        try:
            sh = np.asarray(shares, dtype=float).ravel()
            floor = float(share_floor)
            for c, s in enumerate(sh):
                if not np.isfinite(s):
                    continue
                if s < 0.5 * floor:
                    rail.append(f"class {c + 1} collapsed: share {s:.4g} < "
                                f"{0.5 * floor:g}")
                elif s < floor:
                    warnings.append(f"class {c + 1} share {s:.4g} below "
                                    f"floor {floor:g}")
        except Exception:
            pass
        if len(betas) > 1:
            try:
                arrs = [np.asarray(b, dtype=float).ravel() for b in betas]
                if all(a.size == arrs[0].size for a in arrs):
                    maxdiff = max(float(np.max(np.abs(a - arrs[0])))
                                  for a in arrs[1:])
                    if np.isfinite(maxdiff) and maxdiff < 1e-6:
                        rail.append("classes identical on betas "
                                    f"(max |diff|={maxdiff:.1e})")
            except Exception:
                pass
        if membership_coefs is not None:
            try:
                G = np.asarray(membership_coefs, dtype=float)
                gmax = float(np.max(np.abs(G))) if G.size else 0.0
                if np.isfinite(gmax) and gmax >= 50.0:
                    rail.append("membership gamma saturated "
                                f"(max |gamma|={gmax:.3g})")
            except Exception:
                pass
    return {"rail": rail, "railed": bool(rail), "warnings": warnings}
