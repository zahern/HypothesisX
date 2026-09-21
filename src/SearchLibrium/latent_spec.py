"""latent_spec.py — VIP and near-identification diagnostics for latent-class specs.

A latent-class model is only as good as its ``class_params_spec``: the
recovery test in this repo shows coefficients recover when the spec mirrors
the true per-class structure, and collapse when it does not (noise variables
everywhere, near-duplicate classes, or tiny shares). This module provides the
two tools needed to *handle* the spec iteratively:

1. ``vip_table`` — Variable Importance Profile: per-(class, variable)
   importance from the fitted model (|t|-statistics with standard-error
   fallbacks), i.e. which variables earn their place in which class.
2. ``near_unidentified_report`` — the near-unidentified test: degenerate
   shares, exploding coefficients, near-duplicate (label-switched) classes,
   weak parameters, low assignment confidence, and — when a Hessian is
   supplied — a Cole (2020)/Gimenez et al. (2004) eigenvalue-ratio +
   variance-decomposition-proportion test.
3. ``propose_class_params_spec`` — turns (1)+(2) into a concrete refreshed
   ``class_params_spec``: drop noise, drop/merge degenerate classes, keep
   identifiability guards.
4. ``membership_vip_table`` + ``membership_identification_report`` — the same
   battery for the class-allocation (membership) equation, centred on the
   collapse pathology: any single covariate whose log-odds swing can saturate
   the softmax and empty a class.
5. ``propose_member_params_spec`` — a collapse-proof ``member_params_spec``
   (swing-based covariate selection, standardisation + bound + floor
   recommendations).

Memory note: everything here operates on fitted 1-D vectors (coefficients,
standard errors, shares) plus single-pass reductions of the ``(N, C)``
posterior. The ``(N, J, K)`` design is never touched or copied.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from scipy.optimize import linear_sum_assignment as _lsa
    _HAVE_SCIPY_LSA = True
except Exception:  # pragma: no cover
    _lsa = None
    _HAVE_SCIPY_LSA = False


# ---------------------------------------------------------------------------
# internal helpers
# ---------------------------------------------------------------------------

def _as_list_of_arrays(class_betas):
    if isinstance(class_betas, list):
        return [np.asarray(b, dtype=float).ravel() for b in class_betas]
    arr = np.asarray(class_betas, dtype=float)
    return [arr[c].ravel() for c in range(arr.shape[0])]


def _class_var_lists(model):
    """Return (per-class var-name lists, union var list) for a fitted model."""
    specs = getattr(model, "_class_specs", None)
    varnames = [str(v) for v in (getattr(model, "varnames", None) or [])]
    n_classes = int(getattr(model, "n_classes", 0))
    per_class, union, seen = [], [], set()
    for c in range(n_classes):
        names = []
        if specs is not None and c < len(specs):
            for i in np.asarray(specs[c]).ravel().tolist():
                try:
                    nm = varnames[int(i)]
                except Exception:
                    continue
                names.append(nm)
                if nm not in seen:
                    seen.add(nm)
                    union.append(nm)
        elif varnames:
            names = list(varnames)
            for nm in names:
                if nm not in seen:
                    seen.add(nm)
                    union.append(nm)
        per_class.append(names)
    return per_class, union


def _as_1d_float(x):
    if x is None:
        return np.array([], dtype=float)
    try:
        return np.asarray(x, dtype=float).ravel()
    except Exception:
        return np.array([], dtype=float)


def _flat_coef_lookup(model):
    """Map (class, var) -> (coef, se, p) using coeff_names when possible."""
    names = list(getattr(model, "coeff_names", None) or [])
    est = _as_1d_float(getattr(model, "coeff_est", getattr(model, "coef_", None)))
    se = _as_1d_float(getattr(model, "stderr", None))
    pv = _as_1d_float(getattr(model, "pvalues", None))
    lut = {}
    for i, nm in enumerate(names):
        s = str(nm)
        if s.startswith("class_"):
            try:
                rest = s.split("class_", 1)[1]
                c = int(rest.split("_", 1)[0]) - 1
                var = rest.split("_", 1)[1]
            except Exception:
                continue
        else:
            continue
        lut[(c, var)] = (
            float(est[i]) if i < est.size else np.nan,
            float(se[i]) if i < se.size else np.nan,
            float(pv[i]) if i < pv.size else np.nan,
        )
    return lut


def _coef_matrix(model):
    """(C, V) coefficient matrix over the union of spec vars (missing = 0)."""
    per_class, union = _class_var_lists(model)
    betas = _as_list_of_arrays(getattr(model, "class_betas", []))
    C = int(getattr(model, "n_classes", len(betas)))
    V = len(union)
    B = np.zeros((C, V), dtype=float)
    pos = {v: j for j, v in enumerate(union)}
    for c in range(min(C, len(betas))):
        names = per_class[c] if c < len(per_class) else []
        b = betas[c]
        for j, nm in enumerate(names):
            if j < b.size and nm in pos:
                B[c, pos[nm]] = b[j]
    return B, per_class, union


# ---------------------------------------------------------------------------
# 1. VIP — variable importance profile
# ---------------------------------------------------------------------------

def vip_table(model, normalize=True):
    """Per-(class, variable) importance profile of a fitted latent-class model.

    Importance is |t| = |coef| / SE where a finite positive SE exists;
    otherwise it falls back to the within-class |coef| share (flagged via
    ``se_missing``). Returns a DataFrame sorted by class then importance.
    """
    B, per_class, union = _coef_matrix(model)
    lut = _flat_coef_lookup(model)
    _shares_raw = getattr(model, "class_probs", None)
    shares = (np.asarray(_shares_raw, dtype=float).ravel()
              if _shares_raw is not None else np.array([], dtype=float))
    C = B.shape[0]
    rows = []
    for c in range(C):
        names = per_class[c] if c < len(per_class) else []
        tvals, fallback = [], []
        for j, nm in enumerate(names):
            coef = float(B[c, union.index(nm)]) if nm in union else 0.0
            se = p = np.nan
            if (c, nm) in lut:
                _, se, p = lut[(c, nm)]
            if np.isfinite(se) and se > 0:
                t = abs(coef) / se
            else:
                t = np.nan
                fallback.append(nm)
            tvals.append(t)
        tvals = np.asarray(tvals, dtype=float)
        if np.all(~np.isfinite(tvals)) and len(names):
            # no usable SEs: rank by within-class |coef| share instead
            denom = float(np.abs(B[c]).sum()) or 1.0
            imp = np.abs(B[c, [union.index(n) for n in names]]) / denom
            how = "coef-share (no SE)"
        else:
            imp = np.where(np.isfinite(tvals), tvals, 0.0)
            how = "|t|"
        tot = float(imp.sum()) or 1.0
        for nm, t, im in zip(names, tvals, imp / tot if normalize else imp):
            coef = float(B[c, union.index(nm)]) if nm in union else 0.0
            rows.append({
                "class": c,
                "class_share": float(shares[c]) if c < shares.size else np.nan,
                "variable": nm,
                "coef": coef,
                "t_stat": float(t) if np.isfinite(t) else np.nan,
                "importance": float(im),
                "se_missing": bool(nm in fallback),
                "measure": how,
            })
    df = pd.DataFrame(rows, columns=["class", "class_share", "variable",
                                     "coef", "t_stat", "importance",
                                     "se_missing", "measure"])
    if len(df):
        df = df.sort_values(["class", "importance"],
                            ascending=[True, False]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# 2. Near-unidentified test
# ---------------------------------------------------------------------------

def _cosine_pairs(B):
    nrm = np.linalg.norm(B, axis=1, keepdims=True)
    nrm = np.where(nrm > 0, nrm, 1.0)
    S = (B / nrm) @ (B / nrm).T
    pairs = []
    C = B.shape[0]
    for a in range(C):
        for b in range(a + 1, C):
            pairs.append((a, b, float(S[a, b])))
    return pairs


def _vdp_report(hessian, ratio_thresh=1e-3, vdp_thresh=0.5, names=None):
    """Cole (2020)/Gimenez et al. (2004) near-identification flags.

    A direction k is near-non-identified when lambda_k/lambda_max falls below
    ``ratio_thresh``; a parameter is implicated when its variance-decomposition
    share ``(v_kj^2/lambda_k)/Var(theta_j)`` exceeds ``vdp_thresh``.
    """
    H = 0.5 * (np.asarray(hessian, dtype=float)
               + np.asarray(hessian, dtype=float).T)
    out = {"ratio_min": np.nan, "flag_dirs": [], "flags": {}}
    try:
        w, V = np.linalg.eigh(H)
    except Exception:
        return out
    w = np.abs(w)
    lam_max = float(w.max()) if w.size and w.max() > 0 else np.nan
    if not np.isfinite(lam_max) or lam_max <= 0:
        return out
    try:
        var = np.diag(np.linalg.pinv(H))
    except Exception:
        var = np.full(H.shape[0], np.nan)
    out["ratio_min"] = float((w / lam_max).min())
    for k, lam in enumerate(w):
        if lam / lam_max >= ratio_thresh:
            continue
        v = V[:, k]
        with np.errstate(divide="ignore", invalid="ignore"):
            share = (v ** 2 / max(lam, 1e-300)) / np.where(var > 0, var, np.nan)
        for j, s in enumerate(share):
            if np.isfinite(s) and s > vdp_thresh:
                nm = names[j] if names and j < len(names) else f"p{j}"
                out["flags"].setdefault(nm, []).append(
                    {"dir": int(k), "ratio": float(lam / lam_max),
                     "vdp_share": float(s)})
        out["flag_dirs"].append(int(k))
    return out


def near_unidentified_report(model, hessian=None, min_share=0.01,
                             max_abs_beta=200.0, dup_cos_thresh=0.98,
                             weak_p=0.10, ratio_thresh=1e-3,
                             vdp_thresh=0.5):
    """Run the near-unidentified test battery on a fitted latent-class model.

    Returns a dict with per-check flags plus an overall ``verdict`` of
    ``"ok"`` | ``"near-unidentified"`` | ``"degenerate"`` and ``reasons``.
    Operates only on fitted vectors (+ one streaming pass over ``posterior``);
    the ``(N, J, K)`` design is never touched.
    """
    B, per_class, union = _coef_matrix(model)
    _shares_raw = getattr(model, "class_probs", None)
    shares = (np.asarray(_shares_raw, dtype=float).ravel()
              if _shares_raw is not None else np.array([], dtype=float))
    C = B.shape[0]
    rep = {"verdict": "ok", "reasons": [], "checks": {}}

    # -- degenerate shares / exploding coefficients -------------------------
    tiny = [int(c) for c in range(C)
            if c >= shares.size or not np.isfinite(shares[c])
            or shares[c] < min_share]
    big = []
    for c in range(C):
        names = per_class[c] if c < len(per_class) else []
        for j, nm in enumerate(names):
            v = float(B[c, union.index(nm)]) if nm in union else 0.0
            if not np.isfinite(v) or abs(v) > max_abs_beta:
                big.append((c, nm, v))
    rep["checks"]["tiny_share_classes"] = tiny
    rep["checks"]["extreme_coefs"] = [(c, nm, float(v)) for c, nm, v in big]
    if tiny or big:
        rep["verdict"] = "degenerate"
        if tiny:
            rep["reasons"].append(f"collapsed/tiny shares in classes {tiny} "
                                  f"(< {min_share})")
        if big:
            rep["reasons"].append(f"{len(big)} coefficient(s) beyond "
                                  f"|beta|={max_abs_beta} or non-finite")

    # -- near-duplicate classes (label-switch / split artefacts) -------------
    dup = [(a, b, s) for a, b, s in _cosine_pairs(B) if s > dup_cos_thresh]
    rep["checks"]["duplicate_class_pairs"] = [
        {"a": a, "b": b, "cosine": s} for a, b, s in dup]
    if dup and rep["verdict"] == "ok":
        rep["verdict"] = "near-unidentified"
    for a, b, s in dup:
        rep["reasons"].append(f"classes {a}~{b} near-duplicate "
                              f"(cosine={s:.4f})")

    # -- weak parameters -----------------------------------------------------
    pv = _as_1d_float(getattr(model, "pvalues", None))
    se = _as_1d_float(getattr(model, "stderr", None))
    n_par = int(_as_1d_float(getattr(model, "coeff_est", None)).size or 0)
    if pv.size:
        weak = int(((pv > weak_p) | ~np.isfinite(pv)).sum())
    elif se.size:
        weak = int((~np.isfinite(se) | (se <= 0)).sum())
    else:
        weak = -1  # unknown: no SE/p-values stored
    rep["checks"]["weak_or_unidentified_params"] = weak
    rep["checks"]["n_class_params"] = n_par
    if weak > 0 and n_par > 0 and weak / max(n_par, 1) > 0.8:
        if rep["verdict"] == "ok":
            rep["verdict"] = "near-unidentified"
        rep["reasons"].append(f"{weak}/{n_par} params weak "
                              f"(p>{weak_p} or bad SE)")

    # -- assignment confidence (one streaming pass over the posterior) --------
    post = getattr(model, "posterior", None)
    if post is not None:
        try:
            post = np.asarray(post, dtype=float)
            rep["checks"]["mean_max_posterior"] = float(
                np.nanmean(np.nanmax(post, axis=1)))
            rep["checks"]["mean_posterior_entropy"] = float(
                np.nanmean(-np.nansum(
                    np.where(post > 0, post * np.log(post), 0.0), axis=1)))
        except Exception:
            rep["checks"]["mean_max_posterior"] = float("nan")
            rep["checks"]["mean_posterior_entropy"] = float("nan")
    else:
        rep["checks"]["mean_max_posterior"] = float("nan")
        rep["checks"]["mean_posterior_entropy"] = float("nan")

    # -- Hessian eigen-ratio / VDP (only when a Hessian is supplied) ----------
    cond = getattr(model, "cond_number", None)
    try:
        rep["checks"]["cond_number"] = (
            float(cond) if cond is not None and np.isfinite(float(cond))
            else float("nan"))
    except Exception:
        rep["checks"]["cond_number"] = float("nan")
    if hessian is not None:
        names = list(getattr(model, "coeff_names", None) or [])
        vdp = _vdp_report(hessian, ratio_thresh=ratio_thresh,
                          vdp_thresh=vdp_thresh, names=names)
        rep["checks"]["hessian_ratio_min"] = vdp["ratio_min"]
        rep["checks"]["hessian_vdp_flags"] = vdp["flags"]
        if vdp["flags"]:
            if rep["verdict"] == "ok":
                rep["verdict"] = "near-unidentified"
            rep["reasons"].append(
                "Hessian eigendirections near-singular: "
                + ", ".join(sorted(vdp["flags"])))
    else:
        rep["checks"]["hessian_ratio_min"] = float("nan")
        rep["checks"]["hessian_vdp_flags"] = {}

    return rep


# ---------------------------------------------------------------------------
# 3. Label alignment (for comparing specs across refits)
# ---------------------------------------------------------------------------

def align_classes(ref_coefs, ref_shares, other_coefs, other_shares):
    """Hungarian alignment of two class labellings (sidesteps label switching).

    Matches on shares first, breaking ties by coefficient SSE. Returns a dict
    ``ref_class -> other_class``. Falls back to identity without scipy.
    """
    ref_coefs = [np.asarray(b, dtype=float).ravel() for b in ref_coefs]
    other_coefs = [np.asarray(b, dtype=float).ravel() for b in other_coefs]
    ref_shares = np.asarray(ref_shares, dtype=float)
    other_shares = np.asarray(other_shares, dtype=float)
    C = len(ref_coefs)
    if (not _HAVE_SCIPY_LSA or len(other_coefs) != C
            or ref_shares.size != C or other_shares.size != C):
        return {c: c for c in range(C)}
    cost = np.zeros((C, C))
    L = max(max((b.size for b in ref_coefs), default=0),
            max((b.size for b in other_coefs), default=0))
    for a in range(C):
        ra = np.zeros(L)
        ra[:ref_coefs[a].size] = ref_coefs[a]
        for b in range(C):
            ob = np.zeros(L)
            ob[:other_coefs[b].size] = other_coefs[b]
            cost[a, b] = abs(float(ref_shares[a]) - float(other_shares[b]))
            denom = (np.abs(ra).sum() + np.abs(ob).sum()) / 2.0 + 1e-12
            cost[a, b] += float(np.abs(ra - ob).sum()) / denom
    rows, cols = _lsa(cost)
    return {int(r): int(c) for r, c in zip(rows, cols)}


# ---------------------------------------------------------------------------
# 4. Spec proposer
# ---------------------------------------------------------------------------

def propose_class_params_spec(model, vip=None, report=None, keep_t=1.0,
                              keep_p=0.30, min_share=0.01,
                              merge_cos_thresh=0.995, always_keep_top1=True):
    """Propose a refreshed ``class_params_spec`` from VIP + diagnostics.

    Rules (per surviving class):
      * keep variables with |t| >= ``keep_t`` (or p <= ``keep_p`` when SEs
        are missing/unreliable);
      * always keep the single top-importance variable (unless the class is
        dropped) and any intercept markers ('_inter'/'intercept') from the
        original spec;
      * drop classes with share < ``min_share`` (caller must refit with fewer
        classes — returned in ``notes``);
      * never auto-merge: near-duplicate pairs are reported for the caller.

    Returns ``(spec, notes)`` where ``spec`` is a list of per-class variable
    lists in the model's native naming (ready for ``class_params_spec``) and
    ``notes`` records every decision.
    """
    if vip is None:
        vip = vip_table(model)
    elif not isinstance(vip, pd.DataFrame):
        vip = pd.DataFrame(vip)
    if report is None:
        report = near_unidentified_report(model, min_share=min_share)
    per_class, _ = _class_var_lists(model)
    _shares_raw = getattr(model, "class_probs", None)
    shares = (np.asarray(_shares_raw, dtype=float).ravel()
              if _shares_raw is not None else np.array([], dtype=float))
    C = int(getattr(model, "n_classes", len(per_class)))

    orig_specs = getattr(model, "_class_specs", None)
    varnames = [str(v) for v in (getattr(model, "varnames", None) or [])]

    spec, notes = [], {"dropped_classes": [], "merge_suggestions": [],
                       "kept": {}, "verdict": report.get("verdict", "?")}
    tiny = set(report.get("checks", {}).get("tiny_share_classes", []))
    for c in range(C):
        share = float(shares[c]) if c < shares.size else np.nan
        if c in tiny or not np.isfinite(share) or share < min_share:
            notes["dropped_classes"].append(
                {"class": c, "share": share,
                 "reason": f"share {share:.4f} < {min_share}"})
            continue
        sub = vip[vip["class"] == c].copy() if len(vip) else vip
        keep = []
        if len(sub):
            use_t = bool((~sub["t_stat"].isna()).any())
            if use_t:
                keep = sub.loc[sub["t_stat"] >= keep_t, "variable"].tolist()
            else:
                keep = sub.loc[sub["importance"] >= keep_p,
                               "variable"].tolist() if "importance" in sub else []
            if always_keep_top1 and not keep:
                keep = [sub.iloc[0]["variable"]]
        # preserve intercept markers from the original spec
        try:
            orig_idx = (np.asarray(orig_specs[c]).ravel().tolist()
                        if orig_specs is not None and c < len(orig_specs) else [])
            for _i in orig_idx:
                _nm = varnames[int(_i)] if 0 <= int(_i) < len(varnames) else ""
                if _nm in ("_inter", "intercept") and _nm not in keep:
                    keep.append(_nm)
        except Exception:
            pass
        # de-duplicate, keep order
        seen, ordered = set(), []
        for v in keep:
            if v not in seen:
                seen.add(v)
                ordered.append(v)
        spec.append(ordered)
        notes["kept"][c] = {"share": share, "vars": ordered}
    for d in sorted(report.get("checks", {}).get("duplicate_class_pairs", []),
                     key=lambda r: (r["a"], r["b"])):
        if float(d.get("cosine", 0.0)) < merge_cos_thresh:
            continue
        notes["merge_suggestions"].append(
            {"classes": [d["a"], d["b"]], "cosine": float(d["cosine"]),
             "note": "near-duplicate classes (original indices) — consider "
                     "refitting with one fewer class instead of auto-merging"})
    notes["n_classes_proposed"] = len(spec)
    return spec, notes


# ---------------------------------------------------------------------------
# 5. Membership VIP — variable importance in the class-allocation equation
# ---------------------------------------------------------------------------

def _membership_arrays(model):
    """Return (G, SE, T, varnames, X) for the membership equation.

    G has shape (C-1, Km) over non-base classes; SE/T may be all-NaN when the
    model was fit without standard errors. X is the (n, Km) membership frame
    or None. Never copies the choice design.
    """
    G = getattr(model, "class_gammas", None)
    if G is None:
        return None, None, None, [], None
    G = np.asarray(G, dtype=float)
    Km = int(getattr(model, "K_membership", G.shape[-1] if G.ndim > 1 else 0))
    if G.ndim == 1:
        G = G.reshape(1, -1)
    C = int(getattr(model, "n_classes", G.shape[0] + 1))
    varnames = [str(v) for v in (getattr(model, "membership_vars", None) or [])]
    if len(varnames) < Km:
        varnames = varnames + [f"mem_{k}" for k in range(len(varnames), Km)]
    se = _as_1d_float(getattr(model, "gamma_se", None))
    tstats = _as_1d_float(getattr(model, "gamma_t_stats", None))
    n_exp = max(C - 1, 0) * Km
    SE = np.full((max(C - 1, 0), Km), np.nan)
    TT = np.full((max(C - 1, 0), Km), np.nan)
    if se.size == n_exp and n_exp:
        SE = se.reshape(C - 1, Km)
    if tstats.size == n_exp and n_exp:
        TT = tstats.reshape(C - 1, Km)
    X = getattr(model, "X_membership", None)
    X = np.asarray(X, dtype=float) if X is not None else None
    if X is not None and (X.ndim != 2 or X.shape[1] != Km):
        X = None
    return G, SE, TT, varnames[:Km], X


def membership_vip_table(model, saturate_thresh=6.0):
    """VIP for the membership (class-allocation) equation.

    One row per (non-base class, covariate): gamma, SE, t, importance, plus
    ``swing`` — the covariate's maximum log-odds swing across individuals,
    ``|gamma| * (max(x) - min(x))``. A swing above ``saturate_thresh``
    (default 6 log-points, ~400:1 odds) means this one variable can saturate
    the softmax and empty a class on its own: the collapse pathology.
    """
    G, SE, TT, varnames, X = _membership_arrays(model)
    if G is None or G.size == 0:
        return pd.DataFrame(columns=["class", "covariate", "gamma", "se",
                                     "t_stat", "importance", "swing",
                                     "saturating", "se_missing"])
    C1, Km = G.shape
    if X is not None:
        with np.errstate(all="ignore"):
            _rng = (np.nanmax(X, axis=0) - np.nanmin(X, axis=0))
        _rng = np.where(np.isfinite(_rng), _rng, np.nan)
    else:
        _rng = np.full(Km, np.nan)
    shares = np.asarray(getattr(model, "class_probs", None),
                        dtype=float).ravel()
    rows = []
    for c in range(C1):
        for k in range(Km):
            g = float(G[c, k])
            se = float(SE[c, k]) if SE.shape == G.shape else np.nan
            t = float(TT[c, k]) if TT.shape == G.shape else np.nan
            if not np.isfinite(t):
                t = (abs(g) / se) if np.isfinite(se) and se > 0 else np.nan
            nm = varnames[k] if k < len(varnames) else f"mem_{k}"
            swing = abs(g) * float(_rng[k]) if np.isfinite(_rng[k]) else np.nan
            rows.append({
                "class": c,
                "class_share": float(shares[c]) if c < shares.size else np.nan,
                "covariate": nm,
                "gamma": g,
                "se": se,
                "t_stat": t,
                "importance": abs(t) if np.isfinite(t) else 0.0,
                "swing": swing,
                "saturating": bool(np.isfinite(swing) and swing > saturate_thresh),
                "se_missing": bool(not np.isfinite(se) or se <= 0),
            })
    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values(["class", "importance"],
                            ascending=[True, False]).reset_index(drop=True)
    return df


def _implied_prior_shares(model):
    """Mean class priors implied by the fitted membership equation."""
    G, _, _, _, X = _membership_arrays(model)
    C = int(getattr(model, "n_classes", 0))
    if G is None or X is None or C < 2:
        return None
    try:
        logits = np.zeros((X.shape[0], C))
        logits[:, :G.shape[0]] = X @ G.T
        logits -= logits.max(axis=1, keepdims=True)
        exp_l = np.exp(logits)
        priors = exp_l / np.clip(exp_l.sum(axis=1, keepdims=True), 1e-300, None)
        return np.asarray(priors.mean(axis=0), dtype=float).ravel()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 7. Identification penalty — teach the spec loop that bad specs are bad
# ---------------------------------------------------------------------------

def identification_penalty(report, *, w_degenerate=100.0, w_near=10.0,
                           w_tiny_class=10.0, w_extreme_coef=5.0,
                           w_duplicate_pair=25.0, w_weak_param=1.0,
                           w_saturating_cov=10.0, w_pinned_gamma=5.0,
                           w_hess_flag=10.0):
    """Scalar >= 0 penalty encoding how bad a spec is (0 for clean specs).

    Accepts either a utility report (:func:`near_unidentified_report`) or a
    membership report (:func:`membership_identification_report`) — whichever
    check keys are present contribute. Add the result to BIC/AIC when ranking
    candidate specs so degenerate or near-unidentified specs lose even when
    their raw likelihood looks competitive.
    """
    if not isinstance(report, dict):
        return 0.0
    checks = report.get("checks", {}) or {}
    pen = 0.0
    verdict = report.get("verdict")
    if verdict == "degenerate":
        pen += float(w_degenerate)
    elif verdict == "near-unidentified":
        pen += float(w_near)
    tiny = (checks.get("tiny_share_classes", [])
            or checks.get("tiny_prior_classes", []))
    pen += float(w_tiny_class) * len(tiny or [])
    pen += float(w_extreme_coef) * len(checks.get("extreme_coefs", []) or [])
    pen += float(w_duplicate_pair) * len(
        checks.get("duplicate_class_pairs", []) or [])
    weak = checks.get("weak_or_unidentified_params", -1)
    try:
        weak = float(weak)
    except Exception:
        weak = -1.0
    if weak > 0:
        pen += float(w_weak_param) * weak
    pen += float(w_saturating_cov) * len(
        checks.get("saturating_covariates", []) or [])
    pen += float(w_pinned_gamma) * len(
        checks.get("bound_pinned_gammas", []) or [])
    pen += float(w_hess_flag) * len(
        checks.get("hessian_vdp_flags", {}) or {})
    return float(pen)


def score_spec(model, hessian=None, bic=None, min_share=0.01,
               penalty_kwargs=None):
    """Score a fitted spec as BIC + identification penalty (lower is better).

    Runs both the utility and membership near-unidentified batteries and adds
    their penalties to the model's BIC, so spec comparison automatically
    punishes collapse, saturation and weak identification. Returns a dict with
    ``bic``, ``penalty``, ``penalized_bic`` and both reports.
    """
    rep_u = near_unidentified_report(model, min_share=min_share,
                                     hessian=hessian)
    rep_m = membership_identification_report(model, min_share=min_share)
    kw = dict(penalty_kwargs or {})
    pen = identification_penalty(rep_u, **kw) + identification_penalty(rep_m,
                                                                       **kw)
    base = (float(bic) if bic is not None
            else float(getattr(model, "bic", np.nan)))
    out = base + pen if np.isfinite(base) else pen
    try:
        model.spec_score_ = {"bic": base, "penalty": pen,
                             "penalized_bic": float(out)}
    except Exception:
        pass
    return {"bic": base, "penalty": pen, "penalized_bic": float(out),
            "utility": rep_u, "membership": rep_m}


def membership_identification_report(model, min_share=0.01,
                                     saturate_thresh=6.0):
    """Near-unidentified test specialised to the membership equation.

    Flags (a) saturating covariates — any single variable whose swing can move
    log-odds by more than ``saturate_thresh``, i.e. collapse a class by
    itself; (b) implied prior shares below ``min_share``; (c) gammas pinned at
    an active ``gamma_max_abs`` bound. Returns a verdict dict like
    :func:`near_unidentified_report`.
    """
    rep = {"verdict": "ok", "reasons": [], "checks": {}}
    G, _, _, varnames, _ = _membership_arrays(model)
    if G is None or G.size == 0:
        rep["checks"]["has_membership"] = False
        return rep
    rep["checks"]["has_membership"] = True
    vip = membership_vip_table(model, saturate_thresh=saturate_thresh)
    sat = vip[vip["saturating"] == True] if len(vip) else vip  # noqa: E712
    rep["checks"]["saturating_covariates"] = [
        {"class": int(r["class"]), "covariate": str(r["covariate"]),
         "gamma": float(r["gamma"]), "swing": float(r["swing"])}
        for _, r in sat.iterrows()]
    if len(sat):
        rep["verdict"] = "near-unidentified"
        rep["reasons"].append(
            f"{len(sat)} membership term(s) can saturate assignment "
            f"(swing > {saturate_thresh} log-points)")
    prior_shares = _implied_prior_shares(model)
    if prior_shares is not None:
        rep["checks"]["implied_prior_shares"] = [float(s) for s in prior_shares]
        tiny = [int(c) for c, s in enumerate(prior_shares)
                if not np.isfinite(s) or s < min_share]
        rep["checks"]["tiny_prior_classes"] = tiny
        if tiny:
            rep["verdict"] = "degenerate"
            rep["reasons"].append(
                f"implied prior shares collapse classes {tiny} (< {min_share})")
    else:
        rep["checks"]["implied_prior_shares"] = []
        rep["checks"]["tiny_prior_classes"] = []
    _B = getattr(model, "gamma_max_abs", None)
    pinned = []
    if _B is not None:
        try:
            _b = float(_B)
            if np.isfinite(_b) and _b > 0:
                for c in range(G.shape[0]):
                    for k in range(G.shape[1]):
                        if abs(float(G[c, k])) >= 0.99 * _b:
                            nm = (varnames[k] if k < len(varnames)
                                  else f"mem_{k}")
                            pinned.append({"class": c, "covariate": nm,
                                           "gamma": float(G[c, k])})
        except Exception:
            pass
    rep["checks"]["bound_pinned_gammas"] = pinned
    if pinned and rep["verdict"] == "ok":
        rep["verdict"] = "near-unidentified"
    if pinned:
        rep["reasons"].append(
            f"{len(pinned)} gamma(s) pinned at the |gamma|<={_B} bound — "
            "the bound is doing the identifying, widen it only with a "
            "standardised spec")
    return rep


# ---------------------------------------------------------------------------
# 6. Membership spec proposer
# ---------------------------------------------------------------------------

def propose_member_params_spec(model, vip=None, report=None, keep_t=1.0,
                               min_share=0.01, saturate_thresh=6.0,
                               gamma_max_abs=None, share_floor=None):
    """Propose a collapse-proof ``member_params_spec``.

    Rules:
      * covariates are compared by *swing* (|gamma| x range), not raw gamma,
        so unstandardised scales cannot hide a saturating variable;
      * drop saturating covariates (swing > ``saturate_thresh``) unless that
        would empty the class equation — then keep the least-saturating
        top-importance covariate and flag it;
      * keep covariates with |t| >= ``keep_t`` plus the top-1 per class;
      * preserve '_inter' markers from the original spec;
      * recommend (not force) ``membership_standardize=True``,
        ``gamma_max_abs`` and ``share_floor`` values that make the proposed
        spec feasible: ``share_floor`` must satisfy
        ``share_floor <= 1/n_classes``.

    Returns ``(spec, notes)`` with ``spec`` as a per-class variable list
    (base class last, usually ``[]`` or ``['_inter']``) ready for
    ``member_params_spec``.
    """
    if vip is None:
        vip = membership_vip_table(model, saturate_thresh=saturate_thresh)
    elif not isinstance(vip, pd.DataFrame):
        vip = pd.DataFrame(vip)
    if report is None:
        report = membership_identification_report(
            model, min_share=min_share, saturate_thresh=saturate_thresh)

    G, _, _, varnames, _ = _membership_arrays(model)
    C = int(getattr(model, "n_classes", 0))
    Km = int(getattr(model, "K_membership", 0))
    orig = getattr(model, "member_params_spec", None)

    def _orig_markers(c):
        try:
            if orig is None:
                return []
            _o = list(orig)
            _per_class = bool(_o) and isinstance(_o[0], (list, tuple, np.ndarray))
            arr = list(_o[c]) if (_per_class and c < len(_o)) else []
            return [str(v) for v in arr if str(v) == "_inter"]
        except Exception:
            return []

    spec, notes = [], {"kept": {}, "dropped_saturating": [],
                       "verdict": report.get("verdict", "?"),
                       "recommendations": {}}
    for c in range(max(C - 1, 0)):
        sub = vip[vip["class"] == c].copy() if len(vip) else vip
        keep = []
        if len(sub):
            use_t = bool((~sub["t_stat"].isna()).any())
            key = "t_stat" if use_t else "importance"
            thresh = keep_t if use_t else 0.0
            cand = sub.loc[sub[key] >= thresh].copy()
            # drop saturating covariates first
            if "swing" in cand.columns:
                _sat = cand[cand["swing"] > saturate_thresh]
                for _, r in _sat.iterrows():
                    notes["dropped_saturating"].append(
                        {"class": c, "covariate": str(r["covariate"]),
                         "gamma": float(r["gamma"]),
                         "swing": float(r["swing"])})
                cand = cand.loc[~(cand["swing"] > saturate_thresh)]
            keep = cand["covariate"].tolist()
            if not keep:
                # keep the least-saturating top-importance covariate over
                # dropping the whole equation (flagged, not silent)
                sub2 = sub.sort_values("importance", ascending=False)
                keep = [sub2.iloc[0]["covariate"]]
                notes["dropped_saturating"].append(
                    {"class": c, "covariate": str(keep[0]),
                     "note": "kept as last resort despite saturation risk"})
        for _m in _orig_markers(c):
            if _m not in keep:
                keep.append(_m)
        seen, ordered = set(), []
        for v in keep:
            if v not in seen:
                seen.add(v)
                ordered.append(v)
        spec.append(ordered)
        notes["kept"][c] = {"vars": ordered}
    # base class: no free gammas (reference); preserve an '_inter' marker only
    # if the original spec carried one there.
    try:
        _base_marks = _orig_markers(C - 1) if C > 0 else []
    except Exception:
        _base_marks = []
    spec.append([m for m in _base_marks if m == "_inter"])

    # feasibility + standardisation recommendations
    notes["recommendations"]["membership_standardize"] = (
        getattr(model, "_memb_standardize_", None) is None)
    if gamma_max_abs is None:
        _cur = getattr(model, "gamma_max_abs", None)
        try:
            _cur = float(_cur) if _cur is not None else None
        except Exception:
            _cur = None
        gamma_max_abs = _cur if (_cur is not None and _cur > 0) else 3.0
    notes["recommendations"]["gamma_max_abs"] = gamma_max_abs
    notes["recommendations"]["gamma_max_abs_note"] = (
        "box bound is only meaningful on standardised covariates — refit with "
        "membership_standardize=True if the current spec was not standardised")
    _floor = share_floor if share_floor is not None else min_share
    notes["recommendations"]["share_floor"] = (
        float(_floor) if C > 0 and float(_floor) <= 1.0 / C else None)
    if C > 0 and float(_floor or 0.0) > 1.0 / C:
        notes["recommendations"]["share_floor_note"] = (
            f"share_floor={_floor} infeasible for {C} classes "
            f"(must be <= {1.0 / C:.4f}); left unset")
    notes["n_classes"] = C
    return spec, notes
