"""
compare_mxl_vs_gse_mxl.py
=========================
Compare three model specifications for handling unobserved heterogeneity:

  A) Standard Mixed Logit (MXL)
     All 4 variables random with normal mixing; COST uses negative lognormal
     (via data negation + 'ln' distribution).

  B) GSE-Informed Mixed Logit
     MNL gradient diagnostics identify TIME and COST as the heterogeneous
     variables. HEADWAY and SEATS are kept fixed.
     COST uses negative lognormal; TIME uses normal mixing.

  C) GSE-Informed Latent Class
     Gradient clustering (from MNL scores) suggests 3 latent segments.
     A Latent-Class MNL is fitted with membership initialised from
     the gradient-based cluster assignments.

Comparison metrics: LL, AIC, BIC, num_params, estimation time, parameter estimates.
"""

from __future__ import annotations

import time
import numpy as np
from scipy import stats
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

from SearchLibrium.multinomial_logit import MultinomialLogit
from SearchLibrium.MixedLogit import MixedLogit
from SearchLibrium.latent_class import LatentClassMixedLogit
from SearchLibrium.sample_data import load_swiss_metro_data


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def compute_mnl_individual_gradients(model: MultinomialLogit) -> np.ndarray:
    """Per-choice-occasion MNL score contributions at the MLE. (N_occ, K)."""
    betas = model.coeff_est
    X = model.X
    y = model.y
    avail = model.avail
    p = model.compute_probabilities(betas, X, avail)
    ymp = y - p
    Kf, Kftrans = model.Kf, model.Kftrans
    g = np.einsum("nj,njk->nk", ymp, X[:, :, model.fxidx]) if Kf > 0 \
        else np.zeros((ymp.shape[0], 0))
    if Kftrans > 0:
        transpos = [model.varnames.tolist().index(v) for v in model.transvars]
        X_trans = X[:, :, transpos].reshape(model.N, len(model.alts), len(transpos))
        lambdas = betas[Kf + Kftrans:]
        Xtrans_lmda = model.trans_func(X_trans, lambdas)
        gtrans = np.einsum("nj,njk->nk", ymp, Xtrans_lmda)
        der_Xtrans_lmda = model.transform_deriv(X_trans, lambdas)
        B_trans = betas[Kf:Kf + Kftrans]
        der_XBtrans = np.einsum("njk,k->njk", der_Xtrans_lmda, B_trans)
        gtrans_lmda = np.einsum("nj,njk->nk", ymp, der_XBtrans)
        g = np.concatenate((g, gtrans, gtrans_lmda), axis=1) if g.size \
            else np.concatenate((gtrans, gtrans_lmda), axis=1)
    return g


def fit_one_mxl(df, varnames, randvars, label, base_alt, n_draws=500):
    """Fit a single MixedLogit and return timing + results dict."""
    t0 = time.perf_counter()
    m = MixedLogit()
    m.setup(
        X=df[varnames].values,
        y=df["CHOICE"].astype(int).values,
        varnames=varnames,
        alts=df["alt"].values,
        ids=df["custom_id"].values,
        panels=df["ID"].values,
        randvars=randvars,
        base_alt=base_alt,
        n_draws=n_draws,
        mnl_init=True,
        return_hess=True,
        return_grad=True,
        ftol=1e-4,
        gtol=1e-4,
    )
    m.fit()
    elapsed = time.perf_counter() - t0
    n_params = len(m.coeff_est)
    return {
        "label": label,
        "model": m,
        "loglik": m.loglik,
        "aic": m.aic,
        "bic": m.bic,
        "n_params": n_params,
        "time": elapsed,
        "converged": m.converged,
        "coeff_est": m.coeff_est,
        "coeff_names": list(m.Xnames) if hasattr(m, "Xnames") else [],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(seed: int = 42):
    np.random.seed(seed)
    t_total_start = time.perf_counter()

    # ---- Data ----------------------------------------------------------------
    df = load_swiss_metro_data()
    alts = sorted(df["alt"].unique().tolist())
    base_alt = "SM"
    n_occ = df["custom_id"].nunique()
    n_ind = df["ID"].nunique()
    n_panels = 9

    print("=" * 78)
    print("  Model Comparison: Standard MXL vs GSE-Informed MXL")
    print("=" * 78)
    print(f"  Swiss Metro  |  {n_occ} occasions  |  {n_ind} individuals  |  P = {n_panels}")

    # ---- Prepare data with negative lognormal COST --------------------------
    df["COST_NEG"] = -df["COST"]
    base_varnames = ["TIME", "COST_NEG", "HEADWAY", "SEATS"]

    # ---- Step 0: Fit MNL + gradient diagnostics -----------------------------
    print("\n--- Gradient Diagnostics (MNL) ---")
    t_mnl = time.perf_counter()
    mnl = MultinomialLogit()
    mnl.setup(
        X=df[["TIME", "COST", "HEADWAY", "SEATS"]].values,
        y=df["CHOICE"].astype(int).values,
        varnames=["TIME", "COST", "HEADWAY", "SEATS"],
        alts=df["alt"].values,
        ids=df["custom_id"].values,
        base_alt=base_alt,
        return_hess=True,
        return_grad=True,
    )
    mnl.fit()
    mnl_ll = mnl.loglik
    t_mnl_elapsed = time.perf_counter() - t_mnl

    # Extract gradients for diagnostics
    occ_map = (
        df[["custom_id", "ID"]]
        .drop_duplicates()
        .sort_values("custom_id")
        .reset_index(drop=True)
    )
    unique_ind = occ_map["ID"].values
    ind_to_occ = {}
    for i, ind_id in enumerate(unique_ind):
        ind_to_occ.setdefault(ind_id, []).append(i)

    g_occ = compute_mnl_individual_gradients(mnl)
    individual_list = sorted(ind_to_occ.keys())
    g_ind = np.zeros((len(individual_list), g_occ.shape[1]))
    for idx, ind_id in enumerate(individual_list):
        g_ind[idx, :] = g_occ[ind_to_occ[ind_id], :].sum(axis=0)
    g_avg = g_ind / n_panels

    # Score variance diagnostics
    mnl_names = list(mnl.Xnames)
    score_var = np.var(g_avg, axis=0)
    rel_var = score_var / (score_var.max() + 1e-12)

    print(f"  MNL LL = {mnl_ll:.4f}  |  time = {t_mnl_elapsed:.2f}s")
    print(f"  Score variance (descending):")
    order = np.argsort(score_var)[::-1]
    for k in order:
        is_het = "HET" if rel_var[k] > 0.01 else "FIX"
        print(f"    {str(mnl_names[k]):>12s}  var={score_var[k]:10.2f}  "
              f"rel={rel_var[k]:.4f}  -> {is_het}")

    # Determine gradient-informed specification
    hetero_vars = [str(mnl_names[k]) for k in order if rel_var[k] > 0.01]
    # Map to MXL variable names (COST -> COST_NEG)
    hetero_vars_mxl = ["COST_NEG" if v == "COST" else v for v in hetero_vars]
    fixed_vars_mxl = [v for v in base_varnames if v not in hetero_vars_mxl]

    print(f"\n  GSE-informed: random = {hetero_vars_mxl}, fixed = {fixed_vars_mxl}")

    # ---- Latent class info from gradient clustering -------------------------
    g_std = (g_ind - g_ind.mean(axis=0)) / (g_ind.std(axis=0) + 1e-8)
    n_classes = 3
    km = KMeans(n_clusters=n_classes, random_state=seed, n_init=10)
    labels = km.fit_predict(g_std)
    class_sizes = np.bincount(labels)
    membership_prior = np.zeros((len(individual_list), n_classes))
    for i, lab in enumerate(labels):
        membership_prior[i, lab] = 0.9
        for c in range(n_classes):
            if c != lab:
                membership_prior[i, c] = 0.1 / (n_classes - 1)
    membership_prior = membership_prior / membership_prior.sum(axis=1, keepdims=True)

    # =========================================================================
    #  Model A: Standard MXL (all random, normal mixing)
    # =========================================================================
    print("\n" + "=" * 78)
    print("  MODEL A: Standard Mixed Logit (all 4 vars random, normal)")
    print("=" * 78)

    randvars_std = {"TIME": "n", "COST_NEG": "n", "HEADWAY": "n", "SEATS": "n"}
    res_a = fit_one_mxl(df, base_varnames, randvars_std, "Standard MXL (all normal)", base_alt)

    print(f"  LL = {res_a['loglik']:.4f}  |  AIC = {res_a['aic']:.1f}  |  "
          f"BIC = {res_a['bic']:.1f}  |  params = {res_a['n_params']}  |  "
          f"time = {res_a['time']:.1f}s")

    # =========================================================================
    #  Model B: Standard MXL with negative lognormal COST
    # =========================================================================
    print("\n" + "=" * 78)
    print("  MODEL B: Standard MXL (all 4 vars random, neg-lognormal COST)")
    print("=" * 78)

    randvars_std_ln = {"TIME": "n", "COST_NEG": "ln", "HEADWAY": "n", "SEATS": "n"}
    res_b = fit_one_mxl(df, base_varnames, randvars_std_ln,
                        "Standard MXL (neg-ln COST)", base_alt)

    print(f"  LL = {res_b['loglik']:.4f}  |  AIC = {res_b['aic']:.1f}  |  "
          f"BIC = {res_b['bic']:.1f}  |  params = {res_b['n_params']}  |  "
          f"time = {res_b['time']:.1f}s")

    # =========================================================================
    #  Model C: GSE-Informed MXL (gradient-selected randvars, neg-ln COST)
    # =========================================================================
    print("\n" + "=" * 78)
    print("  MODEL C: GSE-Informed MXL (gradient-selected randvars)")
    print("=" * 78)

    # Build randvars from gradient diagnostics
    randvars_gse = {}
    for v in hetero_vars_mxl:
        if v == "COST_NEG":
            randvars_gse[v] = "ln"   # negative lognormal
        else:
            randvars_gse[v] = "n"    # normal for time-like vars

    print(f"  randvars = {randvars_gse}")
    res_c = fit_one_mxl(df, base_varnames, randvars_gse,
                        "GSE-Informed MXL", base_alt)

    print(f"  LL = {res_c['loglik']:.4f}  |  AIC = {res_c['aic']:.1f}  |  "
          f"BIC = {res_c['bic']:.1f}  |  params = {res_c['n_params']}  |  "
          f"time = {res_c['time']:.1f}s")

    # =========================================================================
    #  Model D: Standard Latent Class (all vars, uniform membership init)
    # =========================================================================
    print("\n" + "=" * 78)
    print("  MODEL D: Standard Latent Class (3 classes, all vars)")
    print("=" * 78)

    t_d = time.perf_counter()
    lc_std = LatentClassMixedLogit(n_classes=n_classes, maxiter=100, random_state=seed)
    try:
        lc_std.setup(
            X=df[base_varnames].values,
            y=df["CHOICE"].astype(int).values,
            varnames=base_varnames,
            ids=df["custom_id"].values,
            alts=df["alt"].values,
        )
        lc_std.fit(em_method="squarem")
        t_d_elapsed = time.perf_counter() - t_d
        res_d = {
            "label": "Standard Latent Class",
            "loglik": lc_std.loglik,
            "aic": lc_std.aic,
            "bic": lc_std.bic,
            "n_params": lc_std.num_params,
            "time": t_d_elapsed,
            "converged": lc_std.converged,
        }
        print(f"  LL = {res_d['loglik']:.4f}  |  AIC = {res_d['aic']:.1f}  |  "
              f"BIC = {res_d['bic']:.1f}  |  params = {res_d['n_params']}  |  "
              f"time = {res_d['time']:.1f}s")
    except Exception as e:
        print(f"  LC setup failed: {e}")
        res_d = {"label": "Standard Latent Class", "loglik": np.nan, "aic": np.nan,
                 "bic": np.nan, "n_params": 0, "time": 0, "converged": False}

    # =========================================================================
    #  Model E: GSE-Informed Latent Class (gradient-init membership)
    # =========================================================================
    print("\n" + "=" * 78)
    print("  MODEL E: GSE-Informed Latent Class (gradient-init membership)")
    print("=" * 78)

    t_e = time.perf_counter()
    lc_gse = LatentClassMixedLogit(n_classes=n_classes, maxiter=100, random_state=seed)
    try:
        lc_gse.setup(
            X=df[base_varnames].values,
            y=df["CHOICE"].astype(int).values,
            varnames=base_varnames,
            ids=df["custom_id"].values,
            alts=df["alt"].values,
        )
        # Inject gradient-based membership prior
        lc_gse.membership_prior = membership_prior
        lc_gse.fit(em_method="squarem")
        t_e_elapsed = time.perf_counter() - t_e
        res_e = {
            "label": "GSE-Informed Latent Class",
            "loglik": lc_gse.loglik,
            "aic": lc_gse.aic,
            "bic": lc_gse.bic,
            "n_params": lc_gse.num_params,
            "time": t_e_elapsed,
            "converged": lc_gse.converged,
        }
        print(f"  LL = {res_e['loglik']:.4f}  |  AIC = {res_e['aic']:.1f}  |  "
              f"BIC = {res_e['bic']:.1f}  |  params = {res_e['n_params']}  |  "
              f"time = {res_e['time']:.1f}s")
    except Exception as e:
        print(f"  GSE-LC failed: {e}")
        res_e = {"label": "GSE-Informed Latent Class", "loglik": np.nan, "aic": np.nan,
                 "bic": np.nan, "n_params": 0, "time": 0, "converged": False}

    # =========================================================================
    #  Summary comparison table
    # =========================================================================
    print("\n" + "=" * 78)
    print("  COMPARISON SUMMARY")
    print("=" * 78)

    all_results = [res_a, res_b, res_c, res_d, res_e]
    # Add MNL baseline
    all_results.insert(0, {
        "label": "MNL (baseline)",
        "loglik": mnl_ll,
        "aic": 2 * len(mnl.coeff_est) - 2 * mnl_ll,
        "bic": np.log(n_ind) * len(mnl.coeff_est) - 2 * mnl_ll,
        "n_params": len(mnl.coeff_est),
        "time": t_mnl_elapsed,
        "converged": True,
    })

    print(f"\n  {'Model':<35s}  {'LL':>10s}  {'AIC':>10s}  {'BIC':>10s}  "
          f"{'Params':>7s}  {'Time':>7s}  {'dLL vs MNL':>12s}  {'vs best':>10s}")
    print(f"  {'-'*35}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*7}  {'-'*7}  "
          f"{'-'*12}  {'-'*10}")

    best_ll = max(r["loglik"] for r in all_results if not np.isnan(r["loglik"]))
    best_aic = min(r["aic"] for r in all_results if not np.isnan(r["aic"]))
    best_bic = min(r["bic"] for r in all_results if not np.isnan(r["bic"]))

    for r in all_results:
        dll = r["loglik"] - mnl_ll if not np.isnan(r["loglik"]) else np.nan
        aic_gap = r["aic"] - best_aic if not np.isnan(r["aic"]) else np.nan
        bic_gap = r["bic"] - best_bic if not np.isnan(r["bic"]) else np.nan
        best_str = f"AIC+{aic_gap:.0f}" if aic_gap > 0 else "*** BEST ***"
        if np.isnan(aic_gap):
            best_str = "FAILED"

        print(f"  {r['label']:<35s}  {r['loglik']:10.2f}  {r['aic']:10.1f}  "
              f"{r['bic']:10.1f}  {r['n_params']:7d}  {r['time']:6.1f}s  "
              f"{dll:+12.2f}  {best_str:>10s}")

    # =========================================================================
    #  Parameter comparison
    # =========================================================================
    print(f"\n{'='*78}")
    print(f"  PARAMETER ESTIMATES")
    print(f"{'='*78}")

    # Print coefficient comparison for MXL models
    for res in [res_a, res_b, res_c]:
        if res["n_params"] == 0:
            continue
        print(f"\n  {res['label']}:")
        names = res["coeff_names"]
        vals = res["coeff_est"]
        if len(names) == len(vals):
            for n, v in zip(names, vals):
                print(f"    {str(n):>20s}  {v:12.6f}")
        else:
            print(f"    names={len(names)}, vals={len(vals)} -- alignment mismatch")

    # =========================================================================
    #  Visual comparison
    # =========================================================================
    models_plot = [r for r in all_results if r["n_params"] > 0 and not np.isnan(r["loglik"])]
    labels_plot = [r["label"] for r in models_plot]
    ll_plot = [r["loglik"] for r in models_plot]
    aic_plot = [r["aic"] for r in models_plot]
    bic_plot = [r["bic"] for r in models_plot]
    time_plot = [r["time"] for r in models_plot]
    params_plot = [r["n_params"] for r in models_plot]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # LL / AIC / BIC
    ax = axes[0]
    x = np.arange(len(labels_plot))
    w = 0.25
    ax.bar(x - w, ll_plot, w, label="LL", color="steelblue")
    ax.bar(x, aic_plot, w, label="AIC", color="darkorange")
    ax.bar(x + w, bic_plot, w, label="BIC", color="lightcoral")
    ax.set_xticks(x)
    ax.set_xticklabels(labels_plot, rotation=45, ha="right", fontsize=8)
    ax.set_title("Log-Likelihood / AIC / BIC")
    ax.legend(fontsize=8)
    ax.axhline(y=mnl_ll, color="gray", linestyle="--", lw=0.8, label="MNL LL")

    # Time
    ax = axes[1]
    bars = ax.bar(labels_plot, time_plot, color=["lightgray", "steelblue", "steelblue",
                                                   "darkorange", "green", "green"][:len(labels_plot)])
    ax.set_title("Estimation Time (s)")
    ax.set_xticklabels(labels_plot, rotation=45, ha="right", fontsize=8)
    for bar, t in zip(bars, time_plot):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{t:.1f}s", ha="center", fontsize=8)

    # Params
    ax = axes[2]
    bars = ax.bar(labels_plot, params_plot, color=["lightgray", "steelblue", "steelblue",
                                                     "darkorange", "green", "green"][:len(labels_plot)])
    ax.set_title("Number of Parameters")
    ax.set_xticklabels(labels_plot, rotation=45, ha="right", fontsize=8)
    for bar, p in zip(bars, params_plot):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                str(p), ha="center", fontsize=9)

    fig.suptitle("Model Comparison: Standard MXL vs GSE-Informed Specifications",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    out_path = "mxl_comparison.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nFigure saved to '{out_path}'")

    t_total = time.perf_counter() - t_total_start
    print(f"\n  Total time: {t_total:.1f}s")

    return all_results


if __name__ == "__main__":
    results = main()
