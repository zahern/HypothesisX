"""
wtp_gradient_comparison.py
==========================
Willingness-to-Pay comparison: Mixed Logit vs Gradient-Structural-Equation.

Both methods estimate individual-level parameters for WTP:
  WTP_k = -beta_k / beta_cost

Methods
-------
A) Mixed Logit (MXL):  beta_n = posterior conditional means (simulation-based).

B) Gradient-Structural-Equation (GSE):
   For normally-distributed parameters:
       beta_n = mu_prior + Sigma_prior @ g_n
   where g_n is the individual MNL score aggregated across occasions.
   The structural model operates on the transformed scale for lognormal
   cost, using MNL's MNL-based Hinv (@ g_n) as the first-order correction.

Cost specification
------------------
Negative lognormal COST: the COST variable is negated, then given 'ln'
distribution. Effective cost coefficient = -exp(mu + sigma * z) <= 0.
"""

from __future__ import annotations

import time
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

from SearchLibrium.multinomial_logit import MultinomialLogit
from SearchLibrium.MixedLogit import MixedLogit
from SearchLibrium.sample_data import load_swiss_metro_data


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def compute_mnl_individual_gradients(model: MultinomialLogit) -> np.ndarray:
    """Per-occasion MNL score contributions at the MLE.  Returns (N_occ, K)."""
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(seed: int = 42):
    np.random.seed(seed)
    t0 = time.perf_counter()

    # ---- Data ----------------------------------------------------------------
    df = load_swiss_metro_data()
    varnames = ["TIME", "COST", "HEADWAY", "SEATS"]
    alts = sorted(df["alt"].unique().tolist())
    base_alt = "SM"
    n_occ = df["custom_id"].nunique()
    n_ind = df["ID"].nunique()
    n_panels = 9

    print("=" * 78)
    print("  WTP Comparison: Mixed Logit  vs  Gradient-Structural-Equation")
    print("=" * 78)
    print(f"  Data: Swiss Metro  |  {n_occ} choice occasions  |  "
          f"{n_ind} individuals  |  P = {n_panels}")

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

    # =========================================================================
    #  Step 1:  Estimate Simple MNL
    # =========================================================================
    print("\n--- Step 1: Estimate MNL ---")
    t_mnl_start = time.perf_counter()
    mnl = MultinomialLogit()
    mnl.setup(
        X=df[varnames].values,
        y=df["CHOICE"].astype(int).values,
        varnames=varnames,
        alts=df["alt"].values,
        ids=df["custom_id"].values,
        base_alt=base_alt,
        return_hess=True,
        return_grad=True,
    )
    mnl.fit()
    t_mnl = time.perf_counter() - t_mnl_start
    beta_mnl = mnl.coeff_est
    mnl_names = list(mnl.Xnames)
    print(f"  MNL LL = {mnl.loglik:.4f}  |  time = {t_mnl:.1f}s  |  K = {len(beta_mnl)}")
    for k, (name, b) in enumerate(zip(mnl_names, beta_mnl)):
        print(f"    {name:>12s}  {b:12.6f}")

    # ---- MNL individual gradients, aggregate to individual level ------------
    g_occ = compute_mnl_individual_gradients(mnl)
    individual_list = sorted(ind_to_occ.keys())
    g_ind = np.zeros((len(individual_list), g_occ.shape[1]))
    for idx, ind_id in enumerate(individual_list):
        g_ind[idx, :] = g_occ[ind_to_occ[ind_id], :].sum(axis=0)
    print(f"  MNL grad shape: occ={g_occ.shape}  ind={g_ind.shape}")

    # OPG Hessian inverse for later use
    H = g_occ.T @ g_occ
    H[H == 0] = 1e-12
    Hinv_mnl = np.linalg.pinv(H)

    # =========================================================================
    #  Step 2:  Estimate Mixed Logit (negative lognormal COST)
    # =========================================================================
    print("\n--- Step 2: Estimate Mixed Logit (negative lognormal COST) ---")
    df_mxl = df.copy()
    df_mxl["COST_NEG"] = -df_mxl["COST"]
    mxl_varnames = ["TIME", "COST_NEG", "HEADWAY", "SEATS"]
    randvars = {"TIME": "n", "COST_NEG": "ln", "HEADWAY": "n", "SEATS": "n"}
    rvdist_list = list(randvars.values())

    t_mxl_start = time.perf_counter()
    mxl = MixedLogit()
    mxl.setup(
        X=df_mxl[mxl_varnames].values,
        y=df_mxl["CHOICE"].astype(int).values,
        varnames=mxl_varnames,
        alts=df_mxl["alt"].values,
        ids=df_mxl["custom_id"].values,
        panels=df_mxl["ID"].values,
        randvars=randvars,
        base_alt=base_alt,
        n_draws=500,
        mnl_init=True,
        return_hess=True,
        return_grad=True,
        ftol=1e-4,
        gtol=1e-4,
    )
    mxl.fit()
    t_mxl = time.perf_counter() - t_mxl_start

    if not hasattr(mxl, "pch2_res") or mxl.pch2_res is None:
        mxl.compute_fitted_params(mxl.y, mxl.p, mxl.panel_info, mxl.Br)
    beta_mxl_ind = mxl.pch2_res  # (N_ind, Kr)  natural-scale posterior means
    print(f"  MXL LL = {mxl.loglik:.4f}  |  time = {t_mxl:.1f}s")

    # ---- Extract MXL prior on transformed scale ------------------------------
    beta_mxl_full = mxl.coeff_est
    Kr = mxl.Kr
    rx_mean = beta_mxl_full[:Kr]
    rx_sd   = abs(beta_mxl_full[Kr:Kr + mxl.Kbw])

    print(f"  MXL prior (transformed scale):")
    for k in range(Kr):
        print(f"    {mxl_varnames[k]:>12s}  mean={rx_mean[k]:10.4f}  sd={rx_sd[k]:.4f}  "
              f"dist={rvdist_list[k]}  post_mean={np.mean(beta_mxl_ind[:, k]):.6f}")

    # Compute prior means and vars on natural scale
    prior_mean_nat = np.zeros(Kr)
    prior_var_nat = np.zeros(Kr)
    for k in range(Kr):
        mu = rx_mean[k]
        sd = rx_sd[k]
        if rvdist_list[k] == "ln":
            s2 = sd * sd
            prior_mean_nat[k] = np.exp(mu + s2 / 2)
            prior_var_nat[k] = np.exp(2 * mu + s2) * (np.exp(s2) - 1)
        else:
            prior_mean_nat[k] = mu
            prior_var_nat[k] = sd * sd
    Sigma_prior = np.diag(prior_var_nat)

    # =========================================================================
    #  Step 3:  Gradient-Structural-Equation (GSE) individual parameters
    # =========================================================================
    print("\n--- Step 3: Gradient-Structural-Equation (GSE) ---")
    t_gse_start = time.perf_counter()

    # Map MNL params -> MXL random vars
    mnl_var_to_idx = {v: mnl_names.index(v) if v in mnl_names
                      else mnl_names.index("COST") for v in mxl_varnames}
    mnl_rand_idx = [mnl_var_to_idx[v] for v in mxl_varnames]
    g_sub = g_ind[:, mnl_rand_idx]  # (N_ind, Kr)

    # ---- GSE with scale-appropriate correction -------------------------------
    # For normal  (n):  beta_n = mu + sd^2 * g_n              [natural scale]
    # For lognormal (ln): use MNL Hinv-scaled correction on log scale
    #   g_log = g_nat * exp(mu)                               [chain rule]
    #   mu_n  = mu + min(sd^2 * g_log, cap)                   [capped]
    #   beta_n = exp(mu_n)

    beta_gse_nat = np.zeros((len(individual_list), Kr))

    # For stability, cap the correction magnitude
    cap_log = 20.0  # prevents exp overflow

    for k in range(Kr):
        mu_k = rx_mean[k]
        sd_k = rx_sd[k]
        g_k  = g_sub[:, k]

        if rvdist_list[k] == "ln":
            g_log = g_k * np.exp(mu_k)           # chain rule
            correction = sd_k * sd_k * g_log
            correction = np.clip(correction, -cap_log, cap_log)
            beta_gse_nat[:, k] = np.exp(mu_k + correction)
        else:
            # For normal: use the MNL Hessian inverse for better scaling
            # Hinv_scaled = estimate the parameter variance from MNL Hessian
            # Use a mixture: partial Hinv + Sigma_prior for regularization
            # beta_n = mu + (Sigma_prior / (1 + lambda)) * g_n
            # where lambda = mean(abs(g_n)) / mean(abs(mu)) regularizes
            reg = np.mean(np.abs(g_k)) / (np.abs(mu_k) + 1e-6)
            alpha = 1.0 / (1.0 + 0.1 * reg)  # damping factor
            correction = sd_k * sd_k * g_k * alpha
            correction = np.clip(correction, -10.0, 10.0)
            beta_gse_nat[:, k] = mu_k + correction

    t_gse = time.perf_counter() - t_gse_start

    # ---- Evaluate GSE log-likelihood -----------------------------------------
    X_mnl = mnl.X
    y_mnl = mnl.y
    avail_mnl = mnl.avail
    eps = 1e-30

    ll_gse_ind = np.zeros(len(individual_list))
    for idx in range(len(individual_list)):
        occ_rows = ind_to_occ[individual_list[idx]]
        b = beta_gse_nat[idx, :]
        b_full = beta_mnl.copy()
        for j, k in enumerate(mnl_rand_idx):
            b_full[k] = b[j]

        acc = 0.0
        for r in occ_rows:
            Xr = X_mnl[r, :, mnl.fxidx]
            br = b_full[mnl.fxidx]
            if Xr.shape[1] == len(br):
                V = Xr.dot(br)
            else:
                V = Xr.T.dot(br)
            V = V - V.max()
            eV = np.exp(V)
            if avail_mnl is not None:
                eV = eV * avail_mnl[r]
            denom = np.sum(eV)
            if denom <= eps:
                denom = eps
            p = eV / denom
            lik = np.sum(y_mnl[r] * p)
            if lik <= eps:
                lik = eps
            acc += np.log(lik)
        ll_gse_ind[idx] = acc
    ll_gse = np.sum(ll_gse_ind)

    # Also compute MXL-posterior LL (using MXL post means on MNL structure)
    ll_mxlpost_ind = np.zeros(len(individual_list))
    for idx in range(len(individual_list)):
        occ_rows = ind_to_occ[individual_list[idx]]
        b = beta_mxl_ind[idx, :]
        b_full = beta_mnl.copy()
        for j, k in enumerate(mnl_rand_idx):
            b_full[k] = b[j]
        acc = 0.0
        for r in occ_rows:
            Xr = X_mnl[r, :, mnl.fxidx]
            br = b_full[mnl.fxidx]
            if Xr.shape[1] == len(br):
                V = Xr.dot(br)
            else:
                V = Xr.T.dot(br)
            V = V - V.max()
            eV = np.exp(V)
            if avail_mnl is not None:
                eV = eV * avail_mnl[r]
            denom = np.sum(eV)
            if denom <= eps:
                denom = eps
            p = eV / denom
            lik = np.sum(y_mnl[r] * p)
            if lik <= eps:
                lik = eps
            acc += np.log(lik)
        ll_mxlpost_ind[idx] = acc
    ll_mxlpost = np.sum(ll_mxlpost_ind)

    print(f"  GSE computation time    : {t_gse:.3f} s")
    print(f"  GSE LL (MNL structure)  : {ll_gse:.4f}")
    print(f"  MXL post LL (MNL struct): {ll_mxlpost:.4f}")

    # =========================================================================
    #  WTP Computation
    # =========================================================================
    print("\n" + "=" * 78)
    print("  WTP ANALYSIS  (WTP_k = beta_k / beta_COST)")
    print("=" * 78)

    idx_time = 0
    idx_cost = 1
    idx_head = 2
    idx_seat = 3

    # ---- MXL WTP (natural scale) -----
    cost_mxl = beta_mxl_ind[:, idx_cost]
    wtp_mxl_time    = beta_mxl_ind[:, idx_time]  / cost_mxl
    wtp_mxl_headway = beta_mxl_ind[:, idx_head]  / cost_mxl
    wtp_mxl_seats   = beta_mxl_ind[:, idx_seat]  / cost_mxl

    # ---- GSE WTP (natural scale) ----
    cost_gse = beta_gse_nat[:, idx_cost]
    wtp_gse_time    = beta_gse_nat[:, idx_time]  / cost_gse
    wtp_gse_headway = beta_gse_nat[:, idx_head]  / cost_gse
    wtp_gse_seats   = beta_gse_nat[:, idx_seat]  / cost_gse

    # ---- MNL point WTP ----
    cost_mnl = abs(beta_mnl[1])
    wtp_mnl_time    = -beta_mnl[0] / cost_mnl
    wtp_mnl_headway = -beta_mnl[2] / cost_mnl
    wtp_mnl_seats   =  beta_mnl[3] / cost_mnl

    # ---- Clean: filter extreme WTP values (> 1e6 or NaN/inf) ---------------
    attrs = {
        "TIME":    (wtp_mnl_time,    wtp_mxl_time,    wtp_gse_time),
        "HEADWAY": (wtp_mnl_headway, wtp_mxl_headway, wtp_gse_headway),
        "SEATS":   (wtp_mnl_seats,   wtp_mxl_seats,   wtp_gse_seats),
    }

    # ---- WTP Distribution Summary -------------------------------------------
    print(f"\n  {'':>10s}  {'MNL':>12s}  {'MXL Mean':>12s}  {'MXL SD':>12s}  "
          f"{'MXL Med':>12s}  {'GSE Mean':>12s}  {'GSE SD':>12s}  "
          f"{'GSE Med':>12s}  {'Corr':>8s}  {'MAD':>10s}")
    print(f"  {'':>10s}  {'':>12s}  {'':>12s}  {'':>12s}  "
          f"{'':>12s}  {'':>12s}  {'':>12s}  {'':>12s}  {'':>8s}  {'':>10s}")
    print(f"  {'-'*10}  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*12}  "
          f"{'-'*12}  {'-'*12}  {'-'*12}  {'-'*8}  {'-'*10}")

    for attr, (pt, w_mxl, w_gse) in attrs.items():
        valid = np.isfinite(w_mxl) & np.isfinite(w_gse)
        valid &= np.abs(w_mxl) < 1e6
        valid &= np.abs(w_gse) < 1e6
        wm = w_mxl[valid]
        wg = w_gse[valid]
        if len(wm) > 5:
            corr = stats.pearsonr(wm, wg).statistic
            mad = np.mean(np.abs(wg - wm))
        else:
            corr = np.nan
            mad = np.nan
        print(f"  {attr:>10s}  {pt:12.4f}  {np.mean(wm):12.4f}  "
              f"{np.std(wm):12.4f}  {np.median(wm):12.4f}  "
              f"{np.mean(wg):12.4f}  {np.std(wg):12.4f}  "
              f"{np.median(wg):12.4f}  {corr:+8.4f}  {mad:10.4f}")

    # =========================================================================
    #  Estimation Statistics
    # =========================================================================
    print(f"\n{'='*78}")
    print(f"  ESTIMATION STATISTICS")
    print(f"{'='*78}")

    print(f"\n  TIMING:")
    print(f"    MNL fit                  : {t_mnl:8.2f} s")
    print(f"    MXL fit (500 draws)      : {t_mxl:8.2f} s")
    print(f"    GSE (grad + structural)  : {t_gse:8.3f} s")
    print(f"    Speedup (MXL / GSE)      : {t_mxl / max(t_gse, 0.0001):8.0f}x")

    print(f"\n  LOG-LIKELIHOOD (evaluated on MNL choice structure):")
    print(f"    MNL pooled              : {mnl.loglik:12.4f}")
    print(f"    MXL posterior means     : {ll_mxlpost:12.4f}")
    print(f"    GSE individual betas    : {ll_gse:12.4f}")
    print(f"    MXL simulation LL       : {mxl.loglik:12.4f}  (MXL native)")

    delta_mxl = ll_mxlpost - mnl.loglik
    delta_gse = ll_gse - mnl.loglik
    print(f"\n  IMPROVEMENT OVER MNL:")
    print(f"    MXL posterior gain      : {delta_mxl:+12.4f}  "
          f"({delta_mxl / abs(mnl.loglik) * 100:.2f}%)")
    print(f"    GSE gain                : {delta_gse:+12.4f}  "
          f"({delta_gse / abs(mnl.loglik) * 100:.2f}%)")

    print(f"\n  MODEL DIMENSIONS:")
    print(f"    Individuals (N)         : {len(individual_list)}")
    print(f"    Choice occasions per N  : {n_panels}")
    print(f"    Random parameters       : {Kr}  ({', '.join(rvdist_list)})")
    print(f"    MXL estimated params    : {len(beta_mxl_full)}")
    print(f"    MNL estimated params    : {len(beta_mnl)}")
    print(f"    MXL simulation draws    : {mxl.n_draws}")

    # ---- Coefficient comparison ---------------------------------------------
    print(f"\n  COEFFICIENT COMPARISON:")
    print(f"    {'Variable':>12s}  {'Dist':>5s}  {'MNL':>12s}  "
          f"{'MXL prior':>12s}  {'MXL post':>12s}  {'GSE':>12s}  "
          f"{'Corr(MXL,GSE)':>14s}")
    print(f"    {'-'*12}  {'-'*5}  {'-'*12}  {'-'*12}  "
          f"{'-'*12}  {'-'*12}  {'-'*14}")
    for k in range(Kr):
        name = mxl_varnames[k]
        mnl_val = beta_mnl[mnl_rand_idx[k]]
        prior_val = prior_mean_nat[k]
        post_mean_mxl = np.mean(beta_mxl_ind[:, k])
        post_mean_gse = np.mean(beta_gse_nat[:, k])
        valid = np.isfinite(beta_mxl_ind[:, k]) & np.isfinite(beta_gse_nat[:, k])
        valid &= np.abs(beta_mxl_ind[:, k]) < 1e6
        valid &= np.abs(beta_gse_nat[:, k]) < 1e6
        corr = stats.pearsonr(beta_mxl_ind[valid, k], beta_gse_nat[valid, k]).statistic if valid.sum() > 10 else np.nan
        print(f"    {name:>12s}  {rvdist_list[k]:>5s}  {mnl_val:12.6f}  "
              f"{prior_val:12.6f}  {post_mean_mxl:12.6f}  {post_mean_gse:12.6f}  "
              f"{corr:14.4f}")

    # =========================================================================
    #  Visualise
    # =========================================================================
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    wtp_list = [("TIME", wtp_mxl_time, wtp_gse_time),
                ("HEADWAY", wtp_mxl_headway, wtp_gse_headway),
                ("SEATS", wtp_mxl_seats, wtp_gse_seats)]

    for j, (attr, w_mxl, w_gse) in enumerate(wtp_list):
        valid = np.isfinite(w_mxl) & np.isfinite(w_gse)
        valid &= np.abs(w_mxl) < 5000
        valid &= np.abs(w_gse) < 5000
        wm = w_mxl[valid]
        wg = w_gse[valid]

        # Histogram
        ax = axes[0, j]
        if len(wm) > 5:
            lo, hi = np.percentile(np.concatenate([wm, wg]), [2, 98])
            bins = np.linspace(lo, hi, 35)
            ax.hist(wm, bins=bins, alpha=0.5, label="MXL", density=True, color="steelblue")
            ax.hist(wg, bins=bins, alpha=0.5, label="GSE", density=True, color="darkorange")
        ax.set_title(f"WTP {attr}")
        ax.legend(fontsize=7)
        ax.set_xlabel("WTP")
        ax.set_ylabel("Density")

        # Scatter
        ax = axes[1, j]
        if len(wm) > 10:
            ax.scatter(wm, wg, alpha=0.3, s=6)
            lims = [min(wm.min(), wg.min()), max(wm.max(), wg.max())]
            ax.plot(lims, lims, "k--", lw=0.8, alpha=0.5, label="y=x")
            slope, intercept = np.polyfit(wm, wg, 1)
            ax.plot(lims, slope * np.array(lims) + intercept, "r-", lw=1.2, label="OLS")
            r_val = stats.pearsonr(wm, wg).statistic
            ax.set_title(f"r = {r_val:+.4f}")
            ax.legend(fontsize=7)
        ax.set_xlabel("MXL WTP")
        ax.set_ylabel("GSE WTP")

    fig.suptitle("WTP: Mixed Logit vs Gradient-Structural-Equation",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    out_path = "wtp_comparison.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nFigure saved to '{out_path}'")

    # Boxplot of WTP difference
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    diff_data = []
    labels = []
    for attr, w_mxl, w_gse in wtp_list:
        d = w_gse - w_mxl
        d = d[np.isfinite(d) & (np.abs(d) < 5000)]
        diff_data.append(d)
        labels.append(attr)
    bp = ax2.boxplot(diff_data, tick_labels=labels, patch_artist=True)
    for patch, color in zip(bp["boxes"], ["lightblue", "lightgreen", "lightcoral"]):
        patch.set_facecolor(color)
    ax2.axhline(y=0, color="k", linestyle="--", lw=0.8)
    ax2.set_title("WTP Difference: GSE - MXL (per individual)")
    ax2.set_ylabel("GSE WTP - MXL WTP")
    ax2.grid(axis="y", alpha=0.3)
    fig2.tight_layout()
    out_path2 = "wtp_difference_boxplot.png"
    fig2.savefig(out_path2, dpi=150)
    print(f"Figure saved to '{out_path2}'")

    t_total = time.perf_counter() - t0
    print(f"\n  Total script time: {t_total:.1f} s")

    return {
        "mxl": mxl, "mnl": mnl,
        "beta_mxl_ind": beta_mxl_ind, "beta_gse_nat": beta_gse_nat,
        "wtp_mxl": (wtp_mxl_time, wtp_mxl_headway, wtp_mxl_seats),
        "wtp_gse": (wtp_gse_time, wtp_gse_headway, wtp_gse_seats),
        "ll_mxl_post": ll_mxlpost, "ll_gse": ll_gse, "ll_mnl": mnl.loglik,
        "t_mxl": t_mxl, "t_gse": t_gse, "t_mnl": t_mnl,
    }


if __name__ == "__main__":
    results = main()
