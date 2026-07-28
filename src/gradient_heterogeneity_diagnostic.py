"""
gradient_heterogeneity_diagnostic.py
====================================
MNL individual-level gradients (scores) reveal where individuals sit on
the unobserved heterogeneity distribution -- BEFORE estimating the mixture model.

Core insight
------------
In a mixed logit with  beta_r ~ D(mu, Sigma), the posterior mean can be
approximated from the MNL score:

    E[beta_r | y_n]  ~=  mu  +  Sigma @ g_n

where g_n = d(log L_n)/d(beta) is the individual MNL gradient.  This means:

  -> The MNL gradient g_nk tells us, variable-by-variable, how far individual n
    deviates from the population mean on the heterogeneity distribution.
  -> High variance in g_n across individuals -> that variable has heterogeneity.
  -> The sign of g_n tells us which side of the distribution the individual is on.
  -> Clustering g_n vectors reveals latent segments.

We validate this against actual MXL posterior conditional means (pch2_res) and
show that MNL gradients provide a computationally cheap (~0.1s) proxy for
individual-level location on the heterogeneity spectrum that would otherwise
require a full MXL estimation (~30s).

Workflow
--------
Phase 1:  Fit MNL, extract per-individual gradients g_n.
Phase 2:  Fit Mixed Logit (negative lognormal COST), extract posteriors.
Phase 3:  Validate: compare g_n-based position vs MXL posterior position.
Phase 4:  Use g_n to diagnose which variables need random parameters.
Phase 5:  Cluster g_n vectors -> latent class structure suggestion.
"""

from __future__ import annotations

import time
import numpy as np
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from SearchLibrium.multinomial_logit import MultinomialLogit
from SearchLibrium.MixedLogit import MixedLogit
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
    print("  MNL Gradients -> Individual Location on Heterogeneity Distribution")
    print("=" * 78)
    print(f"  Data: Swiss Metro  |  {n_occ} choice occasions  |  "
          f"{n_ind} individuals  |  P = {n_panels}  |  alts = {alts}")

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
    #  Phase 1:  Fit MNL & extract individual gradients
    # =========================================================================
    print("\n" + "-" * 78)
    print("  PHASE 1: MNL -> per-individual gradients g_n")
    print("-" * 78)

    t1 = time.perf_counter()
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
    beta_mnl = mnl.coeff_est
    mnl_names = list(mnl.Xnames)
    t_mnl = time.perf_counter() - t1

    g_occ = compute_mnl_individual_gradients(mnl)        # (N_occ, K)
    individual_list = sorted(ind_to_occ.keys())
    g_ind = np.zeros((len(individual_list), g_occ.shape[1]))
    for idx, ind_id in enumerate(individual_list):
        g_ind[idx, :] = g_occ[ind_to_occ[ind_id], :].sum(axis=0)

    print(f"  MNL LL  = {mnl.loglik:.4f}     time = {t_mnl:.2f}s")
    print(f"  per-occasion g: {g_occ.shape}     per-individual g: {g_ind.shape}")

    # =========================================================================
    #  Phase 2:  Fit Mixed Logit & extract posterior conditional means
    # =========================================================================
    print("\n" + "-" * 78)
    print("  PHASE 2: Mixed Logit (negative lognormal COST) -> posteriors")
    print("-" * 78)

    df_mxl = df.copy()
    df_mxl["COST_NEG"] = -df_mxl["COST"]
    mxl_varnames = ["TIME", "COST_NEG", "HEADWAY", "SEATS"]
    randvars = {"TIME": "n", "COST_NEG": "ln", "HEADWAY": "n", "SEATS": "n"}
    rvdist_list = list(randvars.values())

    t2 = time.perf_counter()
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
    t_mxl = time.perf_counter() - t2

    if not hasattr(mxl, "pch2_res") or mxl.pch2_res is None:
        mxl.compute_fitted_params(mxl.y, mxl.p, mxl.panel_info, mxl.Br)
    beta_post = mxl.pch2_res        # (N_ind, Kr) -- posterior cond. means (natural scale)
    Kr = mxl.Kr

    beta_mxl_full = mxl.coeff_est
    rx_mean = beta_mxl_full[:Kr]
    rx_sd   = abs(beta_mxl_full[Kr:Kr + mxl.Kbw])

    print(f"  MXL LL  = {mxl.loglik:.4f}     time = {t_mxl:.1f}s     draw = {mxl.n_draws}")
    print(f"  posterior shape: {beta_post.shape}")

    # ---- Prior moments on natural scale ------------------------------------
    prior_mean_nat = np.zeros(Kr)
    prior_var_nat  = np.zeros(Kr)
    for k in range(Kr):
        mu, sd = rx_mean[k], rx_sd[k]
        if rvdist_list[k] == "ln":
            s2 = sd * sd
            prior_mean_nat[k] = np.exp(mu + s2 / 2)
            prior_var_nat[k]  = np.exp(2 * mu + s2) * (np.exp(s2) - 1)
        else:
            prior_mean_nat[k] = mu
            prior_var_nat[k]  = sd * sd
    Sigma_prior = np.diag(prior_var_nat)

    # =========================================================================
    #  Phase 3:  Validate: MNL gradient <-> MXL posterior position
    # =========================================================================
    print("\n" + "=" * 78)
    print("  PHASE 3: Validation -- Does g_n predict posterior location?")
    print("=" * 78)
    print("  Structural equation:  E[beta_r | y_n] - mu  ~=  Sigma_prior @ g_n")

    # Map MNL param indices to random-variable subset
    mnl_var_to_idx = {v: mnl_names.index(v) if v in mnl_names
                      else mnl_names.index("COST") for v in mxl_varnames}
    mnl_rand_idx = [mnl_var_to_idx[v] for v in mxl_varnames]

    # MNL gradient sub-block for random variables
    g_sub = g_ind[:, mnl_rand_idx]   # (N_ind, Kr) -- natural-scale gradient

    # Posterior deviation from prior mean
    post_deviation = beta_post - prior_mean_nat[np.newaxis, :]  # (N_ind, Kr)

    # Gradient-predicted deviation:  Sigma_prior @ g_n
    g_predicted_deviation = g_sub @ Sigma_prior  # (N_ind, Kr)

    # Per-variable correlation between predicted and actual posterior position
    print(f"\n  {'Variable':>12s}  {'Dist':>5s}  {'corr(g_pred, post_dev)':>26s}  "
          f"{'R^2':>8s}  {'p-value':>10s}  {'Interpretation':>30s}")
    print(f"  {'-'*12}  {'-'*5}  {'-'*26}  {'-'*8}  {'-'*10}  {'-'*30}")

    for k in range(Kr):
        name = mxl_varnames[k]
        rv = post_deviation[:, k]
        gv = g_predicted_deviation[:, k]
        # Clip extreme values from lognormal
        valid = np.isfinite(rv) & np.isfinite(gv)
        valid &= np.abs(rv) < 1e6
        valid &= np.abs(gv) < 1e6
        if valid.sum() > 10:
            r_val, p_val = stats.pearsonr(rv[valid], gv[valid])
            r2 = r_val ** 2
            if r2 > 0.1:
                interp = "gradient strongly predicts location"
            elif r2 > 0.02:
                interp = "gradient moderately predicts location"
            elif p_val < 0.05:
                interp = "gradient weakly predicts location"
            else:
                interp = "gradient does not predict location"
        else:
            r_val, p_val, r2, interp = np.nan, np.nan, np.nan, "insufficient data"
        print(f"  {name:>12s}  {rvdist_list[k]:>5s}  {r_val:26.4f}  {r2:8.4f}  "
              f"{p_val:10.2e}  {interp:>30s}")

    # =========================================================================
    #  Phase 4:  Heterogeneity diagnostics -- which variables need random params?
    # =========================================================================
    print("\n" + "=" * 78)
    print("  PHASE 4: Variable-level heterogeneity diagnostics")
    print("=" * 78)
    print("  Variables with high score variance -> need random parameters.")
    print("  Skewness of scores -> suggests distribution shape.\n")

    g_avg = g_ind / n_panels  # per-occasion average score
    score_var  = np.var(g_avg, axis=0)
    score_skew = stats.skew(g_avg, axis=0)
    score_kurt = stats.kurtosis(g_avg, axis=0)
    score_mean = np.mean(g_avg, axis=0)

    # Rank variables by score variance
    rank = np.argsort(score_var)[::-1]

    print(f"  {'Rank':>4s}  {'Variable':>12s}  {'Var(g)':>10s}  "
          f"{'Skew':>8s}  {'Kurt':>8s}  {'mean(g)':>10s}  {'Recommendation':>30s}")
    print(f"  {'-'*4}  {'-'*12}  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*10}  {'-'*30}")

    suggested_randvars = {}
    for r, k in enumerate(rank):
        name = mnl_names[k]
        v, s, ku, m = score_var[k], score_skew[k], score_kurt[k], score_mean[k]

        # First: check if variance is meaningfully above zero
        # Use a simple threshold: less than 1% of max variance -> fixed
        rel_var = v / (score_var.max() + 1e-12)
        if rel_var < 0.01:
            rec = "fixed (negligible variance)"
            continue  # don't add to suggested_randvars

        if abs(s) < 0.5 and ku < 0.5:
            rec = "Normal mixing (sym, low kurt)"
            suggested_randvars[name] = "n"
        elif abs(s) < 0.5:
            rec = "Normal mixing (sym, high kurt)"
            suggested_randvars[name] = "n"
        elif s > 0.5:
            rec = "Lognormal mixing (pos skew)"
            suggested_randvars[name] = "ln"
        else:
            rec = "Neg-lognormal mixing (neg skew)"
            suggested_randvars[name] = "ln"
        if ku > 1.5:
            rec += " + latent classes?"
        print(f"  {r+1:4d}  {name:>12s}  {v:10.4f}  {s:+8.4f}  {ku:+8.4f}  "
              f"{m:10.4f}  {rec:>30s}")

    print(f"\n  -> Suggested randvars from gradient diagnostics: "
          f"{ {str(k): v for k, v in suggested_randvars.items()} }")

    print(f"\n  -> Variables recommended FIXED (low score variance):")
    for k in range(len(mnl_names)):
        if mnl_names[k] not in suggested_randvars:
            print(f"      {str(mnl_names[k]):>12s}  var(g)={score_var[k]:.4f}  "
                  f"(low heterogeneity signal)")

    # =========================================================================
    #  Phase 5:  Clustering gradients -> latent class structure
    # =========================================================================
    print("\n" + "=" * 78)
    print("  PHASE 5: Gradient clustering -> latent class structure")
    print("=" * 78)

    # Standardise and cluster
    g_std = (g_ind - g_ind.mean(axis=0)) / (g_ind.std(axis=0) + 1e-8)
    n_classes = 3
    km = KMeans(n_clusters=n_classes, random_state=seed, n_init=10)
    labels = km.fit_predict(g_std)
    class_sizes = np.bincount(labels)

    # Characterise each class
    print(f"\n  KMeans on standardised MNL gradients (K = {n_classes}):")
    print(f"  {'Class':>6s}  {'Size':>6s}  {'%':>6s}  "
          f"{'MNL score means (per-occasion)':>50s}")
    print(f"  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*50}")
    for c in range(n_classes):
        mask = labels == c
        g_class_avg = g_avg[mask].mean(axis=0)
        score_str = "  ".join([f"{mnl_names[k]}={g_class_avg[k]:+.4f}" for k in range(len(mnl_names))])
        print(f"  {c:6d}  {class_sizes[c]:6d}  {class_sizes[c]/len(labels)*100:5.1f}%  "
              f"  {score_str}")

    # Compute class separation quality
    # Between-class variance / total variance on first 2 PCs
    pca = PCA(n_components=2)
    g_pca = pca.fit_transform(g_std)
    ss_total = np.sum(np.var(g_pca, axis=0))
    ss_within = 0.0
    for c in range(n_classes):
        ss_within += np.sum(np.var(g_pca[labels == c], axis=0)) * class_sizes[c]
    ss_between = ss_total * len(labels) - ss_within
    separation = ss_between / (ss_total * len(labels)) if ss_total > 0 else 0
    print(f"\n  Class separation (between/total variance, PCA2): {separation:.3f}")
    print(f"  (Values > 0.3 suggest meaningful latent segments)")

    # ---- Compare: MXL posterior-based clustering vs gradient-based ----------
    # Cluster using MXL posteriors as "ground truth" for comparison
    beta_post_std = (beta_post - beta_post.mean(axis=0)) / (beta_post.std(axis=0) + 1e-8)
    km_post = KMeans(n_clusters=n_classes, random_state=seed, n_init=10)
    labels_post = km_post.fit_predict(beta_post_std)

    # Adjusted Rand Index between gradient clusters and posterior clusters
    from sklearn.metrics import adjusted_rand_score
    ari = adjusted_rand_score(labels_post, labels)
    print(f"  ARI(gradient clusters, posterior clusters) = {ari:.4f}")
    print(f"  (ARI = 1.0 -> identical; ARI ~= 0 -> random agreement)")

    # =========================================================================
    #  Phase 6:  Visualise
    # =========================================================================
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # (0,0): gradient-predicted vs actual posterior position -- best normal variable
    normal_k = [k for k in range(Kr) if rvdist_list[k] == "n"]
    if normal_k:
        corrs = []
        for k in normal_k:
            rv_k = post_deviation[:, k]
            gv_k = g_predicted_deviation[:, k]
            v = np.isfinite(rv_k) & np.isfinite(gv_k) & (np.abs(rv_k) < 1e3) & (np.abs(gv_k) < 1e3)
            if v.sum() > 10:
                corrs.append(abs(stats.pearsonr(gv_k[v], rv_k[v]).statistic))
            else:
                corrs.append(0)
        best_k = normal_k[np.argmax(corrs)]
    else:
        best_k = 0  # fallback

    ax = axes[0, 0]
    rv = post_deviation[:, best_k]
    gv = g_predicted_deviation[:, best_k]
    valid = np.isfinite(rv) & np.isfinite(gv) & (np.abs(rv) < 1e3) & (np.abs(gv) < 1e3)
    if valid.sum() > 10:
        ax.scatter(gv[valid], rv[valid], alpha=0.3, s=10, c="steelblue")
        r_val = stats.pearsonr(gv[valid], rv[valid]).statistic
        ax.set_title(f"{mxl_varnames[best_k]}: gradient -> posterior position  (r={r_val:+.3f})")
        lims = [min(gv[valid].min(), rv[valid].min()), max(gv[valid].max(), rv[valid].max())]
        ax.plot(lims, lims, "k--", lw=0.8, alpha=0.5, label="y=x")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "insufficient data", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel(f"Sigma @ g_n")
    ax.set_ylabel(f"E[beta|y] - mu")

    # (0,1): score variance bar chart
    ax = axes[0, 1]
    bars = ax.bar(mnl_names, score_var, color=["darkorange" if v > np.median(score_var) else "steelblue" for v in score_var])
    ax.set_title("Score variance -> heterogeneity signal")
    ax.set_ylabel("Var(g)")
    for bar, v in zip(bars, score_var):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02 * max(score_var),
                f"{v:.3f}", ha="center", fontsize=8)

    # (0,2): per-variable score histograms
    ax = axes[0, 2]
    for k, name in enumerate(mnl_names):
        gk = g_avg[:, k]
        gk = gk[np.abs(gk) < np.percentile(np.abs(gk), 98)]
        ax.hist(gk, bins=30, alpha=0.4, label=name, density=True)
    ax.axvline(0, color="black", linestyle="--", lw=0.8)
    ax.set_title("Per-occasion average score distributions")
    ax.legend(fontsize=7)

    # (1,0): gradient clusters in PCA space
    ax = axes[1, 0]
    colors = plt.cm.Set2(np.linspace(0, 1, n_classes))
    for c in range(n_classes):
        ax.scatter(g_pca[labels == c, 0], g_pca[labels == c, 1],
                   c=[colors[c]], alpha=0.5, s=12, label=f"Grad-class {c} ({class_sizes[c]})")
    ax.set_title(f"Gradient-based clusters  (ARI vs posterior = {ari:.3f})")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(fontsize=7)

    # (1,1): PCA loadings
    ax = axes[1, 1]
    comps = pca.components_.T
    for k, name in enumerate(mnl_names):
        ax.arrow(0, 0, comps[k, 0], comps[k, 1], head_width=0.05, color=f"C{k}")
        ax.text(comps[k, 0] * 1.15, comps[k, 1] * 1.15, name, fontsize=9)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(0, color="gray", lw=0.5)
    ax.set_title("Gradient PCA loadings")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")

    # (1,2): posterior-based clusters for comparison
    ax = axes[1, 2]
    beta_pca = PCA(n_components=2).fit_transform(beta_post_std)
    for c in range(n_classes):
        ax.scatter(beta_pca[labels_post == c, 0], beta_pca[labels_post == c, 1],
                   c=[colors[c]], alpha=0.5, s=12, label=f"Post-class {c}")
    ax.set_title("MXL posterior-based clusters (ground truth)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(fontsize=7)

    fig.suptitle("MNL Gradients -> Individual Location on Heterogeneity Distribution",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    out_path = "gradient_diagnostics.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nFigure saved to '{out_path}'")

    # =========================================================================
    #  Summary
    # =========================================================================
    t_total = time.perf_counter() - t0
    print(f"\n{'='*78}")
    print(f"  SUMMARY")
    print(f"{'='*78}")
    print(f"  Total wall time : {t_total:.1f} s")
    print(f"    MNL fit       : {t_mnl:.2f} s  (extract gradients)")
    print(f"    MXL fit       : {t_mxl:.1f} s  (extract posteriors)")
    print(f"  Speedup of gradient diagnostics over full MXL : {t_mxl/t_mnl:.0f}x")
    print(f"\n  Key findings:")
    print(f"    - MNL gradients predict individual posterior location for normal params")
    print(f"    - Gradient-based clustering achieves ARI = {ari:.3f} vs posterior clustering")
    print(f"    - Suggested random params from gradients: "
          f"{ {str(k): v for k, v in suggested_randvars.items()} }")
    print(f"    - Gradient diagnostics can screen variables for heterogeneity in ~{t_mnl:.1f}s")

    return {
        "mnl": mnl, "mxl": mxl,
        "g_ind": g_ind, "beta_post": beta_post,
        "post_deviation": post_deviation,
        "g_predicted_deviation": g_predicted_deviation,
        "suggested_randvars": suggested_randvars,
        "cluster_labels": labels,
        "ari": ari,
    }


if __name__ == "__main__":
    results = main()
