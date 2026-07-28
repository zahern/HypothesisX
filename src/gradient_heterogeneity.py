"""
gradient_heterogeneity.py
==========================
Uses MNL individual-level gradients (scores) to influence / detect
heterogeneity, then contrasts them with posterior conditional means from a
Mixed Logit with negative lognormal mixing distributions.

Structural modelling equation
-----------------------------
For a random-parameter model  beta_r ~ D(mu, Sigma), the posterior mean
can be approximated via a one-step Newton correction from the MNL MLE:

    E[beta_r | y_n, X_n]  ~=  beta_mnl  +  Sigma_mxl @ g_n

where:
  - beta_mnl  : MNL point estimates (for the random-parameter subset)
  - g_n       : individual MNL score (gradient) contribution at the MLE
                g_nk = sum_j (y_nj - P_nj) * X_njk
  - Sigma_mxl : covariance matrix of the random parameters from the MXL

Implementation
--------------
1. Estimate a simple MNL on Swiss Metro data (choice-occasion level).
2. Aggregate per-occasion MNL gradients to per-individual level.
3. Estimate a Mixed Logit with lognormal-distributed coefficients.
   (Negative lognormal is achieved by negating the COST variable before
    estimation -- the fitted lognormal then implies a non-positive
    effective coefficient.)
4. Compute posterior conditional means (individual-level fitted random
   parameters) from the MXL.
5. Contrast  g_n  with  Sigma^{-1} @ (posterior_deviation).
"""

from __future__ import annotations

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

from SearchLibrium.multinomial_logit import MultinomialLogit
from SearchLibrium.MixedLogit import MixedLogit
from SearchLibrium.sample_data import load_swiss_metro_data


# ---------------------------------------------------------------------------
# Helper: individual-level MNL gradient (per choice occasion)
# ---------------------------------------------------------------------------
def compute_mnl_individual_gradients(model: MultinomialLogit) -> np.ndarray:
    """Re-evaluate the MNL at its MLE and return per-observation scores.

    Returns
    -------
    g : ndarray of shape (N, K)
        Individual gradient (score) contributions g_nk = d log L_n / d beta_k.
        N = number of choice occasions.
    """
    betas = model.coeff_est
    X = model.X
    y = model.y
    avail = model.avail
    p = model.compute_probabilities(betas, X, avail)
    ymp = y - p

    Kf = model.Kf
    Kftrans = model.Kftrans

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

    # ---- 1. Load Swiss Metro data ------------------------------------------
    df = load_swiss_metro_data()
    varnames = ["TIME", "COST", "HEADWAY", "SEATS"]
    alts = sorted(df["alt"].unique().tolist())
    base_alt = "SM"

    print("=" * 72)
    print("  MNL Gradients  ->  Heterogeneity Influence")
    print("=" * 72)
    n_unique_ids = df["custom_id"].nunique()
    n_individuals = df["ID"].nunique()
    print(f"Dataset: Swiss Metro  |  {n_unique_ids} choice occasions  |  "
          f"{n_individuals} individuals  |  alternatives = {alts}")

    # ---- 2. Build individual-to-occasion mapping ---------------------------
    # Each custom_id belongs to exactly one ID.  We need this to aggregate
    # MNL scores from occasion level up to individual level.
    occ_map = (
        df[["custom_id", "ID"]]
        .drop_duplicates()
        .sort_values("custom_id")
        .reset_index(drop=True)
    )
    unique_ind  = occ_map["ID"].values         # corresponding individual ids

    # Build a mapping: individual_id -> list of occasion row indices (0-based)
    ind_to_occ = {}
    for i, ind_id in enumerate(unique_ind):
        ind_to_occ.setdefault(ind_id, []).append(i)

    

        # ---- 3. Estimate Simple MNL -------------------------------------------
    print("\n--- Step 1: Estimate simple MNL ---")
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
    mnl.summarise()
    beta_mnl = mnl.coeff_est
    print(f"\nMNL log-likelihood : {mnl.loglik:.4f}")
    mnl_names = list(mnl.Xnames)
    print(f"MNL params ({len(mnl_names)}): {mnl_names}")

    # ---- 4. Compute & aggregate individual MNL gradients ------------------
    print("\n--- Step 2: Per-occasion MNL gradients, aggregate to individual ---")
    g_occ = compute_mnl_individual_gradients(mnl)  # (8316, K)
    print(f"Occasion-level gradient shape : {g_occ.shape}")

    # Aggregate to individual level: sum scores within each individual
    individual_list = sorted(ind_to_occ.keys())
    K = g_occ.shape[1]
    g_ind = np.zeros((len(individual_list), K))
    for idx, ind_id in enumerate(individual_list):
        g_ind[idx, :] = g_occ[ind_to_occ[ind_id], :].sum(axis=0)

    print(f"Individual-level gradient shape : {g_ind.shape}")

    # ---- 5. Estimate Mixed Logit with (negative) lognormals ---------------
    print("\n--- Step 3: Estimate Mixed Logit with negative lognormals ---")
    df_mxl = df.copy()
    df_mxl["COST_NEG"] = -df_mxl["COST"]
    mxl_varnames = ["TIME", "COST_NEG", "HEADWAY", "SEATS"]
    randvars = {
        "TIME": "n",
        "COST_NEG": "ln",  # negative lognormal via negated data
        "HEADWAY": "n",
        "SEATS": "n",
    }

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
    mxl.summarise()

    # ---- 6. Extract posterior conditional means ---------------------------
    print("\n--- Step 4: Extract posterior conditional means ---")
    if not hasattr(mxl, "pch2_res") or mxl.pch2_res is None:
        mxl.compute_fitted_params(mxl.y, mxl.p, mxl.panel_info, mxl.Br)
    fitted_random = mxl.pch2_res  # (N_ind, Kr)
    print(f"Posterior conditional means shape : {fitted_random.shape}")

    mxl_names = list(mxl.Xnames)
    print(f"MXL params ({len(mxl_names)}): {mxl_names}")

    # ---- 7. Align MNL and MXL parameters ----------------------------------
    print("\n--- Step 5: Align parameters between MNL and MXL ---")
    # MNL params: ['TIME','COST','HEADWAY','SEATS']
    # MXL params: ['TIME','COST_NEG','HEADWAY','SEATS', 'sd.TIME',...]
    # We need MNL gradient sub-block for the random variables.
    # Map: MXL random var name -> MNL parameter index
    mxl_vars = ["TIME", "COST_NEG", "HEADWAY", "SEATS"]
    mnl_var_to_idx = {v: mnl_names.index(v) if v in mnl_names
                      else mnl_names.index("COST") for v in mxl_vars}
    mnl_random_idx = [mnl_var_to_idx[v] for v in mxl_vars]
    Kr = len(mnl_random_idx)
    print(f"Random variables           : {mxl_vars}")
    print(f"MNL param indices          : {mnl_random_idx}")
    print(f"Kr = {Kr}, fitted_random cols = {fitted_random.shape[1]}")

    # ---- 8. Ensure consistent individual ordering --------------------------
    # MXL posteriors are ordered by individual ID (sorted). We need the
    # same ordering for our aggregated MNL gradients.
    # individual_list is sorted, same as MXL's internal panel ordering.
    # Verify shapes match:
    if g_ind.shape[0] != fitted_random.shape[0]:
        print(f"WARNING: g_ind rows ({g_ind.shape[0]}) != fitted_random rows "
              f"({fitted_random.shape[0]}). Truncating to min.")
        n = min(g_ind.shape[0], fitted_random.shape[0])
        g_ind = g_ind[:n, :]
        fitted_random = fitted_random[:n, :]

    # ---- 9. Build prior covariance Sigma from MXL estimates ----------------
    # The structural equation uses the PRIOR covariance of the random params.
    # For 'n': Var = sd^2 (on natural scale)
    # For 'ln': Var = exp(2*mu + sd^2) * (exp(sd^2) - 1) (on natural scale)
    print("\n--- Step 6: Build prior covariance Sigma from MXL ---")
    Kr = mxl.Kr
    beta_mxl_full = mxl.coeff_est
    Kr_est = mxl.Kr
    # MXL parameter order: [Bf(0) | Br_mean(Kr) | chol(Kchol) | Br_sd(Kbw) | ...]
    # Since Kr=4 and Kbw=4 (uncorrelated), Bf=None (all vars are random):
    #   [Br_mean: 4] [chol: 0] [Br_sd: 4]
    br_mean = beta_mxl_full[:Kr_est]              # [-0.0343, -7.9091, -0.0295, 1.2695]
    br_sd   = beta_mxl_full[Kr_est:Kr_est + mxl.Kbw]  # [0.0492, 4.6999, 0.0388, 1.1557]

    # Distributions per random variable
    rvdist_list = list(randvars.values())  # ['n', 'ln', 'n', 'n']
    print(f"MXL means (transformed scale): {np.array2string(br_mean, precision=4)}")
    print(f"MXL SDs   (transformed scale): {np.array2string(br_sd, precision=4)}")
    print(f"MXL distributions: {rvdist_list}")

    # Build prior variance vector on the NATURAL scale
    # (use abs for SDs -- optimisation may leave sign flipped)
    prior_var = np.zeros(Kr)
    prior_mean_natural = np.zeros(Kr)
    for k in range(Kr):
        mu_k = br_mean[k]
        sd_k = abs(br_sd[k])
        if rvdist_list[k] == "ln":
            s2 = sd_k * sd_k
            prior_mean_natural[k] = np.exp(mu_k + s2 / 2)
            prior_var[k] = np.exp(2 * mu_k + s2) * (np.exp(s2) - 1)
        else:
            prior_mean_natural[k] = mu_k
            prior_var[k] = sd_k * sd_k

    Sigma_prior = np.diag(prior_var)  # (Kr, Kr)
    print(f"Prior mean (natural scale):  {np.array2string(prior_mean_natural, precision=6)}")
    print(f"Prior var  (natural scale):  {np.array2string(prior_var, precision=6)}")

    # ---- 10. Structural equation: posterior_dev ~ Sigma_prior @ g_n --------
    print("\n--- Step 7: Structural equation contrast ---")
    g_mnl_sub = g_ind[:, mnl_random_idx]       # (N_ind, Kr)
    beta_mnl_sub = beta_mnl[mnl_random_idx]     # (Kr,)
    post_dev = fitted_random - prior_mean_natural[np.newaxis, :]  # deviation from prior mean

    # Also provide posterior covariance for comparison
    Sigma_post = np.cov(fitted_random.T)

    # Structural predictions using both Sigma_prior and sigma_post
    Sigma_prior_g = g_mnl_sub @ Sigma_prior  # (N_ind, Kr)
    Sigma_post_g = g_mnl_sub @ Sigma_post

    # ---- 11. Contrast diagnostics ------------------------------------------
    for label, Sigma_g in [("Sigma_prior", Sigma_prior_g), ("Sigma_post", Sigma_post_g)]:
        print(f"\n  --- Using {label} ---")
        for k in range(Kr):
            name = mxl_vars[k]
            r = stats.pearsonr(post_dev[:, k], Sigma_g[:, k])
            r2 = r.statistic ** 2
            print(f"    {name:12s}  corr(post_dev, Sigma@g) = {r.statistic:+.4f}  "
                  f"R2 = {r2:.4f}  (p = {r.pvalue:.4e})")

    # ---- 12. Visualise ----------------------------------------------------
    # Pick the better Sigma (prior or posterior based on overall highest |r|)
    r_prior_avg = np.mean([abs(stats.pearsonr(post_dev[:, k], Sigma_prior_g[:, k]).statistic) for k in range(Kr)])
    r_post_avg  = np.mean([abs(stats.pearsonr(post_dev[:, k], Sigma_post_g[:, k]).statistic)  for k in range(Kr)])
    best_Sigma_g = Sigma_prior_g if r_prior_avg >= r_post_avg else Sigma_post_g
    best_label = "prior" if r_prior_avg >= r_post_avg else "posterior"

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()
    for k in range(Kr):
        ax = axes[k]
        ax.scatter(best_Sigma_g[:, k], post_dev[:, k], alpha=0.3, s=8)
        ax.set_xlabel(f"Sigma_{best_label} @ g_n  [{mxl_vars[k]}]")
        ax.set_ylabel(f"posterior deviation  [{mxl_vars[k]}]")
        slope, intercept = np.polyfit(best_Sigma_g[:, k], post_dev[:, k], 1)
        xl = np.array(ax.get_xlim())
        ax.plot(xl, slope * xl + intercept, "r--", lw=1)
        r_val = stats.pearsonr(post_dev[:, k], best_Sigma_g[:, k]).statistic
        ax.set_title(f"{mxl_vars[k]}  (r = {r_val:+.3f})")
    fig.suptitle("Structural Equation:  E[b | y] - mu  ~  Sigma @ g_n",
                 fontsize=13)
    fig.tight_layout()
    out_path = "gradient_heterogeneity_contrast.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nFigure saved to '{out_path}' (using Sigma_{best_label})")

    # ---- 13. Gradient-based heterogeneity scores (normals only) --------
    print("\n--- Step 8: Gradient-based heterogeneity scores ---")
    # Exclude lognormal (COST_NEG) from score -- its prior variance dominates
    normal_mask = np.array([d != "ln" for d in rvdist_list])
    g_normal = g_mnl_sub[:, normal_mask]
    Sigma_normal = np.diag(prior_var[normal_mask])
    # Scale-normalised score:  g_n' @ sqrt(Sigma) @ sqrt(Sigma) @ g_n
    # Using Sigma_chol to get a properly-scaled quadratic form
    het_score = np.sum(g_normal * (g_normal @ Sigma_normal), axis=1)
    post_dev_normal = post_dev[:, normal_mask]
    post_norm = np.linalg.norm(post_dev_normal, axis=1)
    r_overall = stats.pearsonr(post_norm, het_score)
    print(f"  (Normals only) corr(|post_dev|, het_score) = {r_overall.statistic:+.4f}  "
          f"(p = {r_overall.pvalue:.4e})")

    top_n = 5
    top_idx = np.argsort(het_score)[-top_n:][::-1]
    print(f"\n  Top {top_n} individuals by gradient heterogeneity score:")
    for i in top_idx:
        print(f"    idx={i:4d}  score={het_score[i]:.4f}  "
              f"|post_dev|={post_norm[i]:.4f}")

    return {
        "mnl": mnl,
        "mxl": mxl,
        "g_ind": g_ind,
        "g_occ": g_occ,
        "fitted_random": fitted_random,
        "post_dev": post_dev,
        "Sigma_prior": Sigma_prior,
        "Sigma_post": Sigma_post,
        "Sigma_prior_g": Sigma_prior_g,
        "het_score": het_score,
    }


if __name__ == "__main__":
    results = main()
