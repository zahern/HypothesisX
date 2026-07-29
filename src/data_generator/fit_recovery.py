"""
fit_recovery.py
===============
Generates latent class choice data with membership variables, then fits
via a self-contained EM algorithm to recover both utility coefficients
(betas) and membership coefficients (gammas).

No SearchLibrium import required — uses numpy + scipy only.

Usage:
    python fit_recovery.py          # quick recovery test
    python fit_recovery.py --full   # full-scale recovery test
"""

from __future__ import annotations

import sys, os, time
import numpy as np
from scipy.special import softmax, logsumexp
from scipy.optimize import minimize, linear_sum_assignment

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data_generator.latent_class_gen import AdvancedLatentClassGenerator


# ===========================================================================
#  Self-contained EM for latent-class MNL with membership equation
# ===========================================================================
class LCMembershipMLE:
    """Latent-class multinomial logit with membership covariates.

    Fit via EM with L2 regularization and multiple random restarts.
    """

    def __init__(self, n_classes, X, y, choice_ids, alts, varnames,
                 class_varnames, membership_vars, maxiter=300, tol=1e-6,
                 l2_penalty=0.1, n_init=3):
        self.C = n_classes
        self.varnames = list(varnames)
        self.class_varnames = [list(cv) for cv in class_varnames]
        self.membership_vars = list(membership_vars)
        self.l2_penalty = l2_penalty
        self.n_init = n_init

        y = np.asarray(y, dtype=float)
        choice_ids = np.asarray(choice_ids)
        alts = np.asarray(alts)

        order = np.lexsort((alts, choice_ids))
        X = X[order]
        y = y[order]
        choice_ids = choice_ids[order]

        unique_ids, counts = np.unique(choice_ids, return_counts=True)
        self.J = int(counts[0])
        self.N = len(unique_ids)

        self.X = X.reshape(self.N, self.J, X.shape[1])
        self.y = y.reshape(self.N, self.J)

        self._class_idx = []
        self._Kc = []
        for cv in class_varnames:
            idx = [varnames.index(v) for v in cv]
            self._class_idx.append(np.array(idx, dtype=int))
            self._Kc.append(len(idx))

        memb_idx = [varnames.index(v) for v in membership_vars]
        self.X_memb = np.zeros((self.N, len(memb_idx)))
        for n in range(self.N):
            self.X_memb[n] = X[n * self.J, memb_idx]
        self.Km = len(memb_idx)

        self.maxiter = maxiter
        self.tol = tol

    def _choice_probs(self, class_betas):
        probs = np.zeros((self.N, self.C, self.J))
        for c in range(self.C):
            bc = np.asarray(class_betas[c], dtype=float)
            Xc = self.X[:, :, self._class_idx[c]]
            utils = Xc @ bc
            utils = utils - utils.max(axis=1, keepdims=True)
            exp_u = np.exp(utils)
            denom = np.clip(exp_u.sum(axis=1, keepdims=True), 1e-300, None)
            probs[:, c, :] = exp_u / denom
        return probs

    def _log_choice(self, class_betas):
        probs = self._choice_probs(class_betas)
        chosen = np.clip((probs * self.y[:, None, :]).sum(axis=2), 1e-300, None)
        return np.log(chosen), probs

    def _membership_priors(self, gammas):
        if self.Km == 0:
            return np.tile(np.full(self.C, 1.0 / self.C), (self.N, 1))
        logits = np.zeros((self.N, self.C))
        for c in range(self.C - 1):
            logits[:, c] = self.X_memb @ gammas[c]
        logits -= logits.max(axis=1, keepdims=True)
        exp_l = np.exp(logits)
        return exp_l / np.clip(exp_l.sum(axis=1, keepdims=True), 1e-300, None)

    def _weighted_m_step(self, beta0, weights, c):
        Xc = self.X[:, :, self._class_idx[c]]
        weights = np.asarray(weights, dtype=float)
        sum_w = np.sum(weights)
        if sum_w < 1e-6:
            return beta0  # no effective data for this class

        l2 = self.l2_penalty

        def obj_and_grad(beta):
            utils = Xc @ beta
            utils = utils - utils.max(axis=1, keepdims=True)
            exp_u = np.exp(utils)
            denom = np.clip(exp_u.sum(axis=1, keepdims=True), 1e-300, None)
            probs = exp_u / denom
            chosen = np.clip((probs * self.y).sum(axis=1), 1e-300, None)
            ll = np.sum(weights * np.log(chosen))
            diff = (self.y - probs) * weights[:, None]
            grad = np.einsum("nj,njk->k", diff, Xc)
            # L2: -0.5 * lambda * ||beta||^2
            reg = 0.5 * l2 * np.sum(beta * beta)
            return -(ll - reg), -grad + l2 * beta

        result = minimize(
            obj_and_grad,
            np.asarray(beta0, dtype=float),
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": 80},
        )
        return result.x

    def _membership_m_step(self, gamma0, weights):
        if self.Km == 0:
            return gamma0

        weights = np.asarray(weights, dtype=float)
        gamma0 = np.asarray(gamma0, dtype=float).ravel()

        def obj_and_grad(gamma_flat):
            g = gamma_flat.reshape(self.C - 1, self.Km)
            priors = self._membership_priors(g)
            log_prior = np.log(np.clip(priors, 1e-300, None))
            ll = np.sum(weights * log_prior)
            residuals = weights - priors
            grad = np.zeros((self.C - 1, self.Km))
            for c in range(self.C - 1):
                grad[c] = self.X_memb.T @ residuals[:, c]
            return -ll, -grad.ravel()

        result = minimize(
            obj_and_grad,
            gamma0,
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": 80},
        )
        return result.x.reshape(self.C - 1, self.Km)

    def _em_once(self, seed=0):
        rng = np.random.default_rng(seed)

        betas = [rng.normal(scale=0.3, size=kc) for kc in self._Kc]
        gammas = rng.normal(scale=0.1, size=(self.C - 1, self.Km)) if self.Km > 0 else None

        prev_ll = -np.inf
        posterior = np.full((self.N, self.C), 1.0 / self.C)

        for it in range(1, self.maxiter + 1):
            log_choice, _ = self._log_choice(betas)
            priors = self._membership_priors(gammas)
            log_joint = log_choice + np.log(np.clip(priors, 1e-300, None))
            log_marg = logsumexp(log_joint, axis=1, keepdims=True)
            posterior = np.exp(log_joint - log_marg)
            ll = float(log_marg.sum())

            if abs(ll - prev_ll) < self.tol and it > 5:
                return {"betas": betas, "gammas": gammas, "posterior": posterior,
                        "loglik": ll, "converged": True, "iterations": it}

            prev_ll = ll

            for c in range(self.C):
                betas[c] = self._weighted_m_step(betas[c], posterior[:, c], c)

            if gammas is not None and self.Km > 0:
                gammas = self._membership_m_step(gammas, posterior)

        return {"betas": betas, "gammas": gammas, "posterior": posterior,
                "loglik": ll, "converged": False, "iterations": it}

    def fit(self):
        best = None
        for init in range(self.n_init):
            result = self._em_once(seed=init * 42 + 7)
            if best is None or result["loglik"] > best["loglik"]:
                best = result

        self.betas = best["betas"]
        self.gammas = best["gammas"]
        self.posterior = best["posterior"]
        self.loglik = best["loglik"]
        self.converged = best["converged"]
        self.iterations = best["iterations"]
        return self


# ===========================================================================
#  Helpers
# ===========================================================================
def build_class_varnames(gen):
    spec = []
    for k in range(gen.K):
        active = list(gen.shared_vars)
        active += gen.class_specific_vars.get(k, [])
        active += gen.noise_vars
        active += gen.weak_vars
        active += gen.collinear_vars
        spec.append(active)
    return spec


def all_utility_varnames(gen):
    names = []
    for k in range(gen.K):
        for v in gen.shared_vars:
            if v not in names:
                names.append(v)
        for v in gen.class_specific_vars.get(k, []):
            if v not in names:
                names.append(v)
    for v in gen.noise_vars:
        if v not in names:
            names.append(v)
    for v in gen.weak_vars:
        if v not in names:
            names.append(v)
    for v in gen.collinear_vars:
        if v not in names:
            names.append(v)
    return names


def align_classes(gen, model, class_varnames):
    est_dicts = []
    for c in range(gen.K):
        d = {}
        for idx, name in enumerate(class_varnames[c]):
            d[name] = float(model.betas[c][idx])
        est_dicts.append(d)

    cost = np.zeros((gen.K, gen.K))
    for kt in range(gen.K):
        for ke in range(gen.K):
            sse = 0.0
            for v in gen.shared_vars:
                tv = gen.parameters[kt].get(v, 0.0)
                ev = est_dicts[ke].get(v, 0.0)
                sse += (tv - ev) ** 2
            for v in gen.class_specific_vars.get(kt, []):
                tv = gen.parameters[kt].get(v, 0.0)
                ev = est_dicts[ke].get(v, 0.0)
                sse += (tv - ev) ** 2
            cost[kt, ke] = sse

    row_ind, col_ind = linear_sum_assignment(cost)
    return {true: est for true, est in zip(row_ind, col_ind)}, est_dicts


# ===========================================================================
#  Main
# ===========================================================================
def main(quick=True, seed=42):
    if quick:
        n_individuals = 800
        n_choice_tasks = 3
        n_membership_vars = 2
        membership_scale = 2.0
        scale_separation = 3.0
        maxiter = 300
    else:
        n_individuals = 5000
        n_choice_tasks = 1
        n_membership_vars = 3
        membership_scale = 2.5
        scale_separation = 3.0
        maxiter = 300

    print("=" * 78)
    print("  LATENT CLASS MLE RECOVERY (with Membership Variables)")
    print("=" * 78)

    gen = AdvancedLatentClassGenerator(
        n_classes=3,
        n_alternatives=3,
        n_individuals=n_individuals,
        n_choice_tasks=n_choice_tasks,
        scale_separation=scale_separation,
        n_noise_vars=3,
        n_weak_vars=1,
        n_collinear_vars=0,
        n_membership_vars=n_membership_vars,
        membership_scale=membership_scale,
        random_state=seed,
    )

    df, true_classes = gen.generate()
    print(f"\n  Generated: {gen.N} individuals x {gen.T} tasks x {gen.J} alts = {len(df)} rows")
    for k in range(gen.K):
        nk = (true_classes == k).sum()
        print(f"    Class {k}: {nk} ({nk / gen.N:.1%})")

    class_varnames = build_class_varnames(gen)

    print(f"\n  --- True Utility Betas ---")
    for k in range(gen.K):
        for v in class_varnames[k]:
            tv = gen.parameters[k].get(v, 0.0)
            print(f"    Class {k}  {v:>25s} = {tv:+.4f}")

    print(f"\n  --- True Membership Gammas ---")
    if gen.gammas is not None:
        for c in range(gen.K - 1):
            for m in range(gen.n_membership_vars):
                print(f"    Class {c}  {gen.membership_var_names[m]} = {gen.gammas[c, m]:+.4f}")
        print(f"    Class {gen.K - 1} (reference) all = 0")

    # Build column list for X
    uvars = all_utility_varnames(gen)
    all_vars = uvars + gen.membership_var_names

    print(f"\n{'=' * 78}")
    print(f"  FITTING via EM (self-contained)")
    print(f"{'=' * 78}")

    t0 = time.perf_counter()
    model = LCMembershipMLE(
        n_classes=gen.K,
        X=df[all_vars].values,
        y=df["choice"].astype(int).values,
        choice_ids=df["choice_id"].values,
        alts=df["alternative"].values,
        varnames=all_vars,
        class_varnames=class_varnames,
        membership_vars=gen.membership_var_names,
        maxiter=maxiter,
        tol=1e-6,
        l2_penalty=0.2,
        n_init=5,
    )
    model.fit()
    elapsed = time.perf_counter() - t0

    print(f"\n  Converged   : {model.converged}")
    print(f"  Iterations  : {model.iterations}")
    print(f"  Log-lik     : {model.loglik:.4f}")
    print(f"  Time        : {elapsed:.1f}s")

    # Align
    mapping, est_dicts = align_classes(gen, model, class_varnames)
    print(f"\n  --- Class Alignment ---")
    for kt in range(gen.K):
        print(f"    True class {kt}  <->  Estimated class {mapping[kt]}")

    # Compare betas
    print(f"\n{'=' * 78}")
    print(f"  UTILITY COEFFICIENT RECOVERY")
    print(f"{'=' * 78}")

    all_beta_errors = []
    for kt in range(gen.K):
        ke = mapping[kt]
        print(f"\n  True Class {kt} <-> Estimated Class {ke}:")
        print(f"  {'Variable':>25s}  {'True':>10s}  {'Estimated':>10s}  {'Error':>10s}")
        print(f"  {'-' * 25}  {'-' * 10}  {'-' * 10}  {'-' * 10}")
        for v in class_varnames[kt]:
            tv = gen.parameters[kt].get(v, 0.0)
            ev = est_dicts[ke].get(v, 0.0)
            err = ev - tv
            all_beta_errors.append(err)
            print(f"  {v:>25s}  {tv:10.4f}  {ev:10.4f}  {err:+10.4f}")

    all_beta_errors = np.array(all_beta_errors)
    beta_rmse = np.sqrt(np.mean(all_beta_errors ** 2))
    beta_mae = np.mean(np.abs(all_beta_errors))
    print(f"\n  Beta RMSE = {beta_rmse:.6f},  MAE = {beta_mae:.6f}")

    # Compare gammas
    print(f"\n{'=' * 78}")
    print(f"  MEMBERSHIP COEFFICIENT RECOVERY")
    print(f"{'=' * 78}")

    if gen.gammas is not None and model.gammas is not None:
        Z = gen._Z_matrix
        true_logits = np.zeros((gen.N, gen.K))
        for c in range(gen.K - 1):
            true_logits[:, c] = Z @ gen.gammas[c]
        true_probs = softmax(true_logits, axis=1)

        est_logits = np.zeros((gen.N, gen.K))
        for c in range(gen.K - 1):
            est_logits[:, c] = Z @ model.gammas[c]
        est_probs_raw = softmax(est_logits, axis=1)

        inv_map = {v: k for k, v in mapping.items()}
        est_probs = np.zeros_like(est_probs_raw)
        for ke in range(gen.K):
            kt = inv_map[ke]
            est_probs[:, kt] = est_probs_raw[:, ke]

        gamma_rmse = np.sqrt(np.mean((true_probs - est_probs) ** 2))
        print(f"  Membership probability RMSE = {gamma_rmse:.6f}")

        ref_est = mapping[gen.K - 1]

        def _gamma_at(idx):
            if idx >= gen.K - 1:
                return np.zeros(gen.n_membership_vars)
            return model.gammas[idx]

        aligned_gammas = np.zeros((gen.K - 1, gen.n_membership_vars))
        for kt in range(gen.K - 1):
            ke = mapping[kt]
            aligned_gammas[kt] = _gamma_at(ke) - _gamma_at(ref_est)

        gamma_errors = []
        print(f"\n  --- Aligned Gamma Coefficients ---")
        for c in range(gen.K - 1):
            for m in range(gen.n_membership_vars):
                tv = gen.gammas[c, m]
                ev = aligned_gammas[c, m]
                err = ev - tv
                gamma_errors.append(err)
                print(f"    Class {c} {gen.membership_var_names[m]:>15s}:  "
                      f"True={tv:+7.4f}  Est={ev:+7.4f}  Err={err:+7.4f}")

        gamma_errors_arr = np.array(gamma_errors)
        gamma_rmse_dir = np.sqrt(np.mean(gamma_errors_arr ** 2))
        gamma_mae = np.mean(np.abs(gamma_errors_arr))
        print(f"\n  Gamma RMSE (direct) = {gamma_rmse_dir:.6f},  MAE = {gamma_mae:.6f}")

    # Summary
    print(f"\n{'=' * 78}")
    print(f"  RECOVERY SUMMARY")
    print(f"{'=' * 78}")
    print(f"  Beta  RMSE: {beta_rmse:.6f}")
    print(f"  Beta  MAE : {beta_mae:.6f}")
    if gen.gammas is not None and model.gammas is not None:
        print(f"  Gamma RMSE (prob):  {gamma_rmse:.6f}")
        print(f"  Gamma RMSE (direct):{gamma_rmse_dir:.6f}")
        print(f"  Gamma MAE  (direct):{gamma_mae:.6f}")

    return gen, model, mapping, beta_rmse


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    results = main(quick=not args.full, seed=args.seed)
