"""
latent_class_recovery_test.py
=============================
1. Generates a latent class dataset with class-specific variable specifications
   (some variables only appear in certain classes — true beta = 0 elsewhere).
2. Fits SearchLibrium's LatentClassMixedLogit with class_params_spec to match
   the true data-generating process.
3. Compares estimated vs true coefficients per class.
4. Reports recovery metrics.

This validates that SearchLibrium can correctly identify class membership
and recover class-specific utility parameters when the specification is
correctly informed.
"""

from __future__ import annotations

import sys
import time
import numpy as np
import pandas as pd
from scipy.special import softmax
import matplotlib.pyplot as plt

# Add the data_generator directory to path
sys.path.insert(0, "data_generator")

from SearchLibrium.latent_class import LatentClassMixedLogit


# ---------------------------------------------------------------------------
#  Data Generation (inline, updated from latent_class_gen.py)
# ---------------------------------------------------------------------------
class LatentClassGenerator:
    """Generate latent class choice data with class-specific variable sets.

    Each class has:
      - 3 shared variables (price, travel_time, waiting_time) — present in all
        classes but with different coefficient values
      - 2 class-specific variables — only influence utility in their class
      - 3 noise variables — true beta = 0 everywhere

    Usage with SearchLibrium requires class_params_spec that mirrors the
    true structure so the model knows which vars are active per class.
    """

    def __init__(
        self,
        n_classes=3,
        n_alternatives=3,
        n_individuals=2000,
        n_choice_tasks=5,
        class_probs=None,
        scale_separation=2.0,
        n_noise_vars=3,
        rare_class=False,
        random_state=42,
    ):
        self.rng = np.random.default_rng(random_state)
        self.K = n_classes
        self.J = n_alternatives
        self.N = n_individuals
        self.T = n_choice_tasks
        self.scale_separation = scale_separation
        self.n_noise_vars = n_noise_vars

        if class_probs is None:
            if rare_class:
                probs = np.ones(n_classes)
                probs[-1] = 0.05
                probs[:-1] = (1 - 0.05) / (n_classes - 1)
                self.class_probs = probs
            else:
                self.class_probs = np.ones(n_classes) / n_classes
        else:
            self.class_probs = np.array(class_probs)

        self._define_variables()
        self._generate_parameters()

    def _define_variables(self):
        self.shared_vars = ["price", "travel_time", "waiting_time"]

        self.class_specific_vars = {
            0: ["comfort_level", "seat_space"],
            1: ["brand_reputation", "loyalty_points"],
            2: ["eco_rating", "carbon_emissions"],
        }

        self.noise_vars = [f"noise_{i}" for i in range(self.n_noise_vars)]

        # Convenience: full variable list per class (for class_params_spec)
        self.class_varnames = {}
        for k in range(self.K):
            self.class_varnames[k] = (
                self.shared_vars + self.class_specific_vars.get(k, [])
            )

        # All variable names (union)
        self.all_varnames = list(self.shared_vars)
        for vs in self.class_specific_vars.values():
            for v in vs:
                if v not in self.all_varnames:
                    self.all_varnames.append(v)
        self.all_varnames += self.noise_vars

    def _generate_parameters(self):
        self.parameters = {}
        for k in range(self.K):
            params = {}
            for var in self.shared_vars:
                params[var] = self.rng.normal(
                    loc=k * self.scale_separation, scale=1.0
                )
            for var in self.class_specific_vars.get(k, []):
                params[var] = self.rng.normal(
                    loc=(k + 1) * self.scale_separation, scale=1.0
                )
            for var in self.noise_vars:
                params[var] = 0.0
            self.parameters[k] = params

    def _generate_features(self):
        X = {}
        for var in self.all_varnames:
            X[var] = self.rng.normal(size=(self.N, self.T, self.J))
        return X

    def generate(self):
        classes = self.rng.choice(self.K, size=self.N, p=self.class_probs)
        X = self._generate_features()
        rows = []

        for n in range(self.N):
            k = classes[n]
            for t in range(self.T):
                utilities = np.zeros(self.J)
                for j in range(self.J):
                    for var, beta in self.parameters[k].items():
                        utilities[j] += beta * X[var][n, t, j]
                probs = softmax(utilities)
                choice = self.rng.choice(self.J, p=probs)
                for j in range(self.J):
                    row = {
                        "individual": n,
                        "choice_task": t,
                        "alternative": j,
                        "choice": 1 if j == choice else 0,
                        "true_class": k,
                    }
                    for var in self.all_varnames:
                        row[var] = X[var][n, t, j]
                    rows.append(row)

        df = pd.DataFrame(rows)
        # Create unique choice-task ID: each task has exactly J alternatives
        df["choice_id"] = df["individual"] * self.T + df["choice_task"]
        return df, classes


# ---------------------------------------------------------------------------
#  Recovery Test
# ---------------------------------------------------------------------------
def main(seed: int = 42):
    np.random.seed(seed)

    # ---- Generate data ----------------------------------------------------
    print("=" * 78)
    print("  Latent Class Recovery Test")
    print("=" * 78)

    gen = LatentClassGenerator(
        n_classes=3,
        n_alternatives=3,
        n_individuals=5000,
        n_choice_tasks=1,
        scale_separation=3.0,    # stronger class separation
        n_noise_vars=3,
        random_state=seed,
    )
    df, true_classes = gen.generate()
    n_obs = df["individual"].nunique()

    print(f"  Generated: {n_obs} individuals x {gen.T} tasks x {gen.J} alts")
    print(f"  Classes: {gen.K}  |  shares: {gen.class_probs}  |  T = {gen.T} (1 obs/person)")
    print(f"  Variables: {len(gen.all_varnames)} total")
    for k in range(gen.K):
        print(f"    Class {k}: {gen.class_varnames[k]}")

    # ---- Prepare class_params_spec -----------------------------------------
    # Must match the generator's structure: only active variables per class.
    class_params_spec = [gen.class_varnames[k] for k in range(gen.K)]

    # ---- Fit Latent Class model --------------------------------------------
    print(f"\n{'='*78}")
    print(f"  Fitting SearchLibrium LatentClassMixedLogit")
    print(f"{'='*78}")

    t0 = time.perf_counter()
    lc = LatentClassMixedLogit(
        n_classes=gen.K,
        maxiter=200,
        class_maxiter=100,
        tol=1e-6,
        random_state=seed,
    )
    lc.setup(
        X=df[gen.all_varnames].values,
        y=df["choice"].astype(int).values,
        varnames=gen.all_varnames,
        ids=df["choice_id"].values,
        alts=df["alternative"].values,
        class_params_spec=class_params_spec,
    )
    lc.fit(em_method="squarem")
    elapsed = time.perf_counter() - t0

    print(f"\n  LC converged : {lc.converged}")
    print(f"  Iterations   : {lc.total_iter}")
    print(f"  Log-lik      : {lc.loglik:.4f}")
    print(f"  AIC          : {lc.aic:.1f}")
    print(f"  BIC          : {lc.bic:.1f}")
    print(f"  Time         : {elapsed:.1f}s")

    # Debug: what did we actually get?
    print(f"\n  DEBUG: class_betas type = {type(lc.class_betas)}")
    if hasattr(lc, 'class_betas') and lc.class_betas is not None:
        for c, cb in enumerate(lc.class_betas):
            print(f"    class {c} betas: {np.array2string(np.asarray(cb), precision=4, suppress_small=True)}")
    print(f"  DEBUG: coeff_names type = {type(lc.coeff_names)}")
    if hasattr(lc, 'coeff_names'):
        cn = lc.coeff_names
        if isinstance(cn, (list, np.ndarray)):
            if len(cn) > 0 and isinstance(cn[0], (list, np.ndarray)):
                for c, names in enumerate(cn):
                    print(f"    class {c} names: {[str(n) for n in names]}")
            else:
                print(f"    flat names: {[str(n) for n in cn]}")

    # ---- Class share recovery ---------------------------------------------
    print(f"\n{'='*78}")
    print(f"  CLASS SHARE RECOVERY")
    print(f"{'='*78}")
    print(f"  {'':>8s}  {'True':>8s}  {'Estimated':>10s}  {'Error':>8s}")
    true_shares = np.bincount(true_classes) / len(true_classes)
    est_shares = lc.class_probs if hasattr(lc, "class_probs") else np.ones(gen.K) / gen.K
    for k in range(gen.K):
        print(f"  Class {k}:  {true_shares[k]:8.4f}  {est_shares[k]:10.4f}  "
              f"{est_shares[k] - true_shares[k]:+8.4f}")

    # Align estimated classes to true classes using coefficient matching
    # (more reliable than share-based matching when shares are similar)
    from scipy.optimize import linear_sum_assignment

    # Build per-class estimated dictionaries
    # coeff_names is a flat list: ['class_1_price', 'class_1_travel_time', ...]
    # class_betas is a list of arrays, one per class, in order
    est_dicts = []
    K_per_class = len(gen.class_varnames[0])  # 5 vars per class
    for c in range(gen.K):
        d = {}
        start = c * K_per_class
        for j in range(K_per_class):
            idx = start + j
            if idx < len(lc.coeff_names) and j < len(lc.class_betas[c]):
                raw_name = str(lc.coeff_names[idx])
                # Strip 'class_X_' prefix
                prefix = f"class_{c+1}_"
                if raw_name.startswith(prefix):
                    var_name = raw_name[len(prefix):]
                else:
                    var_name = raw_name
                d[var_name] = float(lc.class_betas[c][j])
        est_dicts.append(d)

    print(f"  Parsed estimated coefficients:")
    for c in range(gen.K):
        d = {k: float(f"{v:.3f}") for k, v in est_dicts[c].items()}
        print(f"    Class {c}: {d}")

    # Align estimated classes to true classes (label switching via Hungarian)
    from scipy.optimize import linear_sum_assignment
    cost = np.zeros((gen.K, gen.K))
    for k_true in range(gen.K):
        for k_est in range(gen.K):
            sse = 0.0
            for var in gen.shared_vars:
                tv = gen.parameters[k_true].get(var, 0.0)
                ev = est_dicts[k_est].get(var, 0.0)
                sse += (tv - ev) ** 2
            cost[k_true, k_est] = sse
    row_ind, col_ind = linear_sum_assignment(cost)
    inv_mapping = {true: est for true, est in zip(row_ind, col_ind)}  # true -> est

    print(f"\n  Label alignment (coefficient-based):")
    for k_true in range(gen.K):
        print(f"    True class {k_true} -> Estimated class {inv_mapping[k_true]}")

    # ---- Coefficient recovery ---------------------------------------------
    print(f"\n{'='*78}")
    print(f"  COEFFICIENT RECOVERY")
    print(f"{'='*78}")

    all_errors = []
    for k_true in range(gen.K):
        k_est = inv_mapping[k_true]
        true_params = gen.parameters[k_true]
        est_dict = est_dicts[k_est]

        print(f"\n  True Class {k_true}  <->  Estimated Class {k_est}:")
        print(f"  {'Variable':>20s}  {'True':>10s}  {'Estimated':>10s}  "
              f"{'Error':>10s}  {'AbsErr':>10s}")
        print(f"  {'-'*20}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")

        active_vars = gen.class_varnames[k_true]
        for var in gen.all_varnames:
            true_val = true_params.get(var, 0.0)
            est_val = est_dict.get(str(var), 0.0)
            err = est_val - true_val
            all_errors.append(err)
            marker = " *" if abs(err) > 0.5 else ""
            if var in active_vars or var in gen.noise_vars:
                print(f"  {var:>20s}  {true_val:10.4f}  {est_val:10.4f}  "
                      f"{err:+10.4f}  {abs(err):10.4f}{marker}")

    # Summary stats
    all_errors = np.array(all_errors)
    rmse = np.sqrt(np.mean(all_errors ** 2))
    mae = np.mean(np.abs(all_errors))
    print(f"\n  RECOVERY SUMMARY:")
    print(f"    RMSE  = {rmse:.6f}")
    print(f"    MAE   = {mae:.6f}")
    print(f"    MaxAE = {np.max(np.abs(all_errors)):.4f}")

    # ---- Plot -------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for k_true in range(gen.K):
        k_est = inv_mapping[k_true]
        ax = axes[k_true]
        true_vals = []
        est_vals = []
        labels = []
        for var in gen.class_varnames[k_true]:
            if var in gen.parameters[k_true]:
                true_vals.append(gen.parameters[k_true][var])
                est_vals.append(est_dicts[k_est].get(str(var), 0.0))
                labels.append(var)

        ax.bar(np.arange(len(labels)) - 0.2, true_vals, 0.35, label="True",
               color="steelblue")
        ax.bar(np.arange(len(labels)) + 0.2, est_vals, 0.35, label="Estimated",
               color="darkorange")
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_title(f"Class {k_true} (share={true_shares[k_true]:.2f})")
        ax.axhline(y=0, color="gray", lw=0.5)
        if k_true == 0:
            ax.legend(fontsize=8)

    fig.suptitle("Latent Class Coefficient Recovery", fontsize=13, fontweight="bold")
    fig.tight_layout()
    out_path = "latent_class_recovery.png"
    fig.savefig(out_path, dpi=150)
    print(f"\n  Figure saved to '{out_path}'")

    return {
        "generator": gen,
        "model": lc,
        "inv_mapping": inv_mapping,
        "rmse": rmse,
        "mae": mae,
        "true_classes": true_classes,
    }


if __name__ == "__main__":
    results = main()
