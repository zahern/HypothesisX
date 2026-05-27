"""SA+PBIL: Simulated Annealing coupled with Population-Based Incremental Learning.

Reference:
    Taco-Morales, M.F. (2026). SA + PBIL: Coupling Simulated Annealing with
    Population-Based Incremental Learning for Discrete Choice Model Specification
    Search. QUT Master's Thesis, May 2026.

Algorithm overview
------------------
The standard SA perturbation direction (add/remove) is guided by a probability
matrix P that learns from accepted solutions.  For each candidate variable five
decisions are tracked:

    1. Inclusion     — is the variable in the model?
    2. Randomness    — does the variable have a random coefficient?
    3. Distribution  — which distribution is used for the random coefficient?
    4. Correlation   — is the random parameter correlated with others?
    5. Box-Cox       — is a Box-Cox transformation applied to the variable?

After every accepted move the matrix is updated using a temperature-coupled
learning rate  l(T):

    p  <-  (1 - l) * p  +  l * 1[significant]

The direction of each perturbation is guided by the PBIL probability:

    p_add = p_bar_out / (p_bar_out + (1 - p_bar_in))

where p_bar_out is the mean probability of variables currently outside the
relevant set (candidates for addition) and p_bar_in is the mean probability
of variables currently inside the set (candidates for removal).

Works with all model types registered in ModelRegistry:
    multinomial, mixed_logit, random_regret, mixed_random_regret,
    ordered_logit, ordered_probit, nested_logit, and any future additions.

Pre-specified (analyst-forced) variables are tracked by the matrix but their
probabilities are NOT updated — their membership is controlled entirely by
the Parameters constraints.
"""

import logging
import os
from datetime import datetime

import numpy as np

try:
    from siman import SA
except ImportError:
    from .siman import SA


# ---------------------------------------------------------------------------
# Learning-rate bounds  (Table 1 from Taco-Morales 2026)
# ---------------------------------------------------------------------------
_L_BOUNDS = {
    "inclusion":    (0.02, 0.25),
    "random":       (0.02, 0.15),
    "distribution": (0.01, 0.05),
    "correlation":  (0.02, 0.15),
    "boxcox":       (0.02, 0.15),
}

_P_LOW  = 0.05   # minimum probability clamp
_P_HIGH = 0.95   # maximum probability clamp


# ---------------------------------------------------------------------------
# Probability Matrix
# ---------------------------------------------------------------------------

class ProbabilityMatrix:
    """Tracks five decision-level probabilities for each candidate variable.

    Parameters
    ----------
    varnames : list[str]
        All candidate variable names (typically param.asvarnames).
    distributions : list[str]
        Available random-coefficient distributions (e.g. ['n','ln','tn','u','t']).
    """

    def __init__(self, varnames, distributions):
        self.varnames      = list(varnames)
        self.distributions = list(distributions)
        n_distr = max(len(distributions), 1)

        # Scalar decisions — initialised at 0.5 (uninformative prior)
        self.p_inclusion = {v: 0.5 for v in varnames}
        self.p_random    = {v: 0.5 for v in varnames}
        self.p_corr      = {v: 0.5 for v in varnames}
        self.p_bc        = {v: 0.5 for v in varnames}

        # Distribution probabilities — initialised uniformly
        self.p_distr = {
            v: {d: 1.0 / n_distr for d in distributions}
            for v in varnames
        }

    # ------------------------------------------------------------------
    def learning_rate(self, decision_type, t, tI):
        """Temperature-coupled learning rate for a given decision type.

        l(T) = l_min + (l_max - l_min) * (1 - T / T_I)

        Clipped to [l_min, l_max] for numeric safety.
        """
        l_min, l_max = _L_BOUNDS[decision_type]
        if tI <= 0:
            return l_max
        progress = max(0.0, min(1.0, 1.0 - t / tI))
        return l_min + (l_max - l_min) * progress

    @staticmethod
    def _clamp(p):
        return max(_P_LOW, min(_P_HIGH, float(p)))

    # ------------------------------------------------------------------
    def update_inclusion(self, var, indicator, t, tI):
        lr = self.learning_rate("inclusion", t, tI)
        self.p_inclusion[var] = self._clamp(
            (1.0 - lr) * self.p_inclusion[var] + lr * indicator
        )

    def update_random(self, var, indicator, t, tI):
        lr = self.learning_rate("random", t, tI)
        self.p_random[var] = self._clamp(
            (1.0 - lr) * self.p_random[var] + lr * indicator
        )

    def update_distribution(self, var, distr, indicator, t, tI):
        if distr not in self.p_distr.get(var, {}):
            return
        lr = self.learning_rate("distribution", t, tI)
        self.p_distr[var][distr] = self._clamp(
            (1.0 - lr) * self.p_distr[var][distr] + lr * indicator
        )

    def update_correlation(self, var, indicator, t, tI):
        lr = self.learning_rate("correlation", t, tI)
        self.p_corr[var] = self._clamp(
            (1.0 - lr) * self.p_corr[var] + lr * indicator
        )

    def update_boxcox(self, var, indicator, t, tI):
        lr = self.learning_rate("boxcox", t, tI)
        self.p_bc[var] = self._clamp(
            (1.0 - lr) * self.p_bc[var] + lr * indicator
        )

    # ------------------------------------------------------------------
    def summary(self):
        """Return a list of dicts for logging / inspection."""
        rows = []
        for v in self.varnames:
            rows.append({
                "var":     v,
                "p_incl":  round(self.p_inclusion[v], 3),
                "p_rand":  round(self.p_random[v],    3),
                "p_corr":  round(self.p_corr[v],      3),
                "p_bc":    round(self.p_bc[v],        3),
                "p_distr": {d: round(p, 3) for d, p in self.p_distr[v].items()},
            })
        return rows


# ---------------------------------------------------------------------------
# SA+PBIL Solver
# ---------------------------------------------------------------------------

class SAPBIL(SA):
    """Simulated Annealing coupled with Population-Based Incremental Learning.

    Extends the base :class:`SA` class by replacing the uniform-random
    perturbation direction with a PBIL-guided direction derived from a
    probability matrix that is updated from accepted solutions.

    Parameters
    ----------
    param : Parameters
        Problem definition (from search.py).
    init_sol : Solution or None
        Optional warm-start solution.
    ctrl : tuple
        ``(tI, tF, max_temp_steps, max_iter)``
    idnum : int or str
        Run identifier for log files.
    **kwargs
        Forwarded to the :class:`SA` base class constructor.
    """

    def __init__(self, param, init_sol, ctrl, idnum=0, **kwargs):
        super().__init__(param, init_sol, ctrl, idnum=idnum, **kwargs)

        varnames      = list(param.asvarnames or [])
        distributions = list(param.distr or ["n", "ln", "tn", "u", "t"])
        self.prob_matrix = ProbabilityMatrix(varnames, distributions)

        # Protected (pre-specified) variables — PBIL does NOT update these
        self._ps_asvars   = set(getattr(param, "ps_asvars",  []) or [])
        self._ps_randvars = set(
            (getattr(param, "ps_randvars", {}) or {}).keys()
        )
        self._ps_bcvars   = set(getattr(param, "ps_bcvars",  []) or [])
        self._ps_corvars  = set(getattr(param, "ps_corvars", []) or [])

        self._pbil_updates = 0   # diagnostic: how many PBIL updates happened

    # ------------------------------------------------------------------
    # Output files  (override to use "sapbil_" prefix and correct header)
    # ------------------------------------------------------------------
    def open_files(self):
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"sapbil_{self.idnum}_{ts}"
        base_dir = getattr(self, "_output_dir", "sa_runs")
        self._run_dir = os.path.join(base_dir, run_name)
        os.makedirs(self._run_dir, exist_ok=True)

        def _open(filename, mode="w"):
            return open(
                os.path.join(self._run_dir, filename),
                mode, encoding="utf-8", buffering=1
            )

        self.results_file  = _open("results.txt")
        self.progress_file = _open("progress.csv")
        self.progress_file.write(
            "step,temperature,current_obj,best_obj,n_accepted,elapsed_s\n"
        )
        self.debug_file   = _open("perturbations.csv")
        self.debug_file.write("step,obj_values,accepted\n")
        self.archive_file = _open("archive.txt")
        self.best_file    = _open("best.txt")

        criterions = getattr(self.param, "criterions", [])
        models     = getattr(self.param, "models_avail", [])
        print("SearchLibrium — SA+PBIL Run",        file=self.results_file)
        print(f"Run ID       : {self.idnum}",        file=self.results_file)
        print(f"Started      : {ts}",                file=self.results_file)
        print(f"Output dir   : {self._run_dir}",     file=self.results_file)
        print(f"Criterions   : {criterions}",        file=self.results_file)
        print(f"Models       : {models}",            file=self.results_file)
        print(
            f"tI={self.tI}, tF={self.tF}, "
            f"steps={self.max_temp_steps}, iter/step={self.max_iter}",
            file=self.results_file,
        )
        print("─" * 60, file=self.results_file)

    # ------------------------------------------------------------------
    # Main perturbation override
    # ------------------------------------------------------------------
    def perturb_solution(self, sol):
        """PBIL-guided SA perturbation — replaces the uniform-random version."""
        baseline_sig    = self.setup_signature(sol)
        curr_score      = [sol.obj(i) for i in range(self.nb_crit)]
        n_perturbations = np.random.randint(1, 6)
        max_attempts    = 15

        new_sol = sol
        for _attempt in range(max_attempts):
            new_sol = self.copy_solution(sol)
            for _ in range(n_perturbations):
                new_sol = self._pbil_perturb(new_sol)
            if new_sol is None:
                return sol

            new_sol = self.apply_constraints(new_sol)
            new_sol = self.repair_solution_for_clarity(new_sol)

            if self.setup_signature(new_sol) != baseline_sig:
                break
            # Widen budget slightly if the neighbourhood is hard to escape
            n_perturbations = min(n_perturbations + 1, 8)

        if self.setup_signature(new_sol) == baseline_sig:
            return sol

        # Evaluate the candidate
        new_sol, converged = self.evaluate(new_sol)
        if not converged:
            self.not_converged += 1
            return self.current_sol

        new_score = [new_sol.obj(i) for i in range(self.nb_crit)]
        if self.accept_change(curr_score, new_score):
            accd = True
            self.no_impr = 0
            self.accepted += 1
            self.current_sol = new_sol
            self.update_best(new_sol)
            # ── PBIL update after every accepted move ────────────────────
            self._update_probability_matrix(new_sol)
        else:
            accd = False
            self.not_accepted += 1

        self.log_kpi(new_sol, self.debug_file, accd)
        return self.current_sol

    # ------------------------------------------------------------------
    # PBIL-guided single perturbation step
    # ------------------------------------------------------------------
    def _pbil_perturb(self, sol):
        """Apply one PBIL-guided perturbation to *sol* and return it."""
        dims = self._available_dimensions(sol)
        if not dims:
            return sol

        dim = np.random.choice(dims)
        if dim == "inclusion":
            return self._pbil_inclusion(sol)
        elif dim == "random":
            return self._pbil_random(sol)
        elif dim == "distribution":
            return self._pbil_distribution(sol)
        elif dim == "correlation":
            return self._pbil_correlation(sol)
        elif dim == "boxcox":
            return self._pbil_boxcox(sol)
        return sol

    # ------------------------------------------------------------------
    def _available_dimensions(self, sol):
        """Return which PBIL decision dimensions are currently actionable."""
        pm   = self.prob_matrix
        dims = ["inclusion"]   # always available

        if self.param.allow_random:
            can_add_rand = any(
                v in sol.get("asvars", []) and v not in sol.get("randvars", {})
                for v in pm.varnames
            )
            can_rem_rand = bool(sol.get("randvars"))
            if can_add_rand or can_rem_rand:
                dims.append("random")

            if sol.get("randvars") and len(pm.distributions) > 1:
                dims.append("distribution")

        if self.param.allow_corvars and self.param.allow_random:
            cor_eligible = [
                v for v in sol.get("randvars", {})
                if v not in sol.get("bcvars", [])
            ]
            if len(cor_eligible) >= 2 or sol.get("corvars"):
                dims.append("correlation")

        if self.param.allow_bcvars:
            can_add_bc = any(
                v in sol.get("asvars", [])
                and v not in sol.get("bcvars",  [])
                and v not in sol.get("corvars", [])
                for v in pm.varnames
            )
            can_rem_bc = bool(sol.get("bcvars"))
            if can_add_bc or can_rem_bc:
                dims.append("boxcox")

        return dims

    # ------------------------------------------------------------------
    # Inclusion dimension
    # ------------------------------------------------------------------
    def _pbil_inclusion(self, sol):
        pm       = self.prob_matrix
        in_vars  = [v for v in sol.get("asvars", [])  if v in pm.p_inclusion]
        out_vars = [v for v in pm.varnames             if v not in sol.get("asvars", [])]

        p_bar_in  = float(np.mean([pm.p_inclusion[v] for v in in_vars]))  if in_vars  else 0.5
        p_bar_out = float(np.mean([pm.p_inclusion[v] for v in out_vars])) if out_vars else 0.5

        denom = p_bar_out + (1.0 - p_bar_in)
        p_add = p_bar_out / denom if denom > 1e-12 else 0.5

        if np.random.random() < p_add and out_vars:
            # Add: sample proportionally to p_inclusion
            probs = np.array([pm.p_inclusion[v] for v in out_vars], dtype=float)
            probs = probs / (probs.sum() + 1e-12)
            var   = np.random.choice(out_vars, p=probs)
            return self.add_asvar(var, sol)

        elif in_vars:
            # Remove: sample proportionally to (1 - p_inclusion), skip pre-specified
            non_ps = [v for v in in_vars if v not in self._ps_asvars]
            if not non_ps or len(in_vars) <= 1:
                return sol
            probs = np.array([1.0 - pm.p_inclusion[v] for v in non_ps], dtype=float)
            if probs.sum() < 1e-12:
                probs = np.ones(len(non_ps), dtype=float)
            probs = probs / probs.sum()
            var   = np.random.choice(non_ps, p=probs)
            return self.remove_asvar(var, sol)

        return sol

    # ------------------------------------------------------------------
    # Randomness dimension
    # ------------------------------------------------------------------
    def _pbil_random(self, sol):
        pm       = self.prob_matrix
        in_rand  = list(sol.get("randvars", {}).keys())
        out_rand = [
            v for v in sol.get("asvars", [])
            if v in pm.p_random and v not in in_rand
        ]

        p_bar_in  = float(np.mean([pm.p_random[v] for v in in_rand]))  if in_rand  else 0.5
        p_bar_out = float(np.mean([pm.p_random[v] for v in out_rand])) if out_rand else 0.5

        denom = p_bar_out + (1.0 - p_bar_in)
        p_add = p_bar_out / denom if denom > 1e-12 else 0.5

        if np.random.random() < p_add and out_rand:
            # Add a random coefficient
            probs = np.array([pm.p_random[v] for v in out_rand], dtype=float)
            probs = probs / (probs.sum() + 1e-12)
            var   = np.random.choice(out_rand, p=probs)
            self.add_randvar(var, sol)
            # Override with PBIL-guided distribution
            if var in sol.get("randvars", {}):
                sol["randvars"][var] = self._pbil_sample_distribution(var, sol)

        elif in_rand:
            # Remove a random coefficient (skip pre-specified)
            non_ps = [v for v in in_rand if v not in self._ps_randvars]
            if non_ps:
                probs = np.array([1.0 - pm.p_random[v] for v in non_ps], dtype=float)
                if probs.sum() < 1e-12:
                    probs = np.ones(len(non_ps), dtype=float)
                probs = probs / probs.sum()
                var   = np.random.choice(non_ps, p=probs)
                self.remove_randvar(var, sol)

        return sol

    # ------------------------------------------------------------------
    # Distribution dimension
    # ------------------------------------------------------------------
    def _pbil_distribution(self, sol):
        pm       = self.prob_matrix
        randvars = sol.get("randvars", {})
        # Only change distributions for non-pre-specified variables
        candidates = [v for v in randvars if v not in self._ps_randvars]
        if not candidates:
            return sol

        # Uniform selection of which variable to change
        var       = np.random.choice(candidates)
        new_distr = self._pbil_sample_distribution(var, sol)
        if new_distr != randvars.get(var):
            self.change_distribution(var, new_distr, sol)

        return sol

    def _pbil_sample_distribution(self, var, sol):
        """Sample a distribution for *var* proportionally to its PBIL probabilities.

        The current distribution is excluded from sampling to encourage change
        (the caller may re-check and accept the same distribution if no other
        option exists).
        """
        pm    = self.prob_matrix
        distrs = pm.distributions
        probs  = np.array(
            [pm.p_distr[var].get(d, 1.0 / max(len(distrs), 1)) for d in distrs],
            dtype=float,
        )
        # Zero out the currently assigned distribution to prefer a change
        current = sol.get("randvars", {}).get(var)
        if current in distrs and len(distrs) > 1:
            probs[distrs.index(current)] = 0.0
        if probs.sum() < 1e-12:
            probs = np.ones(len(distrs), dtype=float)
        probs = probs / probs.sum()
        return np.random.choice(distrs, p=probs)

    # ------------------------------------------------------------------
    # Correlation dimension
    # ------------------------------------------------------------------
    def _pbil_correlation(self, sol):
        pm       = self.prob_matrix
        eligible = [
            v for v in sol.get("randvars", {})
            if v in pm.p_corr and v not in sol.get("bcvars", [])
        ]
        in_cor  = [v for v in sol.get("corvars", []) if v in eligible]
        out_cor = [v for v in eligible if v not in in_cor]

        p_bar_in  = float(np.mean([pm.p_corr[v] for v in in_cor]))  if in_cor  else 0.5
        p_bar_out = float(np.mean([pm.p_corr[v] for v in out_cor])) if out_cor else 0.5

        denom = p_bar_out + (1.0 - p_bar_in)
        p_add = p_bar_out / denom if denom > 1e-12 else 0.5

        if np.random.random() < p_add and len(out_cor) >= 1:
            # Add correlation: follow the same logic as perturb_add_corfeature
            new_corvars = sorted(set(sol.get("corvars", [])).union(eligible))
            sol["corvars"] = new_corvars if len(new_corvars) >= 2 else []
            sol["corvars"] = [v for v in self.param.varnames if v in sol["corvars"]]
            sol["bcvars"]  = [v for v in sol.get("bcvars", []) if v not in sol["corvars"]]

        elif in_cor:
            # Remove correlation for one variable
            non_ps = [v for v in in_cor if v not in self._ps_corvars]
            if non_ps:
                probs = np.array([1.0 - pm.p_corr[v] for v in non_ps], dtype=float)
                if probs.sum() < 1e-12:
                    probs = np.ones(len(non_ps), dtype=float)
                probs = probs / probs.sum()
                var   = np.random.choice(non_ps, p=probs)
                self.remove_corvar(var, sol)

        return sol

    # ------------------------------------------------------------------
    # Box-Cox dimension
    # ------------------------------------------------------------------
    def _pbil_boxcox(self, sol):
        pm     = self.prob_matrix
        in_bc  = [v for v in sol.get("bcvars", [])  if v in pm.p_bc]
        out_bc = [
            v for v in sol.get("asvars", [])
            if v in pm.p_bc
            and v not in sol.get("bcvars",  [])
            and v not in sol.get("corvars", [])
        ]

        p_bar_in  = float(np.mean([pm.p_bc[v] for v in in_bc]))  if in_bc  else 0.5
        p_bar_out = float(np.mean([pm.p_bc[v] for v in out_bc])) if out_bc else 0.5

        denom = p_bar_out + (1.0 - p_bar_in)
        p_add = p_bar_out / denom if denom > 1e-12 else 0.5

        if np.random.random() < p_add and out_bc:
            probs = np.array([pm.p_bc[v] for v in out_bc], dtype=float)
            probs = probs / (probs.sum() + 1e-12)
            var   = np.random.choice(out_bc, p=probs)
            self.add_bcvar(var, sol)

        elif in_bc:
            non_ps = [v for v in in_bc if v not in self._ps_bcvars]
            if non_ps:
                probs = np.array([1.0 - pm.p_bc[v] for v in non_ps], dtype=float)
                if probs.sum() < 1e-12:
                    probs = np.ones(len(non_ps), dtype=float)
                probs = probs / probs.sum()
                var   = np.random.choice(non_ps, p=probs)
                self.remove_bcvar(var, sol)

        return sol

    # ------------------------------------------------------------------
    # PBIL matrix update after an accepted solution
    # ------------------------------------------------------------------
    def _update_probability_matrix(self, sol):
        """Update the probability matrix based on an accepted solution.

        For each variable in the full candidate set the significance of its
        coefficients in the fitted model determines the update indicators
        for each decision level.
        """
        pm      = self.prob_matrix
        t, tI   = self.t, self.tI

        # Build a {coeff_name: bool_significant} map from the fitted model
        sig = self._build_significance_map(sol)

        asvars   = set(sol.get("asvars",  []) or [])
        randvars = dict(sol.get("randvars", {}) or {})
        bcvars   = set(sol.get("bcvars",  []) or [])
        corvars  = set(sol.get("corvars", []) or [])

        for var in pm.varnames:
            # Pre-specified variables: skip all PBIL updates
            if var in self._ps_asvars:
                continue

            in_model  = var in asvars
            is_random = var in randvars
            is_bc     = var in bcvars
            is_corr   = var in corvars

            # ── 1. Inclusion ─────────────────────────────────────────────
            if in_model:
                mean_sig = sig.get(var, False)
                sd_sig   = sig.get(f"sd.{var}", False)
                if mean_sig or (is_random and sd_sig):
                    incl_ind = 1.0
                elif (not mean_sig) and (not is_random or (not mean_sig and not sd_sig)):
                    incl_ind = 0.0
                else:
                    incl_ind = 0.5   # ambiguous — no change to prior
            else:
                # Not in model → reduce inclusion probability
                incl_ind = 0.0

            pm.update_inclusion(var, incl_ind, t, tI)

            if not in_model:
                continue   # decisions 2-5 only apply to in-model variables

            # ── 2. Randomness ─────────────────────────────────────────────
            if var not in self._ps_randvars:
                if is_random:
                    rand_ind = 1.0 if sig.get(f"sd.{var}", False) else 0.0
                else:
                    rand_ind = 0.0
                pm.update_random(var, rand_ind, t, tI)

            if not is_random:
                continue   # decisions 3-5 only apply when variable is random

            # ── 3. Distribution ──────────────────────────────────────────
            if var not in self._ps_randvars:
                current_distr = randvars[var]
                mean_sig_d    = sig.get(var,          False)
                sd_sig_d      = sig.get(f"sd.{var}",  False)
                distr_ind     = 1.0 if (mean_sig_d and sd_sig_d) else 0.0
                pm.update_distribution(var, current_distr, distr_ind, t, tI)

            # ── 4. Correlation ───────────────────────────────────────────
            if not is_bc and var not in self._ps_corvars:
                if is_corr:
                    chol_sig = any(
                        (
                            sig.get(f"chol.{var}.{v2}", False)
                            or sig.get(f"chol.{v2}.{var}", False)
                        )
                        for v2 in corvars
                        if v2 != var
                    )
                    corr_ind = 1.0 if chol_sig else 0.0
                else:
                    corr_ind = 0.0
                pm.update_correlation(var, corr_ind, t, tI)

            # ── 5. Box-Cox ───────────────────────────────────────────────
            if not is_corr and var not in self._ps_bcvars:
                if is_bc:
                    bc_ind = 1.0 if sig.get(f"lambda.{var}", False) else 0.0
                else:
                    bc_ind = 0.0
                pm.update_boxcox(var, bc_ind, t, tI)

        self._pbil_updates += 1

    # ------------------------------------------------------------------
    def _build_significance_map(self, sol):
        """Build a ``{coeff_name: bool}`` significance map from a fitted model.

        Returns an empty dict if the model or its p-values are unavailable.
        """
        model = sol.get("model")
        if model is None:
            return {}
        try:
            pvalues     = np.array(model.pvalues)
            coeff_names = list(model.coeff_names) if model.coeff_names is not None else []
            p_thresh    = getattr(self.param, "p_val", 0.05)
            return {
                name: bool(pv <= p_thresh)
                for name, pv in zip(coeff_names, pvalues)
            }
        except Exception as exc:
            logging.debug("SA+PBIL: could not build significance map: %s", exc)
            return {}

    # ------------------------------------------------------------------
    # Finalise: print probability matrix summary
    # ------------------------------------------------------------------
    def finalise(self):
        super().finalise()

        header = (
            f"\nSA+PBIL — probability matrix after {self._pbil_updates} update(s):\n"
            f"  {'Variable':<22s}  {'P(incl)':>8s}  {'P(rand)':>8s}  "
            f"{'P(corr)':>8s}  {'P(bc)':>6s}  P(distr)"
        )
        lines = [header]

        for row in self.prob_matrix.summary():
            distr_str = " ".join(f"{d}={v}" for d, v in row["p_distr"].items())
            lines.append(
                f"  {row['var']:<22s}  {row['p_incl']:>8.3f}  "
                f"{row['p_rand']:>8.3f}  {row['p_corr']:>8.3f}  "
                f"{row['p_bc']:>6.3f}  {distr_str}"
            )

        for line in lines:
            print(line)
            try:
                print(line, file=self.results_file)
            except Exception:
                logging.debug(
                    "SA+PBIL: unable to write probability matrix to results_file"
                )
