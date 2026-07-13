"""HS+PBIL: Harmony Search coupled with Population-Based Incremental Learning.

Reference:
    Taco-Morales, M.F. (2026). SA + PBIL: Coupling Simulated Annealing with
    Population-Based Incremental Learning for Discrete Choice Model Specification
    Search. QUT Master's Thesis, May 2026.

This module extends the PBIL probability-guided perturbation mechanism from
SAPBIL to the Harmony Search algorithm.  Instead of uniform-random
perturbations during pitch adjustment, the PBIL probability matrix guides
add/remove decisions for variables, increasing the probability of choosing
directions that led to significant parameters in previously accepted solutions.
"""

import copy
import logging
import math
import numpy as np
import time as _time

try:
    from .harmony import HarmonySearch
    from .sapbil import ProbabilityMatrix
except ImportError:
    from harmony import HarmonySearch
    from sapbil import ProbabilityMatrix

logger = logging.getLogger(__name__)


# pylint: disable=too-many-ancestors,attribute-defined-outside-init
class HSPBIL(HarmonySearch):
    """Harmony Search with PBIL-guided pitch adjustment.

    Overrides the standard uniform-random pitch perturbations with
    probability-weighted decisions that learn from accepted solutions.
    """

    def __init__(self, param, init_sol, ctrl, idnum=0, **kwargs):
        super().__init__(param, ctrl=ctrl, idnum=idnum)

        varnames = list(param.asvarnames or [])
        distributions = list(param.distr or ["n", "ln", "tn", "u", "t"])
        self.prob_matrix = ProbabilityMatrix(varnames, distributions)

        self._ps_asvars = set(getattr(param, "ps_asvars", []) or [])
        self._ps_randvars = set(
            (getattr(param, "ps_randvars", {}) or {}).keys()
        )
        self._ps_bcvars = set(getattr(param, "ps_bcvars", []) or [])
        self._ps_corvars = set(getattr(param, "ps_corvars", []) or [])

        self._pbil_updates = 0

    # ------------------------------------------------------------------
    # Output files  (override to use "hspbil_" prefix)
    # ------------------------------------------------------------------

    def create_output_files(self, param, **kwargs):
        run_id = kwargs.get('idnum', 0)
        ts = _time.strftime("%Y%m%d_%H%M%S")
        run_name = f"hspbil_{run_id}_{ts}" if run_id else f"hspbil_{ts}"

        try:
            self.results_file = open(f"{run_name}_results.txt", "w")
        except Exception:
            self.results_file = open(f"hspbil_results.txt", "w")

        try:
            self.progress_file = open(f"{run_name}_progress.txt", "w")
        except Exception:
            self.progress_file = open(f"hspbil_progress.txt", "w")

        print("SearchLibrium - HS+PBIL Run", file=self.results_file)
        print(f"Run ID: {run_id}", file=self.results_file)
        print(f"Start time: {_time.strftime('%Y-%m-%d %H:%M:%S')}", file=self.results_file)
        print("-" * 72, file=self.results_file)
        self.results_file.flush()

        print("iteration,score", file=self.progress_file)
        self.progress_file.flush()

    # ------------------------------------------------------------------
    # PBIL significance helpers (same as SAPBIL)
    # ------------------------------------------------------------------

    def _build_significance_map(self, sol):
        model = sol.get("model")
        if model is None:
            return {}
        try:
            pvalues = np.array(model.pvalues)
            coeff_names = list(model.coeff_names) if model.coeff_names is not None else []
            p_thresh = getattr(self.param, "p_val", 0.05)
            return {
                name: bool(pv <= p_thresh)
                for name, pv in zip(coeff_names, pvalues)
            }
        except Exception as exc:
            logger.debug("HS+PBIL: could not build significance map: %s", exc)
            return {}

    def _update_probability_matrix(self, sol):
        pm = self.prob_matrix

        sig = self._build_significance_map(sol)

        asvars = set(sol.get("asvars", []) or [])
        randvars = dict(sol.get("randvars", {}) or {})
        bcvars = set(sol.get("bcvars", []) or [])
        corvars = set(sol.get("corvars", []) or [])

        t = 1.0
        tI = 1.0

        for var in pm.varnames:
            if var in self._ps_asvars:
                continue

            in_model = var in asvars
            is_random = var in randvars
            is_bc = var in bcvars
            is_corr = var in corvars

            if in_model:
                mean_sig = sig.get(var, False)
                sd_sig = sig.get(f"sd.{var}", False)
                if mean_sig or (is_random and sd_sig):
                    incl_ind = 1.0
                elif (not mean_sig) and (not is_random or (not mean_sig and not sd_sig)):
                    incl_ind = 0.0
                else:
                    incl_ind = 0.5
            else:
                incl_ind = 0.0

            pm.update_inclusion(var, incl_ind, t, tI)

            if not in_model:
                continue

            if var not in self._ps_randvars:
                if is_random:
                    rand_ind = 1.0 if sig.get(f"sd.{var}", False) else 0.0
                else:
                    rand_ind = 0.0
                pm.update_random(var, rand_ind, t, tI)

            if not is_random:
                continue

            if var not in self._ps_randvars:
                current_distr = randvars[var]
                mean_sig_d = sig.get(var, False)
                sd_sig_d = sig.get(f"sd.{var}", False)
                distr_ind = 1.0 if (mean_sig_d and sd_sig_d) else 0.0
                pm.update_distribution(var, current_distr, distr_ind, t, tI)

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

            if not is_corr and var not in self._ps_bcvars:
                if is_bc:
                    bc_ind = 1.0 if sig.get(f"lambda.{var}", False) else 0.0
                else:
                    bc_ind = 0.0
                pm.update_boxcox(var, bc_ind, t, tI)

        self._pbil_updates += 1

    # ------------------------------------------------------------------
    # PBIL-guided inclusion perturbation
    # ------------------------------------------------------------------

    def _pbil_inclusion(self, sol):
        """Apply one PBIL-guided inclusion add/remove step."""
        pm = self.prob_matrix
        in_vars = [v for v in sol.get("asvars", []) if v in pm.p_inclusion]
        out_vars = [v for v in pm.varnames if v not in sol.get("asvars", [])]

        p_bar_in = float(np.mean([pm.p_inclusion[v] for v in in_vars])) if in_vars else 0.5
        p_bar_out = float(np.mean([pm.p_inclusion[v] for v in out_vars])) if out_vars else 0.5

        denom = p_bar_out + (1.0 - p_bar_in)
        p_add = p_bar_out / denom if denom > 1e-12 else 0.5

        if np.random.random() < p_add and out_vars:
            probs = np.array([pm.p_inclusion[v] for v in out_vars], dtype=float)
            probs = probs / (probs.sum() + 1e-12)
            var = np.random.choice(out_vars, p=probs)
            return self.add_asvar(var, sol)
        elif in_vars:
            non_ps = [v for v in in_vars if v not in self._ps_asvars]
            if not non_ps or len(in_vars) <= 1:
                return sol
            probs = np.array([1.0 - pm.p_inclusion[v] for v in non_ps], dtype=float)
            if probs.sum() < 1e-12:
                probs = np.ones(len(non_ps), dtype=float)
            probs = probs / probs.sum()
            var = np.random.choice(non_ps, p=probs)
            return self.remove_asvar(var, sol)
        return sol

    # ------------------------------------------------------------------
    # PBIL override of key perturbation methods from SearchBase
    # ------------------------------------------------------------------

    def perturb_asfeature(self, sol):
        if sol['asvars'] is None or len(sol['asvars']) == 0:
            return self.perturb_add_asfeature(sol)
        if np.random.random() < 0.5:
            return self._pbil_inclusion(sol)
        return super().perturb_asfeature(sol)

    def perturb_isfeature(self, sol):
        return super().perturb_isfeature(sol)

    # ------------------------------------------------------------------
    # Override improvise to track iteration counter for PBIL logging
    # ------------------------------------------------------------------

    def improvise(self):
        best, current = [], []
        for iter in range(self.maxiter):
            sine_iter = max(0, np.sign(math.sin(iter)))
            self.harm_rate = (self.min_harm + ((self.max_harm - self.min_harm) / self.maxiter) * iter) * sine_iter
            self.pitch = (self.min_pitch + ((self.max_pitch - self.min_pitch) / self.maxiter) * iter) * sine_iter

            new_sol = self.build_solution(self.memory, self.harm_rate)
            curr_sol, converged = self.pitch_adjustment(new_sol, self.pitch)
            if converged:
                self.insert_solution(curr_sol)
                try:
                    self._update_probability_matrix(curr_sol)
                except Exception:
                    pass

        all_val, obj_val = self.log_convergence(self.memory)
        if self.generate_plots:
            try:
                self.plot_results(self.memory, all_val, obj_val)
            except Exception as exc:
                logger.warning(f"Convergence plot failed (non-fatal, search result unaffected): {exc}")

    def run_search(self, existing_sols=None):
        import time as _time

        self.start = _time.time()

        from .search import get_unique as _get_unique
        try:
            from .search import get_unique
        except ImportError:
            from search import get_unique

        self.create_output_files(self.param, idnum=getattr(self, 'idnum', 0))

        existing_memory = self.screen_solutions(existing_sols)
        generated_memory = self.initialize_memory(self.max_mem)
        init_memory = generated_memory + existing_memory
        unique_memory = get_unique(init_memory, 0)
        for sol in unique_memory:
            sol.data['is_initial_sol'] = True

        memory_sorted = self.sort_memory(unique_memory)
        memory = memory_sorted[: self.max_mem]
        self.memory = memory.copy()

        if self.memory:
            self.best_sol = self.copy_solution(self.sort_memory(self.memory)[0])

        print(f"HS+PBIL[{getattr(self, 'idnum', 0)}] Memory initialised.  "
              f"Starting improvisation ({self.maxiter} iterations) ...")
        self.improvise()

        improved = self.sort_memory(self.memory.copy())
        if improved:
            self.best_sol = self.copy_solution(improved[0])

        print(f"HS+PBIL[{getattr(self, 'idnum', 0)}]. Search complete")
        logger.info("Search ended at: {}".format(str(_time.ctime())))

        try:
            pm_summary = self.prob_matrix.summary()
            print("\nHS+PBIL -- final probability matrix:", file=self.results_file)
            for row in pm_summary:
                print(f"  {row}", file=self.results_file)
        except Exception:
            pass

        return improved
