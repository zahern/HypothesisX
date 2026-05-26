"""Bandit-guided Simulated Annealing for SearchLibrium.

This module provides:
- PerturbationBandit: Thompson-sampling policy over perturbation action types
- BanditSA: SA variant that uses the bandit to pick perturbation actions
"""

import logging

import numpy as np

try:
    from siman import SA
except ImportError:
    from .siman import SA


class PerturbationBandit:
    """Thompson Sampling over perturbation action-type arms."""

    def __init__(self, arm_names, prior_alpha=1.0, prior_beta=1.0, epsilon=0.05, rng=None):
        self.arm_names = list(arm_names)
        self.n_arms = len(self.arm_names)
        self.alpha = np.full(self.n_arms, float(prior_alpha), dtype=float)
        self.beta = np.full(self.n_arms, float(prior_beta), dtype=float)
        self.counts = np.zeros(self.n_arms, dtype=int)
        self.epsilon = float(epsilon)
        self.rng = rng if rng is not None else np.random.default_rng()

    def select_arm(self, available_indices):
        if not available_indices:
            raise ValueError("No available bandit arms to select from.")

        if self.rng.random() < self.epsilon:
            return int(self.rng.choice(available_indices))

        samples = {
            idx: self.rng.beta(self.alpha[idx], self.beta[idx])
            for idx in available_indices
        }
        return max(samples, key=samples.get)

    def update(self, arm_index, reward):
        reward = float(reward)
        self.counts[arm_index] += 1

        if reward >= 0.0:
            self.alpha[arm_index] += 1.0 + reward
        else:
            self.beta[arm_index] += 1.0 + abs(reward)

    def summary(self):
        rows = []
        total = np.maximum(self.alpha + self.beta, 1e-12)
        means = self.alpha / total
        for idx, name in enumerate(self.arm_names):
            rows.append(
                {
                    "arm": name,
                    "alpha": float(self.alpha[idx]),
                    "beta": float(self.beta[idx]),
                    "mean": float(means[idx]),
                    "count": int(self.counts[idx]),
                }
            )
        rows.sort(key=lambda row: row["mean"], reverse=True)
        return rows


class BanditSA(SA):
    """SA variant with Thompson-sampling perturbation selection."""

    def __init__(
        self,
        param,
        init_sol,
        ctrl,
        idnum=0,
        bandit_prior_alpha=1.0,
        bandit_prior_beta=1.0,
        bandit_epsilon=0.05,
        **kwargs,
    ):
        super().__init__(param, init_sol, ctrl, idnum=idnum, **kwargs)

        self._actions = self._build_action_table()
        self.bandit = PerturbationBandit(
            arm_names=[name for name, _ in self._actions],
            prior_alpha=bandit_prior_alpha,
            prior_beta=bandit_prior_beta,
            epsilon=bandit_epsilon,
        )
        self.bandit_history = []

    def _build_action_table(self):
        return [
            ("add_asfeature", self.perturb_add_asfeature),
            ("remove_asfeature", self.perturb_remove_asfeature),
            ("add_isfeature", self.perturb_add_isfeature),
            ("remove_isfeature", self.perturb_remove_isfeature),
            ("add_randfeature", self.perturb_add_randfeature),
            ("remove_randfeature", self.perturb_remove_randfeature),
            ("add_bcfeature", self.perturb_add_bcfeature),
            ("remove_bcfeature", self.perturb_remove_bcfeature),
            ("add_corfeature", self.perturb_add_corfeature),
            ("remove_corfeature", self.perturb_remove_corfeature),
            ("change_model", self.perturb_model_t),
            ("change_distribution", self.perturb_distribution),
        ]

    def _available_arm_indices(self, sol):
        available = []

        if self.param.asvarnames and len(sol.get("asvars", [])) < len(self.param.asvarnames):
            available.append(0)  # add_asfeature
        if len(sol.get("asvars", [])) > 1:
            available.append(1)  # remove_asfeature

        if self.param.isvarnames:
            is_candidates = [v for v in self.param.isvarnames if v not in sol.get("isvars", [])]
            if is_candidates:
                available.append(2)  # add_isfeature
        if sol.get("isvars"):
            available.append(3)  # remove_isfeature

        if self.param.allow_random:
            add_r_candidates = [
                v for v in sol.get("asvars", []) if v not in sol.get("randvars", {})
            ]
            if add_r_candidates:
                available.append(4)  # add_randfeature

            rem_r_candidates = [
                v for v in sol.get("randvars", {}) if v not in self.param.ps_randvars
            ]
            if rem_r_candidates:
                available.append(5)  # remove_randfeature

            dist_candidates = [
                v for v in sol.get("randvars", {}) if v not in self.param.ps_randvars
            ]
            if dist_candidates and len(getattr(self.param, "distr", []) or []) > 1:
                available.append(11)  # change_distribution

        if self.param.allow_bcvars:
            add_bc_candidates = [
                v
                for v in sol.get("asvars", [])
                if v not in sol.get("bcvars", []) and v not in self.param.ps_corvars
            ]
            if add_bc_candidates:
                available.append(6)  # add_bcfeature

            rem_bc_candidates = [
                v for v in sol.get("bcvars", []) if v not in self.param.ps_bcvars
            ]
            if rem_bc_candidates:
                available.append(7)  # remove_bcfeature

        if self.param.allow_corvars and self.param.allow_random:
            cor_eligible = [
                v for v in sol.get("randvars", {}) if v not in sol.get("bcvars", [])
            ]
            if len(cor_eligible) >= 2:
                available.append(8)  # add_corfeature

            rem_cor_candidates = [
                v for v in sol.get("corvars", []) if v not in self.param.ps_corvars
            ]
            if rem_cor_candidates:
                available.append(9)  # remove_corfeature

        if self.param.avail_models is not None and len(self.param.avail_models) > 1:
            available.append(10)  # change_model

        return sorted(set(available))

    def _compute_bandit_reward(self, old_obj, new_obj, accepted, converged):
        if not converged:
            return -1.0

        sign = self.param.sign_crit(0)
        # Positive value means objective improved under current criterion direction.
        delta = sign * (new_obj - old_obj)
        scale = max(abs(old_obj), 1.0)
        normalized = delta / scale

        if accepted:
            if normalized > 0:
                return min(1.0, 0.1 + 10.0 * normalized)
            return max(-1.0, 10.0 * normalized)

        return -0.2

    def _apply_selected_action(self, sol):
        baseline_sig = self.setup_signature(sol)
        max_attempts = 12
        last_arm_idx = None

        for _ in range(max_attempts):
            candidate = self.copy_solution(sol)
            available = self._available_arm_indices(candidate)
            if not available:
                return sol, None, False

            arm_idx = self.bandit.select_arm(available)
            last_arm_idx = arm_idx
            _, action = self._actions[arm_idx]
            result = action(candidate)
            if result is not None:
                candidate = result

            candidate = self.apply_constraints(candidate)
            candidate = self.repair_solution_for_clarity(candidate)
            if self.setup_signature(candidate) != baseline_sig:
                return candidate, arm_idx, True

        return sol, last_arm_idx, False

    def perturb_solution(self, sol):
        curr_score = [sol.obj(i) for i in range(self.nb_crit)]

        new_sol, arm_idx, changed = self._apply_selected_action(sol)
        if not changed:
            if arm_idx is not None:
                self.bandit.update(arm_idx, -0.5)
                self.bandit_history.append(
                    {
                        "step": int(self.step),
                        "arm": self._actions[arm_idx][0],
                        "accepted": False,
                        "converged": False,
                        "reward": -0.5,
                        "reason": "no_change",
                    }
                )
            return self.current_sol

        new_sol, converged = self.evaluate(new_sol)
        if not converged:
            self.not_converged += 1
            reward = self._compute_bandit_reward(curr_score[0], curr_score[0], False, False)
            self.bandit.update(arm_idx, reward)
            self.bandit_history.append(
                {
                    "step": int(self.step),
                    "arm": self._actions[arm_idx][0],
                    "accepted": False,
                    "converged": False,
                    "reward": float(reward),
                    "reason": "non_converged",
                }
            )
            return self.current_sol

        new_score = [new_sol.obj(i) for i in range(self.nb_crit)]
        accepted = bool(self.accept_change(curr_score, new_score))

        if accepted:
            self.no_impr = 0
            self.accepted += 1
            self.current_sol = new_sol
            self.update_best(new_sol)
        else:
            self.not_accepted += 1

        reward = self._compute_bandit_reward(curr_score[0], new_score[0], accepted, converged)
        self.bandit.update(arm_idx, reward)
        self.bandit_history.append(
            {
                "step": int(self.step),
                "arm": self._actions[arm_idx][0],
                "accepted": bool(accepted),
                "converged": bool(converged),
                "reward": float(reward),
                "old_obj": float(curr_score[0]),
                "new_obj": float(new_score[0]),
            }
        )

        self.log_kpi(new_sol, self.debug_file, accepted)
        return self.current_sol

    def finalise(self):
        super().finalise()

        lines = ["Bandit arm summary (sorted by posterior mean):"]
        for row in self.bandit.summary():
            lines.append(
                "  {arm:<20s} mean={mean:.4f} alpha={alpha:.2f} beta={beta:.2f} count={count}".format(
                    **row
                )
            )

        for line in lines:
            print(line)
            try:
                print(line, file=self.results_file)
            except Exception:
                logging.debug("Unable to write bandit summary to results_file")
