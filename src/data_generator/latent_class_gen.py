import numpy as np
import pandas as pd
from scipy.special import softmax


class AdvancedLatentClassGenerator:
    def __init__(
        self,
        n_classes=3,
        n_alternatives=3,
        n_individuals=2000,
        n_choice_tasks=5,
        class_probs=None,
        scale_separation=2.5,
        n_noise_vars=5,
        n_weak_vars=3,
        n_collinear_vars=3,
        rare_class=False,
        random_state=42,
        n_membership_vars=3,
        membership_scale=2.0,
    ):
        self.rng = np.random.default_rng(random_state)

        self.K = n_classes
        self.J = n_alternatives
        self.N = n_individuals
        self.T = n_choice_tasks
        self.random_state = random_state

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

        self.scale_separation = scale_separation
        self.n_noise_vars = n_noise_vars
        self.n_weak_vars = n_weak_vars
        self.n_collinear_vars = n_collinear_vars
        self.n_membership_vars = n_membership_vars
        self.membership_scale = membership_scale

        self._define_variables()
        self._generate_parameters()
        self._generate_membership_coefficients()

    # --------------------------------
    # Variable Names
    # --------------------------------
    def _define_variables(self):

        self.shared_vars = [
            "price",
            "travel_time",
            "waiting_time"
        ]

        self.class_specific_vars = {
            0: ["comfort_level", "seat_space"],
            1: ["brand_reputation", "loyalty_points"],
            2: ["eco_rating", "carbon_emissions"]
        }

        self.noise_vars = [f"random_noise_{i}" for i in range(self.n_noise_vars)]
        self.weak_vars = [f"weak_signal_{i}" for i in range(self.n_weak_vars)]
        self.collinear_vars = [f"collinear_var_{i}" for i in range(self.n_collinear_vars)]
        self.membership_var_names = [f"mem_var_{i}" for i in range(self.n_membership_vars)]

    # --------------------------------
    # Utility Parameters (betas)
    # --------------------------------
    def _generate_parameters(self):

        self.parameters = {}

        for k in range(self.K):
            params = {}

            for var in self.shared_vars:
                params[var] = self.rng.normal(
                    loc=k * self.scale_separation,
                    scale=1.0
                )

            for var in self.class_specific_vars.get(k, []):
                params[var] = self.rng.normal(
                    loc=(k + 1) * self.scale_separation,
                    scale=1.0
                )

            for var in self.weak_vars:
                params[var] = self.rng.normal(
                    loc=0,
                    scale=0.05
                )

            for var in self.noise_vars:
                params[var] = 0.0

            self.parameters[k] = params

    # --------------------------------
    # Membership Coefficients (gammas)
    # --------------------------------
    def _generate_membership_coefficients(self):
        if self.n_membership_vars == 0:
            self.gammas = None
            return

        self.gammas = np.zeros((self.K - 1, self.n_membership_vars))

        for c in range(self.K - 1):
            for m in range(self.n_membership_vars):
                self.gammas[c, m] = self.rng.normal(
                    loc=(c - (self.K - 2) / 2.0) * self.membership_scale,
                    scale=1.0
                )

    # --------------------------------
    # Feature Generation
    # --------------------------------
    def _generate_features(self):

        all_vars = set(self.shared_vars)
        for v in self.class_specific_vars.values():
            all_vars.update(v)

        all_vars.update(self.noise_vars)
        all_vars.update(self.weak_vars)
        all_vars.update(self.collinear_vars)

        X = {}

        for var in all_vars:
            X[var] = self.rng.normal(size=(self.N, self.T, self.J))

        for i, var in enumerate(self.collinear_vars):
            X[var] = 0.8 * X["price"] + 0.2 * self.rng.normal(
                size=(self.N, self.T, self.J)
            )

        return X

    def _generate_membership_features(self):
        if self.n_membership_vars == 0:
            return None

        Z = {}
        for i, name in enumerate(self.membership_var_names):
            Z[name] = self.rng.normal(size=self.N)

        return Z

    # --------------------------------
    # Class Assignment
    # --------------------------------
    def _assign_classes(self):
        if self.gammas is None:
            return self.rng.choice(
                self.K,
                size=self.N,
                p=self.class_probs
            )

        Z = self._Z_matrix
        logits = np.zeros((self.N, self.K))

        for c in range(self.K - 1):
            logits[:, c] = Z @ self.gammas[c]

        probs = softmax(logits, axis=1)

        classes = np.zeros(self.N, dtype=int)
        for n in range(self.N):
            classes[n] = self.rng.choice(self.K, p=probs[n])

        return classes

    # --------------------------------
    # Data Generation
    # --------------------------------
    def generate(self):

        Z = self._generate_membership_features()
        if Z is not None:
            Z_names = list(Z.keys())
            self._Z_matrix = np.column_stack([Z[name] for name in Z_names])
        else:
            self._Z_matrix = None

        classes = self._assign_classes()

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
                        "true_class": k
                    }

                    for var in X:
                        row[var] = X[var][n, t, j]

                    if Z is not None:
                        for name in Z_names:
                            row[name] = Z[name][n]

                    rows.append(row)

        df = pd.DataFrame(rows)
        df["choice_id"] = df["individual"] * self.T + df["choice_task"]
        return df, classes
