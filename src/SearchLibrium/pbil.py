"""
pbil.py

Shared probability-matrix engine for PBIL-guided search.

This module is intentionally dependency-free (no `self`, no reference to
any Search/SA/HS class) so it can be imported by any algorithm that wants
to opt into probability-guided perturbation via a `pbil_enabled` flag,
without creating a circular import between algorithm modules.

The matrix always exists and is always read from, regardless of whether
learning is enabled. Only the update step is conditional on the caller.
This keeps "guided" and "non-guided" search running through the exact
same neighbour-generation code path, differing only in whether the
probabilities are static or adaptive — which is what makes a frozen-matrix
run a valid, like-for-like control condition for an enabled-matrix run.
"""

import numpy as np
import re

# Global probability floor/ceiling. Keeps every probability strictly
# inside (0, 1) so no decision ever becomes fully deterministic — the
# search must always retain some chance of exploring the opposite choice.
P_MIN, P_MAX = 0.05, 0.95

# Per-decision learning-rate bounds, temperature-coupled (higher early,
# lower late — same idea as SA's cooling schedule). Bounds differ by
# decision type on purpose: inclusion/exclusion is the highest-leverage,
# most frequently revisited decision and should adapt fastest; finer
# decisions like which distribution to use should stay conservative so a
# single lucky/unlucky draw doesn't overwrite an otherwise stable prior.
_LR_BOUNDS = {
    "inclusion":    (0.02, 0.25),
    "random":       (0.02, 0.15),
    "distribution": (0.01, 0.05),
    "correlation":  (0.02, 0.15),
    "boxcox":       (0.02, 0.15),
}


def _clamp(p):
    return max(P_MIN, min(P_MAX, float(p)))


def learning_rate(decision_type, t_current, t_initial):
    """Higher learning rate early in the run, decaying toward lr_min as
    t_current approaches 0. Mirrors SA's temperature schedule so both
    mechanisms cool down together."""
    lr_min, lr_max = _LR_BOUNDS[decision_type]
    if t_initial <= 0:
        return lr_max
    progress = max(0.0, min(1.0, 1.0 - t_current / t_initial))
    return lr_min + (lr_max - lr_min) * progress


def _normalize_dist(dist_dict):
    """Keeps the per-variable distribution dict a valid probability
    distribution (sums to 1) at all times, not just at sampling time.
    This matters for diagnostics/logging: a printed matrix should always
    show real probabilities, not raw unnormalized weights."""
    total = sum(dist_dict.values())
    if total <= 0:
        n = len(dist_dict)
        return {d: 1.0 / n for d in dist_dict}
    return {d: _clamp(v / total) for d, v in dist_dict.items()}


def initialize_prob_matrix(asvarnames, all_bcvars, distributions):
    """
    Builds the starting matrix: every decision at a neutral, uninformative
    prior (0.5, or uniform across distributions). This neutral start is
    what makes a frozen (non-learning) run behave as an unbiased baseline
    rather than one that's secretly already tilted toward some outcome.

    boxcox is set to None — not 0.0 or 0.5 — for variables outside
    all_bcvars. Encoding ineligibility directly in the matrix means a
    variable that can never take a Box-Cox transform can never be
    accidentally sampled for one, without relying on a separate filter
    elsewhere to remember that exclusion.
    """
    n_distr = max(len(distributions), 1)
    matrix = {}
    for var in asvarnames:
        matrix[var] = {
            "inclusion":  0.5,
            "random":     0.5,
            "dist":       {d: 1.0 / n_distr for d in distributions},
            "correlated": 0.5,
            "boxcox":     0.5 if var in all_bcvars else None,
        }
    return matrix

def initialize_prob_matrix_lc(asvarnames, isvarnames, num_classes, all_bcvars, distributions):
    """
    Builds one probability matrix per latent class instead of a single
    flat matrix. Every class gets an entry per utility variable (asvar).
    Every class except the last (base/reference class, whose membership
    is implicit and never estimated) also gets an entry per membership
    variable (isvar), since C classes only have C-1 membership equations.

    Returns a list of length num_classes, each element a dict keyed by
    variable name. Membership entries only carry 'inclusion' — gamma
    coefficients are plain fixed effects, so distribution/random/
    correlation/box-cox dimensions don't apply to them.
    """
    n_distr = max(len(distributions), 1)
    matrices = []
    for c in range(num_classes):
        matrix = {}
        for var in asvarnames:
            matrix[var] = {
                "inclusion":  0.5,
                "random":     0.5,
                "dist":       {d: 1.0 / n_distr for d in distributions},
                "correlated": 0.5,
                "boxcox":     0.5 if var in all_bcvars else None,
            }
        if c != num_classes - 1:  # base class has no membership equation
            for var in isvarnames:
                matrix[var] = {"inclusion": 0.5}
        matrices.append(matrix)
    return matrices

def update_prob_matrix(matrix, sol, t_current, t_initial,
                        sig_vars, insig_vars,
                        ps_asvars, ps_randvars, ps_bcvars, ps_corvars):
    """
    Updates the matrix from one accepted solution.

    Two rules that shape every branch below:

    1. A variable not present in `sol` is left completely untouched.
       Absence from one particular accepted solution is not evidence that
       the variable is bad — it may simply not have been tried yet. Only
       variables that were actually evaluated (present in the solution,
       with a computed significance) move their probability up or down.
       An "ambiguous" significance result (present, but neither clearly
       significant nor clearly not) also leaves the probability untouched
       rather than pulling it toward 0.5 — no new evidence means no
       update, not a push toward uncertainty.

    2. Pre-specified (analyst-forced) variables are skipped entirely.
       Their status is fixed by the analyst, not discovered by the
       search, so their probabilities are tracked (for consistent
       logging) but never learned.
    """
    asvars   = set(sol.get("asvars", []) or [])
    randvars = dict(sol.get("randvars", {}) or {})
    bcvars   = set(sol.get("bcvars", []) or [])
    corvars  = set(sol.get("corvars", []) or [])

    for var, p in matrix.items():
        if var in ps_asvars:
            continue

        in_model  = var in asvars
        if not in_model:
            continue

        is_random = var in randvars
        is_bc     = var in bcvars
        is_corr   = var in corvars

        # -- 1. Inclusion --
        sig_mean, sig_sd     = var in sig_vars["mean"],   var in sig_vars["sd"]
        insig_mean, insig_sd = var in insig_vars["mean"], var in insig_vars["sd"]
        if sig_mean or (is_random and sig_sd):
            lr = learning_rate("inclusion", t_current, t_initial)
            p["inclusion"] = _clamp((1 - lr) * p["inclusion"] + lr * 1.0)
        elif insig_mean and (not is_random or insig_sd):
            lr = learning_rate("inclusion", t_current, t_initial)
            p["inclusion"] = _clamp((1 - lr) * p["inclusion"] + lr * 0.0)
        # else: ambiguous, left untouched

        # -- 2. Randomness --
        if var not in ps_randvars:
            indicator = 1.0 if (is_random and var in sig_vars["sd"]) else 0.0
            lr = learning_rate("random", t_current, t_initial)
            p["random"] = _clamp((1 - lr) * p["random"] + lr * indicator)

        if not is_random:
            continue

        # -- 3. Distribution --
        if var not in ps_randvars:
            current_distr = randvars[var]
            indicator = 1.0 if (var in sig_vars["mean"] and var in sig_vars["sd"]) else 0.0
            lr = learning_rate("distribution", t_current, t_initial)
            p["dist"][current_distr] = _clamp(
                (1 - lr) * p["dist"][current_distr] + lr * indicator
            )
            p["dist"] = _normalize_dist(p["dist"])

        # -- 4. Correlation --
        if not is_bc and var not in ps_corvars:
            indicator = 1.0 if any(
                f"chol.{var}.{v2}" in sig_vars["chol"] or f"chol.{v2}.{var}" in sig_vars["chol"]
                for v2 in corvars if v2 != var
            ) else 0.0
            lr = learning_rate("correlation", t_current, t_initial)
            p["correlated"] = _clamp((1 - lr) * p["correlated"] + lr * indicator)

        # -- 5. Box-Cox --
        if p["boxcox"] is not None and not is_corr and var not in ps_bcvars:
            indicator = 1.0 if var in sig_vars["lambda"] else 0.0
            lr = learning_rate("boxcox", t_current, t_initial)
            p["boxcox"] = _clamp((1 - lr) * p["boxcox"] + lr * indicator)
    print(summarize_prob_matrix_table( matrix))
    
    return matrix

def update_prob_matrix_lc(matrices, sol, t_current, t_initial, ps_asvars, ps_isvars):
    """Per-class analogue of update_prob_matrix. Each class's matrix
    learns independently: a variable can become more likely in class 1
    while becoming less likely in class 2, based on that class's own
    p-value for that variable.

    Utility entries carry a 'dist' sub-key (from initialize_prob_matrix_lc);
    membership entries don't — used here to tell the two apart without
    needing a separate isvarnames lookup per entry.
    """
    model = sol.get("model")
    if model is None:
        return matrices

    class_params_spec = sol.get('class_params_spec', None)
    member_params_spec = sol.get('member_params_spec', None)
    class_sig, member_sig = classify_significance_lc(sol)
    lr = learning_rate("inclusion", t_current, t_initial)
    num_classes = len(matrices)

    for c in range(num_classes):
        matrix = matrices[c]
        in_class_vars = set(class_params_spec[c]) if class_params_spec is not None else set()
        in_member_vars = (
            set(member_params_spec[c])
            if (member_params_spec is not None and c < len(member_params_spec))
            else set()
        )
        sig_c = class_sig.get(c, {})
        sig_m = member_sig.get(c, {})

        for var, p in matrix.items():
            is_membership_entry = "dist" not in p
            if is_membership_entry:
                if var in ps_isvars or var not in in_member_vars or var not in sig_m:
                    continue  # not present this round / no p-value → no update
                indicator = 1.0 if sig_m[var] else 0.0
                p["inclusion"] = _clamp((1 - lr) * p["inclusion"] + lr * indicator)
            else:
                if var in ps_asvars or var not in in_class_vars or var not in sig_c:
                    continue
                indicator = 1.0 if sig_c[var] else 0.0
                p["inclusion"] = _clamp((1 - lr) * p["inclusion"] + lr * indicator)
    print(summarize_prob_matrix_table_lc( matrices))
    return matrices

    
def summarize_prob_matrix_table(prob_matrix: dict, top_n: int = 15, all_randvars: list = None, all_corvars: list = None) -> str:
    """
    Return a formatted summary of the probability matrix,
    sorted by inclusion probability descending.
    """
    rows = sorted(
        prob_matrix.items(),
        key=lambda kv: kv[1]["inclusion"],
        reverse=True
    )[:top_n]

    header = (
        f"\n{'Variable':<14} {'Incl':>6} {'Rand':>6} "
        f"{'n':>5} {'ln':>5} {'tn':>5} {'u':>5} {'t':>5} "
        f"{'Corr':>6} {'BC':>6}"
    )
    sep   = "-" * 74
    lines = [header, sep]

    for var, p in rows:
        d  = p["dist"]
        bc = f"{p['boxcox']:.3f}" if p["boxcox"] is not None else "  N/A"
        if all_randvars is not None and var not in all_randvars:
            lines.append(
                f"{var:<14} {p['inclusion']:>6.3f} {'N/A':>6} "
                f"{'N/A':>5} {'N/A':>5} {'N/A':>5} {'N/A':>5} {'N/A':>5} "
                f"{'N/A':>6} {bc:>6}"
            )
        else:
            corr_val = f"{p['correlated']:>6.3f}" if all_corvars is None or var in all_corvars else "   N/A"
            lines.append(
                f"{var:<14} {p['inclusion']:>6.3f} {p['random']:>6.3f} "
                f"{d.get('n',0):>5.3f} {d.get('ln',0):>5.3f} "
                f"{d.get('tn',0):>5.3f} {d.get('u',0):>5.3f} "
                f"{d.get('t',0):>5.3f} "
                f"{corr_val} {bc:>6}"
            )

    return "\n".join(lines)


def build_significance_map(sol, p_val_threshold=0.05):
    """Maps every fitted coefficient name to whether its p-value clears
    the significance threshold. Returns {} if the solution has no fitted
    model yet (e.g. it failed to converge), so callers can treat a missing
    model the same as "nothing significant" without a separate check."""
    model = sol.get("model")
    if model is None:
        return {}
    try:
        pvalues = np.array(model.pvalues)
        coeff_names = list(model.coeff_names) if model.coeff_names is not None else []
        return {name: bool(pv <= p_val_threshold) for name, pv in zip(coeff_names, pvalues)}
    except Exception:
        return {}


def classify_significance(sol, sig):
    """
    Turns the raw {coeff_name: bool} map into the sig_vars/insig_vars sets
    that update_prob_matrix expects. Kept separate from update_prob_matrix
    itself so the significance classification (what counts as "in the
    model", "significant", etc.) can be tested and adjusted independently
    of the probability-update arithmetic.
    """
    asvars = sol.get("asvars", []) or []
    sig_vars = {
        "mean":   {v for v in asvars if sig.get(v, False)},
        "sd":     {v for v in asvars if sig.get(f"sd.{v}", False)},
        "chol":   {k for k, ok in sig.items() if k.startswith("chol.") and ok},
        "lambda": {v for v in asvars if sig.get(f"lambda.{v}", False)},
    }
    insig_vars = {
        "mean":   {v for v in asvars if not sig.get(v, False)},
        "sd":     {v for v in asvars if not sig.get(f"sd.{v}", False)},
        "chol":   {k for k, ok in sig.items() if k.startswith("chol.") and not ok},
        "lambda": {v for v in asvars if not sig.get(f"lambda.{v}", False)},
    }
    return sig_vars, insig_vars



def classify_significance_lc(sol, p_val_threshold=0.05):
    """Per-class analogue of classify_significance. Parses the fitted
    model's coeff_names ("class_{c}_{var}") and gamma_names
    ("gamma_class_{c}_{var}") back into per-class significance maps.

    Returns (class_sig, member_sig), each a dict keyed by class index
    (0-based) -> {var: bool_significant}.
    """
    model = sol.get("model")
    if model is None:
        return {}, {}

    class_sig = {}
    coeff_names = list(getattr(model, "coeff_names", None) or [])
    pvalues = np.array(getattr(model, "pvalues", None)) if getattr(model, "pvalues", None) is not None else np.array([])
    for name, pv in zip(coeff_names, pvalues):
        m = re.match(r'^class_(\d+)_(.+)$', str(name))
        if not m:
            continue
        c = int(m.group(1)) - 1
        var = m.group(2)
        class_sig.setdefault(c, {})[var] = bool(pv <= p_val_threshold)

    member_sig = {}
    gamma_names = list(getattr(model, "gamma_names", None) or [])
    gamma_pvalues = getattr(model, "gamma_p_values", None)
    gamma_pvalues = np.array(gamma_pvalues) if gamma_pvalues is not None else np.array([])
    for name, pv in zip(gamma_names, gamma_pvalues):
        m = re.match(r'^gamma_class_(\d+)_(.+)$', str(name))
        if not m:
            continue
        c = int(m.group(1)) - 1
        var = m.group(2)
        member_sig.setdefault(c, {})[var] = bool(pv <= p_val_threshold)

    return class_sig, member_sig


def sample_variable(var, matrix, all_bcvars, ps_asvars, ps_randvars,
                     ps_bcvars, ps_corvars, rng=np.random):
    """
    Draws one variable's full spec (included? random? which distribution?
    correlated? box-cox?) directly from the matrix's learned/frozen
    probabilities. Pre-specified variables bypass the coin flip entirely
    and take the analyst's forced value, matching the same override rule
    used everywhere else in the search.

    Correlation and box-cox are kept mutually exclusive (a variable can't
    be both in this design), so if both draws succeed, box-cox is the one
    dropped.
    """
    p = matrix[var]
    included = True if var in ps_asvars else (rng.rand() < p["inclusion"])
    if not included:
        return None

    is_random = True if var in ps_randvars else (rng.rand() < p["random"])
    dist = None
    if is_random:
        if var in ps_randvars and ps_randvars[var] != "any":
            dist = ps_randvars[var]
        else:
            dists = list(p["dist"].keys())
            probs = np.array([p["dist"][d] for d in dists])
            probs = probs / probs.sum()
            dist = rng.choice(dists, p=probs)

    is_corr = False
    if is_random:
        is_corr = True if var in ps_corvars else (rng.rand() < p["correlated"])

    has_bc = False
    if p["boxcox"] is not None and var in all_bcvars and not is_corr:
        has_bc = True if var in ps_bcvars else (rng.rand() < p["boxcox"])
    if is_corr and has_bc:
        has_bc = False

    return {"var": var, "is_random": is_random, "dist": dist,
            "is_corr": is_corr, "has_bc": has_bc}



def sample_solution_from_matrix(matrix, all_bcvars, ps_asvars, ps_randvars,
                                 ps_bcvars, ps_corvars, rng=np.random):
    """
    Generates a full candidate specification from the matrix in one pass —
    every variable sampled independently. Used wherever a search algorithm
    needs a brand-new solution informed by what's been learned so far,
    rather than a purely random one.
    """
    specs = [
        sample_variable(v, matrix, all_bcvars, ps_asvars, ps_randvars,
                         ps_bcvars, ps_corvars, rng)
        for v in matrix
    ]
    return [s for s in specs if s is not None]


def summarize_prob_matrix(matrix):
    """Flattens the matrix into one row per variable, for logging/reporting."""
    rows = []
    for var, p in matrix.items():
        rows.append({
            "var":     var,
            "p_incl":  round(p["inclusion"], 3),
            "p_rand":  round(p["random"], 3),
            "p_corr":  round(p["correlated"], 3),
            "p_bc":    round(p["boxcox"], 3) if p["boxcox"] is not None else None,
            "p_distr": {d: round(v, 3) for d, v in p["dist"].items()},
        })
    return rows

def summarize_prob_matrix_table_lc(matrices, top_n=15, all_randvars=None, all_corvars=None):
    """Same format as summarize_prob_matrix_table, but one table per class.
    Utility vs membership entries are told apart by the 'dist' key —
    utility entries carry it (from initialize_prob_matrix_lc), membership
    entries don't — so no isvarnames lookup is needed here."""
    num_classes = len(matrices)
    sections = []

    header = (
        f"\n{'Variable':<14} {'Incl':>6} {'Rand':>6} "
        f"{'n':>5} {'ln':>5} {'tn':>5} {'u':>5} {'t':>5} "
        f"{'Corr':>6} {'BC':>6}"
    )
    sep = "-" * 74

    for c, matrix in enumerate(matrices):
        is_base = (c == num_classes - 1)
        label = f"CLASS {c + 1}" + ("  (Base — no membership equation)" if is_base else "")
        lines = [f"\n=== {label} — Utility Variables ===", header, sep]

        util_rows = sorted(
            ((v, p) for v, p in matrix.items() if "dist" in p),
            key=lambda kv: kv[1]["inclusion"], reverse=True
        )[:top_n]
        for var, p in util_rows:
            d = p["dist"]
            bc = f"{p['boxcox']:.3f}" if p["boxcox"] is not None else "  N/A"
            if all_randvars is not None and var not in all_randvars:
                lines.append(
                    f"{var:<14} {p['inclusion']:>6.3f} {'N/A':>6} "
                    f"{'N/A':>5} {'N/A':>5} {'N/A':>5} {'N/A':>5} {'N/A':>5} "
                    f"{'N/A':>6} {bc:>6}"
                )
            else:
                corr_val = f"{p['correlated']:>6.3f}" if all_corvars is None or var in all_corvars else "   N/A"
                lines.append(
                    f"{var:<14} {p['inclusion']:>6.3f} {p['random']:>6.3f} "
                    f"{d.get('n',0):>5.3f} {d.get('ln',0):>5.3f} "
                    f"{d.get('tn',0):>5.3f} {d.get('u',0):>5.3f} "
                    f"{d.get('t',0):>5.3f} "
                    f"{corr_val} {bc:>6}"
                )

        member_rows = sorted(
            ((v, p) for v, p in matrix.items() if "dist" not in p),
            key=lambda kv: kv[1]["inclusion"], reverse=True
        )
        if member_rows:
            lines.append(f"\n--- {label} — Membership Variables ---")
            lines.append(f"{'Variable':<14} {'Incl':>6}")
            lines.append("-" * 22)
            for var, p in member_rows:
                lines.append(f"{var:<14} {p['inclusion']:>6.3f}")

        sections.append("\n".join(lines))

    return "\n".join(sections)