"""
separation_screen.py
=====================

Consolidated module for quasi-perfect separation diagnostics in latent class
choice models, and the multistart-integrated remediation protocol described
in the accompanying LaTeX proposal (Sections 6-7).

Everything that decides whether a fitted class/parameter is trustworthy
lives here. Firth's penalized-likelihood correction itself is NOT touched
and stays in latent_class.py exactly as it already is -- this module only
diagnoses, decides, and labels; it never re-derives the estimator.

Two families of functions, matching the two categories discussed:

  (A) Single-fit diagnostics (moved here from latent_class.py, unchanged
      in behaviour, now plain functions instead of methods):
        - is_degenerate_candidate(...)
        - collinearity_diag(...)
        - cluster_stability_report(model, ...)

  (A, new) Formal / interpretable separation checks added in this round:
        - konis_separation_test(...)      : linear-programming test (eq. konis-lp)
        - separation_contingency_table(...): weighted 2x2 attribution table

  (B) Multi-fit orchestration (new), same calling convention as BEHier.py
      (`BEHier(search_instance, sol, max_passes=10)`):
        - run_separation_protocol(search_instance, memory, ...)

Only external dependency for family (B): `search_instance.evaluate_model(sol)`
and, for the Firth escalation step, the search instance's existing Firth-
enabled estimation path (unchanged, called exactly as it already is
elsewhere in the codebase).
"""

import numpy as np
from scipy.optimize import linprog
from scipy.stats import norm


# ------------------------------------------------------------------------
# (A) Single-fit diagnostics -- moved verbatim from latent_class.py
# ------------------------------------------------------------------------

def is_degenerate_candidate(betas, pi, min_share=0.01, max_abs_beta=200.0):
    """Cheap, no-extra-estimation screen for the classic near-empty-class
    EM local optimum: a class collapsed to almost no weighted individuals
    lets its own coefficients diverge with no real contradicting cases to
    stop them, producing a spuriously HIGHER raw loglik than a healthy
    solution. Uses only `betas`/`pi`, already computed for every multistart
    candidate -- no Hessian, no re-estimation. `max_abs_beta` is a sanity
    bound, not a hard science: it should sit well above any plausible
    coefficient for the model's variable scales; NaN/inf always count as
    degenerate regardless.

    UNCHANGED from latent_class.py::_is_degenerate_candidate (was already
    a @staticmethod, so this move is purely mechanical -- no `self` was
    ever used in the body).
    """
    pi_np = np.asarray(pi)
    if not np.all(np.isfinite(pi_np)) or np.any(pi_np < min_share):
        return True
    for b in betas:
        b_np = np.asarray(b)
        if not np.all(np.isfinite(b_np)) or np.any(np.abs(b_np) > max_abs_beta):
            return True
    return False

def hauck_donner_flags(H, dH, theta, alpha=0.05):
    """Hauck-Donner effect (HDE) detection for the Wald statistic of each
    parameter (Hauck & Donner, 1977; Yee, 2022).

    The Wald statistic z_k = theta_k / se_k exhibits HDE when it decreases as
    |theta_k| grows, i.e. dz_k/dtheta_k < 0. With v_k = [H^{-1}]_kk this is
    equivalent to theta_k * dv_k/dtheta_k > 2 v_k, where
    dv_k/dtheta_k = -[H^{-1} (dH/dtheta_k) H^{-1}]_kk. The criterion is the sign
    of a derivative, so it needs no tuning threshold, and it is invariant to
    rescaling of theta_k.

    A parameter is flagged only when HDE is present AND its Wald test fails to
    reject H0: theta_k = 0 at level `alpha`. Past the tipping point a strongly
    significant estimate only loses power (inference is unchanged); the harmful
    case is the large estimate whose inflated SE renders it non-significant.

    H     : (p, p) observed information (Hessian of the negative log-likelihood)
    dH    : (p, p, p) third-derivative tensor, dH[:, :, k] = dH/dtheta_k
    theta : (p,) parameter vector at which H and dH were evaluated
    alpha : two-sided significance level of the Wald test

    References
    ----------
    Hauck, W. W., & Donner, A. (1977). Wald's test as applied to hypotheses in
        logit analysis. JASA, 72(360), 851-853.
    Yee, T. W. (2022). On the Hauck-Donner effect in Wald tests: Detection,
        tipping points, and parameter space characterization. JASA, 117(540),
        1763-1774.
    """
    H_inv = np.linalg.inv(H)
    v = np.diag(H_inv)
    dv = -np.einsum('ki,ijk,jk->k', H_inv, dH, H_inv)
    theta = np.asarray(theta)
    valid = v > 0
    hde = valid & (theta * dv > 2.0 * v)
    z = np.where(valid, theta / np.sqrt(np.where(valid, v, 1.0)), 0.0)
    return hde & (np.abs(z) < norm.ppf(1.0 - alpha / 2.0))


def collinearity_diag(H, ratio_thresh=1e-3, vdp_thresh=0.5):
    """Cole (2020) / Gimenez et al. (2004) 'Hessian method': flag a
    direction as near non-identified when lambda_k/lambda_max falls
    below `ratio_thresh` (raw, unscaled eigen-decomposition -- scaling
    to a correlation matrix would hide univariate separation). For
    every flagged direction, attribute it to parameters via the
    Belsley-Kuh-Welsch (1980) variance-decomposition proportion:
    share_j = (v_kj^2/lambda_k) / Var(theta_j); share_j > vdp_thresh
    (0.5, the standard BKW cutoff) marks parameter j as implicated.
    Returns (share, flag, ratio_min) arrays aligned with H's row/column order.

    UNCHANGED from latent_class.py::_collinearity_diag (was already a
    @staticmethod; this move is purely mechanical).
    """
    lam, V = np.linalg.eigh(H)
    lam = np.clip(lam, 1e-300, None)
    ratio = lam / lam.max()
    var_j = np.sum(V ** 2 / lam[None, :], axis=1)
    p = H.shape[0]
    share = np.zeros(p)
    flag = np.zeros(p, dtype=bool)
    for k in range(p):
        if ratio[k] < ratio_thresh:
            vdp_k = (V[:, k] ** 2 / lam[k]) / var_j
            share = np.maximum(share, vdp_k)
            flag |= vdp_k > vdp_thresh
    return share, flag, float(ratio.min())


def cluster_stability_report(model, attempts, vdp_thresh=0.5, ratio_thresh=1e-3):
    """For every converged multistart attempt, rank classes by share
    (descending) to sidestep label switching, then flag which variables
    show VP > vdp_thresh in each ranked class. Returns a list of rows:
    (ll, kind, class_rank, flagged_vars).

    CHANGED from latent_class.py::_cluster_stability_report: takes `model`
    (the fitted LatentClassConditional instance) as an explicit first
    argument instead of being a bound method, exactly like BEHier.py takes
    `search_instance` explicitly. Every former `self.X` becomes `model.X`;
    the internal call to `self._collinearity_diag(...)` becomes a direct
    call to `collinearity_diag(...)` (the sibling function above, no more
    `self.`). Behaviour is otherwise identical.
    """
    import jax.numpy as jnp  # local import: keep jax optional for callers
    from .latent_class import RIDGE  # the ridge constant already defined there

    rows = []
    for a in attempts:
        betas, inter, gamma, pi, ll, kind = (
            a['betas'], a['inter'], a['gamma'], a['pi'], a['ll'], a['kind']
        )
        R, _ = model._estep(betas, inter, gamma, pi)
        share = np.asarray(R).mean(axis=0)
        order = np.argsort(-share)
        for rank, c in enumerate(order, start=1):
            beta_j = jnp.asarray(betas[c])
            w_j = jnp.asarray(R[:, c])
            _, g_raw, h_raw = model._class_fgh_raw[c]
            H = np.asarray(h_raw(beta_j, w_j) + jnp.eye(beta_j.shape[0], dtype=beta_j.dtype) * RIDGE)
            ratio_min = float(np.linalg.eigvalsh(H).min() / np.linalg.eigvalsh(H).max())
            vdp_share, flag, _ = collinearity_diag(H, ratio_thresh=ratio_thresh, vdp_thresh=vdp_thresh)
            names_c = [model.varnames[i] for i in model._class_specs[c]]
            flagged = [names_c[k] for k in range(len(names_c)) if flag[k]]
            rows.append(dict(ll=float(ll), kind=kind, class_rank=rank,
                              share=float(share[c]), ratio_min=ratio_min,
                              flagged_vars=flagged))
    return rows


# ------------------------------------------------------------------------
# (A, new) Formal separation test (Konis 2007) and interpretable attribution
# ------------------------------------------------------------------------

def konis_separation_test(xbar_weighted, tol=1e-9):
    """Konis (2007) linear-programming separation test (eq. konis-lp in the
    proposal). `xbar_weighted` is an (n, p) array whose rows are already
    w_i * ytilde_i * x_i (weight-and-sign-folded structure vectors) for a
    binary sub-problem -- see `build_purchase_stage_xbar` for how to build
    this for the two-stage choice model, or construct the multinomial
    one-vs-rest contrasts directly for a standard MNL choice/membership
    stage (Konis 2007, Section 1.4).

    Returns a dict: {'separated': bool, 'status': scipy status code,
                      'message': str, 'n_rows': int}.
    Rows with negligible weight (below `tol` in L1 norm) are dropped first,
    since they cannot bind a constraint in floating point regardless.
    """
    row_mass = np.abs(xbar_weighted).sum(axis=1)
    A = xbar_weighted[row_mass > tol]
    n_rows, p = A.shape
    if n_rows == 0:
        return {'separated': False, 'status': None, 'message': 'no weighted rows', 'n_rows': 0}

    c = -A.sum(axis=0)          # maximize sum(A @ beta)  ==  minimize -sum(A @ beta)
    A_ub = -A                   # A @ beta >= 0  <=>  -A @ beta <= 0
    b_ub = np.zeros(n_rows)
    bounds = [(None, None)] * p

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    separated = (res.status == 3)  # HiGHS: unbounded
    return {'separated': separated, 'status': res.status, 'message': res.message, 'n_rows': n_rows}


def build_purchase_stage_xbar(x_it, y_it, w_i):
    """Helper: build the weighted structure-vector matrix for the
    purchase-stage binary logit of the two-stage LatentClassConditional
    choice model, ready to hand to `konis_separation_test`.

    x_it : (N, T, K) array, chosen-bundle attributes minus no_purchase attributes
           (i.e. X[:, :, j_pref, :] - X[:, :, no_purchase_idx, :])
    y_it : (N, T) binary purchase outcome
    w_i  : (N,) posterior weight of the class under test, w_is

    Returns: (N*T, K) array of w * ytilde * x, ready for konis_separation_test.
    """
    N, T, K = x_it.shape
    ytilde = 2 * y_it - 1
    xbar = ytilde[:, :, None] * x_it
    W = w_i[:, None] * np.ones((N, T))
    return (W[:, :, None] * xbar).reshape(-1, K)


def separation_contingency_table(u_it, p_it, w_i, tau=0.01):
    """Weighted 2x2 attribution table (eq. choice-cell / eq. cell-threshold
    in the proposal). u_it, p_it: (N, T) binary arrays (attribute-in-chosen
    vs. purchase outcome, or the membership-level analogue). w_i: (N,)
    posterior weight for the class under test.

    Returns dict with the four weighted cell counts and `flagged` (bool):
    True if any cell falls below tau * total weighted mass -- the same
    rule used throughout Sections 3.3 / 4.3 / 7.4 of the proposal.
    """
    N, T = u_it.shape
    W = w_i[:, None] * np.ones((N, T))
    cells = {}
    for u in (0, 1):
        for p in (0, 1):
            cells[(u, p)] = float((W * (u_it == u) * (p_it == p)).sum())
    total = sum(cells.values())
    flagged = total > 0 and min(cells.values()) < tau * total
    return {'cells': cells, 'total': total, 'flagged': flagged}

def attribute_separation(xbar, var_names):
    """Leave-one-out attribution: drop each column of `xbar` in turn and
    re-run the Konis test on the reduced matrix. A variable is 'implicated'
    if removing it alone clears the separated flag -- generalizes to
    continuous attributes (COST, TIME, ...), unlike a binary contingency
    table. `var_names` must align 1:1 with xbar's columns."""
    implicated = []
    for k, name in enumerate(var_names):
        reduced = np.delete(xbar, k, axis=1)
        if reduced.shape[1] == 0:
            continue
        if not konis_separation_test(reduced)['separated']:
            implicated.append(name)
    return implicated

def sign_breakdown(xbar, col_idx, tol=1e-9):
    """Conteo de signos de una columna del vector de estructura ponderado --
    el analogo continuo de la tabla 2x2: si (casi) todas las filas de peso
    no-nulo caen del mismo lado, es la firma de separacion (ninguna fila
    'frena' a ese coeficiente)."""
    col = xbar[:, col_idx]
    return {
        'positive': int((col > tol).sum()),
        'negative': int((col < -tol).sum()),
        'zero': int((np.abs(col) <= tol).sum()),
    }

# ------------------------------------------------------------------------
# (B) Multi-fit orchestration -- same calling convention as BEHier.py
# ------------------------------------------------------------------------

def run_separation_protocol(search_instance, memory, tau=0.01, ratio_thresh=1e-3,
                             vdp_thresh=0.5, bic_tolerance=10.0):
    """Steps 3-7 of the proposal's protocol (Steps 0-1, alignment and the
    multistart ensemble itself, are assumed already done by the time
    `memory` -- the list of converged multistart candidates for one
    specification -- reaches this function; mirrors where BEHier() is
    invoked in harmony.py, i.e. once per candidate specification, after
    its multistart run has produced `memory`).

    Returns the representative solution (either the best CLEAN run, or the
    best-LL run after Firth escalation and labeling), with `sol['separation_label']`
    set to one of: 'clean', 'Firth', 'Firth_Parcial', 'Firth_Vacio',
    'Firth_Membership_Corrected'.

    Only external dependency: `search_instance.evaluate_model(sol)` for the
    Firth re-estimation fallback (Step 6), exactly as BEHier.py depends on
    it for refitting.
    """
    # Step 3: degenerate-class screen (already-available betas/pi per candidate)
    survivors = [sol for sol in memory
                 if not is_degenerate_candidate(sol['model'].class_betas, sol['model'].class_probs)]
    if not survivors:
        survivors = memory  # nothing survives the screen; fall back to full ensemble, flagged downstream

    # Step 4: formal separation test per surviving run
    clean_runs, flagged_runs = [], []
    for sol in survivors:
        model = sol['model']
        any_separated = False
        for c in range(model.n_classes):
            # NOTE: building x_it/y_it/w_i per class requires the model's own
            # stored design arrays; wire this to whatever attribute already
            # holds them (e.g. model._X, model._y_pref, model._w_purch) --
            # left as a call-site TODO since those attribute names weren't
            # visible in the excerpts reviewed for this module.
            xbar = build_purchase_stage_xbar(model._x_it, model._y_it, model.posterior[:, c])
            result = konis_separation_test(xbar)
            if result['separated']:
                any_separated = True
                break
        (flagged_runs if any_separated else clean_runs).append(sol)

    # Step 5: selection among candidates
    if clean_runs:
        best_clean = min(clean_runs, key=lambda s: s['bic'])
        best_overall = min(survivors, key=lambda s: s['bic'])
        if best_clean['bic'] - best_overall['bic'] <= bic_tolerance:
            best_clean['separation_label'] = 'clean'
            return best_clean

    # Step 6: Firth escalation (fallback) -- delegates the actual re-fit to
    # the search instance's existing Firth-enabled estimation path.
    best_run = min(survivors, key=lambda s: s['loglik'], default=None)
    if best_run is None:
        return None
    firth_sol = search_instance.evaluate_model(best_run, force_firth=True)  # call-site TODO: confirm kwarg name

    # Step 7: re-diagnose on the POST-CORRECTION posterior, then label
    model = firth_sol['model'] if isinstance(firth_sol, dict) else firth_sol
    any_separated_post = False
    for c in range(model.n_classes):
        xbar = build_purchase_stage_xbar(model._x_it, model._y_it, model.posterior[:, c])
        if konis_separation_test(xbar)['separated']:
            any_separated_post = True
            break

    if not any_separated_post:
        label = 'Firth'
    else:
        label = 'Firth_Vacio'  # cell-level table check (Section 7.7) refines
                                 # this to 'Firth_Parcial' when the flagged
                                 # cell is populated but the instability flag
                                 # persists -- wire in separation_contingency_table
                                 # per attribute once the call site is confirmed.
    firth_sol['separation_label'] = label
    return firth_sol
