"""
BEHier — Backward Elimination with Hierarchical Rules
=======================================================

In simple terms, what this module does:

  1. Takes an already-estimated solution and looks at which coefficients
     are NOT statistically significant.
  2. Removes the least significant one first — but respecting hierarchical
     groups (an intercept, a random parameter's mean/sd pair, a correlation
     block, a Box-Cox transform) so that related pieces are removed or kept
     together instead of breaking a group apart incorrectly.
  3. Re-estimates the model and checks whether the goodness-of-fit (BIC)
     improved. If it did, the change is kept and the process repeats with
     the next worst coefficient. If it did not improve, the change is
     reverted and that coefficient is skipped instead.
  4. Anything the user explicitly prespecified (forced to stay in the
     model, i.e. any `ps_*` setting) is NEVER removed, no matter how
     insignificant it tests.
  5. This repeats until every remaining coefficient is significant (or a
     prespecified exception), or `max_passes` is reached.

This module is self-contained: it can be added or removed without touching
anything else. Its only external dependency is `search_instance.evaluate_model(sol)`,
used to re-fit the model after each change — that fitting engine is not
duplicated here on purpose.
"""

import copy
import math
import random
import numpy as np
from scipy.stats import chi2


# ------------------------------------------------------------------------
# Read-only helpers
# ------------------------------------------------------------------------

def get_components(sol):
    return sol['asvars'], sol['isvars'], sol['randvars'], sol['bcvars'], \
        sol['corvars'], sol['asc_ind']


def update_objectives(param, sol):
    for i in range(param.nb_crit):
        metric = param.criterions[i][0]
        sol.update_objective(i, sol[metric])


# ------------------------------------------------------------------------
# Removal helpers — uniform naming: all "remove" functions use a leading
# underscore, all take `param` explicitly instead of relying on `self`.
# Each one respects any variable marked as prespecified (`ps_*`): those
# are never dropped, regardless of significance.
# ------------------------------------------------------------------------

def _remove_insig_asvars(asvars, insig, bcvars, pval, pval_member,
                          class_params_spec, member_params_spec, param):
    # Keep significant as-variables, i.e., those with significant pvals
    asvars_sig = [var for var in asvars if var not in insig]
    asvars_sig.extend(param.ps_asvars)  # prespecified: always kept

    # Replace insignificant alt-spec coefficient with generic coefficient
    insig_altspec = []
    for var in param.asvarnames:
        altspec = [name for name in insig if name.startswith(var)]
        insig_altspec.extend(altspec)
    insig_altspec_vars = [var for var in insig_altspec if var not in param.asvarnames]

    rem_asvars = []

    if insig_altspec_vars:
        gen_var = [var for sublist in insig_altspec_vars for var in sublist.split("_")]
        gen_coeff = [var for var in param.asvarnames if var in gen_var]

        if asvars_sig:
            redund_vars = [var for var in gen_coeff if any(var in sublist for sublist in asvars_sig)]
            asvars_sig.extend([var for var in gen_coeff if var not in redund_vars])
            rem_asvars = sorted(list(set(asvars_sig)))
        else:
            rem_asvars = gen_coeff
    if not rem_asvars:
        rem_asvars = sorted(list(set(asvars_sig)))

    rem_class_params_spec = copy.deepcopy(class_params_spec)
    rem_member_params_spec = copy.deepcopy(member_params_spec)

    return rem_asvars, rem_class_params_spec, rem_member_params_spec


def _remove_insig_isvars(isvars, insig, param):
    insig_isvars = []
    for var in param.isvarnames:
        insig_isvar = [name for name in insig if name.startswith(var)]
        insig_isvars.extend(insig_isvar)

    remove_isvars = []
    remove_isvars.extend(part.split(".") for part in insig_isvars)

    remove_isvar = [var for var in remove_isvars if var in isvars]

    dict_insig_isvar = {var: remove_isvar.count(var) for var in remove_isvar}

    rem_isvar = [k for k, v in dict_insig_isvar.items() if v == (len(param.choice_set) - 1)]

    isvars_revised = [var for var in isvars if var not in rem_isvar]
    isvars_revised.extend(param.ps_isvars)  # prespecified: always kept

    rem_isvars = sorted(list(set(isvars_revised)))
    return rem_isvars


def _remove_insig_randvars(insig, randvars, rem_asvars, param):
    insig_sd = [var for var in insig if var.startswith('sd.')]
    insig_sd_rem = [str(var).replace('sd.', '') for var in insig_sd]

    remove_rv = [var for var in insig_sd_rem if
                 var not in param.ps_randvars.keys() or var not in rem_asvars]

    rem_rand_vars = {var: val for var, val in randvars.items() if var in rem_asvars and var not in remove_rv}
    rem_rand_vars.update({var: val for var, val in param.ps_randvars.items()
                          if var in rem_asvars and val != 'f'})  # prespecified: always kept

    for var in param.ps_corvars:
        if var in rem_asvars and var not in rem_rand_vars.keys():
            if remove_rv:
                rem_rand_vars.update({var: np.random.choice(remove_rv)})

    return rem_rand_vars


def _remove_insig_bcvars(insig, bcvars, rem_asvars, param):
    ns_lambda = [x for x in insig if x.startswith('lambda.')]
    ns_bctransvar = [str(i).replace('lambda.', '') for i in ns_lambda
                      if str(i).replace('lambda.', '') not in param.ps_bcvars]  # prespecified: never marked for removal
    rem_bcvars = [var for var in bcvars if var in rem_asvars and var not in ns_bctransvar
                      and var not in param.ps_corvars]
    return rem_bcvars


def _remove_insig_corvars(insig, corvars, rem_randvars, rem_bcvars, all_pvalues, all_coeff_names, param):
    """
    Remove insignificant correlated variables using a row-based Cholesky
    criterion: a variable is removed from corvars only if NONE of its
    off-diagonal correlations with other corvars are significant. If at
    least one correlation is significant, the variable stays. Prespecified
    (`ps_corvars`) variables are always kept regardless of significance.
    """
    if not corvars or len(corvars) < 2:
        return []

    Kc = len(corvars)

    pval_dict = dict(zip(all_coeff_names, all_pvalues))

    matrix = np.ones((Kc, Kc))
    for i, vi in enumerate(corvars):
        for j, vj in enumerate(corvars):
            if j <= i:
                key1 = f"chol.{vi}.{vj}"
                key2 = f"chol.{vj}.{vi}"
                pval = pval_dict.get(key1, pval_dict.get(key2, 1.0))
                matrix[i][j] = pval
                matrix[j][i] = pval

    sig_value = param.p_val
    matrix_diag = np.diag(matrix).copy()
    np.fill_diagonal(matrix, sig_value)

    rows_insig = ~np.any(matrix < sig_value, axis=1)

    np.fill_diagonal(matrix, matrix_diag)

    if np.all(rows_insig):
        # Even if every row is insignificant, prespecified corvars must stay
        rem_corvars = [var for var in corvars if var in param.ps_corvars
                       and var in rem_randvars.keys() and var not in rem_bcvars]
        return rem_corvars if len(rem_corvars) >= 2 else []

    rem_corvars = [var for var, insig_row in zip(corvars, rows_insig)
                if (not insig_row or var in param.ps_corvars)  # prespecified: always kept
                and var in rem_randvars.keys()
                and var not in rem_bcvars]

    return rem_corvars if len(rem_corvars) >= 2 else []

def _parse_lc_coeff_name(name):
    """Split a latent-class coefficient name into (kind, class_idx, base_var).

    kind is one of 'class_fixed', 'member_gamma', 'member_intercept', or
    'other' (anything not matching the latent-class naming convention).
    class_idx is 0-based. base_var is None for member_intercept.
    """
    if name.startswith('gamma_intercept_class_'):
        return 'member_intercept', int(name.rsplit('_', 1)[-1]) - 1, None
    if name.startswith('gamma_class_'):
        rest = name[len('gamma_class_'):]
        class_str, var = rest.split('_', 1)
        return 'member_gamma', int(class_str) - 1, var
    if name.startswith('class_'):
        rest = name[len('class_'):]
        class_str, var = rest.split('_', 1)
        return 'class_fixed', int(class_str) - 1, var
    return 'other', None, None


def _lrt_within_range(loglik_child, loglik_parent, df, alpha=0.05):
    """True when the loglik drop from parent to child is no larger than
    what `df` lost parameters would explain under H0 — i.e. the child's
    loglik looks reliable and does not need more multistart attempts."""
    if loglik_child is None or not math.isfinite(loglik_child) or df <= 0:
        return False
    drop = loglik_parent - loglik_child
    if drop <= 0:
        return True
    return (2.0 * drop) <= chi2.ppf(1 - alpha, df)


def _build_lc_init_betas(search_instance, model, old_class_spec, new_class_spec):
    """Per-class warm start: {name: value} maps restricted to the variables
    that survive in `new_class_spec`, sourced from model.varnames/_class_specs
    (setup()'s resolved order) so it's correct regardless of old_class_spec's
    own ordering."""
    new_maps = []
    for c in range(len(new_class_spec)):
        if getattr(model, 'varnames', None) is not None and getattr(model, '_class_specs', None) is not None:
            names_c = [model.varnames[i] for i in model._class_specs[c]]
        else:
            names_c = list(old_class_spec[c])
        dom_map = dict(zip(names_c, model.class_betas[c]))
        new_maps.append(search_instance._match_betas(dom_map, new_class_spec[c]))
    return new_maps


def _behier_latent(search_instance, sol, max_passes=10):
    """Latent-class counterpart of BEHier: surgical, per-class elimination
    of insignificant class-fixed and membership-gamma coefficients, using
    an LRT gate to decide how many multistart attempts each re-fit needs.
    """
    param = search_instance.param
    p_val = param.p_val
    ps_asvars = set(getattr(param, 'ps_asvars', []))
    ps_isvars = set(getattr(param, 'ps_isvars', []))

    if sol.get('model') is None or getattr(sol['model'], 'pvalues', None) is None:
        print("No p-values available for _behier_latent. Returning original solution.")
        return sol

    def insignificant(ref_sol):
        ref_model = ref_sol['model']
        out = []
        n_phi = ref_model.n_classes - 1
        beta_pvalues = np.asarray(ref_model.pvalues)[n_phi:n_phi + len(ref_model.coeff_names)]

        print(f"[DEBUG insignificant] class-beta coeff_names/pvalues ({len(ref_model.coeff_names)}, n_phi offset={n_phi}):")
        for _n, _p in zip(ref_model.coeff_names, beta_pvalues):
            print(f"    {_n:<30} p={float(_p):.4f}")

        pval_dict = dict(zip(ref_model.coeff_names, beta_pvalues))
        class_specs = ref_sol['class_params_spec']

        for name, pv in zip(ref_model.coeff_names, beta_pvalues):
            if float(pv) <= p_val:
                continue
            kind, c_idx, var = _parse_lc_coeff_name(name)
            if kind != 'class_fixed' or var in ps_asvars or var.startswith('intercept.'):
                continue
            out.append((kind, c_idx, var, float(pv)))

        # ASC: remove all jointly per class, or keep all — never partially.
        for c_idx, spec in enumerate(class_specs):
            asc_vars = [v for v in spec if str(v).startswith('intercept.') and v not in ps_asvars]
            if not asc_vars:
                continue
            asc_pvals = [(v, float(pval_dict.get(f'class_{c_idx + 1}_{v}', 0.0))) for v in asc_vars]
            if all(pv_ > p_val for _, pv_ in asc_pvals):
                out.extend(('class_fixed', c_idx, v, pv_) for v, pv_ in asc_pvals)
                print(f"    [Intercept] class {c_idx + 1}: all ASC insignificant — removing jointly.")
            elif any(pv_ > p_val for _, pv_ in asc_pvals):
                sig = [v for v, pv_ in asc_pvals if pv_ <= p_val]
                print(f"    [Intercept] class {c_idx + 1}: {sig} still significant — keeping all ASC.")

        gamma_names = getattr(ref_model, 'gamma_names', None)
        gamma_names = [] if gamma_names is None else gamma_names
        gamma_pvals = getattr(ref_model, 'gamma_p_values', None)
        gamma_pvals = [] if gamma_pvals is None else gamma_pvals
        print(f"[DEBUG insignificant] membership gamma_names/p_values ({len(gamma_names)}):")
        for _n, _p in zip(gamma_names, gamma_pvals):
            print(f"    {_n:<30} p={float(_p):.4f}")

        for name, pv in zip(gamma_names, gamma_pvals):
            if float(pv) <= p_val:
                continue
            kind, c_idx, var = _parse_lc_coeff_name(name)
            if kind == 'member_gamma' and var not in ps_isvars:
                out.append((kind, c_idx, var, float(pv)))
        return out

    def remove_from_spec(ref_sol, items):
        new_class  = [list(arr) for arr in ref_sol['class_params_spec']]
        new_member = [list(arr) for arr in ref_sol['member_params_spec']] \
            if ref_sol.get('member_params_spec') is not None else None
        for kind, c_idx, var, _ in items:
            if kind == 'class_fixed' and var in new_class[c_idx]:
                new_class[c_idx].remove(var)
            elif kind == 'member_gamma' and new_member is not None and var in new_member[c_idx]:
                new_member[c_idx].remove(var)
        return new_class, new_member

    def refit(ref_sol, new_class, new_member, df):
        """Warm-started re-fit of a reduced spec, escalating multistart via
        the LRT gate. Returns (trial_sol, accepted_by_lrt)."""
        old_class = ref_sol['class_params_spec']
        trial = search_instance.copy_solution(ref_sol)
        trial['class_params_spec']  = np.array(new_class,  dtype=object)
        if not any(str(v).startswith('intercept.') for cls in new_class for v in cls):
            trial['asc_ind'] = False  # every class lost its ASC this round — stop setup() from reinjecting        
        trial['member_params_spec'] = np.array(new_member, dtype=object) if new_member is not None else None
        trial['init_class_betas'] = _build_lc_init_betas(search_instance, ref_sol['model'], old_class, new_class)
        search_instance.param.num_classes = len(new_class)

        parent_loglik = float(ref_sol['loglik'])
        converged, loglik = False, float('-inf')
        for n_init_try in (1, 5):
            trial['n_init_override'] = n_init_try
            aic, bic, loglik, mae, _, _, _, _, _, converged, trial = search_instance.evaluate_model(trial)
            trial['aic'], trial['bic'], trial['loglik'], trial['mae'] = aic, bic, loglik, mae
            _drop = parent_loglik - loglik if (converged and math.isfinite(loglik)) else float('nan')
            _crit = chi2.ppf(0.95, df) if df > 0 else float('nan')
            print(f"    [DEBUG refit] n_init={n_init_try} converged={converged} "
                  f"loglik={loglik:.4f} parent_loglik={parent_loglik:.4f} "
                  f"drop={_drop:.4f} 2*drop={2*_drop:.4f} critical(df={df})={_crit:.4f} "
                  f"BIC={float(trial['bic']):.4f}")
            if converged and _lrt_within_range(loglik, parent_loglik, df):
                return trial, True
        return trial, (converged and math.isfinite(loglik))

    # ---- Phase 1: batch — remove every insignificant coeff at once ----
    to_remove = insignificant(sol)
    print(f"[DEBUG] to_remove ({len(to_remove)}): " +
          ", ".join(f"{k} class{c+1}.{v}(p={p:.3f})" for k, c, v, p in to_remove))
    if not to_remove:
        return sol

    new_class, new_member = remove_from_spec(sol, to_remove)
    trial, reliable = refit(sol, new_class, new_member, df=len(to_remove))

    if reliable and float(trial['bic']) < float(sol['bic']):
        print(f"BEHier[latent] batch: {len(to_remove)} coeffs removed, "
              f"BIC {float(sol['bic']):.4f} -> {float(trial['bic']):.4f} — accepted.")
        return trial

    print(f"BEHier[latent] batch not accepted (reliable={reliable}) "
          f"— falling back to sequential elimination.")

    # ---- Phase 2: sequential, worst p-value first ----
    current = sol
    remaining = sorted(to_remove, key=lambda it: -it[3])
    cleanup_pass = 0

    while remaining and cleanup_pass < max_passes:
        item = remaining.pop(0)
        kind, c_idx, var, _ = item
        cur_class  = current['class_params_spec']
        cur_member = current.get('member_params_spec')

        still_present = (kind == 'class_fixed' and var in cur_class[c_idx]) or \
                         (kind == 'member_gamma' and cur_member is not None and var in cur_member[c_idx])
        if not still_present:
            cleanup_pass += 1
            continue

        new_class, new_member = remove_from_spec(current, [item])
        print(f"    [DEBUG pass {cleanup_pass + 1}] trying to remove {kind} class {c_idx+1} "
              f"'{var}' (p={item[3]:.4f}) — baseline BIC={float(current['bic']):.4f}")
        trial, reliable = refit(current, new_class, new_member, df=1)

        if reliable and float(trial['bic']) < float(current['bic']):
            print(f"BEHier[latent] pass {cleanup_pass + 1}: {kind} class {c_idx + 1} "
                  f"'{var}' removed — BIC improved to {float(trial['bic']):.4f}.")
            current = trial
        else:
            print(f"BEHier[latent] pass {cleanup_pass + 1}: {kind} class {c_idx + 1} "
                  f"'{var}' — BIC did not improve, kept.")

        cleanup_pass += 1

    return current


# ------------------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------------------

def BEHier(search_instance, sol, max_passes=10):
    """
    Backward Elimination with Hierarchical Rules.

    Iteratively removes the least significant coefficient — respecting
    hierarchical groups (intercepts, random mean/sd pairs, correlation
    blocks, Box-Cox transforms) and any prespecified (`ps_*`) exception —
    re-estimating after each removal and keeping the change only if it
    improves BIC. Stops when every remaining coefficient is significant
    (or prespecified) or when `max_passes` is reached.

    `search_instance` is only used to call `evaluate_model(sol)`.
    """
    param = search_instance.param

    if getattr(param, 'latent_class', False):
        return _behier_latent(search_instance, sol, max_passes)

    np_state  = np.random.get_state()
    rnd_state = random.getstate()

    if sol['pvalues'] is None or len(sol['pvalues']) == 0 or len(sol['coeff_names']) == 0:
        print("No p-values or coefficient names available for BEHier. Returning original solution.")
        return sol

    as_vars, is_vars, rand_vars, bc_vars, corvars, asc_ind = get_components(sol)
    asvars, isvars, randvars, bcvars = as_vars, is_vars, rand_vars, bc_vars

    intercept_names = [f'intercept.{a}' for a in param.choice_set if a != param.base_alt]

    cleanup_pass = 0
    skipped      = set()
    skipped_vars = set()
    best_spec    = {k: sol[k] for k in ['asvars','isvars','randvars','bcvars','corvars',
                                         'bic','pvalues','coeff_names','model','aic','loglik','mae','asc_ind']}

    print(f"\n{'='*60}")
    print(f"BEHier — Solution {sol['sol_num']} | {sol['model_n']} | BIC: {round(float(sol['bic']), 4)}")
    print(f"{'='*60}")

    while cleanup_pass < max_passes and sol['pvalues'] is not None:

        pvalues     = sol['pvalues']
        coeff_names = sol['coeff_names']
        pval_dict   = dict(zip(coeff_names, pvalues))

        # Pick worst insignificant coefficient not already skipped
        sorted_insig = sorted(
            [(coeff_names[i], float(pvalues[i])) for i in range(len(pvalues))
             if float(pvalues[i]) > param.p_val and coeff_names[i] not in skipped],
            key=lambda x: -x[1])

        if not sorted_insig:
            break

        worst_name, max_pval = sorted_insig[0]
        var_name = worst_name.replace('sd.', '').replace('chol.', '').split('.')[0]

        all_insig = [(str(coeff_names[i]), round(float(pvalues[i]), 3))
                     for i in range(len(pvalues)) if float(pvalues[i]) > param.p_val]

        print(f"\n--- BEHier pass {cleanup_pass + 1} ---")
        print(f"    BIC before   : {round(float(sol['bic']), 4)}")
        print(f"    Insignificant: {all_insig}")
        print(f"    Worst        : {worst_name} (p={round(max_pval, 3)})")

        insig = []

        # ----------------------------------------------------------------
        # Case 1 — Intercepts: remove all jointly or not at all
        # ----------------------------------------------------------------
        if worst_name in intercept_names:
            all_intercepts_insig = all(pval_dict.get(n, 1.0) > param.p_val for n in intercept_names)
            if all_intercepts_insig:
                insig = intercept_names
                print(f"    [Intercept]  : all intercepts insignificant — removing all jointly.")
            else:
                sig = [n for n in intercept_names if pval_dict.get(n, 1.0) <= param.p_val]
                print(f"    [Intercept]  : {sig} still significant — keeping all intercepts.")
                skipped.update(intercept_names)
                continue

        # ----------------------------------------------------------------
        # Case 2 — Fixed parameter
        # ----------------------------------------------------------------
        elif worst_name in asvars and var_name not in randvars and var_name not in bcvars:
            insig = [worst_name]
            print(f"    [Fixed]      : removing {worst_name} (p={round(max_pval, 3)}).")

        # ----------------------------------------------------------------
        # Case 3 — Random parameter (mean or sd)
        # ----------------------------------------------------------------
        elif var_name in randvars and not worst_name.startswith('chol.') and var_name not in corvars:
            sd_name   = f'sd.{var_name}'
            chol_diag = f'chol.{var_name}.{var_name}'
            sd_pval   = pval_dict.get(sd_name, pval_dict.get(chol_diag, 0.0))
            mean_pval = pval_dict.get(var_name, 0.0)
            sd_insig  = sd_pval   > param.p_val
            mean_insig= mean_pval > param.p_val

            if sd_insig:
                # Mean sig, sd insig — OR — Mean insig, sd insig: remove sd
                insig = [sd_name] if sd_name in pval_dict else [chol_diag]
                print(f"    [Random]     : sd insignificant (p={round(sd_pval,3)}) — removing sd of {var_name}.")
            else:
                # Mean insig, sd sig — keep both
                print(f"    [Random]     : mean of {var_name} insig but sd significant (p={round(sd_pval,3)}) — keeping both.")
                skipped.add(worst_name)
                continue

        # ----------------------------------------------------------------
        # Case 4 — Cholesky off-diagonal: remove var from corvars if all cross-chol insig
        # ----------------------------------------------------------------
        elif worst_name.startswith('chol.'):
            parts     = worst_name.split('.')
            v, u      = parts[1], parts[2]
            is_diag   = (v == u)

            if not is_diag:
                if v in skipped_vars:
                    skipped.add(worst_name)
                    print(f"    [Chol]       : {v} already decided — skipping {worst_name}.")
                    continue
                cross_chol = [k for k in pval_dict if k.startswith('chol.') and
                              k != f'chol.{v}.{v}' and
                              (f'.{v}.' in k or k.endswith(f'.{v}'))]
                all_cross_insig = all(pval_dict[k] > param.p_val for k in cross_chol) if cross_chol else True
                if all_cross_insig:
                    insig = cross_chol if cross_chol else [worst_name]
                    print(f"    [Chol]       : all off-diagonal chol for {v} insignificant — removing {v} from corvars.")
                else:
                    sig_cross = [k for k in cross_chol if pval_dict[k] <= param.p_val]
                    print(f"    [Chol]       : {sig_cross} still significant — keeping {v} in corvars.")
                    skipped_vars.add(v)
                    skipped.add(worst_name)
                    continue
            else:
                # Diagonal chol — treat as sd
                sd_insig = pval_dict.get(worst_name, 0.0) > param.p_val
                if sd_insig:
                    insig = [f'sd.{v}']
                    print(f"    [Chol diag]  : {worst_name} insignificant — removing sd of {v}.")
                else:
                    skipped.add(worst_name)
                    continue

        # ----------------------------------------------------------------
        # Case 5 — Boxcox: remove from bcvars, keep in randvars if applicable
        # ----------------------------------------------------------------
        elif var_name in bcvars:
            insig = [worst_name]
            if var_name in randvars:
                print(f"    [Boxcox]     : lambda of {var_name} insig — removing from bcvars, remains in randvars.")
            else:
                print(f"    [Boxcox]     : lambda of {var_name} insig — removing from bcvars, reverting to fixed.")

        else:
            skipped.add(worst_name)
            print(f"    [Skip]       : {worst_name} — no matching case, skipping.")
            continue

        # ----------------------------------------------------------------
        # Apply removal and check if spec changed
        # ----------------------------------------------------------------
        if not insig:
            skipped.add(worst_name)
            continue

        rem_asvars, _, _ = _remove_insig_asvars(asvars, insig, bcvars, pvalues, [], None, None, param)
        if len(rem_asvars) == 0:
            rem_asvars = list(asvars)
        rem_isvars   = _remove_insig_isvars(isvars, insig, param)
        rem_randvars = _remove_insig_randvars(insig, randvars, rem_asvars, param)
        rem_bcvars   = _remove_insig_bcvars(insig, bcvars, rem_asvars, param)
        rem_corvars  = _remove_insig_corvars(
            insig, corvars, rem_randvars, rem_bcvars,
            sol['pvalues'], sol['coeff_names'], param) if worst_name.startswith('chol.') else list(corvars)

        if param.ps_intercept is not None:
            rem_asc_ind = param.ps_intercept  # prespecified: never overridden by significance
        else:
            rem_asc_ind = False if (insig and all(n in intercept_names for n in insig)) else asc_ind

        spec_changed = (
            sorted(rem_asvars)                          != sorted(asvars)   or
            sorted(rem_isvars)                          != sorted(isvars)   or
            dict(sorted(rem_randvars.items()))          != dict(sorted(randvars.items())) or
            sorted(rem_bcvars)                          != sorted(bcvars)   or
            sorted(rem_corvars)                         != sorted(corvars)  or
            rem_asc_ind                                 != asc_ind
        )

        if not spec_changed:
            skipped.add(worst_name)
            print(f"    Result       : no spec change — skipping.")
            continue

        if spec_changed and sorted(rem_corvars) != sorted(corvars):
            skipped_vars = set()

        sol['asvars']  = rem_asvars
        sol['isvars']  = rem_isvars
        sol['randvars']= rem_randvars
        sol['bcvars']  = rem_bcvars
        sol['corvars'] = rem_corvars
        sol['asc_ind'] = rem_asc_ind

        aic, bic, loglik, mae, asvars, isvars, randvars, bcvars, corvars, converged, pvalues, coeff_names, sol = \
            search_instance.evaluate_model(sol)

        asc_ind = sol['asc_ind']
        sol['aic'], sol['bic'], sol['loglik'], sol['mae']     = aic, bic, loglik, mae
        sol['pvalues'], sol['coeff_names']                    = pvalues, coeff_names
        passes = isinstance(loglik, float) and math.isfinite(loglik)

        print(f"    BIC after    : {round(float(bic), 4)} | converged: {converged} | valid: {passes}")

        if passes and float(bic) < float(best_spec['bic']):
            print(f"    Result       : improved — new best BIC: {round(float(bic), 4)}")
            best_spec = {k: sol[k] for k in ['asvars','isvars','randvars','bcvars','corvars',
                                              'bic','pvalues','coeff_names','model','aic','loglik','mae','asc_ind']}
            skipped = set()
            skipped_vars = set()
        else:
            sol.update(best_spec)
            asc_ind = sol['asc_ind']
            asvars, isvars, randvars, bcvars, corvars = \
                sol['asvars'], sol['isvars'], sol['randvars'], sol['bcvars'], sol['corvars']
            skipped.update(intercept_names if worst_name in intercept_names else [worst_name])
            print(f"    Result       : BIC did not improve — reverting, skipping {worst_name}.")

        cleanup_pass += 1

    update_objectives(param, sol)
    np.random.set_state(np_state)
    random.setstate(rnd_state)
    return sol