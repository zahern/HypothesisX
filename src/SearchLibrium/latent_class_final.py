"""
LatentClassFinal -- clean, self-contained latent-class MNL engine.

Design principles
-----------------
- Inherits only from DiscreteChoiceModel.
- Everything configured in setup(); fit() takes no arguments.
- C inferred from class_params_spec (mandatory).
- Base class = last class, 1-indexed at the API (0-indexed internally).
- Internal normalization (z-score for continuous, /sigma for dummies,
  untouched ASCs). Stored state stays normalized; summarise() denormalizes.
- Multistart 10% exact / 30% jitter / 60% random, vmap-batched over chains.
- Per-chain degeneracy flagged alongside CONVERGED/STAGNATED.
- Winner selected from non-degenerate chains only.
- Diagnostics (VDP, ratio_min, Konis, OPG/Hess) on RAW Hessian (no ridge,
  pinv fallback). Ridge reserved for Newton.
- Optional Firth-style corrections (choice / membership) -- off by default.
- OPG computation optional (off by default); Cholesky solve.
"""

import time
import numpy as np
from scipy.stats import norm as _scipy_norm
from scipy.optimize import linear_sum_assignment

import jax
import jax.numpy as jnp
from jax import lax, grad, hessian

jax.config.update("jax_enable_x64", True)

try:
    from _choice_model import DiscreteChoiceModel
    from separation_screen import (
        is_degenerate_candidate, collinearity_diag,
        konis_separation_test, attribute_separation, sign_breakdown, hauck_donner_flags,
    )
except ImportError:
    from ._choice_model import DiscreteChoiceModel
    from .separation_screen import (
        is_degenerate_candidate, collinearity_diag,
        konis_separation_test, attribute_separation, sign_breakdown, hauck_donner_flags,
    )


RIDGE_NEWTON = 1e-8
MIN_COMP = 1e-300
OPG_HESS_RATIO_FLAG = 10.0
DEGEN_MAX_ABS_BETA = 200.0


# =====================================================================
# Batched primitives (vmap-safe, no Python control flow inside)
# =====================================================================

def _fast_flatten(betas, inter, gamma, pi):
    n = betas[0].shape[0]
    parts = [b.reshape(n, -1) for b in betas]
    parts.append(inter.reshape(n, -1))
    if gamma.size > 0:
        parts.append(gamma.reshape(n, -1))
    parts.append(pi.reshape(n, -1))
    return jnp.concatenate(parts, axis=1)


def _fast_unflatten(flat, model):
    n = flat.shape[0]
    Ks = model._Ks
    n_inter = model._n_inter
    C = model.n_classes
    Km = model.K_membership
    pos = 0
    betas = []
    for k in Ks:
        k_int = int(k)
        betas.append(flat[:, pos:pos + k_int])
        pos += k_int
    inter = flat[:, pos:pos + n_inter]; pos += n_inter
    if Km > 0:
        gamma = flat[:, pos:pos + C * Km].reshape(n, C, Km)
        pos += C * Km
    else:
        gamma = jnp.zeros((n, C, 0), dtype=flat.dtype)
    pi = flat[:, pos:pos + C]
    return betas, inter, gamma, pi


def _fast_newton(f_j, g_j, h_j, x0, w, inner_iter, tol=1e-10):
    """Fixed-budget Newton, safe inside vmap. Keeps the last accepted step."""
    Kc = x0.shape[-1]

    def body(carry):
        x, any_accepted, i = carry
        g = g_j(x, w)
        H = h_j(x, w) + RIDGE_NEWTON * jnp.eye(Kc, dtype=x.dtype)
        step_v = jnp.linalg.solve(H, g)
        step_v = jnp.where(jnp.isfinite(step_v), step_v, jnp.zeros_like(step_v))
        f0 = f_j(x, w)

        def ls_b(c):
            t, x_cur, accepted, j = c
            cand = x - t * step_v
            fc = f_j(cand, w)
            ok = jnp.isfinite(fc) & (fc <= f0 + 1e-12)
            x_new = jnp.where(ok, cand, x_cur)
            return t * 0.5, x_new, ok, j + 1

        def ls_c(c):
            _, _, accepted, j = c
            return (j < 20) & (~accepted)

        _, x_ls, accepted, _ = lax.while_loop(
            ls_c, ls_b,
            (jnp.array(1.0, dtype=x.dtype), x,
             jnp.array(False), jnp.array(0)),
        )
        x_next = jnp.where(accepted, x_ls, x)
        return x_next, any_accepted | accepted, i + 1

    def cond(carry):
        _, _, i = carry
        return i < inner_iter

    x_final, any_accepted, _ = lax.while_loop(
        cond, body, (x0, jnp.array(False), jnp.array(0)),
    )
    g_final = g_j(x_final, w)
    gnorm = jnp.linalg.norm(g_final)
    return x_final, any_accepted, gnorm


def _fast_em_step(model, betas, inter, gamma, pi):
    R, _ = model._estep(betas, inter, gamma, pi)

    new_betas = []
    accepted_list = []
    gnorm_list = []
    for c in range(model.n_classes):
        f_j, g_j, h_j = model._class_fgh[c]
        x_c, acc_c, gn_c = _fast_newton(
            f_j, g_j, h_j, betas[c], R[:, c],
            model.newton_inner_iter,
        )
        new_betas.append(x_c)
        accepted_list.append(acc_c)
        gnorm_list.append(gn_c)

    if model._has_membership:
        Km = model.K_membership
        v0 = jnp.concatenate([inter, gamma.ravel()])
        f_m, g_m, h_m = model._mem_fgh
        v1, acc_m, gn_m = _fast_newton(
            f_m, g_m, h_m, v0, R, model.newton_inner_iter,
        )
        gnorm_list.append(gn_m)
        n_inter = model._n_inter
        new_inter = v1[:n_inter]
        new_gamma = v1[n_inter:].reshape(model.n_classes, Km)
        new_pi = pi
    else:
        new_inter, new_gamma = inter, gamma
        new_pi = R.mean(axis=0)
        new_pi = new_pi / jnp.sum(new_pi)

    any_accepted = jnp.stack(accepted_list).any()
    if model._has_membership:
        any_accepted = any_accepted | acc_m

    gnorm_max = jnp.stack(gnorm_list).max()

    # ll must be evaluated at the UPDATED parameters. The previous version
    # returned `ll` from the E-step on the INPUT state, leaving it one EM
    # step behind (new_betas, new_inter, new_gamma, new_pi). That corrupted:
    #   - the SQUAREM acceptance test (`ll_p >= ll2`),
    #   - the fallback value used when SQUAREM fails (`ll_f = ll2`),
    #   - the convergence check `dll = |ll_f - ll_prev|`.
    _, ll_new = model._estep(new_betas, new_inter, new_gamma, new_pi)
    return new_betas, new_inter, new_gamma, new_pi, ll_new, R, any_accepted, gnorm_max


# =====================================================================
# Main class
# =====================================================================

class LatentClassFinal(DiscreteChoiceModel):
    """Latent-class MNL with exact-Newton EM, vmap-batched multistart,
    internal z-score normalization, optional Firth-style corrections."""

    def __init__(self):
        super().__init__()
        self.descr = "LC-Final"

        # Reporting slots (populated by _finalise)
        self.class_betas = None
        self.class_betas_norm = None
        self.class_probs = None
        self.class_gammas = None
        self.posterior = None
        self.loglik = None
        self.aic = None
        self.bic = None
        self.converged = False
        self.total_iter = 0
        self.se_computation_error = None

        self.multistart_loglik = []
        self.multistart_kind = []
        self.multistart_degenerate = []
        self.multistart_shares = []
        self.optimise_membership = True   # compat with _choice_model.summarise

    # ------------------------------------------------------------------
    # SETUP (does everything -- model, membership, normalization, inits)
    # ------------------------------------------------------------------
    def setup(self, X, y, varnames, ids, alts, class_params_spec,
              member_params_spec=None, base_class=None,
              betas0=None, gamma0=None, inter0=None,
              n_init=10, maxiter=1000, newton_inner_iter=5,
              ftol=1e-10, gtol=1e-6, random_state=0,
              choice_correction=False, membership_correction=False,
              panels=False, ind_id=None, avail=None,
              fit_intercept=False, base_alt=None,
              step_tol=1e-8, max_stagnant=20, 
              max_plateau=50, plateau_tol=1e-6,
              compute_opg=False, verbose=1, debug_trace=False):
        # ---- scalars / flags ----
        self.n_init = max(1, int(n_init))
        self.maxiter = int(maxiter)
        self.newton_inner_iter = int(newton_inner_iter)
        self.ftol = float(ftol)
        self.gtol = float(gtol)
        self.tol = self.ftol
        self.random_state = int(random_state)
        self.choice_correction = bool(choice_correction)
        self.membership_correction = bool(membership_correction)
        self.step_tol = float(step_tol)
        self.max_stagnant = int(max_stagnant)
        self.max_plateau = int(max_plateau)
        self.plateau_tol = float(plateau_tol)
        self.compute_opg = bool(compute_opg)
        self.verbose = int(verbose)
        self.debug_trace = bool(debug_trace)

        if class_params_spec is None:
            raise ValueError("class_params_spec is mandatory (list of lists).")

        def _dict_to_spec(d):
            if not isinstance(d, dict):
                return d
            keys = sorted(d.keys(), key=lambda k: int(str(k).split('_')[-1]))
            return [list(d[k]) for k in keys]

        class_params_spec = _dict_to_spec(class_params_spec)
        member_params_spec = _dict_to_spec(member_params_spec)

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        ids = np.asarray(ids)
        alts = np.asarray(alts)
        varnames = list(varnames)

        # ---- fit_intercept: prepend ASC columns ----
        self.base_alt = None
        if fit_intercept:
            self.base_alt = base_alt if base_alt is not None else alts[0]
            asc_alts = [a for a in np.unique(alts) if a != self.base_alt]
            asc_names = [f"intercept.{a}" for a in asc_alts]
            asc_cols = np.column_stack([(alts == a).astype(float) for a in asc_alts])
            X = np.column_stack([asc_cols, X])
            varnames = asc_names + varnames
            for c in range(len(class_params_spec)):
                has = any(v in asc_names for v in class_params_spec[c])
                if not has:
                    class_params_spec[c] = list(asc_names) + list(class_params_spec[c])

        # ---- class count + specs ----
        C = len(class_params_spec)
        self.n_classes = C

        _sort_key = lambda v: (0, str(v)) if str(v).startswith("intercept.") else (1, str(v))
        class_params_spec = [sorted(spec, key=_sort_key) for spec in class_params_spec]

        self.K = X.shape[1]
        self.varnames = varnames

        if avail is None:
            avail = np.ones_like(y, dtype=float)
        else:
            avail = np.asarray(avail, dtype=float)

        # ---- sort into (N, P, J, K) ----
        if panels and ind_id is None:
            raise ValueError("panels=True requires ind_id.")
        ind_id = np.asarray(ind_id) if ind_id is not None else ids

        _, first_idx = np.unique(alts, return_index=True)
        self.alts = alts[np.sort(first_idx)]
        self.J = len(self.alts)

        order = np.lexsort((alts, ids))
        X, y, avail = X[order], y[order], avail[order]
        ids_s = ids[order]; ind_s = ind_id[order]

        uniq_sit, counts = np.unique(ids_s, return_counts=True)
        if np.any(counts != self.J):
            raise ValueError("LatentClassFinal requires balanced long-format data by choice id.")

        n_sit = len(uniq_sit)
        X3 = X.reshape(n_sit, self.J, self.K)
        y3 = y.reshape(n_sit, self.J)
        av3 = avail.reshape(n_sit, self.J)
        ind_per_sit = ind_s.reshape(n_sit, self.J)[:, 0]

        uniq_ind, inv = np.unique(ind_per_sit, return_inverse=True)
        self.N = len(uniq_ind)
        self.ids = uniq_ind
        counts_per_ind = np.bincount(inv, minlength=self.N)
        P = int(counts_per_ind.max())
        self.P = P
        self.sample_size = int(n_sit)

        Xp = np.zeros((self.N, P, self.J, self.K))
        yp = np.zeros((self.N, P, self.J))
        avp = np.zeros((self.N, P, self.J))
        mask = np.zeros((self.N, P))
        slot = np.zeros(self.N, dtype=int)
        for s in range(n_sit):
            i = inv[s]; t = slot[i]
            Xp[i, t] = X3[s]; yp[i, t] = y3[s]; avp[i, t] = av3[s]
            mask[i, t] = 1.0
            slot[i] += 1

        self.X = Xp
        self.y = yp
        self.avail = avp
        self.panel_mask = mask
        self.panel_info = mask

        # ---- class spec indices ----
        self._class_specs, Ks = [], []
        for c, spec in enumerate(class_params_spec):
            idxs = []
            for v in spec:
                if v not in varnames:
                    raise ValueError(f"Class {c+1} variable '{v}' not in varnames.")
                idxs.append(varnames.index(v))
            self._class_specs.append(np.array(idxs, dtype=int))
            Ks.append(len(idxs))
        self._Ks = np.array(Ks, dtype=int)

        # ---- base class (user 1-indexed, internal 0-indexed) ----
        if base_class is None:
            self.base_class = C - 1
        else:
            bc_user = int(base_class)
            if not (1 <= bc_user <= C):
                raise ValueError(f"base_class must be in [1, {C}].")
            self.base_class = bc_user - 1
        self._intercept_free_classes = [c for c in range(C) if c != self.base_class]
        self._n_inter = len(self._intercept_free_classes)

        # ---- membership ----
        self._has_membership = member_params_spec is not None
        self.member_params_spec = member_params_spec
        if self._has_membership:
            if len(member_params_spec) != C:
                raise ValueError(f"member_params_spec must have length {C}.")
            covariate_vars = sorted({v for arr in member_params_spec
                                      for v in arr if v != "_inter"})
            self.membership_vars = covariate_vars
            self.K_membership = len(covariate_vars)
            mem_idx = [varnames.index(v) for v in covariate_vars]
            var_to_col = {v: i for i, v in enumerate(covariate_vars)}

            self._member_mask = np.zeros((C, self.K_membership))
            for c in range(C):
                for v in member_params_spec[c]:
                    if v != "_inter" and v in var_to_col:
                        self._member_mask[c, var_to_col[v]] = 1.0

            # Identification: no variable can appear in ALL classes.
            for v_idx, v in enumerate(covariate_vars):
                if self._member_mask[:, v_idx].all():
                    raise ValueError(
                        f"Membership variable '{v}' appears in all {C} "
                        f"classes. The softmax is invariant to logit shifts, "
                        f"so '{v}' must be excluded from at least one class "
                        f"(typically the base class {self.base_class + 1})."
                    )

            Zm = np.zeros((self.N, self.K_membership))
            for i in range(self.N):
                t0 = int(np.argmax(mask[i]))
                Zm[i] = Xp[i, t0, 0, mem_idx]
            self.X_membership = Zm
        else:
            self.membership_vars = []
            self.K_membership = 0
            self._member_mask = np.zeros((C, 0))
            self.X_membership = None

        # ---- normalization (z-score cont., /sigma dummies, ASCs raw) ----
        Xflat = Xp.reshape(-1, self.K)
        mu = np.zeros(self.K)
        sigma = np.ones(self.K)
        for k, v in enumerate(varnames):
            v_str = str(v)
            if v_str.startswith("intercept.") or v_str == "intercept":
                continue
            col = Xflat[:, k]
            u = np.unique(col[np.isfinite(col)])
            s = float(np.std(col))
            if len(u) <= 2:
                if s > 1e-12:
                    sigma[k] = s
                # mu stays 0
            else:
                if s > 1e-12:
                    sigma[k] = s
                    mu[k] = float(np.mean(col))
        self._norm_mu = mu
        self._norm_sigma = sigma

        # Apply normalization once, never undo
        Xp = (Xp - mu[None, None, None, :]) / sigma[None, None, None, :]
        self.X = Xp
        if self.X_membership is not None and self.K_membership > 0:
            mem_idx_arr = np.array(
                [varnames.index(v) for v in self.membership_vars], dtype=int
            )
            self.X_membership = (
                (self.X_membership - mu[mem_idx_arr][None, :])
                / sigma[mem_idx_arr][None, :]
            )
            self._mem_sigma = sigma[mem_idx_arr]
        else:
            self._mem_sigma = np.ones(0)

        self._class_sigma = [sigma[self._class_specs[c]] for c in range(C)]

        # ---- device arrays ----
        self._Xd = jnp.asarray(self.X)
        self._yd = jnp.asarray(self.y)
        self._avd = jnp.asarray(self.avail)
        self._maskd = jnp.asarray(self.panel_mask)
        self._Zmd = jnp.asarray(self.X_membership) if self.X_membership is not None else None
        self._member_mask_d = jnp.asarray(self._member_mask)

        # ---- resolve betas0/inter0/gamma0 (original scale) -> normalized ----
        self._betas0_norm, self._inter0, self._gamma0_norm = self._resolve_inits(
            betas0, inter0, gamma0
        )

        return self

    def _resolve_inits(self, betas0, inter0, gamma0):
        C = self.n_classes
        # betas
        b_norm = None
        if betas0 is not None:
            if isinstance(betas0, list) and betas0 and isinstance(betas0[0], dict):
                rng = np.random.default_rng(self.random_state + 9999)

                def _get_beta(dom, v):
                    # try exact, then strip 'intercept.', then add it
                    if v in dom:
                        return float(dom[v])
                    if v.startswith("intercept."):
                        short = v.split(".", 1)[1]
                        for alt in (short, f"ASC_{short}", f"asc_{short}"):
                            if alt in dom:
                                return float(dom[alt])
                    else:
                        for alt in (f"intercept.{v}", f"ASC_{v}", f"asc_{v}"):
                            if alt in dom:
                                return float(dom[alt])
                    return float(rng.normal(scale=0.05))

                b_orig = []
                for c in range(C):
                    names_c = [self.varnames[i] for i in self._class_specs[c]]
                    dom = betas0[c] if c < len(betas0) else {}
                    vals = [_get_beta(dom, v) for v in names_c]
                    b_orig.append(np.array(vals, dtype=float))
            else:
                b_orig = [np.asarray(b, dtype=float) for b in betas0]
            b_norm = [b_orig[c] * self._class_sigma[c] for c in range(C)]

        # gammas FIRST (the intercept correction needs them)
        g_norm, g_orig = None, None
        if gamma0 is not None:
            if isinstance(gamma0, list) and gamma0 and isinstance(gamma0[0], dict):
                mem_vars = self.membership_vars
                g_orig = np.zeros((C, self.K_membership))
                for c in range(C):
                    dom = gamma0[c] if c < len(gamma0) else {}
                    for k, v in enumerate(mem_vars):
                        if v in dom:
                            g_orig[c, k] = dom[v]
            else:
                g_orig = np.asarray(gamma0, dtype=float)
            if g_orig.shape == (C, self.K_membership):
                g_norm = g_orig * self._mem_sigma[None, :]

        # intercepts: ALWAYS reported scale -> convert to internal
        inter = None
        if inter0 is not None:
            if isinstance(inter0, dict):
                inter = np.array(
                    [inter0.get(c, 0.0) for c in self._intercept_free_classes],
                    dtype=float,
                )
            else:
                inter_arr = np.asarray(inter0, dtype=float)
                if inter_arr.shape == (self._n_inter,):
                    inter = inter_arr.copy()

            if inter is not None and g_orig is not None and self.K_membership > 0:
                mem_idx = [self.varnames.index(v) for v in self.membership_vars]
                mu_Z = self._norm_mu[mem_idx]

                def _corr(c):
                    return float(np.sum(g_orig[c] * mu_Z))

                corr_base = _corr(self.base_class)
                for i, c in enumerate(self._intercept_free_classes):
                    inter[i] = inter[i] + _corr(c) - corr_base

        return b_norm, inter, g_norm

    # ------------------------------------------------------------------
    # KERNELS
    # ------------------------------------------------------------------
    def _class_ll(self, beta_active, c):
        idx = jnp.asarray(self._class_specs[c])
        Xc = jnp.take(self._Xd, idx, axis=3)
        util = jnp.einsum('npjk,k->npj', Xc, beta_active)
        util = jnp.where(self._avd > 0, util, -1e10)
        util = util - jnp.max(util, axis=2, keepdims=True)
        expu = jnp.exp(util) * self._avd
        denom = jnp.clip(jnp.sum(expu, axis=2, keepdims=True), MIN_COMP)
        logp = util - jnp.log(denom)
        ll_np = jnp.sum(self._yd * logp, axis=2)
        return jnp.sum(ll_np * self._maskd, axis=1)

    def _class_ll_matrix(self, betas):
        return jnp.stack([self._class_ll(betas[c], c) for c in range(self.n_classes)], axis=1)

    def _membership_logits(self, inter, gamma):
        C = self.n_classes
        logits = jnp.zeros((self.N, C))
        for i, c in enumerate(self._intercept_free_classes):
            logits = logits.at[:, c].set(inter[i])
        if self.K_membership > 0:
            gamma_m = gamma * self._member_mask_d
            logits = logits + self._Zmd @ gamma_m.T
        return logits

    def _membership_log_probs(self, inter, gamma):
        logits = self._membership_logits(inter, gamma)
        logits = logits - jnp.max(logits, axis=1, keepdims=True)
        return logits - jnp.log(jnp.clip(jnp.sum(jnp.exp(logits), axis=1, keepdims=True), MIN_COMP))

    def _estep(self, betas, inter, gamma, pi):
        ll_c = self._class_ll_matrix(betas)
        if self._has_membership:
            logH = self._membership_log_probs(inter, gamma)
        else:
            logpi = jnp.log(jnp.clip(pi, MIN_COMP))
            logH = jnp.broadcast_to(logpi[None, :], (self.N, self.n_classes))
        num = ll_c + logH
        denom = jax.scipy.special.logsumexp(num, axis=1, keepdims=True)
        R = jnp.exp(num - denom)
        return R, jnp.sum(denom)
    def _make_trace_cb(self, kinds):
        def _cb(it, ll, dll, gn, done, stag):
            it_i = int(it)
            ll_np   = np.asarray(ll)
            dll_np  = np.asarray(dll)
            gn_np   = np.asarray(gn)
            done_np = np.asarray(done)
            stag_np = np.asarray(stag)
            C = ll_np.shape[0]

            if it_i == 0:
                hdr = f"{'it':>4} |"
                for k in range(C):
                    kk = kinds[k] if k < len(kinds) else "?"
                    hdr += f"  chain {k} ({kk:<12})                            |"
                print(hdr)
                print("-" * len(hdr))

            line = f"{it_i + 1:>4} |"
            for k in range(C):
                tag = ""
                if done_np[k]:
                    tag = " DONE"
                elif stag_np[k] >= self.max_stagnant:
                    tag = " STAG"
                elif gn_np[k] > 1e-3 and dll_np[k] < 1e-8:
                    tag = " grad-stuck"
                line += (f" {ll_np[k]:>11.3f}  {dll_np[k]:>8.2e}  "
                        f"{gn_np[k]:>8.2e}{tag:<12} |")
            print(line)
        return _cb
    def _build_kernels(self):
        class_fgh, class_fgh_raw = [], []
        for c in range(self.n_classes):
            def nll(beta, w, c=c):
                return -jnp.sum(w * self._class_ll(beta, c))
            raw = (jax.jit(nll), jax.jit(grad(nll)), jax.jit(hessian(nll)))
            class_fgh_raw.append(raw)
            if self.choice_correction:
                def info(beta, w, c=c):
                    H = hessian(lambda b: -nll(b, w, c=c))(beta)
                    return -H + jnp.eye(beta.shape[0], dtype=beta.dtype) * 1e-12
                def val(beta, w, c=c):
                    I = info(beta, w, c=c)
                    _, ld = jnp.linalg.slogdet(I)
                    return nll(beta, w, c=c) - 0.5 * ld
                class_fgh.append((jax.jit(val), jax.jit(grad(val)), jax.jit(info)))
            else:
                class_fgh.append(raw)

        n_inter = self._n_inter
        Km = self.K_membership
        C = self.n_classes

        def mem_ll(v, R):
            inter = v[:n_inter]
            gamma = v[n_inter:].reshape(C, Km)
            return jnp.sum(R * self._membership_log_probs(inter, gamma))

        def mem_nll(v, R):
            return -mem_ll(v, R)

        def mem_info(v, R):
            H = hessian(lambda vv: mem_ll(vv, R))(v)
            return -H + jnp.eye(v.shape[0], dtype=v.dtype) * 1e-12

        mem_fgh_raw = (jax.jit(mem_nll), jax.jit(grad(mem_nll)), jax.jit(hessian(mem_nll)))

        if self.membership_correction:
            def mem_val(v, R):
                I = mem_info(v, R)
                _, ld = jnp.linalg.slogdet(I)
                return -(mem_ll(v, R) + 0.5 * ld)
            mem_fgh = (jax.jit(mem_val), jax.jit(grad(mem_val)), jax.jit(mem_info))
        else:
            mem_fgh = mem_fgh_raw

        self._class_fgh = class_fgh
        self._class_fgh_raw = class_fgh_raw
        self._mem_fgh = mem_fgh
        self._mem_fgh_raw = mem_fgh_raw

    # ------------------------------------------------------------------
    # SEPARATION STRUCTURE VECTORS (used by Konis)
    # ------------------------------------------------------------------
    def _separation_xbar(self, c, betas, w_i):
        idx = self._class_specs[c]
        Xc = np.asarray(self.X)[:, :, :, idx]
        yc = np.asarray(self.y)
        avail = np.asarray(self.avail)
        real = np.asarray(self.panel_mask) > 0
        j_chosen = np.argmax(yc, axis=2)
        x_chosen = np.take_along_axis(
            Xc, j_chosen[:, :, None, None], axis=2)[:, :, 0, :]
        rows = []
        for j in range(self.J):
            m = (avail[:, :, j] > 0) & (j_chosen != j) & real
            diff = x_chosen - Xc[:, :, j, :]
            w = w_i[:, None] * m
            rows.append((w[:, :, None] * diff).reshape(-1, diff.shape[-1]))
        return np.concatenate(rows, axis=0)

    def _separation_xbar_membership(self, s, R_last):
        if not self._has_membership:
            return np.zeros((0, 0))
        active = (np.where(self._member_mask[s] > 0)[0]
                  if self.K_membership > 0 else np.array([], dtype=int))
        ones = np.ones((self.N, 1))
        Z = (np.hstack([ones, np.asarray(self.X_membership)[:, active]])
             if active.size else ones)
        R = np.asarray(R_last)
        rows = []
        for sp in range(R.shape[1]):
            if sp == s:
                continue
            rows.append(R[:, s][:, None] * Z)
            rows.append(-R[:, sp][:, None] * Z)
        return np.concatenate(rows, axis=0)

    # ------------------------------------------------------------------
    # FIT -- vmap-batched multistart
    # ------------------------------------------------------------------
    def fit(self):
        t0 = time.time()
        C = self.n_classes
        n_init = self.n_init

        self._build_kernels()

        rng = np.random.default_rng(self.random_state)

        # ---- init schedule: 10% exact, 30% jitter, 60% random ----
        b0 = self._betas0_norm
        i0 = self._inter0
        g0 = self._gamma0_norm
        has_b = b0 is not None
        has_i = i0 is not None
        has_g = g0 is not None

        n_exact_b  = int(np.ceil(0.10 * n_init)) if has_b else 0
        n_jitter_b = int(np.ceil(0.30 * n_init)) if has_b else 0
        jit_scale_b = np.linspace(0.2, 1.0, n_jitter_b) if n_jitter_b else np.array([])

        n_exact_i  = int(np.ceil(0.10 * n_init)) if has_i else 0
        n_jitter_i = int(np.ceil(0.30 * n_init)) if has_i else 0
        jit_scale_i = np.linspace(0.2, 1.0, n_jitter_i) if n_jitter_i else np.array([])

        n_exact_g  = int(np.ceil(0.10 * n_init)) if has_g else 0
        n_jitter_g = int(np.ceil(0.30 * n_init)) if has_g else 0
        jit_scale_g = np.linspace(0.2, 1.0, n_jitter_g) if n_jitter_g else np.array([])

        b_stack, i_list, g_list = [], [], []
        kinds = []
        for k in range(n_init):
            rng = np.random.default_rng(self.random_state + k)
            # betas
            if has_b and k < n_exact_b:
                b_k = [jnp.asarray(b) for b in b0]
                kind_b = "memory"
            elif has_b and k < n_exact_b + n_jitter_b:
                s = float(jit_scale_b[k - n_exact_b])
                b_k = [jnp.asarray(b + rng.normal(scale=s, size=np.asarray(b).shape))
                       for b in b0]
                kind_b = "memory_jitter"
            else:
                #b_k = [jnp.asarray(rng.normal(scale=1.0, size=int(kc)))
                 #      for kc in self._Ks]
                b_k = [jnp.asarray(rng.normal(scale=0.05, size=int(kc)) * self._class_sigma[c])
                       for c, kc in enumerate(self._Ks)]
                kind_b = "random"
            # inter
            if has_i and k < n_exact_i:
                i_k = jnp.asarray(i0)
            elif has_i and k < n_exact_i + n_jitter_i:
                s = float(jit_scale_i[k - n_exact_i])
                i_k = jnp.asarray(i0 + rng.normal(scale=s, size=self._n_inter))
            else:
                #i_k = jnp.asarray(rng.normal(scale=0.5, size=self._n_inter))
                i_k = jnp.asarray(rng.normal(scale=0.01, size=self._n_inter))
            # gamma
            if has_g and k < n_exact_g:
                g_k = jnp.asarray(g0)
            elif has_g and k < n_exact_g + n_jitter_g:
                s = float(jit_scale_g[k - n_exact_g])
                g_k = jnp.asarray(g0 + rng.normal(scale=s, size=(C, self.K_membership)))
            else:
                #g_k = jnp.asarray(rng.normal(scale=0.5, size=(C, self.K_membership)))
                g_k = jnp.asarray(rng.normal(scale=0.01, size=(C, self.K_membership))
                                   * self._mem_sigma[None, :])

            b_stack.append(b_k)
            i_list.append(i_k)
            g_list.append(g_k)
            kinds.append(kind_b)
            
            #print(f"[FASTNEW init {k}] "
             #   f"betas={[np.asarray(b) for b in b_stack[k]]} "
              #  f"inter={np.asarray(i_list[k])} "
               # f"gamma={np.asarray(g_list[k])}", flush=True)
        betas_b = [jnp.stack([b_stack[k][c] for k in range(n_init)]) for c in range(C)]
        inter_b = jnp.stack(i_list)
        gamma_b = jnp.stack(g_list)
        pi_b = jnp.full((n_init, C), 1.0 / C)

        # ---- vmap wrappers ----
        def _one_step(betas, inter, gamma, pi):
            return _fast_em_step(self, betas, inter, gamma, pi)
        step_vmapped = jax.vmap(_one_step, in_axes=([0] * C, 0, 0, 0))

        def _estep_batched(betas, inter, gamma, pi):
            def _e(b, i, g, p):
                return self._estep(b, i, g, p)
            return jax.vmap(_e, in_axes=([0] * C, 0, 0, 0))(betas, inter, gamma, pi)

        # ---- outer body: SQUAREM + kill-on-stagnation + plateau-kill ----
        def outer_body(carry):
            (betas, inter, gamma, pi, ll_prev, done, conv, n_stag, n_plat,
             ll_best, it, gn_prev, last_dll, last_step) = carry
            b1, i1, g1, p1, ll1, _, acc1, gn1 = step_vmapped(betas, inter, gamma, pi)
            b2, i2, g2, p2, ll2, _, acc2, gn2 = step_vmapped(b1, i1, g1, p1)

            th0 = _fast_flatten(betas, inter, gamma, pi)
            th1 = _fast_flatten(b1, i1, g1, p1)
            th2 = _fast_flatten(b2, i2, g2, p2)

            r = th1 - th0
            v = th2 - 2.0 * th1 + th0
            nv = jnp.linalg.norm(v, axis=-1)
            nr = jnp.linalg.norm(r, axis=-1)
            use_sq = nv > 1e-14
            alpha_init = jnp.minimum(-nr / jnp.maximum(nv, 1e-14), -1.0).reshape(-1, 1)

            def ls_cond(c):
                _, _, _, accepted, j = c
                return (j < 10) & (~jnp.all(accepted))
            def ls_body(c):
                alpha, th_acc, ll_acc, accepted, j = c
                th_p = th0 - 2.0 * alpha * r + alpha ** 2 * v
                bp, ip, gp, pp = _fast_unflatten(th_p, self)
                _, ll_p = _estep_batched(bp, ip, gp, pp)
                ok = jnp.isfinite(ll_p) & (ll_p >= ll2)
                take = (~accepted) & ok
                th_acc = jnp.where(take[:, None], th_p, th_acc)
                ll_acc = jnp.where(take, ll_p, ll_acc)
                accepted = accepted | ok
                return (alpha - 1.0) / 2.0, th_acc, ll_acc, accepted, j + 1

            _, th_acc, ll_acc, accepted, _ = lax.while_loop(
                ls_cond, ls_body,
                (alpha_init, th2, ll2,
                 jnp.zeros((n_init,), dtype=bool), jnp.array(0)),
            )
            th_sq = jnp.where(accepted[:, None], th_acc, th2)
            ll_sq = jnp.where(accepted, ll_acc, ll2)
            th_f = jnp.where(use_sq[:, None], th_sq, th2)
            ll_f = jnp.where(use_sq, ll_sq, ll2)

            b_f, i_f, g_f, p_f = _fast_unflatten(th_f, self)

            def _gn_at(b, i, g, p):
                R, _ = self._estep(b, i, g, p)
                gn_max = jnp.array(0.0)
                for c in range(C):
                    _, g_j, _ = self._class_fgh[c]
                    gc = g_j(b[c], R[:, c])
                    gn_max = jnp.maximum(gn_max, jnp.max(jnp.abs(gc)))
                if self._has_membership:
                    _, g_m, _ = self._mem_fgh
                    v = jnp.concatenate([i, g.ravel()])
                    gm = g_m(v, R)
                    gn_max = jnp.maximum(gn_max, jnp.max(jnp.abs(gm)))
                return gn_max

            gn_f = jax.vmap(_gn_at, in_axes=([0] * C, 0, 0, 0))(b_f, i_f, g_f, p_f)

            # ---- convergence (ftol relative to ll scale) ----
            dll = jnp.abs(ll_f - ll_prev)
            ftol_eff = self.ftol * (1.0 + jnp.abs(ll_f))
            converged_here = (dll < ftol_eff) & (gn_f < self.gtol)
            conv_new = conv | converged_here

            # ---- stagnation (as before) ----
            step_norm = jnp.linalg.norm(th_f - th0, axis=-1)
            grad_stuck = (gn_prev - gn_f) < (1e-3 * gn_f + 1e-12)
            stalled = (dll < ftol_eff) & (step_norm < self.step_tol) & grad_stuck
            n_stag_new = jnp.where(stalled, n_stag + 1, 0)
            kill_stag = n_stag_new >= self.max_stagnant

            # ---- plateau kill (Option B) ----
            improved = ll_f > (ll_best + self.plateau_tol)
            n_plat_new = jnp.where(improved, 0, n_plat + 1)
            ll_best_new = jnp.maximum(ll_best, ll_f)
            kill_plat = n_plat_new >= self.max_plateau
            plateau_ok = kill_plat & (gn_f < self.gtol)
            conv_new = conv_new | plateau_ok

            done_new = done | converged_here | kill_stag | kill_plat

            already = done
            b_f = [jnp.where(already[:, None], betas[c], b_f[c]) for c in range(C)]
            i_f = jnp.where(already[:, None], inter, i_f)
            g_f = jnp.where(already[:, None, None], gamma, g_f)
            p_f = jnp.where(already[:, None], pi, p_f)
            ll_f = jnp.where(already, ll_prev, ll_f)
            n_stag_new = jnp.where(already, n_stag, n_stag_new)
            n_plat_new = jnp.where(already, n_plat, n_plat_new)
            ll_best_new = jnp.where(already, ll_best, ll_best_new)

            # Report values for the final iteration of each chain. A chain
            # already marked done has its state frozen by the `where` above,
            # so its true delta this step is 0 — report that, not the raw
            # pre-freeze value, which would be a bogus non-zero.
            # For already-done chains, preserve the dll/step captured at the
            # iteration they first became done (carried via last_dll/last_step
            # in the loop carry). Without this, the winner's final ΔLL is
            # always reported as 0.0 because the outer loop keeps running
            # until the LAST chain stops — long after the winner converged.
            dll_report = jnp.where(already, last_dll, dll)
            step_report = jnp.where(already, last_step, step_norm)

            if self.debug_trace:
                jax.debug.callback(
                    self._trace_cb,
                    it, ll_f, dll, gn_f, done_new, n_stag_new,
                )

            return (b_f, i_f, g_f, p_f, ll_f, done_new, conv_new,
                    n_stag_new, n_plat_new, ll_best_new, it + 1, gn_f,
                    dll_report, step_report)

        def outer_cond(carry):
            (_, _, _, _, _, done, _conv, _ns, _np, _lb, it, _gn,
             _dll, _step) = carry
            return (it < self.maxiter) & (~jnp.all(done))
        
        # === DEBUG TRACE (one-shot) ===
        if self.debug_trace:
            self._trace_cb = self._make_trace_cb(kinds)

        ll_init = jnp.full((n_init,), -jnp.inf)
        done_init = jnp.zeros((n_init,), dtype=bool)
        conv_init = jnp.zeros((n_init,), dtype=bool)
        stag_init = jnp.zeros((n_init,), dtype=jnp.int32)
        plat_init = jnp.zeros((n_init,), dtype=jnp.int32)
        llb_init = jnp.full((n_init,), -jnp.inf)
        gn_init = jnp.full((n_init,), jnp.inf)
        dll_init = jnp.full((n_init,), jnp.inf)
        step_init = jnp.full((n_init,), jnp.inf)

        (betas_f, inter_f, gamma_f, pi_f, ll_f,
         done_f, conv_f, stag_f, plat_f, llb_f, it_f, gn_final,
         dll_final, step_final) = lax.while_loop(
            outer_cond, outer_body,
            (betas_b, inter_b, gamma_b, pi_b,
             ll_init, done_init, conv_init, stag_init, plat_init, llb_init,
             jnp.array(0), gn_init, dll_init, step_init),
        )
        jax.block_until_ready(ll_f)

        # ---- posterior per chain -> shares -> degeneracy screen ----
        R_all, _ = _estep_batched(betas_f, inter_f, gamma_f, pi_f)
        shares_all = np.asarray(R_all.mean(axis=1))

        degen_flags = np.array([
            is_degenerate_candidate(
                [np.asarray(betas_f[c][k]) for c in range(C)],
                shares_all[k],
                min_share=0.01,
                max_abs_beta=DEGEN_MAX_ABS_BETA,
            ) for k in range(n_init)
        ])

        # ---- per-chain Konis separation test ----
        R_np = np.asarray(R_all)                            # (n_init, N, C)
        sep_choice = np.zeros(n_init, dtype=bool)
        sep_member = np.zeros(n_init, dtype=bool)
        for k in range(n_init):
            betas_k = [np.asarray(betas_f[c][k]) for c in range(C)]
            for c in range(C):
                xbar = self._separation_xbar(c, betas_k, R_np[k, :, c])
                if xbar.shape[0] and konis_separation_test(xbar)["separated"]:
                    sep_choice[k] = True
                    break
            if self._has_membership:
                for c in self._intercept_free_classes:
                    xbar_m = self._separation_xbar_membership(c, R_np[k])
                    if xbar_m.shape[0] and konis_separation_test(xbar_m)["separated"]:
                        sep_member[k] = True
                        break
                        # ─────────────────────────────────────────────────────────────
        # [RAW-DUMP-BLOCK-START] All 20 basins, all params
        # ─────────────────────────────────────────────────────────────
        # Solo imprime. Escala NORMALIZADA. Sirve para inspeccionar
        # visualmente label switching, dispersión y modos.
        # ─────────────────────────────────────────────────────────────
        # RAW-DUMP controlado por flag dedicado, independiente de self.verbose.
        # Poner en False para silenciar el dump por-basin (betas/gammas de
        # los 20 starting values) sin tocar el resto de la salida verbose.
        _raw_dump_verbose = False

        if _raw_dump_verbose:
            print("=" * 78)
            print(f"  [RAW-DUMP] Ensemble crudo — {n_init} starting values, "
                  f"K={C}, escala normalizada")
            print("=" * 78)

            print("  Class specs:")
            for c in range(C):
                names_c = [self.varnames[i] for i in self._class_specs[c]]
                print(f"    C{c+1}: {names_c}")
            if self._has_membership:
                print(f"  Membership vars: {self.membership_vars}")
                print(f"  Free-intercept classes: "
                      f"{[c+1 for c in self._intercept_free_classes]}")
            print()

            # Orden ascendente por k (0..19). Cambiar a sorted(range(n_init),
            # key=lambda k: -ll_f[k]) si quieres ver primero los mejores LL.
            for k in range(n_init):
                flags_k = []
                if degen_flags[k]:  flags_k.append("DEGEN")
                if sep_choice[k]:   flags_k.append("SEP_C")
                if sep_member[k]:   flags_k.append("SEP_M")
                if int(plat_f[k]) >= self.max_plateau:
                    flags_k.append("PLAT")
                elif int(stag_f[k]) >= self.max_stagnant:
                    flags_k.append("STAG")
                if not bool(conv_f[k]):
                    flags_k.append("MAXIT")
                tag = "|".join(flags_k) if flags_k else "CLEAN"

                shares_k = np.asarray(R_np[k]).mean(axis=0)
                sh_str = " ".join(f"{s:.3f}" for s in shares_k)

                print(f"  ── sv={k:<2}  ll={float(ll_f[k]):>10.4f}  "
                      f"[{tag}]  shares=[{sh_str}]")

                for c in range(C):
                    names_c = [self.varnames[i] for i in self._class_specs[c]]
                    vals = np.asarray(betas_f[c][k])
                    body = "  ".join(f"{n}={v:>+8.4f}"
                                     for n, v in zip(names_c, vals))
                    print(f"      C{c+1} β:  {body}")

                if self._has_membership and self._n_inter > 0:
                    inter_vals = np.asarray(inter_f[k])
                    body = "  ".join(
                        f"C{c+1}={inter_vals[i]:>+8.4f}"
                        for i, c in enumerate(self._intercept_free_classes))
                    print(f"      inter: {body}")

                if self._has_membership and self.K_membership > 0:
                    gv = np.asarray(gamma_f[k])
                    for c in range(C):
                        active = [(j, v) for j, v in enumerate(self.membership_vars)
                                  if self._member_mask[c, j] > 0]
                        if not active:
                            continue
                        body = "  ".join(f"{v}={gv[c, j]:>+8.4f}"
                                         for j, v in active)
                        print(f"      C{c+1} γ:  {body}")
                print()
            print("=" * 78)
        # ─────────────────────────────────────────────────────────────
        # [RAW-DUMP-BLOCK-END] All 20 basins, all params
        # ─────────────────────────────────────────────────────────────
        # ─────────────────────────────────────────────────────────────
        # [REVIEW-BLOCK] Per-basin block-Hessian ridge screen (clean only)
        # ─────────────────────────────────────────────────────────────
        # Diagnóstico principled de identificabilidad. Para cada basin
        # (cadena) en sus parámetros FINALES ya fitados, computa el
        # Hessiano de bloque por clase (dim Kc ≤ 6) y el de membresía
        # activa (dim n_inter + Σ activas), y evalúa λ_min/λ_max.
        #
        #   ridge real → λ_min/λ_max → 0   (típicamente < 1e-8)
        #   modo sano  → λ_min/λ_max > 1e-3
        #
        # IMPORTANTE: sólo se evalúan basins NO excluidos por Konis
        # (sep_choice / sep_member) ni por degeneración — ésos ya no
        # pueden ser elegidos aguas abajo (`eligible` los descarta), así
        # que diagnosticarlos es cómputo tirado. Entre los no-flagueados,
        # sólo importan los top-N por LL (los otros nunca ganarían).
        #
        # Loop Python sobre kernels jit YA cacheados (sin vmap): vmap
        # fuerza recompilación (~30-40 s) porque el shape batched no
        # coincide con el shape de las llamadas EM. Llamar h_j directo
        # reutiliza la traza existente.
        # ─────────────────────────────────────────────────────────────
        # SWITCH TEMPORAL: poner en False para saltarse el screening completo
        # (benchmark de tiempo). Dejar en True para producción.
        _review_active    = True
        _review_verbose   = True
        _review_ratio_min = 1e-6
        _review_top_n     = min(5, n_init)

        sep_reviewed = np.zeros(n_init, dtype=bool)

                # ─────────────────────────────────────────────────────────────
        # [REVIEW-BLOCK] Per-basin block-Hessian ridge screen (clean only)
        # ─────────────────────────────────────────────────────────────
        # Flujo:
        #   1. Konis/degen → descarta los ya excluidos
        #   2. Pre-filtro NECESARIO: |θ|max > log(100) ≈ 4.605. En escala
        #      normalizada (σ_X = 1), esto significa exp(θ) > 100 →
        #      la clase predice la elección con P > 99% en alguna región
        #      del espacio de covariables. Condición necesaria para que
        #      el Hessiano sea singular. Derivado de MNL, no calibrado.
        #   3. Ordenar los viables por LL descendente
        #   4. Iterar en orden: si |θ| < log(100) → clean automático,
        #      parar. Si |θ| ≥ log(100) → Hessian. Si clean → parar.
        #      Si RIDGE → siguiente. Sin tope: el primer clean es el
        #      winner (LL descendente garantiza que ninguno posterior
        #      puede ganarle).
        #   5. Belsley-Kuh-Welsch para atribuir el ridge a variables.
        # Sin vmap: loop sobre kernels jit YA cacheados (shapes sin batch
        # coinciden con las llamadas EM → hit de caché).
        # ─────────────────────────────────────────────────────────────
        _review_active    = True
        _review_verbose   = self.verbose

        # Numerical-rank criterion: singular at machine precision (sqrt(eps))
        _review_ratio_min = float(np.sqrt(np.finfo(float).eps))
        _review_abs_thresh = float(np.log(100.0))   # ≈ 4.605

        sep_reviewed = np.zeros(n_init, dtype=bool)

        if _review_active or _review_verbose:
            R_np = np.asarray(R_all)
            ridge_class_basin  = np.zeros(n_init, dtype=bool)
            ridge_member_basin = np.zeros(n_init, dtype=bool)
            ratio_by_class     = np.full((n_init, C), np.nan)
            ratio_member       = np.full(n_init,        np.nan)
            culprits_class     = {k: [] for k in range(n_init)}
            culprits_member    = {k: [] for k in range(n_init)}
            _evaluated         = np.zeros(n_init, dtype=bool)
            hde_basin          = np.zeros(n_init, dtype=bool)
            dh_class, dh_mem   = self._hde_kernels()
            

            # ---- Membresía: índices activos + nombres, una vez ----
            if self._has_membership:
                n_inter = self._n_inter
                Km      = self.K_membership
                active_idx = list(range(n_inter))
                for c in range(C):
                    for kk in range(Km):
                        if self._member_mask[c, kk] > 0:
                            active_idx.append(n_inter + c * Km + kk)
                active_idx = np.array(active_idx, dtype=int)
                mem_names_active = []
                for c in self._intercept_free_classes:
                    mem_names_active.append(f"inter_C{c+1}")
                for c in range(C):
                    for kk in range(Km):
                        if self._member_mask[c, kk] > 0:
                            mem_names_active.append(
                                f"C{c+1}:{self.membership_vars[kk]}")
            else:
                active_idx = np.array([], dtype=int)
                mem_names_active = []

            # ---- Orden de evaluación: no-Konis/degen, LL desc ----
            _already_flagged = degen_flags | sep_choice | sep_member
            _viable_idx = np.where(~_already_flagged)[0]
            if _viable_idx.size == 0:
                _order = np.array([], dtype=int)
            else:
                _order = _viable_idx[np.argsort(-np.asarray(ll_f)[_viable_idx])]

            # ---- Evaluación incremental hasta el primer clean ----
            for k in _order:
                _evaluated[k] = True      
                basin_is_ridge = False

                # Clases
                for c in range(C):
                    h_j    = self._class_fgh_raw[c][2]
                    beta_k = jnp.asarray(betas_f[c][k], dtype=jnp.float64)
                    w_k    = jnp.asarray(R_np[k, :, c], dtype=jnp.float64)
                    H      = np.asarray(h_j(beta_k, w_k))
                    ev     = np.linalg.eigvalsh(H)
                    ratio  = float(ev[0]) / max(float(ev[-1]), 1e-300)
                    ratio_by_class[k, c] = ratio
                    if ratio < _review_ratio_min:
                        ridge_class_basin[k] = True
                        basin_is_ridge = True
                        _, flag_bkw, _ = collinearity_diag(
                            H, ratio_thresh=_review_ratio_min)
                        names_c = [self.varnames[i]
                                   for i in self._class_specs[c]]
                        for j, name in enumerate(names_c):
                            if flag_bkw[j]:
                                culprits_class[k].append(f"C{c+1}:{name}")
                    else:
                        hde = hauck_donner_flags(
                            H, np.asarray(dh_class[c](beta_k, w_k)), np.asarray(beta_k))
                        if hde.any():
                            hde_basin[k] = True
                            basin_is_ridge = True
                            for j, i in enumerate(self._class_specs[c]):
                                if hde[j]:
                                    culprits_class[k].append(
                                        f"C{c+1}:{self.varnames[i]} [HDE]")
                            

                # Membresía
                if self._has_membership and active_idx.size > 0:
                    h_m   = self._mem_fgh_raw[2]
                    v_k   = jnp.asarray(np.concatenate(
                                [np.asarray(inter_f[k]),
                                 np.asarray(gamma_f[k]).ravel()]),
                                dtype=jnp.float64)
                    R_k   = jnp.asarray(R_np[k], dtype=jnp.float64)
                    H_m   = np.asarray(h_m(v_k, R_k))
                    H_act = H_m[np.ix_(active_idx, active_idx)]
                    ev_m  = np.linalg.eigvalsh(H_act)
                    ratio = float(ev_m[0]) / max(float(ev_m[-1]), 1e-300)
                    ratio_member[k] = ratio
                    if ratio < _review_ratio_min:
                        ridge_member_basin[k] = True
                        basin_is_ridge = True
                        _, flag_bkw, _ = collinearity_diag(
                            H_act, ratio_thresh=_review_ratio_min)
                        for j, name in enumerate(mem_names_active):
                            if flag_bkw[j]:
                                culprits_member[k].append(name)
                    else:
                        dH_m = np.asarray(dh_mem(v_k, R_k))
                        hde = hauck_donner_flags(
                            H_act, dH_m[np.ix_(active_idx, active_idx, active_idx)],
                            np.asarray(v_k)[active_idx])
                        if hde.any():
                            hde_basin[k] = True
                            basin_is_ridge = True
                            for j, name in enumerate(mem_names_active):
                                if hde[j]:
                                    culprits_member[k].append(f"{name} [HDE]")
                if not basin_is_ridge:
                    break   # primer clean → winner

            if _review_active:
                sep_reviewed = ridge_class_basin | ridge_member_basin | hde_basin

            if _review_verbose:
                n_eval    = int(_evaluated.sum())
                n_flag    = int(sep_reviewed.sum())
                print("=" * 67)
                print(f"  [REVIEW] Block-Hessian ridge + Hauck-Donner screen")
                print(f"    λ_min/λ_max < {_review_ratio_min:.1e} (sqrt eps)  →  RIDGE")
                print(f"    dz/dθ < 0 and Wald n.s. (α = 0.05)        →  HDE")
                print(f"    Evaluated: {n_eval} / {n_init}   "
                      f"Flagged: {n_flag}")
                print("=" * 67)
                for k in range(n_init):
                    if not _evaluated[k]:
                        continue
                    tag = ("RIDGE" if (ridge_class_basin[k] or ridge_member_basin[k])
                           else " HDE " if hde_basin[k] else "  OK ")
                    cls_body = "  ".join(
                        f"C{c+1}:{ratio_by_class[k, c]:.1e}"
                        for c in range(C)
                        if np.isfinite(ratio_by_class[k, c])
                    )
                    mem_body = (f"  mem:{ratio_member[k]:.1e}"
                                if np.isfinite(ratio_member[k]) else "")
                    print(f"  k={k:<2}  ll={float(ll_f[k]):>10.2f}  "
                          f"[{tag}]  {cls_body}{mem_body}")
                    if sep_reviewed[k]:
                        cc = culprits_class.get(k, [])
                        mm = (culprits_member.get(k, [])
                              if self._has_membership else [])
                        if cc:
                            print(f"         choice culprits: "
                                  f"{', '.join(cc)}")
                        if mm:
                            print(f"         member culprits: "
                                  f"{', '.join(mm)}")
                print("=" * 67)
        # ─────────────────────────────────────────────────────────────
        # [REVIEW-BLOCK-END] Per-basin block-Hessian ridge screen
        # ─────────────────────────────────────────────────────────────
        # ---- group restarts by (ll, flags) — basin membership ----
        basin_key = {}
        for k in range(n_init):
            flags_k = []
            if degen_flags[k]:
                worst = int(np.argmin(shares_all[k]))
                flags_k.append(f"DEGENERATED: C{worst + 1} "
                               f"share={shares_all[k][worst]:.4f}")
            if sep_choice[k]:
                flags_k.append("SEPARATED:CHOICE")
            if sep_member[k]:
                flags_k.append("SEPARATED:MEMBER")
            if sep_reviewed[k] and not (sep_choice[k] or sep_member[k]):
                flags_k.append("SEPARATED:REVIEWED")
            if int(plat_f[k]) >= self.max_plateau:
                flags_k.append("PLATEAU")
            elif int(stag_f[k]) >= self.max_stagnant:
                flags_k.append("STAGNATED")
            elif not bool(conv_f[k]):
                flags_k.append("MAXITER")
            basin_key[k] = (round(float(ll_f[k]), 6), tuple(flags_k))

        groups = {}
        for k in range(n_init):
            groups.setdefault(basin_key[k], []).append(k)

        if self.verbose:
            n_basins = len(groups)
            print("=" * 67)
            print(f"  Multistart Ensemble — {n_init} starting values, "
                  f"{n_basins} basin{'s' if n_basins != 1 else ''}")
            print("=" * 67)
            for i, ((ll_key, flags_k), members) in enumerate(groups.items(),
                                                              start=1):
                flag_str = ("  ".join(f"[{f}]" for f in flags_k)
                            if flags_k else "[CLEAN]")
                print(f"  Basin {i}   ll = {ll_key:.4f}   {flag_str}")
                sv_str = ", ".join(str(m) for m in members)
                print(f"    starting values: {sv_str}   ({len(members)})")
            print("  " + "-" * 63)

            # Detail: which variable caused separation, per affected restart.
            for k in range(n_init):
                if sep_choice[k]:
                    betas_k = [np.asarray(betas_f[c][k]) for c in range(C)]
                    for c in range(C):
                        xbar = self._separation_xbar(c, betas_k, R_np[k, :, c])
                        if xbar.shape[0] and konis_separation_test(xbar)["separated"]:
                            names_c = [self.varnames[i]
                                       for i in self._class_specs[c]]
                            culprits = attribute_separation(xbar, names_c)
                            print(f"  Starting Values {k}: SEPARATED:CHOICE in class "
                                  f"{c + 1} -> {culprits}")
                if sep_member[k] and self._has_membership:
                    for c in self._intercept_free_classes:
                        xbar_m = self._separation_xbar_membership(c, R_np[k])
                        if xbar_m.shape[0] and konis_separation_test(xbar_m)["separated"]:
                            active = np.where(self._member_mask[c] > 0)[0]
                            names_m = [f"inter_C{c+1}"] + [self.membership_vars[i] for i in active]
                            culprits = attribute_separation(xbar_m, names_m)
                            print(f"  Starting Values {k}: SEPARATED:MEMBER in class "
                                  f"{c + 1} -> {culprits}")

        # ---- winner: best CLEAN starting value always wins ----
        # A degenerated or separated solution is not interpretable, regardless
        # of how much better its ll looks. Only if NO clean starting value
        # exists do we fall back to the global best (least-bad, flagged).
        best_overall_idx = int(np.argmax(np.asarray(ll_f)))

        # Prefer chains that both (a) carry no degeneracy/separation flag
        # AND (b) actually converged (conv_f=True). A chain killed by
        # plateau or stagnation without meeting gtol has conv_f=False and
        # is second-class even if its ll equals that of a properly
        # converged chain.
        _mask_reviewed = sep_reviewed if _review_active else np.zeros(n_init, dtype=bool)
        eligible = np.where(~degen_flags & ~sep_choice & ~sep_member
                            & ~_mask_reviewed & conv_f)[0]

        if eligible.size == 0:
            # No fully-converged clean chain — fall back to any clean chain.
            eligible = np.where(~degen_flags & ~sep_choice & ~sep_member
                                & ~_mask_reviewed)[0]

        if eligible.size == 0:
            if self.verbose:
                print("[FAST] WARNING: no clean starting value in ensemble; "
                      "picking best ll (least-bad, flagged).")
            winner = best_overall_idx
        else:
            winner = int(eligible[np.argmax(np.asarray(ll_f)[eligible])])
            gap = float(ll_f[best_overall_idx]) - float(ll_f[winner])
            self.winner_gap = gap
            if self.verbose and winner != best_overall_idx:
                print(f"[FAST] best clean starting value is {gap:.1f} ll worse "
                      f"than global best; still picking it (clean > ll).")

        self.winner_kind = kinds[winner] if winner < len(kinds) else 'random'
        self.winner_ll = float(ll_f[winner])

        # Expose the WINNER's own flags, not a reduction over all starting
        # values. Downstream code (search.evaluate_lc) reads these to decide
        # whether to penalise the solution's BIC; using `any()` over the
        # whole ensemble lets a single bad chain contaminate a perfectly
        # clean winner.
        self.winner_sep_choice = bool(sep_choice[winner])
        self.winner_sep_member = bool(sep_member[winner])
        self.winner_degenerate = bool(degen_flags[winner])
        self.winner_sep_reviewed = bool(sep_reviewed[winner])
        self.winner_hde = bool(hde_basin[winner]) if _review_active else False
        self._hde_screened_clean = bool(_review_active and _evaluated[winner]
                                        and not sep_reviewed[winner])

        if self.verbose:
            flags_w = []
            if degen_flags[winner]: flags_w.append("DEGENERATED")
            if sep_choice[winner]: flags_w.append("SEPARATED:CHOICE")
            if sep_member[winner]: flags_w.append("SEPARATED:MEMBER")
            tag = "CLEAN" if not flags_w else " ".join(flags_w)
            w_basin = next((i for i, (_, m) in enumerate(groups.items(), 1)
                            if winner in m), "?")
            print(f"  Selected: starting value {winner}  "
                  f"(basin {w_basin}, ll={float(ll_f[winner]):.4f}, {tag})")
            print(f"  EM iterations (Max across All): {int(it_f)}")
            print("=" * 67)

        self.multistart_separation_choice = sep_choice.tolist()
        self.multistart_separation_member = sep_member.tolist()

        # ---- pack best, finalise ----
        # Explicit stop-reason string, so summarise() can report *why* the
        # winning chain stopped (converged / plateau / stagnated / maxiter).
        if conv_f[winner]:
            stop_reason = "converged"
        elif int(plat_f[winner]) >= self.max_plateau:
            stop_reason = "plateau"
        elif int(stag_f[winner]) >= self.max_stagnant:
            stop_reason = "stagnated"
        elif int(it_f) >= self.maxiter:
            stop_reason = "maxiter"
        else:
            stop_reason = "stopped"

        best = dict(
            betas=[np.asarray(betas_f[c][winner]) for c in range(C)],
            inter=np.asarray(inter_f[winner]),
            gamma=np.asarray(gamma_f[winner]),
            pi=np.asarray(pi_f[winner]),
            loglik=float(ll_f[winner]),
            converged=bool(conv_f[winner]),
            n_iter=int(it_f),
            posterior=np.asarray(R_all[winner]),
            final_delta_ll=float(dll_final[winner]),
            final_step=float(step_final[winner]),
            stop_reason=stop_reason,
        )
        self._finalise(best, time.time() - t0)
        self.multistart_loglik = [float(x) for x in np.asarray(ll_f)]
        self.multistart_kind = list(kinds)
        self.multistart_degenerate = degen_flags.tolist()
        self.multistart_shares = shares_all.tolist()        

    # ------------------------------------------------------------------
    # FINALISE
    # ------------------------------------------------------------------
    def _build_coeff_names(self):
        """Per-class coefficient names, in the same order as _Ks (Kc entries
        per class). Overridable so subclasses that append extra X-independent
        slots (e.g. LatentClassConditionalFinal's v_s) can extend the list
        without touching _class_specs — the latter is indexed against X and
        must stay length-Kc for _class_ll/_compute_se to work."""
        names = []
        for c in range(self.n_classes):
            for v in [self.varnames[i] for i in self._class_specs[c]]:
                names.append(f"class_{c + 1}_{v}")
        return names

    def _finalise(self, best, elapsed):
        C = self.n_classes
        betas_norm = [np.asarray(b) for b in best["betas"]]
        inter = np.asarray(best["inter"])
        gamma = np.asarray(best["gamma"])
        posterior = np.asarray(best["posterior"])

        self.class_betas_norm = betas_norm
        self.class_gammas_norm = gamma
        self.posterior = posterior
        self.class_probs = self._normalize(posterior.mean(axis=0))
        self.loglik = float(best["loglik"])
        self.converged = bool(best["converged"])
        self.total_iter = int(best["n_iter"])
        self.estim_time_sec = float(elapsed)

        # coeff_names
        self.coeff_names = self._build_coeff_names()

        # num_params / AIC / BIC
        n_beta = int(self._Ks.sum())
        n_inter = self._n_inter if self._has_membership else 0
        n_gamma_active = int(self._member_mask.sum()) if self._has_membership else 0
        if self._has_membership:
            self.num_params = n_beta + n_inter + n_gamma_active
        else:
            self.num_params = n_beta + (C - 1)
        self.aic = 2 * self.num_params - 2 * self.loglik
        self.bic = np.log(self.N) * self.num_params - 2 * self.loglik

        self.final_delta_ll = float(best.get("final_delta_ll", 0.0))
        self.final_step     = float(best.get("final_step", 0.0))
        self.stop_reason    = best.get("stop_reason", "unknown")
        self.final_grad_choice = [
            np.asarray(self._class_fgh_raw[c][1](
                jnp.asarray(betas_norm[c]),
                jnp.asarray(self.posterior[:, c])))
            for c in range(C)
        ]
        if self._has_membership:
            v_m = jnp.concatenate([jnp.asarray(inter),
                                    jnp.asarray(gamma).ravel()])
            self.final_grad_member = np.asarray(
                self._mem_fgh_raw[1](v_m, jnp.asarray(self.posterior)))
        else:
            self.final_grad_member = np.zeros(0)
        self.gtol_res = max(
            max((float(np.max(np.abs(g))) for g in self.final_grad_choice
                 if g.size), default=0.0),
            float(np.max(np.abs(self.final_grad_member)))
            if self.final_grad_member.size else 0.0,
        )
        self.ftol_res = 0.0

        # SE + diagnostics (uses normalized betas internally)
        try:
            self._compute_se(betas_norm, inter, gamma)
        except Exception as exc:
            self.se_computation_error = str(exc)
            self.stderr = None
            self.zvalues = None
            self.pvalues = None
            self.se_params = None

        # Denormalized reporting copies
        self._denormalize_report(betas_norm, inter, gamma)

        # Per-class gamma maps for warm-starting the NEXT fit (search memory
        # or next K level). Consumed by Search._warm_start_from_memory via
        # `getattr(m_model, 'gamma_maps', None)` and by
        # Search._capture_dominant_class for K→K+1 expansion.
        #
        # Scale contract (must match LatentClassFinal._resolve_inits):
        #   - '_inter' (intercepts): internal Z-centred logit — read from
        #     self.class_gammas, NOT from self.gamma_params (which has been
        #     shifted by -gamma*mu_Z for reporting).
        #   - variable keys (gammas): denormalized — divided by _mem_sigma,
        #     because _resolve_inits re-normalizes by multiplying by _mem_sigma.
        self.gamma_maps = []
        if getattr(self, 'class_gammas', None) is not None:
            n_inter = self._n_inter
            Km = self.K_membership
            for c in range(C):
                gmap = {}
                if c in self._intercept_free_classes:
                    i_int = self._intercept_free_classes.index(c)
                    gmap['_inter'] = float(self.gamma_params[i_int])
                for k, v in enumerate(self.membership_vars):
                    if self._member_mask[c, k] > 0:
                        idx = n_inter + c * Km + k
                        gmap[v] = float(self.class_gammas[idx]) / float(self._mem_sigma[k])
                self.gamma_maps.append(gmap)

        # Predicted shares
        self.pred_prob, self.obs_prob = self._compute_prop_alts(betas_norm, posterior)

        # ---- Entropy + OCC (classification quality) ----
        R = np.asarray(posterior)
        C_ = self.n_classes
        with np.errstate(divide="ignore", invalid="ignore"):
            logR = np.where(R > 1e-300, np.log(R), 0.0)
            entropy = (1.0 - float(-np.sum(R * logR)) / (self.N * np.log(C_))
                       if C_ > 1 else None)
        self.entropy = entropy
        modal = np.argmax(R, axis=1)
        occ = np.zeros(C_)
        for c in range(C_):
            idx = modal == c
            if idx.sum() == 0:
                occ[c] = np.nan
                continue
            avePP = float(R[idx, c].mean())
            pi_c = float(self.class_probs[c])
            if avePP >= 1.0 or pi_c <= 0.0:
                occ[c] = np.nan
            else:
                occ[c] = (avePP / (1 - avePP)) / (pi_c / (1 - pi_c))
        self.occ = occ
        return self

    def _denormalize_report(self, betas_norm, inter, gamma):
        C = self.n_classes
        class_betas = []
        for c in range(C):
            sig = self._class_sigma[c]
            class_betas.append(betas_norm[c] / sig)

        # Conditional: v_s (último coeff de cada clase) entra linealmente en
        # el logit del purchase, así que NO hereda la cancelación softmax del
        # choice. En escala normalizada quedó centrado; para reportarlo en la
        # misma escala que los β_x hay que sumarle el shift Σ β_orig·μ_k,
        # igual que se hace con los interceptos de membresía.
        if isinstance(self, LatentClassConditionalFinal):
            for c in range(C):
                spec_c = np.asarray(self._class_specs[c])
                mu_c   = self._norm_mu[spec_c]
                shift  = float(np.dot(class_betas[c][:-1], mu_c))
                class_betas[c][-1] += shift

        self.class_betas = class_betas
        self.coeff_est = np.concatenate(class_betas) if class_betas else np.empty(0)

        # se_params layout: [phi(C-1) | betas(sum Ks) | inter | gamma_active]
        if getattr(self, "se_params", None) is None:
            return
        n_phi = C - 1
        params = np.asarray(self.se_params).copy()
        se = np.asarray(self.stderr).copy() if self.stderr is not None else None

        opg = np.asarray(self.opg_se).copy() if self.opg_se is not None else None
        off = n_phi
        for c in range(C):
            kc = int(self._Ks[c])
            sig_c = self._class_sigma[c]
            params[off:off + kc] /= sig_c
            if se is not None:
                se[off:off + kc] /= sig_c
            if opg is not None:
                opg[off:off + kc] /= sig_c
            off += kc
        # también gammas activas si querés consistencia completa
        if self._has_membership and self.K_membership > 0 and opg is not None:
            active = [(c, v) for c in range(C) for v in range(self.K_membership)
                      if self._member_mask[c, v] > 0]
            for c, v in active:
                opg[off + self._n_inter + c * self.K_membership + v] /= self._mem_sigma[v]
        if opg is not None:
            self.opg_se = opg
        # intercepts (n_inter): unchanged
        off += self._n_inter
        # gamma active: divide by corresponding mem_sigma
        if self._has_membership and self.K_membership > 0:
            active = [(c, v) for c in range(C) for v in range(self.K_membership)
                      if self._member_mask[c, v] > 0]
            for c, v in active:
                i = off + c * self.K_membership + v
                s = self._mem_sigma[v]
                params[i] /= s
                if se is not None:
                    se[i] /= s

        self.se_params = params
        if se is not None:
            self.stderr = se

        # gamma_params / gamma_se
        if getattr(self, "gamma_params", None) is not None and len(self.gamma_params) > 0:
            gp = np.asarray(self.gamma_params).copy()
            gs = np.asarray(self.gamma_se).copy() if self.gamma_se is not None else None
            gop = (np.asarray(self.gamma_opg_se).copy()
                   if getattr(self, 'gamma_opg_se', None) is not None else None)
            # First n_inter are intercepts (unchanged); rest are gammas (active)
            active = [(c, v) for c in range(C) for v in range(self.K_membership)
                      if self._member_mask[c, v] > 0]
            for j, (c, v) in enumerate(active):
                s = self._mem_sigma[v]
                idx = self._n_inter + j
                if idx < len(gp):
                    gp[idx] /= s
                    if gs is not None:
                        gs[idx] /= s
                    if gop is not None:
                        gop[idx] /= s
            # Intercept denorm: inter_orig = inter_norm - Σ_k γ_orig[c,k]·μ_Z[k]
            mem_idx = [self.varnames.index(v) for v in self.membership_vars]
            mu_Z = self._norm_mu[mem_idx]
            base_corr = 0.0
            for j, (cc, vv) in enumerate(active):
                if cc == self.base_class:
                    base_corr += gp[self._n_inter + j] * mu_Z[vv]
            for c in range(C):
                if c not in self._intercept_free_classes:
                    continue
                i_int = self._intercept_free_classes.index(c)
                corr = 0.0
                for j, (cc, vv) in enumerate(active):
                    if cc == c:
                        corr += gp[self._n_inter + j] * mu_Z[vv]
                gp[i_int] = gp[i_int] - corr + base_corr
            # Delta-method SE for the denormalized intercepts, using the normalized-scale
            # cov_theta: d(inter_orig_c) = d(inter_c) - sum_k (mu_k/sigma_k) d(gamma_ck)
            #                              + sum_k (mu_k/sigma_k) d(gamma_base,k)
            cov_n = getattr(self, 'cov_theta', None)
            if cov_n is not None and gs is not None:
                cov_n = np.asarray(cov_n)
                Km = self.K_membership
                inter_start = n_phi + int(np.sum(self._Ks))
                gamma_start = inter_start + self._n_inter
                gt = np.asarray(self.gamma_t_stats).copy()
                gpv = np.asarray(self.gamma_p_values).copy()
                params = np.asarray(self.se_params).copy()
                se_v = np.asarray(self.stderr).copy()
                z_v = np.asarray(self.zvalues).copy()
                p_v = np.asarray(self.pvalues).copy()
                cov_o = getattr(self, 'cov_opg', None)
                opg_v = np.asarray(self.opg_se).copy() if self.opg_se is not None else None
                for i_int, c in enumerate(self._intercept_free_classes):
                    g = np.zeros(cov_n.shape[0])
                    g[inter_start + i_int] = 1.0
                    for cc, vv in active:
                        w = mu_Z[vv] / self._mem_sigma[vv]
                        if cc == c:
                            g[gamma_start + cc * Km + vv] -= w
                        elif cc == self.base_class:
                            g[gamma_start + cc * Km + vv] += w
                    var = float(g @ cov_n @ g)
                    se_i = np.sqrt(var) if var > 0 else 0.0
                    z_i = gp[i_int] / se_i if se_i > 0 else np.nan
                    p_i = 2 * (1 - _scipy_norm.cdf(abs(z_i))) if se_i > 0 else np.nan
                    gs[i_int], gt[i_int], gpv[i_int] = se_i, z_i, p_i
                    j = inter_start + i_int
                    params[j], se_v[j], z_v[j], p_v[j] = gp[i_int], se_i, z_i, p_i
                    if cov_o is not None and gop is not None and opg_v is not None:
                        var_o = float(g @ cov_o @ g)
                        gop[i_int] = np.sqrt(var_o) if var_o > 0 else np.nan
                        opg_v[j] = gop[i_int]
                self.gamma_t_stats = gt
                self.gamma_p_values = gpv
                self.se_params = params
                self.stderr = se_v
                self.zvalues = z_v
                self.pvalues = p_v
                if opg_v is not None:
                    self.opg_se = opg_v    
            self.gamma_params = gp
            if gs is not None:
                self.gamma_se = gs
            if gop is not None:
                self.gamma_opg_se = gop 

    @staticmethod
    def _normalize(p):
        p = np.clip(np.asarray(p, dtype=float), 1e-12, None)
        return p / p.sum()

    def _joint_negll(self, betas, inter, gamma, pi):
        _, ll = self._estep(betas, inter, gamma, pi)
        return -ll
    # ------------------------------------------------------------------
    # SE + DIAGNOSTICS
    # ------------------------------------------------------------------
    def _compute_se(self, betas_norm, inter, gamma):
        C = self.n_classes
        Km = self.K_membership
        n_phi = C - 1
        n_beta = int(self._Ks.sum())
        n_inter = self._n_inter
        n_gamma_dense = C * Km

        def unpack(v):
            off = n_phi
            b = []
            for k in self._Ks:
                b.append(v[off:off + int(k)]); off += int(k)
            ii = v[off:off + n_inter]; off += n_inter
            gg = v[off:off + n_gamma_dense].reshape(C, Km) if n_gamma_dense else jnp.zeros((C, 0))
            return b, ii, gg

       

        def _phi_to_pi_inner(phi):
            logits = jnp.zeros(C)
            idx_free = [c for c in range(C) if c != self.base_class]
            for i, c in enumerate(idx_free):
                logits = logits.at[c].set(phi[i])
            logits = logits - jnp.max(logits)
            p = jnp.exp(logits)
            return p / jnp.sum(p)

        def joint_negll_flat(v):
            b, ii, gg = unpack(v)
            if self._has_membership:
                pi_arg = jnp.full(C, 1.0 / C)
            else:
                pi_arg = _phi_to_pi_inner(v[:n_phi])
            return self._joint_negll(b, ii, gg, pi_arg)

        # phi placeholder (only used for shape/position, not identified when
        # membership is active because intercepts ARE the log-odds params)
        phi = np.array([
            np.log(max(self.class_probs[c], 1e-12) /
                   max(self.class_probs[self.base_class], 1e-12))
            for c in range(C) if c != self.base_class
        ])

        theta = np.concatenate([phi,
                                np.concatenate(betas_norm),
                                np.asarray(inter),
                                np.asarray(gamma).ravel()])
        theta_j = jnp.asarray(theta)

        # JIT the Hessian once per model
        if not hasattr(self, "_hess_jit"):
            self._hess_jit = jax.jit(hessian(joint_negll_flat))
        H = np.asarray(self._hess_jit(theta_j))
        try:
            cov = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            cov = np.linalg.pinv(H)
        se_full = np.sqrt(np.clip(np.diag(cov), 0, None))
        with np.errstate(divide="ignore", invalid="ignore"):
            z_full = np.where(se_full > 0, theta / se_full, np.nan)
        p_full = 2 * (1 - _scipy_norm.cdf(np.abs(z_full)))

        # OPG (optional)
        if self.compute_opg:
            # When membership is active, `phi` is a positional placeholder:
            # the intercepts (inter) already ARE the log-odds params, so phi
            # never enters the likelihood and its column in the OPG Jacobian
            # is identically zero. That makes JtJ singular by construction.
            # Under the Conditional's purchase stage (cond ~ 1e15), that zero
            # column poisons every diagonal of pinv(JtJ) with NaN, not just
            # the phi block. Drop phi from the OPG entirely when membership
            # is active — it's not an estimated parameter in that branch.
            has_mem_opg = self._has_membership
            if has_mem_opg:
                theta_opg = theta[n_phi:]
                n_phi_opg = 0
            else:
                theta_opg = theta
                n_phi_opg = n_phi
            theta_opg_j = jnp.asarray(theta_opg)

            def per_individual_ll_flat_opg(v):
                if n_phi_opg == 0:
                    v_full = jnp.concatenate([jnp.zeros(n_phi, dtype=v.dtype), v])
                else:
                    v_full = v
                b, ii, gg = unpack(v_full)
                ll_c = self._class_ll_matrix(b)
                if self._has_membership:
                    logH = self._membership_log_probs(ii, gg)
                else:
                    pi_arg = _phi_to_pi_inner(v_full[:n_phi])
                    logpi = jnp.log(jnp.clip(pi_arg, MIN_COMP))
                    logH = jnp.broadcast_to(logpi[None, :], (self.N, C))
                return jax.scipy.special.logsumexp(ll_c + logH, axis=1)

            jac_fn = jax.jit(jax.jacfwd(per_individual_ll_flat_opg))
            jac = np.asarray(jac_fn(theta_opg_j))
            JtJ = jac.T @ jac
            # Truncated-SVD pseudo-inverse. rcond=1e-6 zeroes any direction
            # whose singular value is below 1e-6 of the largest — exactly the
            # near-singular directions the purchase stage produces when a
            # class is (near-)separated. Without an explicit rcond, the
            # default threshold leaves those directions to be inverted to
            # ~1e15, and the sqrt(diag) then propagates NaN through the
            # whole block. With rcond, unidentified directions get SE = 0
            # (correctly reflecting "no information"), and identified ones
            # get finite values.
            n_opg = JtJ.shape[0]
            ridge_opg = 1e-10 * max(np.trace(JtJ) / n_opg, 1.0)
            try:
                cov_opg = np.linalg.pinv(JtJ + ridge_opg * np.eye(n_opg),
                                          rcond=1e-6)
                if not np.all(np.isfinite(cov_opg)):
                    raise np.linalg.LinAlgError("non-finite pinv")
            except np.linalg.LinAlgError:
                cov_opg = np.zeros((n_opg, n_opg))
            diag_opg = np.diag(cov_opg)
            opg_se_sub = np.where(np.isfinite(diag_opg) & (diag_opg > 0),
                                   np.sqrt(np.clip(diag_opg, 0, None)),
                                   np.nan)

            # Map back to full-theta layout: phi block stays NaN (placeholder,
            # not identified under membership); everything else gets its OPG SE.
            opg_se_full = np.full_like(theta, np.nan)
            if n_phi_opg == 0:
                opg_se_full[n_phi:] = opg_se_sub
            else:
                opg_se_full[:] = opg_se_sub
            s0 = n_phi if n_phi_opg == 0 else 0
            self.cov_opg = np.zeros((len(theta), len(theta)))
            self.cov_opg[s0:, s0:] = cov_opg
        else:
            opg_se_full = np.full_like(theta, np.nan)
            self.cov_opg = None

        # OPG/Hess ratio flag
        with np.errstate(divide="ignore", invalid="ignore"):
            opg_hess_ratio = opg_se_full / np.where(se_full > 1e-12, se_full, np.nan)
        opg_hess_flag = opg_hess_ratio > OPG_HESS_RATIO_FLAG

        self.opg_se = opg_se_full
        self.se_params = theta
        self.cov_theta = cov
        self.stderr = se_full
        self.zvalues = z_full
        self.pvalues = p_full
        self.se_method = "hessian (jax autodiff)"
        
        self.opg_hess_ratio = opg_hess_ratio
        self.opg_hess_flag = opg_hess_flag
        # Non-positive variance flag (Hessian not PD) over estimated parameters only:
        # phi is a placeholder under membership; masked gammas never enter the likelihood
        est_mask = np.ones(len(theta), dtype=bool)
        if self._has_membership:
            est_mask[:n_phi] = False
        if n_gamma_dense:
            est_mask[n_phi + n_beta + n_inter:] = np.asarray(self._member_mask).ravel() > 0
        self.nonpd_flag = (np.diag(cov) <= 0) & est_mask
        H_act = H[np.ix_(est_mask, est_mask)]
        self.cond_number = float(np.linalg.cond(H_act)) if H_act.size else float("nan")
        self.hess_min_eig = (float(np.linalg.eigvalsh(0.5 * (H_act + H_act.T)).min())
                             if H_act.size else float("nan"))

        # gamma params
        gamma_start = n_phi + n_beta
        gamma_theta = theta[gamma_start:]
        gamma_se_d = se_full[gamma_start:]
        gamma_t_d = z_full[gamma_start:]
        gamma_p_d = p_full[gamma_start:]
        gamma_opg_d = opg_se_full[gamma_start:]

        gamma_names, gp, gs, gt, gpv, gopg = [], [], [], [], [], []
        for i, c in enumerate(self._intercept_free_classes):
            gamma_names.append(f"gamma_intercept_class_{c+1}")
            gp.append(gamma_theta[i]); gs.append(gamma_se_d[i])
            gt.append(gamma_t_d[i]); gpv.append(gamma_p_d[i])
            gopg.append(gamma_opg_d[i])
        mem_vars = self.membership_vars or []
        for c in range(C):
            for k in range(Km):
                if self._member_mask[c, k] > 0:
                    idx = n_inter + c * Km + k
                    gamma_names.append(f"gamma_class_{c+1}_{mem_vars[k]}")
                    gp.append(gamma_theta[idx]); gs.append(gamma_se_d[idx])
                    gt.append(gamma_t_d[idx]); gpv.append(gamma_p_d[idx])
                    gopg.append(gamma_opg_d[idx])
        self.gamma_params = np.array(gp)
        self.gamma_se = np.array(gs)
        self.gamma_t_stats = np.array(gt)
        self.gamma_p_values = np.array(gpv)
        self.gamma_names = gamma_names
        self.gamma_opg_se = np.array(gopg)
        g_idx = list(range(n_inter)) + [n_inter + c * Km + k for c in range(C) for k in range(Km)
                                        if self._member_mask[c, k] > 0]
        self.gamma_nonpd_flag = self.nonpd_flag[gamma_start + np.asarray(g_idx, dtype=int)]
        self.class_gammas = gamma_theta

        # Class-share SE (delta method)
        if self._has_membership:
            # d(share_c)/d(inter, gamma) via autodiff
            def _shares_from_v(v):
                inter_v = v[:n_inter]
                gamma_v = v[n_inter:]
                gamma_mat = gamma_v.reshape(C, Km)
                logits = jnp.zeros((self.N, C))
                for i, c in enumerate(self._intercept_free_classes):
                    logits = logits.at[:, c].set(inter_v[i])
                if Km > 0:
                    gm = gamma_mat * self._member_mask_d
                    logits = logits + self._Zmd @ gm.T
                logits = logits - jnp.max(logits, axis=1, keepdims=True)
                priors = jnp.exp(logits)
                priors = priors / jnp.sum(priors, axis=1, keepdims=True)
                return jnp.mean(priors, axis=0)   # (C,)

            jac_fn = jax.jit(jax.jacfwd(_shares_from_v))
            v_flat = np.concatenate([np.asarray(inter),
                                     np.asarray(gamma).ravel()])
            jac = np.asarray(jac_fn(jnp.asarray(v_flat)))  # (C, n_inter+C*Km)
            # cov block for (inter, gamma)
            g_start = n_phi + n_beta
            cov_g = cov[g_start:g_start + n_inter + n_gamma_dense,
                        g_start:g_start + n_inter + n_gamma_dense]
            var_pi = np.einsum('ck,kl,cl->c', jac, cov_g, jac)
            se_pi = np.sqrt(np.clip(var_pi, 0, None))
        else:
            se_pi = self._delta_method_share_se(
                cov[:n_phi, :n_phi] if n_phi else np.zeros((0, 0)),
                self.class_probs)
        self.se_pi = se_pi
        with np.errstate(divide="ignore", invalid="ignore"):
            logit_pi = np.log(np.clip(self.class_probs, 1e-12, 1-1e-12) /
                              np.clip(1 - self.class_probs, 1e-12, None))
            se_logit = np.where((self.class_probs > 0) & (self.class_probs < 1),
                                se_pi / np.clip(self.class_probs * (1 - self.class_probs), 1e-12, None),
                                np.nan)
        lo_logit = logit_pi - 1.96 * se_logit
        hi_logit = logit_pi + 1.96 * se_logit
        self.ci_lo_pi = 1.0 / (1.0 + np.exp(-lo_logit))
        self.ci_hi_pi = 1.0 / (1.0 + np.exp(-hi_logit))

        # RAW per-class Hessians for VDP / ratio_min
        self.se_raw = np.full_like(theta, np.nan)
        self.collin_share = np.full_like(theta, np.nan)
        self.collin_flag = np.zeros_like(theta, dtype=bool)
        self.class_ratio_min = np.full(C, np.nan)
        self.class_ratio_min_abs_flag = np.zeros(C, dtype=bool)
        self.class_ratio_min_rel_flag = np.zeros(C, dtype=bool)

        self.hde_flag = np.zeros_like(theta, dtype=bool)
        dh_class, dh_mem = self._hde_kernels()

        H_list, se_raw_blocks, hde_blocks = [], [], []
        for c, beta_c in enumerate(betas_norm):
            _, _, h_raw = self._class_fgh_raw[c]
            beta_j = jnp.asarray(beta_c)
            w_j = jnp.asarray(self.posterior[:, c])
            H = np.asarray(h_raw(beta_j, w_j))   # NO ridge
            H_list.append(H)
            try:
                info_inv = np.linalg.inv(H)
            except np.linalg.LinAlgError:
                info_inv = np.linalg.pinv(H)
            se_raw_blocks.append(np.sqrt(np.clip(np.diag(info_inv), 0, None)))
            if getattr(self, "_hde_screened_clean", False):
                hde_blocks.append(np.zeros(H.shape[0], dtype=bool))
            else:
                try:
                    hde_blocks.append(hauck_donner_flags(
                        H, np.asarray(dh_class[c](beta_j, w_j)), np.asarray(beta_j)))
                except np.linalg.LinAlgError:
                    hde_blocks.append(np.zeros(H.shape[0], dtype=bool))
            try:
                ev = np.linalg.eigvalsh(H)
                self.class_ratio_min[c] = ev.min() / max(ev.max(), 1e-300)
            except Exception:
                self.class_ratio_min[c] = np.nan

        rel_thresh = self._relative_thresholds(self.class_ratio_min, min_siblings=2)

        off = n_phi
        for c in range(C):
            share_abs, flag_abs, _ = collinearity_diag(H_list[c], ratio_thresh=1e-3)
            if rel_thresh[c] > 0:
                share_rel, flag_rel, _ = collinearity_diag(H_list[c], ratio_thresh=rel_thresh[c])
            else:
                share_rel, flag_rel = np.zeros_like(share_abs), np.zeros_like(flag_abs)
            self.class_ratio_min_abs_flag[c] = flag_abs.any()
            self.class_ratio_min_rel_flag[c] = flag_rel.any()
            kc = H_list[c].shape[0]
            self.se_raw[off:off + kc] = se_raw_blocks[c]
            self.collin_share[off:off + kc] = np.maximum(share_abs, share_rel)
            self.collin_flag[off:off + kc] = flag_abs | flag_rel
            self.hde_flag[off:off + kc] = hde_blocks[c]
            off += kc

        # Membership diagnostics (raw Hessian, active submatrix)
        gamma_se_raw_dense = np.full(n_inter + n_gamma_dense, np.nan)
        gamma_collin_dense = np.full(n_inter + n_gamma_dense, np.nan)
        gamma_flag_dense = np.zeros(n_inter + n_gamma_dense, dtype=bool)
        gamma_hde_dense = np.zeros(n_inter + n_gamma_dense, dtype=bool)
        if self._has_membership and gamma_se_raw_dense.size > 0:
            _, _, h_raw = self._mem_fgh_raw
            v_j = jnp.asarray(gamma_theta)
            R_j = jnp.asarray(self.posterior)
            H_full = np.asarray(h_raw(v_j, R_j))
            active_idx = list(range(n_inter))
            for c in range(C):
                for k in range(Km):
                    if self._member_mask[c, k] > 0:
                        active_idx.append(n_inter + c * Km + k)
            active_idx = np.array(active_idx, dtype=int)
            H_active = H_full[np.ix_(active_idx, active_idx)]
            try:
                info_inv = np.linalg.inv(H_active)
            except np.linalg.LinAlgError:
                info_inv = np.linalg.pinv(H_active)
            se_raw_active = np.sqrt(np.clip(np.diag(info_inv), 0, None))
            share_active, flag_active, _ = collinearity_diag(H_active)
            gamma_se_raw_dense[active_idx] = se_raw_active
            gamma_collin_dense[active_idx] = share_active
            gamma_flag_dense[active_idx] = flag_active
            if not getattr(self, "_hde_screened_clean", False):
                try:
                    dH_full = np.asarray(dh_mem(v_j, R_j))
                    gamma_hde_dense[active_idx] = hauck_donner_flags(
                        H_active, dH_full[np.ix_(active_idx, active_idx, active_idx)],
                        np.asarray(v_j)[active_idx])
                except np.linalg.LinAlgError:
                    pass

        gsr, gsc, gsf, ghf = [], [], [], []
        for i in range(n_inter):
            gsr.append(gamma_se_raw_dense[i])
            gsc.append(gamma_collin_dense[i])
            gsf.append(gamma_flag_dense[i])
            ghf.append(gamma_hde_dense[i])
        for c in range(C):
            for k in range(Km):
                if self._member_mask[c, k] > 0:
                    idx = n_inter + c * Km + k
                    gsr.append(gamma_se_raw_dense[idx])
                    gsc.append(gamma_collin_dense[idx])
                    gsf.append(gamma_flag_dense[idx])
                    ghf.append(gamma_hde_dense[idx])
        self.gamma_se_raw = np.array(gsr)
        self.gamma_collin_share = np.array(gsc)
        self.gamma_collin_flag = np.array(gsf, dtype=bool)
        self.gamma_hde_flag = np.array(ghf, dtype=bool)

        # Firth summary
        classes_flagged = [c for c in range(C) if self.class_ratio_min_rel_flag[c]]
        coeff_class = np.concatenate([np.full(int(k), c) for c, k in enumerate(self._Ks)])
        vars_flagged = [self.coeff_names[i] for i in range(len(self.coeff_names))
                         if coeff_class[i] in classes_flagged
                         and i < len(self.collin_flag) - n_phi
                         and self.collin_flag[n_phi + i]]
        gamma_vars_flagged = [self.gamma_names[i] for i in range(len(self.gamma_collin_flag))
                               if self.gamma_collin_flag[i]]
        self.firth_summary = {
            "needed_correction": bool(classes_flagged or gamma_vars_flagged),
            "classes_flagged": classes_flagged,
            "choice_vars_flagged": vars_flagged,
            "membership_vars_flagged": gamma_vars_flagged,
        }

    def _hde_kernels(self):
        """Third-derivative kernels (dH/dtheta) for HDE detection, built once per model."""
        if getattr(self, '_hde_kernels_cache', None) is None:
            dh_class = [jax.jit(jax.jacfwd(self._class_fgh_raw[c][2]))
                        for c in range(self.n_classes)]
            dh_mem = (jax.jit(jax.jacfwd(self._mem_fgh_raw[2]))
                      if self._has_membership else None)
            self._hde_kernels_cache = (dh_class, dh_mem)
        return self._hde_kernels_cache

    @staticmethod
    def _relative_thresholds(ratio_mins, gap_factor=100.0, min_siblings=2):
        ratio_mins = np.asarray(ratio_mins, dtype=float)
        n = len(ratio_mins)
        thresh = np.zeros(n)
        for i in range(n):
            others = np.delete(ratio_mins, i)
            others = others[np.isfinite(others)]
            if len(others) >= min_siblings:
                thresh[i] = np.median(others) / gap_factor
        return thresh

    def _delta_method_share_se(self, cov_phi, pi):
        C = len(pi)
        if C == 1:
            return np.zeros(1)
        se_pi = np.zeros(C)
        for c in range(C):
            jac = np.array([pi[c] * ((1.0 if c == j else 0.0) - pi[j])
                            for j in range(C - 1)])
            var_c = jac @ cov_phi @ jac
            se_pi[c] = np.sqrt(max(var_c, 0.0))
        return se_pi
    
    def summarise_wald(self, alpha=0.05, file=None):
        """Class-pair Wald diagnostics on the choice block (not a formal test).

        For each pair of classes sharing variables: a joint Wald test on the
        common betas, plus per-variable tests listing those that fail to
        reject equality at `alpha`. Tests run in the normalized scale, where
        `cov_theta` lives; betas are displayed denormalized.
        `file=None` routes to stdout."""
        from scipy.stats import chi2
        C = self.n_classes
        n_phi = C - 1
        rel_tol = 1e-8

        # se_params is denormalized for display; tests use class_betas_norm,
        # the exact point (normalized scale, unshifted) where cov_theta was evaluated.
        beta_disp = np.asarray(self.se_params)
        theta = beta_disp.copy()
        offsets = n_phi + np.concatenate(([0], np.cumsum(self._Ks)[:-1])).astype(int)
        for c in range(C):
            kc = int(self._Ks[c])
            theta[offsets[c]:offsets[c] + kc] = np.asarray(self.class_betas_norm[c])
        cov = np.asarray(self.cov_theta)
        pvals = (np.asarray(self.pvalues) if self.pvalues is not None
                 else np.full(theta.shape, np.nan))

        def p(text=""):
            print(text, file=file)

        def stars(pv):
            if not np.isfinite(pv):
                return ""
            for thr, sym in ((0.001, "(***)"), (0.01, "(**)"), (0.05, "(*)"), (0.1, "(.)")):
                if pv < thr:
                    return sym
            return "(-)"

        def joint_verdict(pv):
            if pv < alpha:
                return "DISTINCT"
            return "BORDERLINE" if pv < 0.10 else "POSSIBLE SPLIT"

        def joint_wald(pairs):
            """Joint Wald on common betas via eigendecomposition; df = effective rank."""
            i1s = np.array([a for a, _ in pairs])
            i2s = np.array([b for _, b in pairs])
            d = theta[i1s] - theta[i2s]
            S = (cov[np.ix_(i1s, i1s)] + cov[np.ix_(i2s, i2s)]
                 - cov[np.ix_(i1s, i2s)] - cov[np.ix_(i2s, i1s)])
            S = 0.5 * (S + S.T)
            ev, V = np.linalg.eigh(S)
            if ev.max() <= 0:
                return None
            keep = ev > rel_tol * ev.max()
            rank = int(keep.sum())
            S_inv = (V[:, keep] / ev[keep]) @ V[:, keep].T
            W = float(d @ S_inv @ d)
            return W, rank, float(chi2.sf(W, rank)), rank < len(d)

        blocks = []
        for c1 in range(C):
            for c2 in range(c1 + 1, C):
                spec1 = list(self._class_specs[c1])
                spec2 = list(self._class_specs[c2])
                common = [v for v in spec1 if v in spec2]
                if not common:
                    continue

                rows, valid, n_na = [], [], 0
                for v in common:
                    i1 = offsets[c1] + spec1.index(v)
                    i2 = offsets[c2] + spec2.index(v)
                    row_base = (self.varnames[v], float(beta_disp[i1]), stars(pvals[i1]),
                                float(beta_disp[i2]), stars(pvals[i2]))
                    var_d = cov[i1, i1] + cov[i2, i2] - 2.0 * cov[i1, i2]
                    if cov[i1, i1] <= 0 or cov[i2, i2] <= 0 or var_d <= 0:
                        n_na += 1
                        rows.append(row_base + (None, None, "n/a"))
                        continue
                    valid.append((i1, i2))
                    W = float((theta[i1] - theta[i2]) ** 2 / var_d)
                    p_val = float(chi2.sf(W, 1))
                    if p_val >= alpha:
                        verdict = "NO DISTINCT" if p_val >= 0.10 else "BORDERLINE"
                        rows.append(row_base + (W, p_val, verdict))

                joint = joint_wald(valid) if valid else None
                n_not_distinct = sum(1 for r in rows if r[7] != "n/a")
                blocks.append((c1, c2, len(common), len(spec1) - len(common),
                               len(spec2) - len(common), joint, n_not_distinct,
                               len(valid), n_na, rows))

        if not blocks:
            return

        width = 78
        p()
        p("=" * width)
        p("  CLASS-PAIR WALD TESTS  (diagnostic, not a formal test)")
        p("=" * width)
        for c1, c2, n_common, only1, only2, joint, n_nd, k_valid, n_na, rows in blocks:
            p()
            p(f"  Class {c1+1} vs Class {c2+1}")
            p(f"  Common: {n_common}  |  Only C{c1+1}: {only1}  |  Only C{c2+1}: {only2}")
            if joint is not None:
                W_j, df_j, p_j, ill = joint
                flag = "  [ill-conditioned]" if ill else ""
                p(f"  Joint Wald (common):  W = {W_j:.2f}   df = {df_j}   "
                  f"p = {p_j:.4f}   -> {joint_verdict(p_j)}{flag}")
            else:
                p("  Joint Wald (common):  n/a")
            na_txt = f" ({n_na} n/a)" if n_na else ""
            p(f"  Not distinct: {n_nd}/{k_valid}{na_txt}")

            if rows:
                p()
                p(f"  {'Var':<22}{'beta_'+str(c1+1):>11}{'':7}{'beta_'+str(c2+1):>11}{'':7}"
                  f"{'Wald':>10}{'p-val':>10}  Verdict")
                p(f"  {'-'*22}{'-'*18}{'-'*18}{'-'*10}{'-'*10}  {'-'*14}")
                for name, b1, s1, b2, s2, W, p_val, verdict in rows:
                    w_txt = f"{W:>10.2f}" if W is not None else f"{'n/a':>10}"
                    p_txt = f"{p_val:>10.4f}" if p_val is not None else f"{'n/a':>10}"
                    p(f"  {name:<22}{b1:>11.4f} {s1:<6}{b2:>11.4f} {s2:<6}"
                      f"{w_txt}{p_txt}  {verdict}")
            p()
            p("-" * width)

        p("  Signif. (individual beta): 0 '(***)' 0.001 '(**)' 0.01 '(*)' 0.05 '(.)' 0.1 '(-)' 1")
        p(f"  Joint verdict: DISTINCT p<{alpha} | BORDERLINE {alpha}<=p<0.10 | "
          f"POSSIBLE SPLIT p>=0.10")
        p("  Betas shown denormalized; tests computed in the normalized scale.")
        p("=" * width)

    def _phi_to_pi(self, phi):
        C = self.n_classes
        logits = jnp.zeros(C)
        idx_free = [c for c in range(C) if c != self.base_class]
        for i, c in enumerate(idx_free):
            logits = logits.at[c].set(phi[i])
        logits = logits - jnp.max(logits)
        p = jnp.exp(logits)
        return p / jnp.sum(p)

    # ------------------------------------------------------------------
    # PREDICTION / NULL
    # ------------------------------------------------------------------
    def _compute_prop_alts(self, betas_norm, posterior):
        J = self.J
        real = self.panel_mask > 0
        total_real = max(float(real.sum()), 1.0)
        obs = np.array([float(self.y[:, :, j][real].sum()) for j in range(J)]) / total_real
        pred = np.zeros(J)
        for c in range(self.n_classes):
            idx = self._class_specs[c]
            Xc = self.X[:, :, :, idx]
            util = np.einsum('npjk,k->npj', Xc, betas_norm[c])
            util = np.where(self.avail > 0, util, -1e10)
            util = util - util.max(axis=2, keepdims=True)
            expu = np.exp(util) * self.avail
            denom = np.clip(expu.sum(axis=2, keepdims=True), 1e-300, None)
            probs = expu / denom
            w = posterior[:, c][:, None, None] * real[:, :, None]
            pred += (probs * w).sum(axis=(0, 1))
        pred = pred / total_real
        return pred, obs

    def class_specific_choice_shares(self):
        n_cols = self.X.shape[3]
        J = self.J
        K = self.n_classes
        real = self.panel_mask > 0
        posterior = np.asarray(self.posterior)
        shares = np.zeros((K, J))
        for c in range(K):
            spec = np.asarray(self._class_specs[c])
            real_idx = spec[spec < n_cols]
            Xc = self.X[:, :, :, real_idx]
            util = np.einsum('npjk,k->npj', Xc, self.class_betas_norm[c])
            util = np.where(self.avail > 0, util, -1e10)
            util = util - util.max(axis=2, keepdims=True)
            expu = np.exp(util) * self.avail
            denom = np.clip(expu.sum(axis=2, keepdims=True), 1e-300, None)
            probs = expu / denom
            w = posterior[:, c][:, None, None] * real[:, :, None]
            num = (probs * w).sum(axis=(0, 1))
            den = w.sum()
            shares[c] = num / den if den > 0 else np.nan
        return shares

    def get_loglik_null(self):
        avail_counts = np.clip(self.avail.sum(axis=2), 1.0, None)
        real = self.panel_mask > 0
        self.loglik_null = float(-np.sum(np.log(avail_counts[real])))
        return self.loglik_null

# =====================================================================
# LatentClassConditionalFinal — purchase-stage extension of LatentClassFinal
# =====================================================================
# Same engine as LatentClassFinal (vmap-batched multistart, internal
# z-score normalization, block-Hessian ridge screen, Konis per chain,
# per-winner flags), extended with a second, nested decision per choice
# occasion: given the preferred scheme (choice, y), whether it is
# purchased (w). Both decisions share the same class-specific utility
# coefficients beta_x; the purchase decision adds one extra, X-independent
# parameter per class (v_s, the "no_purchase" utility), appended AFTER
# beta_x in each class's parameter vector, so _Ks[c] becomes Kc+1.
#
# Every v_s-specific piece follows the same convention as the old
# LatentClassConditional (latent_class.py), so results are directly
# comparable:
#   - v_s is NEVER rescaled: _class_sigma[c] is extended with 1.0 so
#     _resolve_inits, the random-init branch, and _denormalize_report all
#     leave the last slot untouched without any special-casing.
#   - _class_ll splits beta_active into [beta_x, v_s] and adds the
#     two-stage log-likelihood (choice MNL + binary purchase logit)
#     inside the same kernel, so the JAX grad/hessian produced by
#     _build_kernels has the right Kc+1 shape automatically.
#   - _separation_xbar returns [x_chosen, -1] folded by ytilde = 2w-1,
#     matching beta_active's layout.
#   - _finalise / summarise temporarily extend varnames/_class_specs so
#     the parent reader prints "no_purchase" like any other coefficient
#     and the Firth summary's coeff_names stays aligned with _Ks.
# =====================================================================

class LatentClassConditionalFinal(LatentClassFinal):
    """Latent-class conditional logit (Vij et al. 2020), built on the
    LatentClassFinal engine. Same public contract as LatentClassFinal,
    plus `w` (purchase indicator) as the third positional argument of
    setup()."""

    def __init__(self):
        super().__init__()
        self.descr = "LCC-Final"

    def _denormalize_report(self, betas_norm, inter, gamma):
        super()._denormalize_report(betas_norm, inter, gamma)

        C = self.n_classes
        n_phi = C - 1

        # v_s (último coeff de cada clase) entra linealmente en el logit del
        # purchase, así que NO hereda la cancelación softmax del choice: en
        # escala normalizada quedó centrado y hay que devolverlo a la escala
        # original con el shift Σ_k β_orig[k]·μ[k]. Recomponemos class_betas
        # desde betas_norm para no depender de si el padre ya lo shiftó.
        class_betas = [np.asarray(betas_norm[c]) / self._class_sigma[c]
                       for c in range(C)]
        for c in range(C):
            spec_c = np.asarray(self._class_specs[c])
            mu_c   = self._norm_mu[spec_c]
            shift  = float(np.dot(class_betas[c][:-1], mu_c))
            class_betas[c][-1] += shift
        self.class_betas = class_betas
        self.coeff_est = np.concatenate(class_betas) if class_betas else np.empty(0)

        # summarise() imprime desde el vector plano se_params, no desde
        # class_betas → aplicar el mismo shift ahí.
        if getattr(self, 'se_params', None) is None:
            return
        params = np.asarray(self.se_params).copy()
        cov_n = getattr(self, 'cov_theta', None)
        cov_o = getattr(self, 'cov_opg', None)
        se_v = np.asarray(self.stderr).copy() if self.stderr is not None else None
        z_v = np.asarray(self.zvalues).copy() if self.zvalues is not None else None
        p_v = np.asarray(self.pvalues).copy() if self.pvalues is not None else None
        opg_v = np.asarray(self.opg_se).copy() if getattr(self, 'opg_se', None) is not None else None
        for c in range(C):
            spec_c = np.asarray(self._class_specs[c])
            mu_c   = self._norm_mu[spec_c]
            shift  = float(np.dot(class_betas[c][:-1], mu_c))
            start  = n_phi + int(np.sum(self._Ks[:c]))
            idx    = start + int(self._Ks[c]) - 1
            params[idx] += shift
            if cov_n is None or se_v is None:
                continue
            # Delta method: d(v_s_orig) = d(v_s) + sum_k (mu_k / sigma_k) d(beta_k)
            g = np.zeros(np.asarray(cov_n).shape[0])
            g[idx] = 1.0
            g[start:idx] = mu_c / self._class_sigma[c][:-1]
            var = float(g @ np.asarray(cov_n) @ g)
            se_i = np.sqrt(var) if var > 0 else 0.0
            z_i = params[idx] / se_i if se_i > 0 else np.nan
            se_v[idx] = se_i
            if z_v is not None:
                z_v[idx] = z_i
            if p_v is not None:
                p_v[idx] = 2 * (1 - _scipy_norm.cdf(abs(z_i))) if se_i > 0 else np.nan
            if cov_o is not None and opg_v is not None:
                var_o = float(g @ cov_o @ g)
                opg_v[idx] = np.sqrt(var_o) if var_o > 0 else np.nan
        self.se_params = params
        if se_v is not None:
            self.stderr = se_v
            self.zvalues = z_v
            self.pvalues = p_v
        if opg_v is not None:
            self.opg_se = opg_v

    def setup(self, X, y, w, varnames, ids, alts, avail=None,
              class_params_spec=None, member_params_spec=None,
              base_class=None, betas0=None, gamma0=None, inter0=None,
              **kwargs):
        ids_arr = np.asarray(ids)
        alts_arr = np.asarray(alts)
        w_arr = np.asarray(w, dtype=float)
        ind_id_arg = kwargs.get('ind_id')

        super().setup(
            X=X, y=y, varnames=varnames, ids=ids_arr, alts=alts_arr,
            avail=avail, class_params_spec=class_params_spec,
            member_params_spec=member_params_spec, base_class=base_class,
            betas0=betas0, gamma0=gamma0, inter0=inter0, **kwargs
        )

        # ---- reconstruct w into (N, P) using the same slot logic ----
        ind_id_arr = np.asarray(ind_id_arg) if ind_id_arg is not None else ids_arr
        order = np.lexsort((alts_arr, ids_arr))
        ids_s = ids_arr[order]
        ind_s = ind_id_arr[order]
        w_s = w_arr[order]

        uniq_sit, counts = np.unique(ids_s, return_counts=True)
        if np.any(counts != self.J):
            raise ValueError("LatentClassConditionalFinal requires balanced "
                             "long-format data by choice id.")
        n_sit = len(uniq_sit)

        w3 = w_s.reshape(n_sit, self.J)[:, 0]
        ind_per_sit = ind_s.reshape(n_sit, self.J)[:, 0]
        _, inv = np.unique(ind_per_sit, return_inverse=True)

        wp = np.zeros((self.N, self.P))
        slot = np.zeros(self.N, dtype=int)
        for s in range(n_sit):
            i = inv[s]
            t = slot[i]
            wp[i, t] = w3[s]
            slot[i] += 1

        self.w = wp
        self._wd = jnp.asarray(self.w)

        # ---- bump _Ks and extend _class_sigma with 1.0 for v_s ----
        self._Ks = self._Ks + 1
        self._class_sigma = [np.append(sig, 1.0) for sig in self._class_sigma]

        # ---- extend the resolved betas0 inits with a v_s slot ----
        if self._betas0_norm is not None:
            rng = np.random.default_rng(self.random_state + 9999)
            extended = []
            for c in range(self.n_classes):
                dom = (betas0[c]
                       if (isinstance(betas0, list)
                           and c < len(betas0)
                           and isinstance(betas0[c], dict)) else {})
                v_s_init = float(dom.get('no_purchase',
                                          rng.normal(scale=0.05)))
                extended.append(np.concatenate([self._betas0_norm[c],
                                                 [v_s_init]]))
            self._betas0_norm = extended

        return self

    def _class_ll(self, beta_active, c):
        beta_x = beta_active[:-1]
        v_s = beta_active[-1]

        idx = jnp.asarray(self._class_specs[c])
        Xc = jnp.take(self._Xd, idx, axis=3)                     # (N,P,J,Kc)
        util = jnp.einsum('npjk,k->npj', Xc, beta_x)
        util = jnp.where(self._avd > 0, util, -1e10)

        # choice term: standard MNL over J alternatives
        util_c = util - jnp.max(util, axis=2, keepdims=True)
        expu = jnp.exp(util_c) * self._avd
        denom = jnp.clip(jnp.sum(expu, axis=2, keepdims=True), MIN_COMP)
        logp_choice = util_c - jnp.log(denom)
        ll_choice = jnp.sum(self._yd * logp_choice, axis=2)      # (N,P)

        # purchase term: binary logit, chosen-alt utility vs v_s
        util_chosen = jnp.sum(self._yd * util, axis=2)           # (N,P)
        m = jnp.maximum(util_chosen, v_s)
        log_denom_w = m + jnp.log(jnp.exp(util_chosen - m)
                                   + jnp.exp(v_s - m))
        logp_buy = util_chosen - log_denom_w
        logp_nobuy = v_s - log_denom_w
        ll_purchase = self._wd * logp_buy + (1.0 - self._wd) * logp_nobuy

        ll_np = ll_choice + ll_purchase
        return jnp.sum(ll_np * self._maskd, axis=1)               # (N,)

    def _separation_xbar(self, c, betas, w_i):
        idx = self._class_specs[c]
        Xc = np.asarray(self.X)[:, :, :, idx]                     # (N,P,J,Kc)
        yc = np.asarray(self.y)                                   # (N,P,J)
        real = np.asarray(self.panel_mask) > 0                    # (N,P)
        w_purch = np.asarray(self.w)                              # (N,P)
        j_chosen = np.argmax(yc, axis=2)                          # (N,P)
        x_chosen = np.take_along_axis(
            Xc, j_chosen[:, :, None, None], axis=2)[:, :, 0, :]   # (N,P,Kc)

        neg_one = -np.ones(x_chosen.shape[:2] + (1,))
        x_full = np.concatenate([x_chosen, neg_one], axis=2)      # (N,P,Kc+1)
        ytilde = 2 * w_purch - 1
        w = w_i[:, None] * real
        xbar = ytilde[:, :, None] * x_full
        return (w[:, :, None] * xbar).reshape(-1, x_full.shape[-1])

    def get_loglik_null(self):
        avail_counts = np.clip(np.asarray(self._avd).sum(axis=2), 1.0, None)
        real = np.asarray(self._maskd) > 0
        ll_choice_null = -np.sum(np.log(avail_counts[real]))
        ll_purchase_null = np.log(0.5) * int(real.sum())
        self.loglik_null = float(ll_choice_null + ll_purchase_null)
        return self.loglik_null

    def _compute_prop_alts(self, betas, posterior):
        J = self.J
        n_cols = self.X.shape[3]
        real = self.panel_mask > 0
        total_real = max(float(real.sum()), 1.0)

        obs = np.array([float(self.y[:, :, j][real].sum())
                        for j in range(J)]) / total_real

        pred = np.zeros(J)
        for c in range(self.n_classes):
            spec = np.asarray(self._class_specs[c])
            real_idx = spec[spec < n_cols]                        # drop fake
            Xc = self.X[:, :, :, real_idx]
            beta_x = np.asarray(self.class_betas_norm[c])[:len(real_idx)]         # drop v_s
            util = np.einsum('npjk,k->npj', Xc, beta_x)
            util = np.where(self.avail > 0, util, -1e10)
            util = util - util.max(axis=2, keepdims=True)
            expu = np.exp(util) * self.avail
            denom = np.clip(expu.sum(axis=2, keepdims=True), 1e-300, None)
            probs = expu / denom
            w = posterior[:, c][:, None, None] * real[:, :, None]
            pred += (probs * w).sum(axis=(0, 1))
        pred = pred / total_real
        return pred, obs

    def _build_coeff_names(self):
        """Extend the parent's list with the v_s ('no_purchase') slot at the
        end of every class, so coeff_names stays aligned 1:1 with _Ks
        (Kc + 1 entries per class) without ever touching _class_specs —
        the latter must remain length-Kc for _class_ll/_compute_se to work."""
        names = []
        for c in range(self.n_classes):
            for v in [self.varnames[i] for i in self._class_specs[c]]:
                names.append(f"class_{c + 1}_{v}")
            names.append(f"class_{c + 1}_no_purchase")
        return names

    def summarise(self, file=None, dashboard=True):
        extra_idx = len(self.varnames)
        varnames_ext = list(self.varnames) + ["no_purchase"]
        specs_ext = [np.append(np.asarray(spec), extra_idx)
                     for spec in self._class_specs]
        orig_varnames, orig_specs = self.varnames, self._class_specs
        self.varnames, self._class_specs = varnames_ext, specs_ext
        try:
            return super().summarise(file=file, dashboard=dashboard)
        finally:
            self.varnames, self._class_specs = orig_varnames, orig_specs

    def class_specific_choice_shares(self):
        n_cols = self.X.shape[3]
        J = self.J
        K = self.n_classes
        real = self.panel_mask > 0
        posterior = np.asarray(self.posterior)
        w_obs = np.asarray(self.w)

        shares = np.zeros((K, J + 1))
        for c in range(K):
            spec = np.asarray(self._class_specs[c])
            real_idx = spec[spec < n_cols]
            Xc = self.X[:, :, :, real_idx]
            beta_x = np.asarray(self.class_betas[c])[:len(real_idx)]

            util = np.einsum('npjk,k->npj', Xc, beta_x)
            util = np.where(self.avail > 0, util, -1e10)
            util = util - util.max(axis=2, keepdims=True)
            expu = np.exp(util) * self.avail
            denom = np.clip(expu.sum(axis=2, keepdims=True), 1e-300, None)
            probs = expu / denom

            w = posterior[:, c][:, None, None] * real[:, :, None]
            num = (probs * w).sum(axis=(0, 1))
            den = w.sum()
            shares[c, :J] = num / den if den > 0 else np.nan

            w2 = posterior[:, c][:, None] * real
            mass = float(w2.sum())
            shares[c, J] = (float((w2 * w_obs).sum() / mass)
                            if mass > 0 else float('nan'))

        return shares

    def class_specific_purchase_rates(self):
        return self.class_specific_choice_shares()[:, self.J]