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

import jax
import jax.numpy as jnp
from jax import lax, grad, hessian

jax.config.update("jax_enable_x64", True)

try:
    from _choice_model import DiscreteChoiceModel
    from separation_screen import (
        is_degenerate_candidate, collinearity_diag,
        konis_separation_test, attribute_separation, sign_breakdown,
    )
except ImportError:
    from ._choice_model import DiscreteChoiceModel
    from .separation_screen import (
        is_degenerate_candidate, collinearity_diag,
        konis_separation_test, attribute_separation, sign_breakdown,
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
    R, ll = model._estep(betas, inter, gamma, pi)

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
    #_, ll_new = model._estep(new_betas, new_inter, new_gamma, new_pi)
    #return new_betas, new_inter, new_gamma, new_pi, ll_new, R, any_accepted, gnorm_max

    return new_betas, new_inter, new_gamma, new_pi, ll, R, any_accepted, gnorm_max


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
                b_orig = []
                for c in range(C):
                    names_c = [self.varnames[i] for i in self._class_specs[c]]
                    dom = betas0[c] if c < len(betas0) else {}
                    vals = [dom.get(v, float(rng.normal(scale=0.05))) for v in names_c]
                    b_orig.append(np.array(vals, dtype=float))
            else:
                b_orig = [np.asarray(b, dtype=float) for b in betas0]
            b_norm = [b_orig[c] * self._class_sigma[c] for c in range(C)]

        # intercepts (membership)
        inter = None
        if inter0 is not None:
            if isinstance(inter0, dict):
                inter = np.array([inter0.get(c, 0.0) for c in self._intercept_free_classes])
            else:
                inter_arr = np.asarray(inter0, dtype=float)
                if inter_arr.shape == (self._n_inter,):
                    inter = inter_arr
        # gammas
        g_norm = None
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
        if not self._has_membership or self.K_membership == 0:
            return np.zeros((0, 0))
        active = np.where(self._member_mask[s] > 0)[0]
        if active.size == 0:
            return np.zeros((0, 0))
        Z = np.asarray(self.X_membership)[:, active]
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
            dll_report = jnp.where(already, 0.0, dll)
            step_report = jnp.where(already, 0.0, step_norm)

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
                            names_m = [self.membership_vars[i] for i in active]
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
        eligible = np.where(~degen_flags & ~sep_choice & ~sep_member & conv_f)[0]

        if eligible.size == 0:
            # No fully-converged clean chain — fall back to any clean chain.
            eligible = np.where(~degen_flags & ~sep_choice & ~sep_member)[0]

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
        self.coeff_names = []
        for c in range(C):
            for v in [self.varnames[i] for i in self._class_specs[c]]:
                self.coeff_names.append(f"class_{c+1}_{v}")

        # num_params / AIC / BIC
        n_beta = int(self._Ks.sum())
        n_inter = self._n_inter if self._has_membership else 0
        n_gamma_active = int(self._member_mask.sum()) if self._has_membership else 0
        if self._has_membership:
            self.num_params = n_beta + n_inter + n_gamma_active
        else:
            self.num_params = n_beta + (C - 1)
        self.aic = 2 * self.num_params - 2 * self.loglik
        self.bic = np.log(self.sample_size) * self.num_params - 2 * self.loglik

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
                    gmap['_inter'] = float(self.class_gammas[i_int])
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
            entropy = 1.0 - float(-np.sum(R * logR)) / (self.N * np.log(C_))
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
            for j, (c, v) in enumerate(active):
                opg[off + j] /= self._mem_sigma[v]
        if opg is not None:
            self.opg_se = opg
        # intercepts (n_inter): unchanged
        off += self._n_inter
        # gamma active: divide by corresponding mem_sigma
        if self._has_membership and self.K_membership > 0:
            active = [(c, v) for c in range(C) for v in range(self.K_membership)
                      if self._member_mask[c, v] > 0]
            for j, (c, v) in enumerate(active):
                s = self._mem_sigma[v]
                params[off + j] /= s
                if se is not None:
                    se[off + j] /= s

        self.se_params = params
        if se is not None:
            self.stderr = se

        # gamma_params / gamma_se
        if getattr(self, "gamma_params", None) is not None and len(self.gamma_params) > 0:
            gp = np.asarray(self.gamma_params).copy()
            gs = np.asarray(self.gamma_se).copy() if self.gamma_se is not None else None
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
            self.gamma_params = gp
            if gs is not None:
                self.gamma_se = gs

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
            def per_individual_ll_flat(v):
                b, ii, gg = unpack(v)
                ll_c = self._class_ll_matrix(b)
                if self._has_membership:
                    logH = self._membership_log_probs(ii, gg)
                else:
                    pi_arg = _phi_to_pi_inner(v[:n_phi])
                    logpi = jnp.log(jnp.clip(pi_arg, MIN_COMP))
                    logH = jnp.broadcast_to(logpi[None, :], (self.N, C))
                return jax.scipy.special.logsumexp(ll_c + logH, axis=1)

            jac_fn = jax.jit(jax.jacfwd(per_individual_ll_flat))
            jac = np.asarray(jac_fn(theta_j))
            JtJ = jac.T @ jac
            try:
                L = np.linalg.cholesky(JtJ + 1e-12 * np.eye(JtJ.shape[0]))
                cov_opg = np.linalg.inv(L.T) @ np.linalg.inv(L)
            except np.linalg.LinAlgError:
                cov_opg = np.linalg.pinv(JtJ)
            opg_se_full = np.sqrt(np.clip(np.diag(cov_opg), 0, None))
        else:
            opg_se_full = np.full_like(theta, np.nan)

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
        self.cond_number = float(np.linalg.cond(H)) if H.size else float("nan")
        self.opg_hess_ratio = opg_hess_ratio
        self.opg_hess_flag = opg_hess_flag

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

        H_list, se_raw_blocks = [], []
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
            off += kc

        # Membership diagnostics (raw Hessian, active submatrix)
        gamma_se_raw_dense = np.full(n_inter + n_gamma_dense, np.nan)
        gamma_collin_dense = np.full(n_inter + n_gamma_dense, np.nan)
        gamma_flag_dense = np.zeros(n_inter + n_gamma_dense, dtype=bool)
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

        gsr, gsc, gsf = [], [], []
        for i in range(n_inter):
            gsr.append(gamma_se_raw_dense[i])
            gsc.append(gamma_collin_dense[i])
            gsf.append(gamma_flag_dense[i])
        for c in range(C):
            for k in range(Km):
                if self._member_mask[c, k] > 0:
                    idx = n_inter + c * Km + k
                    gsr.append(gamma_se_raw_dense[idx])
                    gsc.append(gamma_collin_dense[idx])
                    gsf.append(gamma_flag_dense[idx])
        self.gamma_se_raw = np.array(gsr)
        self.gamma_collin_share = np.array(gsc)
        self.gamma_collin_flag = np.array(gsf, dtype=bool)

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
        """Per-variable Wald tests between class pairs, choice block only.
        Prints only variables that FAIL to reject equality at `alpha`.
        Prints nothing at all if every shared variable is distinct.
        `file=None` routes to stdout; pass a file handle to also route there."""
        from scipy.stats import chi2
        C = self.n_classes
        n_phi = C - 1
        theta = np.asarray(self.se_params)
        cov = np.asarray(self.cov_theta)

        def p(text=""):
            print(text, file=file)

        # Two-pass: collect rows first, print header only if there's something.
        all_rows = []
        for c1 in range(C):
            for c2 in range(c1 + 1, C):
                spec1 = list(self._class_specs[c1])
                spec2 = list(self._class_specs[c2])
                common = [v for v in spec1 if v in spec2]
                if not common:
                    continue

                off1 = n_phi + int(np.sum(self._Ks[:c1]))
                off2 = n_phi + int(np.sum(self._Ks[:c2]))
                rows = []
                for v in common:
                    i1 = off1 + spec1.index(v)
                    i2 = off2 + spec2.index(v)
                    b1 = float(theta[i1]); b2 = float(theta[i2])
                    var_d = (float(cov[i1, i1]) + float(cov[i2, i2])
                            - 2.0 * float(cov[i1, i2]))
                    if var_d <= 0:
                        continue
                    W = (b1 - b2) ** 2 / var_d
                    p_val = float(1.0 - chi2.cdf(W, 1))

                    if p_val >= alpha:
                        if p_val >= 0.10:
                            verdict = "NO DISTINCT"
                        else:
                            verdict = "BORDERLINE"
                        rows.append((self.varnames[v], b1, b2, W, p_val, verdict))

                if rows:
                    all_rows.append((c1, c2, rows))

        if not all_rows:
            return

        p()
        p("=" * 78)
        p("  CLASS-PAIR WALD TESTS")
        p("=" * 78)
        for c1, c2, rows in all_rows:
            p()
            p(f"  Class {c1+1} vs Class {c2+1}  — Choice-parameters NOT "
            f"significantly different (p >= {alpha})")
            p()
            p(f"  {'Var':<22}{'beta_'+str(c1+1):>12}"
            f"{'beta_'+str(c2+1):>12}{'Wald':>10}{'p-val':>10}  Verdict")
            p(f"  {'-'*22}{'-'*12}{'-'*12}{'-'*10}{'-'*10}  {'-'*16}")
            for name, b1, b2, W, p_val, verdict in rows:
                p(f"  {name:<22}{b1:>12.4f}{b2:>12.4f}"
                f"{W:>10.2f}{p_val:>10.4f}  {verdict}")

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
            util = np.einsum('npjk,k->npj', Xc, self.class_betas[c])
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