"""MixedLogitGSE — Gradient Score Enhanced Mixed Logit.

Extends MixedLogit with gradient-latent loadings.

  beta_nk = mu_k + (gamma_k + tau_k * z_nk) * g_nk + sigma_k * eta_nk

Parameter layout (no transforms, no correlation):
  [Br_b(Kr) | gamma(Kr) | gamma_w(Kr) | Br_w(Kr)]   (12 or 16 params)

Usage:
  mxl = MixedLogitGSE()
  mxl.setup(..., gradient_scores=g_avg, random_gamma=False)
  mxl.fit()

Larch equivalence (kept in sync with test_runs_tours stage-5 mode model):
  The model β_nk = μ_k + γ_k·g_nk + σ_k·η_nk decomposes term-by-term into a
  plain Larch mixed logit, since β_nk multiplies x_k in the utility:
      μ_k·x_k          → ordinary fixed utility term         P('c_k')  * X('x_k')
      γ_k·(g_nk·x_k)   → systematic gradient interaction     P('g_k')  * X('x_k * g')
      σ_k·(η_nk·x_k)   → Normal random coefficient           mixtures.append(Normal('c_k','s_k'))
  random_gamma=True adds τ_k·z_nk·g_nk, i.e. promoting the γ_k loadings to
  random coefficients too (Normal mixtures on the g_k params in the Larch build).
  See stage5_destination_choice._estimate_mixed_logit for the mirrored build.
"""

import numpy as np
import jax.numpy as jnp
import jax

from SearchLibrium.MixedLogit import MixedLogit

try:
    from ._device import device as dev
except ImportError:
    from _device import device as dev


class MixedLogitGSE(MixedLogit):
    """Mixed Logit with Gradient Score Enhanced latent loadings."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.Kgrad = 0
        self.Kgrad_w = 0
        self.gamma_draws = None

    def setup(self, *args, gradient_scores=None, random_gamma=False, **kwargs):
        self.random_gamma = bool(random_gamma)
        if gradient_scores is not None:
            # Standardise each gradient column to mean 0 / std 1 (O(1)) so the
            # γ loadings are on a common scale and can't numerically overflow.
            # Centering matters: an uncentred score folds a spurious γ·mean/std
            # shift into μ; centring keeps μ interpretable as the population mean.
            # A constant (zero-variance) column would divide by ~0 and blow up to
            # ~1e8 — guard it by leaving such columns at 0 (they carry no gradient
            # information anyway).
            g = np.asarray(gradient_scores, dtype=float)
            gmu = g.mean(axis=0, keepdims=True)
            gsd = g.std(axis=0, keepdims=True)
            gsd_safe = np.where(gsd < 1e-8, 1.0, gsd)
            self.gradient_scores = (g - gmu) / gsd_safe
            n_const = int(np.sum(gsd < 1e-8))
            print(f"[MixedLogitGSE] Gradient loadings enabled (Kgrad={len(kwargs.get('randvars', {}))}, "
                  f"random_gamma={self.random_gamma}, "
                  f"raw std range=[{float(gsd.min()):.2f}, {float(gsd.max()):.2f}] -> "
                  f"centred+standardised"
                  + (f", {n_const} constant column(s) zeroed" if n_const else "") + ")")
        else:
            self.gradient_scores = None
        super().setup(*args, **kwargs)

    # ── Hook overrides ─────────────────────────────────────────────

    def _n_coeff_extra(self) -> int:
        return self.Kgrad + self.Kgrad_w

    def _bound_extra(self) -> dict:
        inf = float("inf")
        return {"grad": ((-inf, inf), self.Kgrad),
                "grad_w": ((0, inf), self.Kgrad_w)}

    def _beta_segment_extra(self):
        return (["grad", "grad_w"], [self.Kgrad, self.Kgrad_w])

    def _jax_cache_key_extra(self):
        return (self.Kgrad, self.Kgrad_w)

    def _init_pad_arrays(self):
        pads = []
        if self.Kgrad > 0:
            pads.append(np.repeat(0.1, self.Kgrad))
        if self.Kgrad_w > 0:
            pads.append(np.repeat(0.1, self.Kgrad_w))
        return pads

    def _jax_negloglik_extra_kwargs(self):
        gd = (jnp.array(self.gamma_draws) if self.Kgrad_w > 0
              and self.gamma_draws is not None else None)
        gs = (jnp.array(self.gradient_scores) if self.gradient_scores is not None
              else None)
        return dict(Kgrad=self.Kgrad, Kgrad_w=self.Kgrad_w,
                    gradient_scores=gs, gamma_draws=gd)

    # ── Fit override ───────────────────────────────────────────────

    def fit(self):
        if self.gradient_scores is not None:
            self.Kgrad = self.Kr
            self.Kgrad_w = self.Kr if self.random_gamma else 0
            if self.Kgrad_w > 0:
                # Seed the random-γ (τ·z) draws so a fit is reproducible across
                # runs — unseeded np.random.randn made every fit non-deterministic
                # (and non-comparable in a search). Reuse the model's own seed
                # when set, mixing in Kgrad so it differs from the η draws' seed.
                _seed = getattr(self, "seed", None)
                rng = np.random.default_rng(None if _seed is None
                                            else int(_seed) + 10007 * self.Kgrad)
                self.gamma_draws = rng.standard_normal((self.N, self.Kgrad))
        # Force JAX-only (scipy fallback doesn't handle gradient loadings)
        self._jax = True
        super().fit()

    # ── JAX loglik with GSE terms ──────────────────────────────────

    @staticmethod
    def _jax_mxl_negloglik(betas, X_jax, y_jax, panel_info_jax, draws_jax,
                            fxidx, rvidx, Kf, Kr, Kchol, Kbw, rvdist_names,
                            correlationLength,
                            Kgrad=0, Kgrad_w=0,
                            gradient_scores=None, gamma_draws=None,
                            **kwargs):
        """GSE loglik: β_nk = μ_k + γ_k·g_nk + σ_k·η_nk."""
        # Split
        Bf       = betas[:Kf]
        Br_b     = betas[Kf:Kf + Kr]
        gamma_mu = betas[Kf + Kr:Kf + Kr + Kgrad] if Kgrad > 0 else jnp.array([])
        gamma_w  = (betas[Kf + Kr + Kgrad:Kf + Kr + Kgrad + Kgrad_w]
                    if Kgrad_w > 0 else jnp.array([]))
        offset   = Kf + Kr + Kgrad + Kgrad_w
        chol_v   = betas[offset:offset + Kchol]
        Br_w     = betas[offset + Kchol:offset + Kchol + Kbw]

        # Cholesky
        chol_mat = jnp.zeros((Kr, Kr))
        idx = 0
        for r in range(correlationLength):
            for c in range(r + 1):
                chol_mat = chol_mat.at[r, c].set(chol_v[idx]); idx += 1
        for k in range(Kbw):
            chol_mat = chol_mat.at[correlationLength + k,
                                   correlationLength + k].set(jnp.abs(Br_w[k]))

        # Random coefficients
        N = X_jax.shape[0]
        Br = Br_b[:, None] + jnp.einsum("kl,nlr->nkr", chol_mat, draws_jax[:, :Kr, :])

        # GSE gradient-latent term (standardised scores, O(1))
        if Kgrad > 0 and gradient_scores is not None:
            if Kgrad_w > 0 and gamma_draws is not None:
                gamma_rnd = gamma_mu[None, :] + gamma_w[None, :] * gamma_draws
            else:
                gamma_rnd = gamma_mu[None, :]
            Br = Br + gamma_rnd[:, :, None] * gradient_scores[:, :, None]

        # Distribution transforms
        for k, dist in enumerate(rvdist_names):
            if dist == "ln":
                Br = Br.at[:, k, :].set(jnp.exp(Br[:, k, :]))
            elif dist == "tn":
                Br = Br.at[:, k, :].set(jnp.abs(Br[:, k, :]))
            elif dist == "u":
                Br = Br.at[:, k, :].set(Br_b[k] + Br_w[k] * (draws_jax[:, k, :] - 0.5))

        # Utility
        Xr = X_jax[:, :, :, rvidx]
        if Kf > 0:
            Xf = X_jax[:, :, :, fxidx]
            V = (jnp.einsum("npjk,k->npj", Xf, Bf)[:, :, :, None]
                 + jnp.einsum("npjk,nkr->npjr", Xr, Br))
        else:
            V = jnp.einsum("npjk,nkr->npjr", Xr, Br)

        V   = V - jnp.max(V, axis=2, keepdims=True)
        eV  = jnp.exp(V)
        p   = eV / jnp.sum(eV, axis=2, keepdims=True)

        pch   = jnp.sum(y_jax[:, :, :, None] * p, axis=2)
        pch   = jnp.prod(pch, axis=1)
        pch   = jnp.clip(pch, 1e-300, None)
        sim_p = jnp.mean(pch, axis=1)
        sim_p = jnp.clip(sim_p, 1e-300, None)
        return -jnp.sum(jnp.log(sim_p))

    # ── Scipy compute_probabilities override ────────────────────────

    def compute_probabilities(self, betas, X, panel_info, draws, drawstrans,
                              avail, var_list, chol_mat):
        """GSE override: add gradient-latent term to Br."""
        vals = list(var_list.values())
        Bf   = vals[0]
        Br_b = vals[1]
        # GSE segments sit between Br_b and chol/Br_w
        has_gse = len(vals) > 9
        gamma   = vals[2] if has_gse else np.array([])
        gamma_w = vals[3] if has_gse else np.array([])
        chol_v  = vals[4] if has_gse else vals[2]
        Br_w    = vals[5] if has_gse else vals[3]
        Bftrans = vals[6] if has_gse else vals[4] if len(vals) > 4 else np.array([])
        flmbda  = vals[7] if has_gse else vals[5] if len(vals) > 5 else np.array([])

        if dev.using_gpu:
            Bf = dev.convert_array_gpu(Bf)
            Br_b = dev.convert_array_gpu(Br_b)
            Br_w = dev.convert_array_gpu(Br_w)
            if len(Bftrans) > 0: Bftrans = dev.convert_array_gpu(Bftrans)
            if len(flmbda) > 0: flmbda = dev.convert_array_gpu(flmbda)
            if len(gamma) > 0:
                gamma = dev.convert_array_gpu(gamma)
                gamma_w = dev.convert_array_gpu(gamma_w)

        XBf = np.zeros((self.N, self.P, self.J))
        if dev.using_gpu: XBf = dev.convert_array_gpu(XBf)

        if self.Kf != 0:
            Xf = X[:, :, :, self.fxidx]
            XBf = dev.cust_einsum("npjk,k -> npj", Xf, Bf).astype(float)

        XBr = np.zeros((self.N, self.P, self.J, draws.shape[2]))
        if self.Kr != 0:
            tmp = dev.np.matmul(chol_mat[:self.Kr, :self.Kr], draws)
            Br = Br_b[None, :, None] + tmp

            if self.Kgrad > 0 and self.gradient_scores is not None:
                if (self.Kgrad_w > 0 and self.gamma_draws is not None
                        and len(gamma_w) > 0):
                    g_rnd = gamma[None, :] + gamma_w[None, :] * self.gamma_draws
                else:
                    g_rnd = gamma[None, :] if len(gamma) > 0 else np.zeros((1, self.Kr))
                Br = Br + g_rnd[:, :, None] * self.gradient_scores[:, :, None]

            Br = self.draws_generator.apply_distribution(Br, self.rvdist)
            self.Br = Br
            Xr = X[:, :, :, self.rvidx].astype(float)
            XBr = dev.cust_einsum("npjk,nkr -> npjr", Xr, Br)

        V = XBf[:, :, :, None] + XBr if self.Kf != 0 else XBr
        eV = np.exp(np.clip(V, -700, 700))
        probs = eV / np.clip(eV.sum(axis=2, keepdims=True), 1e-300, None)
        return probs
