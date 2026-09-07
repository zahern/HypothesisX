"""JAX environment helpers for SearchLibrium.

Promoted from the SEQ activity-based pipeline (``larch_jax_enforce.py`` in
``Z:/test_runs_tours/code/``), generalized so any estimator — Larch, native
JAX models, or MetaCountRegressor — shares one CPU-first setup instead of
each pipeline script re-implementing it.

Large discrete-choice models (hundreds of zones × thousands of cases) must
run on JAX with exact autodiff gradients: numba finite-difference gradients
stall (SLSQP at iteration 0), float32 flattens the log-likelihood, and CUDA
Hessians fail on CPU nodes. These helpers:

* :func:`ensure_jax_environment` — set CPU-only float64 env vars + enable
  x64 *before* jax is first imported (must precede any jax array creation).
* :func:`is_large_model` — cases × alts above ``LARCH_JAX_THRESHOLD``
  (default 50k cells) counts as "large".
* :func:`estimate_row_guard` — memory guard for dense n_obs × n_zone IDCA
  expansions; tells you when importance sampling is mandatory.
* :func:`check_gradient_health` — probe ``d_logloss`` at start values:
  finite and non-zero, otherwise estimation will "converge" spuriously.

Environment knobs::

    LARCH_FORCE_JAX=1 (default) — prefer jax; numba only as fallback
    LARCH_JAX_THRESHOLD=50000    — cases×alts above this counts as large
    JAX_ENABLE_X64=1, JAX_PLATFORMS=cpu, JAX_PLATFORM_NAME=cpu,
    CUDA_VISIBLE_DEVICES="", TF_CPP_MIN_LOG_LEVEL=3
"""

from __future__ import annotations

import logging
import os

log = logging.getLogger(__name__)


def ensure_jax_environment() -> bool:
    """Set CPU-only float64 JAX env vars + enable x64.

    Idempotent and safe to call at import time. Returns True if jax is
    importable afterwards.
    """
    os.environ.setdefault('JAX_ENABLE_X64', '1')
    os.environ.setdefault('JAX_PLATFORMS', 'cpu')
    os.environ.setdefault('JAX_PLATFORM_NAME', 'cpu')
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
    os.environ.setdefault('XLA_FLAGS', '--xla_force_host_platform_device_count=1')
    if os.environ.get('JAX_ENABLE_X64', '1') == '0':
        try:
            import jax  # noqa: F401
            return True
        except Exception:
            return False
    try:
        import jax
        try:
            jax.config.update('jax_enable_x64', True)
        except Exception as e:  # noqa: BLE001
            log.warning("jax_utils: jax_enable_x64 failed (%s)", e)
        try:
            import jax.numpy as jnp  # noqa: F401
            # Touch a tiny array to force XLA init now (fail fast, not mid-fit).
            _ = jnp.zeros(1, dtype=jnp.float64).block_until_ready()
        except Exception:
            pass
        if not jax.config.jax_enable_x64:
            log.warning("jax_utils: jax_enable_x64 is False — "
                        "gradients unreliable in float32")
        return True
    except ImportError:
        log.warning("jax_utils: jax not installed — JAX-only models cannot fit "
                    "(pip install 'jax[cpu]')")
        return False
    except Exception as e:  # noqa: BLE001
        log.warning("jax_utils: jax init failed (%s)", e)
        return False


def is_large_model(n_cases=None, n_alts=None, n_cells=None) -> bool:
    """True when cases × alts exceeds ``LARCH_JAX_THRESHOLD`` (default 50k)."""
    try:
        thresh = int(os.environ.get('LARCH_JAX_THRESHOLD', '50000'))
    except Exception:
        thresh = 50000
    try:
        if n_cells is not None:
            return int(n_cells) > thresh
        if n_cases is not None and n_alts is not None:
            return int(n_cases) * int(n_alts) > thresh
    except Exception:
        pass
    # Unknown size → assume large when forcing (safe side: JAX handles both).
    return os.environ.get('LARCH_FORCE_JAX', '1').strip() != '0'


def estimate_row_guard(n_obs, n_zones, sample_size=0) -> dict:
    """Memory guard for dense n_obs × n_zones IDCA expansions.

    A full 371-zone enumeration at 50k observations is ~18.5M rows and OOMs;
    destination models must use importance sampling
    (``sample_size`` ~ 20). Returns a dict with ``full_rows``,
    ``sampled_rows``, ``est_gb`` (float64, ~30 cols), ``use_sampling`` and
    ``oom_risk`` flags.
    """
    full = int(n_obs) * int(n_zones)
    samp = int(n_obs) * int(sample_size) if sample_size and sample_size > 0 else full
    # ~30 float64 cols × 8 bytes ≈ 240 B/row + pandas/xarray overhead ×2.
    est_gb = samp * 240 * 2 / 1e9
    return {
        'n_obs': int(n_obs), 'n_zones': int(n_zones),
        'sample_size': int(sample_size),
        'full_rows': full, 'sampled_rows': samp,
        'est_gb': round(float(est_gb), 2),
        'use_sampling': bool(sample_size and sample_size > 0 and full > samp),
        'oom_risk': bool(est_gb > 4.0 or full > 10_000_000),
    }


def check_gradient_health(model, label="model"):
    """Probe penalized ``d_logloss`` at start values: finite & non-zero?

    Returns ``(ok, grad_norm)``. A dead gradient makes scipy optimizers
    "converge" at iteration 0 with every coefficient left at its prior,
    which downstream code then misreads as a legitimate fit.
    """
    try:
        import numpy as np
        x0 = np.asarray(model.pvals, dtype=float)
        g = np.asarray(model.d_logloss(x0), dtype=float)
        finite = bool(np.all(np.isfinite(g)))
        norm = float(np.linalg.norm(g)) if finite else float('nan')
        ok = finite and norm > 0.0
        if not ok:
            log.warning("jax_utils: %s gradient unhealthy (finite=%s, |g|=%.3e, "
                        "engine=%s)", label, finite, norm,
                        getattr(model, 'compute_engine', '?'))
        return ok, norm
    except Exception as e:  # noqa: BLE001
        log.warning("jax_utils: %s gradient probe failed (%r)", label, e)
        return False, float('nan')


__all__ = [
    "ensure_jax_environment",
    "is_large_model",
    "estimate_row_guard",
    "check_gradient_health",
]
