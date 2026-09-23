"""Optional numba execution engine for SearchLibrium choice models.

This module is the registry / dispatcher. It answers two questions without
fitting anything:

* :func:`engine_supported` — which numba family (if any) covers a model
  instance: ``'mxl'`` (MixedLogit), ``'rrm'`` (fixed RandomRegret),
  ``'mrrm'`` (MixedRandomRegret), or ``None`` (unsupported -> SciPy).
* :func:`make_minimiser_for` — a SciPy-compatible ``minimise_func`` for
  models whose ``fit()`` honours the ``minimise_func`` hook (currently
  MixedLogit). Models without that hook (RRM pair) are driven through the
  first-class ``engine='numba'`` setup kwarg instead; see below.

Uniform usage (all supported models)::

    model.setup(..., engine='numba')   # or: model.engine = 'numba'
    model.fit(...)                     # njit likelihood + L-BFGS-B/BFGS;
                                       # silent fallback to SciPy/JAX when
                                       # the spec is outside the supported
                                       # base case

Global default (run everything on numba instead of JAX)::

    import os
    os.environ['SL_ENGINE'] = 'numba'  # or: numba_engine.set_default_engine('numba')

    model = MixedLogit()   # engine attr unset -> resolves to 'numba'
    model.setup(...); model.fit(...)

An explicit per-model ``engine`` (setup kwarg or attribute) or an explicit
``_jax=True/False`` always wins over the global default; unset means
"follow the global default", and with no global default the legacy
behaviour (JAX-first) is preserved. Models without a numba port
(MNL, Nested, Ordered, ...) run their numpy/scipy path when the global
default is numba — i.e. JAX is switched off, not replaced by njit.

Search-level wiring reads ``params.engine`` (see ``Search.fit_mxl``,
``Search.fit_random_regret``, ``Search.evaluate_mixed_rrm``).

Coverage roadmap
----------------
* MixedLogit — done (``numba_mxl``).
* RandomRegret / MixedRandomRegret — done (``numba_rrm``).
* MultinomialLogit, NestedLogit/MixedNested, OrderedLogit,
  LatentClassMixedLogit, ExplodedLogit — not yet ported; the dispatcher
  returns ``None`` and everything runs exactly as before. Each needs its
  likelihood ported to ``@njit`` the same way (mirror the JAX/fast-path
  math, validate against the numpy path, register here).
"""

_FAMILIES = ('mxl', 'rrm', 'mrrm')

# ---------------------------------------------------------------------------
# Global default engine (SL_ENGINE env var + programmatic override).
# ---------------------------------------------------------------------------
import os as _os

_ENGINE_ENV_VAR = 'SL_ENGINE'

_default_engine_override = None


def set_default_engine(name):
    """Set the process-wide default engine (``'numba'``/``'jax'``/``None``).

    ``None`` (the default) clears the override so the ``SL_ENGINE``
    environment variable — or, when that is also unset, the legacy
    JAX-first behaviour — applies. An explicit per-model ``engine``
    always wins over this default.
    """
    global _default_engine_override
    if name is None:
        _default_engine_override = None
        return None
    norm = str(name).strip().lower() or None
    _default_engine_override = norm
    return norm


def get_default_engine():
    """Return the global default engine name, or ``None`` for legacy JAX-first.

    The programmatic override (:func:`set_default_engine`) wins over the
    ``SL_ENGINE`` environment variable.
    """
    if _default_engine_override is not None:
        return _default_engine_override
    try:
        env = _os.environ.get(_ENGINE_ENV_VAR, '')
        return str(env).strip().lower() or None
    except Exception:
        return None


def numba_as_default():
    """True when the global default engine resolves to ``'numba'``."""
    try:
        return (get_default_engine() or '') == 'numba'
    except Exception:
        return False


def effective_engine(model):
    """Resolve the engine for *model*: explicit attr wins, else global default.

    Returns the normalised engine name (``'numba'``/``'jax'``/...) or
    ``None`` when neither is set (legacy per-model behaviour). Never raises.
    """
    try:
        explicit = getattr(model, 'engine', None)
        if explicit is not None and str(explicit).strip() != '':
            return str(explicit).strip().lower()
    except Exception:
        pass
    try:
        return get_default_engine()
    except Exception:
        return None


def _type_id(model):
    try:
        t = type(model)
        return (t.__module__.split('.')[-1], t.__name__)
    except Exception:
        return (None, None)


def engine_supported(model):
    """Return the numba family key for *model*, or ``None`` if unsupported.

    Never raises; never imports numba (builders do that lazily).
    """
    try:
        mod, name = _type_id(model)
        if (mod, name) == ('MixedLogit', 'MixedLogit'):
            return 'mxl'
        if (mod, name) == ('rrm', 'RandomRegret'):
            return 'rrm'
        if (mod, name) == ('mixedrrm', 'MixedRandomRegret'):
            return 'mrrm'
    except Exception:
        pass
    return None


def make_minimiser_for(model):
    """Return a SciPy-compatible ``minimise_func`` for *model*, or ``None``.

    Only families whose ``fit()`` honours the ``minimise_func`` hook are
    served here (currently MXL). RRM-family models are driven through the
    ``engine='numba'`` setup kwarg instead and intentionally return
    ``None`` here. ``None`` always means "use the standard path".
    """
    try:
        if engine_supported(model) != 'mxl':
            return None
        try:
            from numba_mxl import make_numba_minimiser
        except ImportError:
            from .numba_mxl import make_numba_minimiser
        return make_numba_minimiser(model)
    except Exception:
        return None
