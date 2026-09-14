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
