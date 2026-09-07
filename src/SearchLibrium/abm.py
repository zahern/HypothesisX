"""Activity-based modelling (ABM) stage presets for SearchLibrium.

Ports the orchestration knowledge from the SEQ activity-based pipeline
(``FRAMEWORK_PLAN.md`` + ``main_2.py`` stages in ``Z:/test_runs_tours/code/``)
into declarative search configurations, so a tour-based model system can be
built as a sequence of SearchLibrium searches — one per stage — instead of a
bespoke script per stage.

Stages
------
``frequency``    — activity frequency (ordered logit/probit, count models)
``day_budget``   — day duration budget (censored/selection models)
``duration``     — activity duration (survival/AFT models)
``timing``       — schedule placement / time-of-day (MNL, nested logit)
``destination``  — tour/trip destination choice (Larch MNL, sampled alts)
``mode``         — trip mode choice (Larch nested logit)
``parking``      — parking choice (MNL)

Each preset (:func:`stage_search_config`) reports the candidate
``models`` for ``Parameters``, a recommended ``algorithm``, sampling and
estimation defaults, and which estimator family actually fits what
(``estimator_map`` — some families live in sibling packages:
MetaCountRegressor for counts/durations, lifelines for AFT, Larch for
large destination/mode choice).

Usage::

    from SearchLibrium.abm import stage_search_config
    cfg = stage_search_config("mode", n_zones=371)
    params = Parameters(models=cfg["models"], nests=cfg.get("nests"), ...)
    best = call_search(params, algorithm=cfg["algorithm"])
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

STAGES = ("frequency", "day_budget", "duration", "timing",
          "destination", "mode", "parking")

#: Which package fits what (SearchLibrium = this package, searched natively).
ESTIMATOR_MAP = {
    "ordered_logit": "SearchLibrium (searched)",
    "ordered_probit": "SearchLibrium (searched)",
    "zero_inflated_ordered_probit": "SearchLibrium (searched)",
    "multinomial": "SearchLibrium (searched)",
    "nested_logit": "SearchLibrium (searched)",
    "larch_mnl": "SearchLibrium[larch] (searched)",
    "larch_nested": "SearchLibrium[larch] (searched)",
    "larch_mixed": "SearchLibrium[larch] (searched)",
    "mixed_logit": "SearchLibrium (searched)",
    "random_regret": "SearchLibrium (searched)",
    "count_nb2_zip_zinb": "MetaCountRegressor (external)",
    "tobit_lognormal": "MetaCountRegressor (external)",
    "heckman": "SearchLibrium + statsmodels OLS step",
    "aft_weibull_lognormal": "lifelines or MetaCountRegressor (external)",
    "pylogit_mnl": "pylogit (cross-check only)",
}

_STAGE_DEFAULTS = {
    "frequency": {
        "description": "Activity frequency per purpose/day (ordered outcomes).",
        "models": ["ordered_logit", "ordered_probit"],
        "algorithm": "agds",
        "notes": "Count alternatives (NB2/ZIP/ZINB) via MetaCountRegressor sit "
                 "alongside; SearchLibrium covers the ordered family.",
    },
    "day_budget": {
        "description": "Day time-budget / censoring and participation selection.",
        "models": ["multinomial"],
        "algorithm": "agds",
        "notes": "Tobit/lognormal via MetaCountRegressor; Heckman two-step via "
                 "SearchLibrium.selection_models + OLS correction.",
    },
    "duration": {
        "description": "Activity duration (positive durations, budget-rescaled).",
        "models": ["multinomial"],
        "algorithm": "agds",
        "notes": "AFT Weibull/lognormal/log-logistic via lifelines or "
                 "MetaCountRegressor; MDCEVModel for joint time-use.",
    },
    "timing": {
        "description": "Schedule placement over TOD periods (e.g. early/am_peak/"
                       "midday/pm_peak/evening/late).",
        "models": ["multinomial", "nested_logit", "random_regret", "larch_mnl"],
        "algorithm": "agds",
        "notes": "Provide param.nests for the nested arms; Larch MNL arm suits "
                 "large period × purpose estimations.",
    },
    "destination": {
        "description": "Tour/trip destination choice over many zones.",
        "models": ["larch_mnl", "multinomial"],
        "algorithm": "agds",
        "n_draws": 200,
        "notes": "Importance-sample alternatives (e.g. 20 of 371) — full "
                 "enumeration OOMs (see jax_utils.estimate_row_guard). Skims "
                 "via SearchLibrium.skims; triple-check zone key-spaces.",
    },
    "mode": {
        "description": "Trip mode choice (e.g. DA/SR/Transit/Walk/Bike with "
                       "Motorised/NonMotorised nests).",
        "models": ["larch_nested", "nested_logit", "multinomial", "larch_mnl"],
        "algorithm": "agds",
        "notes": "Larch nested arm is the primary for JAX autodiff; native "
                 "nested arm cross-checks small segments.",
    },
    "parking": {
        "description": "Parking choice for auto trips (OnStreet/Garage/PnR).",
        "models": ["multinomial", "larch_mnl"],
        "algorithm": "agds",
        "notes": "Small choice sets; native MNL usually suffices.",
    },
}


def available_stages():
    """List the ABM stage names with one-line descriptions."""
    return {s: _STAGE_DEFAULTS[s]["description"] for s in STAGES}


def stage_search_config(stage, n_zones=None, nests=None, larch_opts=None):
    """Recommended search configuration for an ABM stage.

    Parameters
    ----------
    stage : one of :data:`STAGES`
    n_zones : int or None — destination/mode zone count; recorded in the
        config and used to flag sampling needs
    nests : optional nest map overriding the stage default expectation
    larch_opts : optional dict forwarded as ``Parameters(larch_opts=...)``

    Returns
    -------
    dict with ``stage/models/algorithm`` plus stage-specific hints
    (``n_draws``, ``needs_nests``, ``needs_sampling``, ``notes``).
    """
    if stage not in _STAGE_DEFAULTS:
        raise ValueError(f"Unknown ABM stage '{stage}'. Choose from {list(STAGES)}.")
    cfg = dict(_STAGE_DEFAULTS[stage])
    cfg["stage"] = stage
    cfg["needs_nests"] = any("nested" in m for m in cfg["models"])
    if nests is not None:
        cfg["nests"] = dict(nests)
    if larch_opts is not None:
        cfg["larch_opts"] = dict(larch_opts)
    if stage == "destination" and n_zones and int(n_zones) > 50:
        cfg["needs_sampling"] = True
        cfg.setdefault("sample_size", 20)
    return cfg


__all__ = [
    "STAGES",
    "ESTIMATOR_MAP",
    "available_stages",
    "stage_search_config",
]
