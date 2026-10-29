import warnings
import argparse
import csv
import faulthandler
import sys
import timeit
from collections import namedtuple
import numpy as np
import pandas as pd
from addicty import Dict
#print('loading devices')

import os

# ---------------------------------------------------------------------------
# Console encoding safety net: the library's dashboards use box-drawing and
# arrow characters that crash print() outright when stdout uses a non-UTF-8
# codec (e.g. cp1252 Windows consoles / redirected output). Degrade gracefully
# to '?' replacements instead of raising UnicodeEncodeError mid-search.
# ---------------------------------------------------------------------------
try:
    import sys as _sys
    _enc = getattr(_sys.stdout, "encoding", None)
    if _sys.stdout is not None and _enc and _enc.lower().replace("-", "") not in ("utf8", "utf"):
        if hasattr(_sys.stdout, "reconfigure"):
            _sys.stdout.reconfigure(errors="replace")
except Exception:
    pass

try:
    from . import misc
except ImportError:
    import misc

try:
    from .banditsa import BanditSA, PerturbationBandit
except ImportError:
    from banditsa import BanditSA, PerturbationBandit

def new_features():
    '''ADDICTY DICT'''
    '''vars you want to ensure are included'''
    pre_spec_constraints = Dict({
        'ps_nest_vars': ['var_NMT_1', 'var_IPT_3'],
        'ps_alt_vars': ['var_ICT_1'],
        'force_include': ['TIME', 'COST'],  # Variables that must always be included
        'force_exclude': [],  # Variables that must never be included
        'force_random': {'TIME': 'n', 'COST': 'ln'},  # Variables that must have random parameters
        'never_random': [],  # Variables that must never have random parameters
        'latent_class_constraints': {
            'class_specific_vars': {
                'class_0': ['TIME'],  # Variables that appear only in class 0
                'class_1': ['COST'],  # Variables that appear only in class 1
            }
        },
        'mixed_model_constraints': {
            # Additional mixed model specific constraints
        }
    })
    print("""
           ADDICTY DICT FUNCTION
           ---------------------
           Defines pre-specified constraints used in this package.

           Returns:
               Dict with keys:
                   ps_nest_vars: variables always included in nests
                   ps_alt_vars: alternative optional variables
                   force_include -> variables that must always be included
                   force_exclude -> variables that must never be included
                   force_random -> dict of variables that must have random parameters with distributions
                   never_random -> variables that must never have random parameters
                   latent_class_constraints -> constraints for latent class models
                   mixed_model_constraints -> constraints for mixed models

           Example usage:
               from yourpackage import new_features
               constraints = new_features()
               print(constraints.force_include)
           """)

def get_version_from_pkg_info():
    """Reads the installed package version via importlib.metadata."""
    try:
        from importlib.metadata import version as _pkg_version
        return _pkg_version("SearchLibrium")
    except Exception:
        pass
    # Fallback: read from egg-info PKG-INFO (editable installs)
    pkg_info_path = os.path.join(os.path.dirname(__file__), "../SearchLibrium.egg-info/PKG-INFO")
    try:
        with open(pkg_info_path, "r") as f:
            for line in f:
                if line.startswith("Version:"):
                    return line.split(":")[1].strip()
    except FileNotFoundError:
        pass
    return "unknown"

__version__ = get_version_from_pkg_info()

def print_version():
    """Print the current version of the package."""
    print(f"Current version: {__version__}")

try:
    from . import _device as dev
except Exception:
    try:
        import _device as dev
    except Exception:
        dev = None

import importlib as _importlib

# Track which optional modules did not load so callers (and ``print_version``)
# can see what is missing in a given environment.
_import_failures = {}


def _import_relative(module_name, names, fallback_globals=None):
    """Import ``names`` from ``module_name``.

    Prefers a package-relative import (correct once SearchLibrium is installed
    as a package); only falls back to a bare top-level import when this file is
    being run as a loose script with no package context (``__package__`` empty).
    Records the *real* error in ``_import_failures`` on failure and returns
    ``None`` -- we never let a bare ``import multinomial_logit`` mask the true
    cause, because that name can never resolve once installed as a package.
    """
    last_err = None
    candidates = []
    if __package__:
        candidates.append(f"{__package__}.{module_name}")
    candidates.append(module_name)
    for target in candidates:
        try:
            mod = _importlib.import_module(target)
        except Exception as e:  # genuine import error (missing dep, syntax, ...)
            last_err = e
            continue
        missing = [n for n in names if not hasattr(mod, n)]
        if missing:
            last_err = AttributeError(
                f"{module_name} is missing expected names {missing}")
            continue
        return {name: getattr(mod, name) for name in names}
    _import_failures[module_name] = last_err
    return None


# Groups of (module, [public names]). Imported independently so that one module
# with an unmet optional dependency (e.g. larch / jax absent in a given
# environment) does not abort the entire public API, and so any warning names
# the *actual* failing module and its real error.
_IMPORT_GROUPS = [
    ('_choice_model', ['DiscreteChoiceModel']),
    ('multinomial_logit', ['MultinomialLogit']),
    ('logistic_regression', ['LogisticRegression']),
    ('MixedLogit', ['MixedLogit']),
    ('coupled_solver', ['CoupledMXLMinimiser', 'make_coupled_minimiser']),
    ('multinomial_nested', ['NestedLogit', 'MultiLayerNestedLogit']),
    ('mixed_nested', ['MixedNested']),
    ('Halton', ['Halton']),
    ('rrm', ['RandomRegret']),
    ('mixedrrm', ['MixedRandomRegret']),
    ('ordered_logit', ['OrderedLogit', 'OrderedLogitLong', 'ExplodedLogit',
                       'MixedExplodedLogit']),
    ('selection_models', ['BinaryProbit', 'HeckmanTwoStep']),
    ('zero_inflated_ordered_probit', ['ZeroInflatedOrderedProbit',
                                      'ZeroInflatedProbit']),
    ('latent_class', ['LatentClassMixedLogit']),
    ('lc_rail_guard', ['lc_result_rail']),
    ('latent_spec', ['vip_table', 'near_unidentified_report', 'align_classes',
                     'propose_class_params_spec']),
    ('mdcev', ['MDCEVFitResult', 'MDCEVModel']),
    ('multinomial_probit', ['MultinomialProbit']),
    ('MixedLogitGSE', ['MixedLogitGSE']),
    ('RandomP', ['RandomParameters']),
    ('constraints_builder', ['ConstraintBuilder', 'create_constraints']),
    ('search', ['Parameters']),
    ('sapbil', ['SAPBIL', 'ProbabilityMatrix']),
    ('hspbil', ['HSPBIL']),
    ('sparseea_agds', ['SparseEAAGDS']),
    ('call_meta', ['call_harmony', 'call_harmony_pbil', 'call_siman',
                   'call_parsa', 'call_search', 'call_sapbil', 'call_banditsa',
                   'call_parcopsa', 'call_agds', 'estimate_ctrl']),
    ('predict', ['DestinationPredictor', 'predict_destination_components',
                 'predict_aggregate_destination_flows']),
    ('sample_data', ['load_electricity_data', 'load_travel_mode_data',
                     'load_swiss_metro_data', 'preview_datasets']),
    ('regularization', ['elasticnet_objective', 'elasticnet_penalty_and_grad',
                        'nl_identification_priors']),
    ('jax_utils', ['ensure_jax_environment', 'is_large_model',
                   'estimate_row_guard', 'check_gradient_health']),
    ('larch_compat', ['apply_larch_patches']),
    ('larch_models', ['LarchMNL', 'LarchNestedLogit', 'LarchMixedLogit',
                      'build_latent_class']),
    ('skims', ['omx_to_dataframe', 'read_omx_skims', 'prepare_skims_from_omx',
               'merge_skim_component']),
    ('destination_distances', ['haversine_km', 'pick_skim_matrices',
                               'load_skim_matrices', 'build_case_alt_distances',
                               'summarize_distance_sources']),
    ('accessibility', ['gravity_accessibility', 'compute_accessibility',
                       'logsum_accessibility']),
    ('abm', ['available_stages', 'stage_search_config']),
    ('mh_choice_estimator', ['ChoiceEstimate', 'ChoiceEstimateResult',
                             'ChoiceSetFrame', 'GaussianRandomWalkProposal',
                             'BlockRandomWalkProposal', 'HHTSCompetingPrior',
                             'chain_diagnostics', 'effective_sample_size',
                             'split_rhat', 'r_hat', 'estimate_dest_choice',
                             'estimate_stage_choice']),
]

for _mod_name, _names in _IMPORT_GROUPS:
    _loaded = _import_relative(_mod_name, _names)
    if _loaded:
        globals().update(_loaded)

if _import_failures:
    _detail = '; '.join(
        f"{m} ({type(e).__name__}: {e})" for m, e in _import_failures.items())
    warnings.warn(f"SearchLibrium: some modules failed to import: {_detail}")

try:
    from .main import print_ascii_art_logo
except Exception:
    print_ascii_art_logo = None


__all__ = [
    "__version__", "print_version", "new_features",
    "DiscreteChoiceModel",
    "MultinomialLogit", "LogisticRegression", "MixedLogit",
    "CoupledMXLMinimiser", "make_coupled_minimiser",
    "NestedLogit", "MultiLayerNestedLogit", "MixedNested", "Halton",
    "RandomRegret", "MixedRandomRegret", "OrderedLogit", "OrderedLogitLong", "ExplodedLogit", "MixedExplodedLogit",
    "BinaryProbit", "HeckmanTwoStep",
    "ZeroInflatedOrderedProbit", "ZeroInflatedProbit",
    "LatentClassMixedLogit", "vip_table", "near_unidentified_report",
    "align_classes", "propose_class_params_spec", "MDCEVFitResult", "MDCEVModel", "MultinomialProbit",
    "MixedLogitGSE", "RandomParameters", "ConstraintBuilder", "create_constraints",
    "Parameters", "misc",
    "ChoiceEstimate", "ChoiceEstimateResult", "ChoiceSetFrame",
    "GaussianRandomWalkProposal",
    "BlockRandomWalkProposal", "HHTSCompetingPrior",
    "chain_diagnostics", "effective_sample_size", "split_rhat", "r_hat",
    "estimate_dest_choice", "estimate_stage_choice",
    "SAPBIL", "ProbabilityMatrix", "HSPBIL", "SparseEAAGDS", "BanditSA", "PerturbationBandit",
    "call_search", "call_siman", "call_harmony", "call_harmony_pbil", "call_sapbil",
    "call_banditsa", "call_parsa", "call_parcopsa", "call_agds", "estimate_ctrl",
    "DestinationPredictor", "predict_destination_components", "predict_aggregate_destination_flows",
    "load_electricity_data", "load_travel_mode_data", "load_swiss_metro_data", "preview_datasets",
    "elasticnet_objective", "elasticnet_penalty_and_grad", "nl_identification_priors",
    "ensure_jax_environment", "is_large_model", "estimate_row_guard", "check_gradient_health",
    "apply_larch_patches", "LarchMNL", "LarchNestedLogit", "LarchMixedLogit",
    "build_latent_class",
    "omx_to_dataframe", "read_omx_skims", "prepare_skims_from_omx", "merge_skim_component",
    "haversine_km", "pick_skim_matrices", "load_skim_matrices",
    "build_case_alt_distances", "summarize_distance_sources",
    "gravity_accessibility", "compute_accessibility", "logsum_accessibility",
    "available_stages", "stage_search_config",
]







