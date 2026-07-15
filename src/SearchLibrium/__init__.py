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

from . import misc

try:
    from banditsa import BanditSA, PerturbationBandit
except ImportError:
    from .banditsa import BanditSA, PerturbationBandit

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

print_version()
try:
    from . import _device as dev
except ImportError:
    #print('Error importing local _device, using global import')
    import _device  as dev
    #print('loaded devices from local import')
try:
    from ._choice_model import DiscreteChoiceModel
    from .multinomial_logit import MultinomialLogit
    from .MixedLogit import MixedLogit
    from .multinomial_nested import NestedLogit, MultiLayerNestedLogit
    from .mixed_nested import MixedNested
    from .Halton import Halton
    from .rrm import RandomRegret
    from .mixedrrm import MixedRandomRegret
    from .ordered_logit import OrderedLogit, OrderedLogitLong
    from .selection_models import BinaryProbit, HeckmanTwoStep
    from .latent_class import LatentClassMixedLogit
    from .mdcev import MDCEVFitResult, MDCEVModel
    from .multinomial_probit import MultinomialProbit
    from .RandomP import RandomParameters
    from .constraints_builder import ConstraintBuilder, create_constraints
    from .search import Parameters

    from . import misc        
    from .call_meta import call_harmony, call_harmony_pbil, call_siman, call_parsa, call_search, call_sapbil, call_banditsa, call_parcopsa, estimate_ctrl
    from .predict import DestinationPredictor, predict_destination_components, predict_aggregate_destination_flows
    from .sample_data import load_electricity_data, load_travel_mode_data, load_swiss_metro_data, preview_datasets

except ImportError as e:
    from _choice_model import DiscreteChoiceModel
    from multinomial_logit import MultinomialLogit
    from MixedLogit import MixedLogit
    from multinomial_nested import NestedLogit, MultiLayerNestedLogit
    from mixed_nested import MixedNested
    from Halton import Halton
    from rrm import RandomRegret
    from mixedrrm import MixedRandomRegret
    from ordered_logit import OrderedLogit, OrderedLogitLong
    from selection_models import BinaryProbit, HeckmanTwoStep
    from latent_class import LatentClassMixedLogit
    from mdcev import MDCEVFitResult, MDCEVModel
    from multinomial_probit import MultinomialProbit
    from RandomP import RandomParameters
    from constraints_builder import ConstraintBuilder, create_constraints
    from search import Parameters    
    from call_meta import call_siman, call_harmony, call_harmony_pbil, call_search, call_sapbil, call_banditsa, call_parsa, call_parcopsa, estimate_ctrl
    from predict import DestinationPredictor, predict_destination_components, predict_aggregate_destination_flows
    from sample_data import load_electricity_data, load_travel_mode_data, load_swiss_metro_data, preview_datasets
try:
    from .main import print_ascii_art_logo
except Exception:
    try:
        from main import print_ascii_art_logo
    except Exception:
        print_ascii_art_logo = None


if print_ascii_art_logo is not None:
    try:
        print_ascii_art_logo()
    except Exception:
        print("SearchLibrium logo skipped; optional display dependencies are missing.")

#print('loaded all')
print('Welcome to SearchLibrium')
new_features()







