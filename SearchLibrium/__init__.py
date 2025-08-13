import warnings
import argparse
import csv
import faulthandler
import sys
import timeit
from collections import namedtuple
import numpy as np
import pandas as pd
#print('loading devices')
print('package version')
import os

def get_version_from_pkg_info():
    """Reads the version from the PKG-INFO file."""
    pkg_info_path = os.path.join(os.path.dirname(__file__), "../SearchLibrium.egg-info/PKG-INFO")
    try:
        with open(pkg_info_path, "r") as f:
            for line in f:
                if line.startswith("Version:"):
                    return line.split(":")[1].strip()
    except FileNotFoundError:
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
    from .multinomial_nested import NestedLogit
    from .Halton import Halton
    from .rrm import RandomRegret
    from .ordered_logit import OrderedLogit, OrderedLogitLong
    from .mixed_logit import MixedLogit

    #print('loaded models')
    from .search import Search
    from .main import print_ascii_art_logo
    from . import misc
except ImportError as e:
    #print(f"Error importing modules: {e}")
    print('trying to import this ')
    from _choice_model import DiscreteChoiceModel
    from multinomial_logit import MultinomialLogit
    from multinomial_nested import NestedLogit
    from Halton import Halton
    from rrm import RandomRegret
    from ordered_logit import OrderedLogit, OrderedLogitLong
    from mixed_logit import MixedLogit
    #from multinomial_logit import MultinomialLogit
    from main import print_ascii_art_logo
    

try:
    from main import print_ascii_art_logo
    print_ascii_art_logo()
except ImportError:
    print("Error importing print_ascii_art_logo from main module. Continuing without logo.") 

#print('loaded all')
print('Welcome to SearchLibrium')




