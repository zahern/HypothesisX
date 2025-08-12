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
try:
    from . import _device as dev
except ImportError:
    #print('Error importing local _device, using global import')
    import _device  as dev
    #print('loaded devices from local import')
try:
    from .multinomial_nested import NestedLogit
    from .Halton import Halton
    from .rrm import RandomRegret
    from .ordered_logit import OrderedLogit, OrderedLogitLong
    from .mixed_logit import MixedLogit
    from .multinomial_logit import MultinomialLogit
    #print('loaded models')
    from .search import Search
    from .main import print_ascii_art_logo
    from . import misc
except ImportError as e:
    #print(f"Error importing modules: {e}")
    from multinomial_nested import NestedLogit
    from Halton import Halton
    from rrm import RandomRegret
    from ordered_logit import OrderedLogit, OrderedLogitLong
    from mixed_logit import MixedLogit
    from multinomial_logit import MultinomialLogit
    from main import print_ascii_art_logo
    

try:
    from main import print_ascii_art_logo
    print_ascii_art_logo()
except ImportError:
    print("Error importing print_ascii_art_logo from main module. Continuing without logo.") 

#print('loaded all')
print('Welcome to SearchLibrium')




