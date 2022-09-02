"""
model_utilities

A Python package with utility functions for ecosystem analysis and modeling
"""

__version__ = "0.1.0"
__author__ = 'Earthshot Science'
__credits__ = 'Earthshot Labs'

from .deepdive_funcs import wood_density_lookup
from .deepdive_funcs import curve_fun
from .deepdive_funcs import curve_fit_func
from .deepdive_funcs import chave_allometry_height
from .deepdive_funcs import mature_biomass_spawn
from .deepdive_funcs import root_shoot_ipcc
from .deepdive_funcs import getNearbyMatureForestPercentiles
from .deepdive_funcs import getWalkerMatureForestPercentiles
from .deepdive_funcs import chave_allometry_noheight
from .deepdive_funcs import PredictIPCC

from .curve_fitting import chapman_richards_set_ymax
from .curve_fitting import logistic_set_ymax
from .curve_fitting import clean_biomass_data
from .curve_fitting import curve_fit_set_ymax
