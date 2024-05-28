import numpy as np
from scipy.optimize import minimize_scalar
from scipy.interpolate import interp1d
from scipy.integrate import quad

start_year = 2022
reticle_limit = 850 #(mm^2)
power_limit = 250 #(W)
bunch_crossing_rate = 40e6 #(1/s)
power_density_limit = power_limit / reticle_limit #(W/ mm^2)

lin_fn = lambda x, k, a: a * x + k
cubic_fn = lambda x, a, k0, k1, k2, k3: k3 * np.power(a*x, 3.0) + k2 * np.power(a*x, 2.0) + k1 * a* x + k0


"""
TECHNOLOGY MODELS
"""
def density_scale_model(year: float):
    """
    Given a year, return the number of integrated transistors which can be expected to fit in
    a square millimeter of silicon (transistors / mm2). Based in IEEE IEDM '22.

    relative (bool): if true, return a scale proportional to the number of transistors available
    in '22.
    """

    year = year - start_year
    fit = np.array([-0.19815559,  0.18717361])
    #simplify expression 
    scale = np.exp(lin_fn(year, 0.0, fit[1]))

    return scale

def transistor_scale_model(year: float):
    """
    Given a year, return the number of integrated transistors which can be expected to fit in
    a square millimeter of silicon (transistors / mm2). Based in IEEE IEDM '22.

    relative (bool): if true, return a scale proportional to the number of transistors available
    in '22.
    """

    year = year - start_year
    fit = np.array([-0.19815559,  0.18717361])
    scale = np.exp(lin_fn(year, *fit))

    return scale

data_range = lambda x: np.linspace(np.min(x), np.max(x), 101)
exp_dist = lambda x, l: l * np.exp(-1 * l * x) * (x > 0)
exp_cdf = lambda x, l: 1 - np.exp(-l * x)
exp_generator = lambda p, l: (-1 / l) * np.log(1 - p)

def expanded_range(arr):
    center = np.mean(arr)
    range = center - np.min(arr)
    expanded_range = 1.20 * range
    new_arr = np.linspace(center - expanded_range, center + expanded_range, len(arr))
    return new_arr

def hard_bounds(x, interp, x_range):
    if x > np.max(x_range):
        return np.array(1.0)
    elif x < np.min(x_range):
        return np.array(0.0)
    else:
        return interp(x)
    
def find_turnon(fit, bracket):
    opt_fn = lambda x: np.abs(0.5 - fit(x))
    soln = minimize_scalar(opt_fn, bracket=bracket)
    return soln