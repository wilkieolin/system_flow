import numpy as np

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

