import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize_scalar

"""
Placeholder object for a node which doesn't classify (reject) data - only passes it
"""
class DummyClassifier:
    def __init__(self):
        self.ratio  = 1
        self.error_matrix = passing_node()

"""
Error matrix representing a dummy classifier (no classification)
"""
def passing_node():
    error_matrix = np.array([[0.0, 0.0], [1.0, 1.0]])
    return error_matrix

def get_rejected(matrix):
    return matrix[0,:]

def get_passed(matrix):
    return matrix[1,:]
"""
Object to estimate the classification statistics of a processing node based on normal processes
"""
"""
Object to estimate the classification statistics of a processing node based on normal processes
"""
class Classifier:
    def __init__(self, ratio, skill, varscale = 1.0):
        self.skill = skill
        self.ratio  = ratio
        self.varscale = varscale

        self.pos = 1 / ratio
        self.neg = 1 - self.pos

        #distribution of Y = 0 (reject) given X (data)
        self.false = lambda x: norm.cdf(x, loc=0.0, scale=varscale)
        #distribution of Y = 1 (accept) given X (data)
        self.true = lambda x: norm.cdf(x, loc=skill, scale=varscale)
        #assume the selectivity we're giving reflects the ratio of the true scores generated
        self.scores = lambda x: self.neg * self.false(x) + self.pos * self.true(x)
        #data accepted given a threshold
        self.accept = lambda x: 1.0 - self.scores(x)
        self.ratio_fn = lambda x: self.accept(x) / self.scores(x)
        #get the data selection threshold
        self.threshold = self.solve_ratio()
        
        self.tn = self.false(self.threshold)
        self.fn = self.true(self.threshold)
        self.tp = (1.0 - self.true(self.threshold))
        self.fp = (1.0 - self.false(self.threshold))

        self.error_matrix = np.array([[self.tn, self.fn], [self.fp, self.tp]])

    def solve_ratio(self):
        opt_fn = lambda x: np.abs(self.ratio - self.ratio_fn(x))
        soln = minimize_scalar(opt_fn, bounds=(0.0, 20.0))
        if soln.success:
            return soln.x
        else:
            print("Solving for classification threshold failed")