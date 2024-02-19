import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize_scalar

"""
Placeholder object for a node which doesn't classify (reject) data - only passes it
"""
class DummyClassifier:
    def __init__(self):
        self.reduction  = 0
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

def reduction_to_samples(reduction, n=1000):
    pos = n * reduction
    neg = n - pos
    samples = [pos, neg]
    return samples

"""
Object to estimate the classification statistics of a processing node based on normal processes
"""
class Classifier:
    def __init__(self, reduction, skill, varscale = 1.0, n = 1000):
        #if the reduction is zero, pass all data
        if reduction < 0.0:
            self.reduction = 0.0
            self.active = False
            self.error_matrix = passing_node()
        
        else:
            self.n = n
            self.pos, self.neg = reduction_to_samples(reduction, n)
            self.reduction  = reduction
            self.skill = skill
            self.varscale = varscale
            self.active = True

            #distribution of Y = 0 (reject) given X (data)
            self.false = lambda x: norm.cdf(x, loc=0.0, scale=varscale)
            #distribution of Y = 1 (accept) given X (data)
            self.true = lambda x: norm.cdf(x, loc=skill, scale=varscale)
            #the distribution of scores depends on the input data to the classifier
            self.scores = lambda x: (self.neg * self.false(x) + self.pos * self.true(x)) / (self.n)
            #get the data selection threshold
            self.threshold = self.solve_reduction()
            
            self.tn = self.false(self.threshold)
            self.fn = self.true(self.threshold)
            self.tp = (1.0 - self.true(self.threshold))
            self.fp = (1.0 - self.false(self.threshold))

            self.error_matrix = np.array([[self.tn, self.fn], [self.fp, self.tp]])

    def solve_reduction(self):
        opt_fn = lambda x: np.abs(self.reduction - self.scores(x))
        soln = minimize_scalar(opt_fn, bounds=(0.0, 20.0))
        if soln.success:
            return soln.x
        else:
            print("Solving for classification threshold failed:")
            print("T:", self.pos, "F:", self.neg, "Ratio:", self.reduction)
            return 0.0
        
    def __call__(self, inputs):
        return inputs * self.error_matrix 