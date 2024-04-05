from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from scipy.stats import norm, permutation_test
from scipy.optimize import minimize_scalar
from scipy.interpolate import PchipInterpolator
from scipy.integrate import quad

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
    samples = [neg, pos]
    return samples

def reduction_to_ratio(reduction):
    return 1.0 / (1.0 - reduction)

def ratio_to_reduction(reduction_ratio):
    return 1.0 - 1.0 / reduction_ratio

def order_test(null_samples, pos_samples):
    score = lambda x: np.sum(x[:,1,:], axis=1)
    null_scores = score(null_samples)
    pos_scores = score(pos_samples)

    p = np.mean(np.reshape(pos_scores, (1, -1)) > np.reshape(null_scores, (-1,1)))
    error = 1 - p
    confusion = np.array([[p, error], [error, p]])

    return confusion

class Classifier(ABC):
    def __init__(self):
        self.error_matrix = passing_node()

    @abstractmethod
    def solve_reduction(self):
        pass

    def __call__(self, inputs):
        mat = inputs * self.error_matrix
        return mat.astype("int")

"""
Placeholder object for a node which doesn't classify (reject) data - only passes it
"""
class DummyClassifier(Classifier):
    def __init__(self):
        super().__init__()
        self.reduction  = 0

    def solve_reduction(self):
        return
    
    def __call__(self, inputs):
        return super().__call__(inputs)




"""
Object to estimate the classification statistics of a processing node based on normal processes
"""
class GaussianClassifier(Classifier):
    def __init__(self, reduction, skill, varscale = 1.0, n = 1000):
        #if the reduction is zero, pass all data
        if reduction <= 0.0:
            self.reduction = 0.0
            self.active = False
            self.error_matrix = passing_node()
        
        else:
            self.reduction  = reduction
            self.skill = skill
            self.varscale = varscale
            self.active = True

            #distribution of Y = 0 (reject) given X (data)
            self.false = lambda x: norm.cdf(x, loc=0.0, scale=1.0)
            #distribution of Y = 1 (accept) given X (data)
            self.true = lambda x: norm.cdf(x, loc=skill, scale=varscale)
            
    def solve_reduction(self, inputs):
        if not self.active:
            self.error_matrix = passing_node()
            return
        
        assert len(inputs) == 2, "Inputs provided must be number of falses and trues in a vector"
        self.neg, self.pos = inputs
        self.n = self.neg + self.pos
        
        #the distribution of scores depends on the input data to the classifier
        self.scores = lambda x: (self.neg * self.false(x) + self.pos * self.true(x)) / (self.n)

        opt_fn = lambda x: np.abs(self.reduction - self.scores(x))
        soln = minimize_scalar(opt_fn, bounds=(0.0, 20.0))
        if soln.success:
            self.threshold = soln.x
        else:
            print("Solving for classification threshold failed:")
            print("T:", self.pos, "F:", self.neg, "Ratio:", self.reduction)
            self.threshold = 0.0
            self.error_matrix = passing_node()

        self.tn = self.false(self.threshold)
        self.fn = self.true(self.threshold)
        self.tp = (1.0 - self.true(self.threshold))
        self.fp = (1.0 - self.false(self.threshold))

        self.error_matrix = np.array([[self.tn, self.fn], [self.fp, self.tp]])
        
    def __call__(self, inputs):
        self.solve_reduction(inputs)
        return super().__call__(inputs)
    
class L1TClassifier(Classifier):
    def __init__(self, reduction, skill_boost: float = 0.0, n_samples: int = 10000):
        super().__init__()
        self.reduction = reduction
        self.rng = np.random.default_rng()
        self.n_samples = n_samples
        self.skill_boost = skill_boost

        self.triggers = ["Jet", "Muon", "EGamma", "Tau"]
        #read trigger data from external files
        #efficiency curves
        self.egamma = pd.read_csv("l1t_data/egamma.csv")
        self.muon = pd.read_csv("l1t_data/single muon.csv")
        self.jet_energy = pd.read_csv("l1t_data/jet energy.csv")
        self.tau = pd.read_csv("l1t_data/isolated tau.csv")

        #trigger rates
        self.rates = pd.read_excel("l1t_data/trigger_rates.xlsx")

        self.jet_rate = self.rates["Jet Rate"][0] 
        self.muon_rate = self.rates["Muon Rate"][0] 
        self.egamma_rate = self.rates["Egamma Rate"][0] 
        self.tau_rate = self.rates["Tau Rate"][0] 

        combined_rates = self.rates["Proportion"]
        self.trigger_choice = lambda: self.rng.choice(np.arange(len(combined_rates)),
                                                      1,
                                                      p = combined_rates)[0]

        #trigger thresholds
        self.jet_threshold = self.rates["Jet Threshold"][0] 
        self.muon_threshold = self.rates["Muon Threshold"][0] 
        self.egamma_threshold = self.rates["Egamma Threshold"][0] 
        self.tau_threshold = self.rates["Tau Threshold"][0] 
        self.thresholds = np.array([self.jet_threshold, 
                                    self.muon_threshold, 
                                    self.egamma_threshold, 
                                    self.tau_threshold])
        

        #define helper functions
        self.exp_dist = lambda x, l: l * np.exp(-1 * l * x) * (x > 0)
        self.exp_cdf = lambda x, l: 1 - np.exp(-l * x)
        self.exp_generator = lambda p, l: (-1 / l) * np.log(1 - p)
        self.data_range = lambda x: np.linspace(np.min(x), np.max(x), 101)

        #fit the efficiency curves to trigger rates
        self.jet_fit, self.jet_l, _ = self.fit_trigger(self.jet_energy, (0.00, 0.40))
        self.muon_fit, self.muon_l, _ = self.fit_trigger(self.muon, (0.00, 0.60))
        self.egamma_fit, self.egamma_l, _ = self.fit_trigger(self.egamma, (0.00, 0.40))
        self.tau_fit, self.tau_l, _ = self.fit_trigger(self.tau, (0.00, 0.40))

        #find the percentile of measurements falling above threshold
        self.jet_prctile = self.exp_cdf(self.jet_threshold, self.jet_l)
        self.muon_prctile = self.exp_cdf(self.muon_threshold, self.muon_l) 
        self.egamma_prctile = self.exp_cdf(self.egamma_threshold, self.egamma_l)
        self.tau_prctile = self.exp_cdf(self.tau_threshold, self.tau_l) 
        self.prctiles = np.array([self.jet_prctile, 
                                  self.muon_prctile, 
                                  self.egamma_prctile, 
                                  self.tau_prctile])
        
        #generate sample distributions
        self.null_samples = np.stack([self.generate_null() for i in range(self.n_samples)])
        self.pos_samples = np.stack([self.generate_positive() for i in range(self.n_samples)])


    """
    Fit an exponential distribution to produce rates matching an individual trigger path
    """
    def fit_trigger(self, data, bounds):
        xs = data["x"]
        ys = data[" y"]
        interpolator = PchipInterpolator(xs, ys)
        #fit to muon - minimize the difference between the trigger efficiency multiplied by the underlying distribution and the trigger rate
        fit = lambda l: np.abs(self.egamma_rate - quad(lambda x: self.exp_dist(x, l) * interpolator(x), np.min(xs), np.max(xs))[0])
        soln = minimize_scalar(fit, bounds = bounds, method="bounded")
        l = soln.x
        fit = lambda x: np.clip(interpolator(x), 0.0, 1.0)

        return fit, l, soln
    
    """
    Generate a distribution of particles centered around the trigger threshold
    """
    def generate_exp(self):
        p = self.rng.uniform(size=(4))
        jet = self.exp_generator(p[0], self.jet_l)
        muon = self.exp_generator(p[1], self.muon_l)
        egamma = self.exp_generator(p[2], self.egamma_l)
        tau = self.exp_generator(p[3], self.tau_l)
        
        e = np.array([jet, muon, egamma, tau])
        z = np.array([self.jet_fit(e[0]),
                    self.muon_fit(e[1]),
                    self.egamma_fit(e[2]),
                    self.tau_fit(e[3])])
        
        res = np.stack((e, z))
        
        return res
    """
    Generate particles from an event which doesn't trigger L1T
    """
    def generate_null(self):
        p = self.rng.uniform(size=(4)) * self.prctiles
        jet = self.exp_generator(p[0], self.jet_l)
        muon = self.exp_generator(p[1], self.muon_l)
        egamma = self.exp_generator(p[2], self.egamma_l)
        tau = self.exp_generator(p[3], self.tau_l)

        e = np.array([jet, muon, egamma, tau])
        z = np.array([self.jet_fit(e[0]),
                    self.muon_fit(e[1]),
                    self.egamma_fit(e[2]),
                    self.tau_fit(e[3])])
        
        res = np.stack((e, z))
        
        return res

    """
    Generate particles from an event which triggers L1T
    """
    def generate_positive(self):
        #select a trigger outcome with rates proportional to those recorded
        outcome_idx = self.trigger_choice()
        outcome = self.rates.iloc[outcome_idx]

        #determine the triggers for that outcome
        trig_egamma = outcome["Egamma"]
        trig_jet = outcome["Jet / Jet energy"]
        trig_muon = outcome["Muon"]
        trig_tau = outcome["Tau"]

        #generate particle energies from the distributions based on whether or not they're above trigger levels
        def transform_p(prctile, outcome):
            p = np.random.uniform()
            if outcome == 1.0:
                #generate an event above the trigger threshold
                return p * (1 - prctile) + prctile
            else:
                #generate an event below the trigger threshold
                return p * prctile
            
        #generate particle energies based on the trigger outcome
        p_jet = transform_p(self.jet_prctile, trig_jet)
        p_muon = transform_p(self.muon_prctile, trig_muon)
        p_egamma = transform_p(self.egamma_prctile, trig_egamma)
        p_tau = transform_p(self.tau_prctile, trig_tau)

        jet = self.exp_generator(p_jet, self.jet_l)
        muon = self.exp_generator(p_muon, self.muon_l)
        egamma = self.exp_generator(p_egamma, self.egamma_l)
        tau = self.exp_generator(p_tau, self.tau_l)

        e = np.array([jet, muon, egamma,  tau,])
        z = np.array([self.jet_fit(e[0]),
                    self.muon_fit(e[1]),
                    self.egamma_fit(e[2]),
                    self.tau_fit(e[3]),
                    ])
        
        res = np.stack((e, z))
        
        return res
    
    def solve_reduction(self, inputs, n_samples: int = 1001):
        assert len(inputs) == 2, "Inputs provided must be number of falses and trues in a vector"
        self.neg, self.pos = inputs
        self.n = self.neg + self.pos
        #how many samples are allowed out?
        n_out = self.n * self.reduction

        ordering_error = order_test(self.null_samples, self.pos_samples + self.skill_boost)
        ordering = inputs * ordering_error
        predictions = np.sum(ordering, axis=1)

        #if we have to cut down the outputs, randomly select from the positive cases
        if n_out < predictions[1]:
            cutoff = n_out / predictions[1]
            rejected = get_passed(ordering) * cutoff
            contingency = np.stack((get_rejected(ordering) + rejected, get_passed(ordering) - rejected), axis=0)
        #otherwise, we have to accept more negatives into the output
        else:
            cutoff = (n_out - predictions[1]) / predictions[0]
            accepted = get_rejected(ordering) * cutoff
            contingency = np.stack((get_rejected(ordering) - accepted, get_passed(ordering) + accepted), axis=0)

        self.contingency = contingency
        #determine how often the ordering of a set of samples is correct
        self.error_matrix = contingency / np.sum(contingency, axis=0)

    def __call__(self, inputs):
        self.solve_reduction(inputs)
        return super().__call__(inputs)