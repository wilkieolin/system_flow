from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from models import *
from scipy.stats import norm, ecdf
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

def contingency_to_error(matrix):
    return matrix / np.sum(matrix, axis=0)

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
    def solve_reduction(self, reduction):
        pass

    def __call__(self, inputs, reduction):
        self.solve_reduction(inputs, reduction)
        mat = inputs * self.error_matrix
        return mat.astype("int")

"""
Placeholder object for a node which doesn't classify (reject) data - only passes it
"""
class DummyClassifier(Classifier):
    def __init__(self):
        super().__init__()

    def solve_reduction(self, inputs, reduction):
        return
    
    def __call__(self, inputs, reduction):
        return super().__call__(inputs, reduction)


"""
Object to estimate the classification statistics of a processing node based on normal processes
"""
class GaussianClassifier(Classifier):
    def __init__(self, skill, varscale = 1.0):
        self.skill = skill
        self.varscale = varscale

        #distribution of Y = 0 (reject) given X (data)
        self.false = lambda x: norm.cdf(x, loc=0.0, scale=1.0)
        #distribution of Y = 1 (accept) given X (data)
        self.true = lambda x: norm.cdf(x, loc=skill, scale=varscale)
            
    def solve_reduction(self, inputs, reduction):
        if reduction == 0.0:
            self.error_matrix = passing_node()
            return
        
        assert len(inputs) == 2, "Inputs provided must be number of falses and trues in a vector"
        self.neg, self.pos = inputs
        self.n = self.neg + self.pos
        
        #the distribution of scores depends on the input data to the classifier
        self.scores = lambda x: (self.neg * self.false(x) + self.pos * self.true(x)) / (self.n)

        opt_fn = lambda x: np.abs(reduction - self.scores(x))
        soln = minimize_scalar(opt_fn, bounds=(0.0, 20.0), method="bounded")
        if soln.success:
            self.threshold = soln.x
        else:
            print("Solving for Gaussian classification threshold failed:")
            print("T:", self.pos, "F:", self.neg, "Ratio:", reduction)
            self.threshold = 0.0
            self.error_matrix = passing_node()

        self.tn = self.false(self.threshold)
        self.fn = self.true(self.threshold)
        self.tp = (1.0 - self.true(self.threshold))
        self.fp = (1.0 - self.false(self.threshold))

        self.error_matrix = np.array([[self.tn, self.fn], [self.fp, self.tp]])
        
    def __call__(self, inputs, reduction):
        return super().__call__(inputs, reduction)
    
class L1TClassifier(Classifier):
    def __init__(self, skill_boost: float = 0.0, n_samples: int = 50000):
        super().__init__()
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
        #and the scores for those samples
        self.null_scores = np.sum(self.null_samples[:,1,:], axis=1)
        self.pos_scores = np.sum(self.pos_samples[:,1,:], axis=1)

        self.negative = lambda x: ecdf(self.null_scores).cdf.evaluate(x)


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
        jet = exp_generator(p[0], self.jet_l)
        muon = exp_generator(p[1], self.muon_l)
        egamma = exp_generator(p[2], self.egamma_l)
        tau = exp_generator(p[3], self.tau_l)
        
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
        jet = exp_generator(p[0], self.jet_l)
        muon = exp_generator(p[1], self.muon_l)
        egamma = exp_generator(p[2], self.egamma_l)
        tau = exp_generator(p[3], self.tau_l)

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

        jet = exp_generator(p_jet, self.jet_l)
        muon = exp_generator(p_muon, self.muon_l)
        egamma = exp_generator(p_egamma, self.egamma_l)
        tau = exp_generator(p_tau, self.tau_l)

        e = np.array([jet, muon, egamma,  tau,])
        z = np.array([self.jet_fit(e[0]),
                    self.muon_fit(e[1]),
                    self.egamma_fit(e[2]),
                    self.tau_fit(e[3]),
                    ])
        
        res = np.stack((e, z))
        
        return res
    
    def solve_reduction(self, inputs, reduction):
        assert len(inputs) == 2, "Inputs provided must be number of falses and trues in a vector"
        neg, pos = inputs
        n = neg + pos
        self.positive = lambda x: ecdf(self.pos_scores + self.skill_boost).cdf.evaluate(x)
        self.scores = lambda x: (neg * self.negative(x) + pos * self.positive(x)) / n

        opt_fn = lambda x: np.abs(reduction - self.scores(x))
        soln = minimize_scalar(opt_fn)
        if soln.success:
            self.threshold = soln.x
        else:
            print("Solving for L1T classification threshold failed:")
            print("T:", self.pos, "F:", self.neg, "Ratio:", reduction)
            self.threshold = 0.0
            self.error_matrix = passing_node()

        self.tn = self.negative(self.threshold)
        self.fn = self.positive(self.threshold)
        self.tp = (1.0 - self.positive(self.threshold))
        self.fp = (1.0 - self.negative(self.threshold))

        self.error_matrix = np.array([[self.tn, self.fn], [self.fp, self.tp]])


    def __call__(self, inputs, reduction):
        return super().__call__(inputs, reduction)
    
class HLTClassifier(Classifier):
    def __init__(self, skill_boost: float = 0.0, n_samples: int = 50000):
        super().__init__()
        self.rng = np.random.default_rng()
        self.n_samples = n_samples
        self.skill_boost = skill_boost

        self.paths = ["B2G", "Higgs", "Muon", "SUSY", "tracking", "tau"]
        #read trigger data from external files
        #efficiency curves
        # data taken and fitted from approved CMS 2018 Data:
        # https://twiki.cern.ch/twiki/bin/view/CMSPublic/HighLevelTriggerRunIIResults
        b2g = pd.read_csv("hlt_data/B2G.csv")
        higgs = pd.read_csv("hlt_data/higgs.csv")
        muon = pd.read_csv("hlt_data/Muon.csv")
        susy = pd.read_csv("hlt_data/SUSY.csv")
        tracking = pd.read_csv("hlt_data/tracking.csv")
        tau = pd.read_csv("hlt_data/tau.csv")

        #trigger rates
        # B2G, higgs, muon, susy, tracking, tau
        # (split objects between muon, tracking, and tau)
        # https://twiki.cern.ch/twiki/bin/view/CMSPublic/HLTplots2018Rates
        self.rates = np.array([96, 235, 202/3, 189, 202/3, 202/3])
        self.rates_norm = self.rates / np.sum(self.rates)

        self.b2g_rate, self.higgs_rate, self.muon_rate, self.susy_rate, self.tracking_rate, self.tau_rate = self.rates_norm
        
        #fit the experimental distribution of objects to the path efficiency data
        b2g_eff, b2g_soln = self.fit_trigger(b2g, self.b2g_rate, [0.0010, 0.0050])
        b2g_turnon = find_turnon(b2g_eff, (200, 600))

        higgs_eff, higgs_soln = self.fit_trigger(higgs, self.higgs_rate, [0.0001, 0.0080])
        higgs_turnon = find_turnon(higgs_eff, (200, 600))

        muon_eff, muon_soln = self.fit_trigger(muon, self.muon_rate, [0.0001, 0.0080])
        muon_turnon = find_turnon(muon_eff, (0, 50))

        susy_eff, susy_soln = self.fit_trigger(susy, self.susy_rate, [0.0001, 0.0080])
        susy_turnon = find_turnon(susy_eff, (100, 120))

        tracking_eff, tracking_soln = self.fit_trigger(tracking, self.tracking_rate, [0.0001, 0.0080])
        #tracking_turnon = find_turnon(tracking_eff, (0, 10))

        tau_eff, tau_soln = self.fit_trigger(tau, self.tau_rate, [0.0001, 0.0080])
        tau_turnon = find_turnon(tau_eff, (0, 40))

        b2g_threshold = b2g_turnon.x
        higgs_threshold = higgs_turnon.x
        muon_threshold = muon_turnon.x
        susy_threshold = susy_turnon.x
        tracking_threshold = 2 #round up tracking to the more reasonable design value of 2 GeV
        tau_threshold = tau_turnon.x

        #find the percentile of measurements above median acceptance level
        b2g_prctile = exp_cdf(b2g_threshold, b2g_soln.x)
        higgs_prctile = exp_cdf(higgs_threshold, higgs_soln.x)
        muon_prctile = exp_cdf(muon_threshold, muon_soln.x)
        susy_prctile = exp_cdf(susy_threshold, susy_soln.x)
        tracking_prctile = exp_cdf(tracking_threshold, tracking_soln.x)
        tau_prctile = exp_cdf(tau_threshold, tau_soln.x) 

        #store these results for use in estimating classifier performance
        self.effiencies = [b2g_eff, higgs_eff, muon_eff, susy_eff, tracking_eff, tau_eff]
        self.fits = [b2g_soln.x, higgs_soln.x, muon_soln.x, susy_soln.x, tracking_soln.x, tau_soln.x]
        self.thresholds = np.array([b2g_threshold, higgs_threshold, muon_threshold, susy_threshold, tracking_threshold, tau_threshold])
        self.prctiles = np.array([b2g_prctile, higgs_prctile, muon_prctile, susy_prctile, tracking_prctile, tau_prctile])
        
        #generate positive and null score distributions
        self.null_evts = np.stack([self.generate_null() for i in range(n_samples)])
        self.pos_evts = np.stack([self.generate_positive() for i in range(n_samples)])

        self.null_scores = self.null_evts[:,2]
        self.pos_scores = self.pos_evts[:,2]

        self.negative = lambda x: ecdf(self.null_scores).cdf.evaluate(x)
        
    def generate_null(self):
        #take a random trigger path
        path = np.random.choice(np.arange(len(self.thresholds)), p = self.rates_norm)
        p = np.random.uniform() * self.prctiles[path]
        l = self.fits[path]
        e = exp_generator(p, l)
        z = self.effiencies[path](e)
        
        return path, e, z
    
    def generate_positive(self):
        #take a random trigger path
        path = np.random.choice(np.arange(len(self.thresholds)), p = self.rates_norm)
        p = 1.0 - np.random.uniform() * self.prctiles[path]
        l = self.fits[path]
        e = exp_generator(p, l)
        z = self.effiencies[path](e)
        
        return path, e, z
    
    def fit_trigger(self, data, empirical_rate, solver_bounds):
        #extract the dynamic range of momenta for the trigger
        xs = data_range(data["momentum"])
        #use a linear interpolation to fit the efficiency curve
        efficiency_fit = lambda x: hard_bounds(x, interp1d(data["momentum"], data[" efficiency"]), xs)
        #estimate the mean proportion of objects a trigger will activate on given an exponential input distribution of objects
        xs2 = expanded_range(xs)
        trigger_rate = lambda l: quad(lambda x: exp_dist(x, l) * efficiency_fit(x), np.min(xs2), np.max(xs2))[0]
        #calculate the gap between that and the empirical rate
        trigger_error = lambda l: np.abs(empirical_rate - trigger_rate(l))
        #minimize the gap
        soln = minimize_scalar(trigger_error, bounds = solver_bounds, method="bounded")
        return efficiency_fit, soln
    
    def solve_reduction(self, inputs, reduction):
        assert len(inputs) == 2, "Inputs provided must be number of falses and trues in a vector"
        neg, pos = inputs
        n = neg + pos
        self.positive = lambda x: ecdf(self.pos_scores + self.skill_boost).cdf.evaluate(x)
        self.scores = lambda x: (neg * self.negative(x) + pos * self.positive(x)) / n

        opt_fn = lambda x: np.abs(reduction - self.scores(x))
        soln = minimize_scalar(opt_fn)
        if soln.success:
            self.threshold = soln.x
        else:
            print("Solving for HLT classification threshold failed:")
            print("T:", self.pos, "F:", self.neg, "Ratio:", reduction)
            self.threshold = 0.0
            self.error_matrix = passing_node()

        self.tn = self.negative(self.threshold)
        self.fn = self.positive(self.threshold)
        self.tp = (1.0 - self.positive(self.threshold))
        self.fp = (1.0 - self.negative(self.threshold))

        self.error_matrix = np.array([[self.tn, self.fn], [self.fp, self.tp]])


    def __call__(self, inputs, reduction):
        return super().__call__(inputs, reduction)
