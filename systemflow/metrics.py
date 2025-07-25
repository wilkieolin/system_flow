"""
Copyright 2025, UChicago Argonne LLC. 
Please refer to 'license' in the root directory for details and disclosures.
"""

import numpy as np
from systemflow.node import Message, ExecutionGraph, Regex
from abc import ABC
import re

from systemflow.node import *

class TotalOps(Metric):
    def __init__(self):
        super().__init__("Total operations", 
                         [],
                         [Regex(r"ops \(n,n\)"),],)
        
    def metric(self, message: Message, properties: dict):
        matches = self.graph_matches(properties)
        ops = np.sum([np.prod(op) for op in matches])
        metrics = {"total ops (n)": ops,}
        
        return metrics
    
class TotalLatency(Metric):
    def __init__(self):
        super().__init__("Total latency", 
                         [Regex(r"latency \(s\)"),],
                         [],)
        
    def metric(self, message: Message, properties: dict):
        matches = self.message_matches(message)
        ops = np.sum(matches)
        metrics = {"total latency (s)": ops,}
        
        return metrics

def precision(matrix):
    tp = matrix[1,1]
    fp = matrix[1,0]
    return tp / (tp + fp)

def recall(matrix):
    tp = matrix[1,1]
    fn = matrix[0,1]
    return tp / (tp + fn)

def f1_score(matrix):
    p = precision(matrix)
    r = recall(matrix)
    f1 = 2 * (p * r) / (p + r)
    return f1

def accuracy(matrix): 
    tp = matrix[1,1]
    tn = matrix[0,0]
    return (tp + tn) / np.sum(matrix)

def serial_parallel_ops(ops: int, parallelism: float):
    assert parallelism >= 0.0 and parallelism <= 1.0, "Must be on the domain [0.0, 1.0]"
    #the area of the serial and parallel ops is the total number required for the algorithm
    serial = np.power(ops, (1 - parallelism))
    parallel = np.power(ops, parallelism)
    return serial, parallel
