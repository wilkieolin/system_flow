import numpy as np

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
