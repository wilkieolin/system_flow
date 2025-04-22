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
