'''Similarity metrics'''
from __future__ import division

import numpy as np

def normalized_l1(A, B):
    '''Normalized similarity between binary encoded vector A
       and binary encoded vector B'''
    assert A.shape == B.shape
    diff = np.sum(abs(A-B), axis=1)
    return 1 - (diff / A.shape[1])


def cos_similarity(u, v):
    denom = (np.linalg.norm(u) * np.linalg.norm(v))
    num = np.dot(u, v)
    if denom == 0:
        return num/1e-10
    else:
       return num / denom