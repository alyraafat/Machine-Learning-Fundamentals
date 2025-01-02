import numpy as np


def euclidean_distance(x: np.ndarray, y: np.ndarray) -> float:
    # check if y is a point or a list of points
    if y.ndim == 2:
        axis = 1
    else:
        axis = None
    return np.sqrt(np.sum((x-y)**2, axis=axis))

def manhattan_distance(x: np.ndarray, y: np.ndarray) -> float:
    # check if y is a point or a list of points
    if y.ndim == 2:
        axis = 1
    else:
        axis = None
    return np.sum(np.abs(x-y), axis=axis)