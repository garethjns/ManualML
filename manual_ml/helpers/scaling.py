"""Scaling functions."""


import numpy as np


def feature_scale(x: np.ndarray) -> np.array:
    """
    x' = (x-xMin) / (xMax-xMin)
    """
    return (x - x.min()) / (x.max() - x.min())


def standard_score(x: np.ndarray) -> np.array:
    """
    x' = (x - u) / sig
    """
    return (x - x.mean()) / x.std()
