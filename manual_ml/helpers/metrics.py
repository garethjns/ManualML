import numpy as np

from typing import Iterable


def mse(y_true: Iterable, y_pred: Iterable) -> float:
    """Mean squared error."""

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return 1 / len(y_true) * np.sum((y_true - y_pred) ** 2)


def log_loss(y_true: Iterable, y_pred: Iterable) -> float:
    """Log loss: -log P(yt|yp) = -(yt log(yp) + (1 - yt) log(1 - yp))"""

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if ~(len(np.unique(y_true)) > 1):
        # Not handling this case, raise error.
        raise ValueError("~(len(np.unique(y_true)) > 1)")

    return 1 / len(y_true) * np.sum(-y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred))


def accuracy(y_true: Iterable, y_pred: Iterable,
             p: int =3) -> float:
    """Calculate accuracy after round to number of decimals specified in p"""

    # This won't fail on mismatched length, only warn, so check.
    if len(y_true) != len(y_pred):
        raise ValueError("len(y_true) != len(y_pred)")

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Round
    y_true = np.round(y_true, p)
    y_pred= np.round(y_pred, p)

    acc = np.sum(y_true == y_pred) / len(y_true)

    return acc
