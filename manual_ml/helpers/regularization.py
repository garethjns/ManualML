"""Regularisation functions and model components."""

import numpy as np


class Regs:
    def reg(self, coeffs: np.ndarray) -> float:
        """
        Calculate regularisation value.

        :param coeffs: Model coefficients.
        :return: Regularisation value.
        """
        # Regularisation
        if (self.params['reg'].lower == 'l1') | (self.params['reg'].lower == 'lasso'):
            reg = lasso(coeffs, self.params['lambda'])

        elif (self.params['reg'].lower == 'l2') | (self.params['reg'].lower == 'ridge'):
            reg = ridge(coeffs, self.params['lambda'])

        else:
            # Off
            reg = 1

        return reg


def ridge(coeffs: np.ndarray,
          a: int =1) -> float:
    """
    Ridge / L2 regularization.

    Objective = RSS + α * (sum of square of coefficients)

    :param coeffs: Model coefficients.
    :return: Regularisation value.
    """
    return a * np.sum(coeffs ** 2)


def lasso(coeffs: np.narray,
          a: int=1) -> float:
    """
    Least Absolute Shrinkage and Selection Operator. / L1 reg regularization.

    Objective = RSS + α * (sum of absolute value of coefficients)

    :param coeffs: Model coefficients.
    :return: Regularisation value.
    """
    return a * np.sum(coeffs)
