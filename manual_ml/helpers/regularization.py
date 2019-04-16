"""Regularisation functions and model components."""

import numpy as np


class Regs:
    def reg(self, coeffs: np.ndarray, n: int) -> float:
        """
        Calculate regularisation value.

        :param coeffs: Model coefficients.
        :param n: n observations.
        :return: Regularisation value.
        """
        # Regularisation
        if self.params['reg'] is not None:
            if (self.params['reg'].lower() == 'l1') | (self.params['reg'].lower == 'lasso'):
                reg = lasso(coeffs)

            elif (self.params['reg'].lower() == 'l2') | (self.params['reg'].lower() == 'ridge'):
                reg = ridge(coeffs)

            else:
                raise ValueError(f"Unknown regularisation type: {self.params['reg']}")

        else:
            # Off
            reg = 0

        return self.params['lambda'] / (2 * n) * reg


def ridge(coeffs: np.ndarray) -> float:
    """
    Ridge / L2 regularization.

    Objective = RSS + α * (sum of square of coefficients)

    :param coeffs: Model coefficients.
    :return: Regularisation value.
    """
    return np.sum(coeffs ** 2)


def lasso(coeffs: np.ndarray) -> float:
    """
    Least Absolute Shrinkage and Selection Operator. / L1 reg regularization.

    Objective = RSS + α * (sum of absolute value of coefficients)

    :param coeffs: Model coefficients.
    :return: Regularisation value.
    """
    return np.sum(coeffs)
