import numpy as np


class Ks:
    """
    Kernels
    """

    def __init__(self,
                 ty='linear',
                 **kwargs):

        self.params = dict([(key, value) for key, value in kwargs.items()])
        self.kernelType = ty.lower()

    def __call__(self, x1, x2):

        if self.kernelType.lower() == 'linear':
            res = self.linear(x1, x2)
        elif self.kernelType.lower() == 'polynomial':
            res = self.polynomial(x1, x2)

        return res

    def linear(self, x1, x2):
        return np.dot(x1, x2)

    def polynomial(self, x1, x2):

        p = self.params.get('p', 2)

        return (1 + np.dot(x1, x2)) ** p

    def gaussian(self, x1, x2):
        sig = self.params.get('sigma', 3)

        return np.exp(-np.linalg.norm(x1 - x2,
                                      axis=0) ** 2 / (2 * (sig ** 2)))

    def RBF(self, x1, x2, gamma=1):
        return np.exp(-gamma * np.abs(x1 - x2) ** 2).squeeze()
