import numpy as np


class Regs():
    """
    Regularisation methods
    """

    def reg(self, coeffs):
        """
        Self here assumed to be ML object
        """
        # Regularisation
        if (self.params['reg'].lower == 'l1') | \
                (self.params['reg'].lower == 'lasso'):
            reg = Regs.LASSO(coeffs, self.params['lambda'])
        elif (self.params['reg'].lower == 'l2') | \
                (self.params['reg'].lower == 'ridge'):
            reg = Regs.ridge(coeffs, self.params['lambda'])
        else:
            reg = 1

        return reg

    @staticmethod
    def ridge(coeffs, a=1):
        """
        Ridge / L2 regularization,
        Objective = RSS + α * (sum of square of coefficients)
        """
        return a * np.sum(coeffs ** 2)

    @staticmethod
    def LASSO(coeffs, a=1):
        """
        LASSO / L1 reg
        LASSO stands for Least Absolute Shrinkage and Selection Operator.
        Objective = RSS + α * (sum of absolute value of coefficients)
        """
        return a * np.sum(coeffs)


class Scales():
    """
    Normalisation methods
    """

    def norm(self, X):
        """
        Assuming self here is ML object. ie. inherited in to ML object and
        called from there.
        """
        if (self.params['norm'] == 'FS') or (self.params['norm'] == True):
            X = self.featureScale(X)
        elif self.params['norm'] == 'SS':
            X = self.standardScore(X)

        return X

    @staticmethod
    def featureScale(X):
        """
        X' = (X-XMin) / (XMax-XMin)
        """
        return (X - X.min()) / (X.max() - X.min())

    @staticmethod
    def standardScore(X):
        """
        X' = (X - u) / sig

        """
        return (X - X.mean()) / X.std()


class Fs():
    """
    General helpers
    """

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def dSigmoid(x):
        return Fs.sigmoid(x) * (1.0 - Fs.sigmoid(x))

    @staticmethod
    def softmax(x):
        return (np.argmax(x,
                          axis=1))
