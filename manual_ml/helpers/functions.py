import numpy as np






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
