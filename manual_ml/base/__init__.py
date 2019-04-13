import pandas as pd
import numpy as np


class MLHelpers:
    def setNames(self, X):
        """
        Assuming self here is ML object - set featureNames property
        """
        if type(X) == pd.DataFrame:
            # Remember feature names as strings
            self.featureNames = [str(s) for s in X.columns]
        else:
            self.featureNames = ['x' + str(s) for s in range(X.shape[1])]

        return self

    @staticmethod
    def stripDF(df):
        """
        Avoids need to type below if statement...
        """
        if (type(df) == pd.DataFrame) or (type(df) == pd.Series):
            df = df.values.squeeze()

        return df.astype(np.float32)

    @staticmethod
    def binClass01(Y):
        for ci, c in enumerate(np.sort(np.unique(Y))):
            Y[Y == c] = ci
        return Y

    @staticmethod
    def binClass1Minus1(Y):
        ci = -3
        for c in np.sort(np.unique(Y)):
            ci += 2
            Y[Y == c] = ci

        return Y
