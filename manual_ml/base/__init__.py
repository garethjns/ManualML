import pandas as pd
import numpy as np

from typing import Union

import matplotlib.pyplot as plt

from manual_ml.helpers.regularization import Regs
from manual_ml.helpers.scaling import standard_score


class BaseModel:
    def set_names(self, x: Union[pd.DataFrame, np.ndarray]) -> None:
        """
        Save df column names to object, or make some up.

        :param x: Array to generate for, or save names from.
        """
        if type(x) == pd.DataFrame:
            # Remember feature names as strings
            self.feature_names = [str(s) for s in x.columns]
        else:
            self.feature_names = ['x' + str(s) for s in range(x.shape[1])]

    @staticmethod
    def strip_df(df: pd.DataFrame) -> np.ndarray:
        """
        Convert dt to array.

        :param df: DataFrame to convert.
        :Retrun: Squeezed vales as float32.
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


class ParametricModel(BaseModel, Regs):
    def __str__(self):
        if self.results is not None:
            return "\n".join([f"Intercept: {self.results['b']}"]
                             + [f"{fn}: {c}" for fn, c in zip(self.feature_names,
                                                              self.results['coefs'])])
        else:
            return "Unfit model."
    def feature_importance(self):
        pd.DataFrame(self.coefs,
                     columns=self.feature_names)

    def predict(self, x: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:

        if type(x) == pd.DataFrame:
            x = x.values

        # Norm?
        if self.params['norm']:
            x = standard_score(x)

        m = self.results['coefs']
        b = self.results['b']

        return np.matmul(x, m) + b

    def plot_history(self,
                     log: bool=False):

        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.plot(self.results['history'])
        ax.set_ylabel('loss')
        ax.set_xlabel('Iteration')

        if log:
            ax.set_xscale('log')
            ax.set_yscale('log')

        plt.show()
