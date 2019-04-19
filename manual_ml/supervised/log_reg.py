import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from manual_ml.base import ParametricModel

from manual_ml.helpers.metrics import log_loss, accuracy
from manual_ml.helpers.functions import Fs
from manual_ml.helpers.regularization import Regs
from manual_ml.helpers.scaling import standard_score


class LogReg(ParametricModel, Regs, Fs):
    def __init__(self,
                 learning_rate: float=1,
                 max_its: int=100,
                 reg: str=None,
                 a: float=0.1,
                 norm: bool=False,
                 learning_rate_decay: float=0,
                 conv_thresh: float=0.00001,
                 conv_steps: int=6):

        self.params = {'learning_rate': learning_rate,
                       'max_its': max_its,
                       'reg': reg,
                       'lambda': a,
                       'norm': norm,
                       'learning_rate_decay': learning_rate_decay,  # PC on each step
                       'conv_thresh': conv_thresh,  # PC
                       'conv_steps': conv_steps}
        self.featureNames = []

    def fit(self, x, y,
            debug: bool=False):

        max_its = self.params['max_its']
        learning_rate = self.params['learning_rate']
        conv_thresh = self.params['conv_thresh']
        conv_steps = self.params['conv_steps']
        learning_rate_decay = self.params['learning_rate_decay']

        # Set feature names
        self.set_names(x)

        # Convert to mats if not
        x = self.strip_df(x)
        y = self.strip_df(y)

        # Norm?
        if self.params['norm']:
            x = standard_score(x)

        # Initialise coefs
        b = 0
        coefs = np.random.normal(0, 1,
                                 size=(x.shape[1]))
        n = x.shape[0]
        stop = 0
        history = []
        i = -1
        while (stop < conv_steps) and (i < (max_its - 1)):
            i += 1

            # Recalc learning_rate
            learning_rate = learning_rate - learning_rate * learning_rate_decay

            h = self.sigmoid(np.matmul(x, coefs) + b)
            # Get reg value
            reg = self.reg(coefs, n)
            loss = log_loss(y, h) + reg
            history.append(loss)

            # Calculate gradients
            m_grad = learning_rate * 1 / n * (np.matmul(np.transpose(x), y - h) + self.params['lambda'] / n * coefs)
            b_grad = learning_rate * 1 / n * np.sum(b * (y - h))

            # Update gradients
            coefs = coefs + m_grad * reg
            b = b + b_grad

            # Check for convergence
            if i > 0:
                if np.abs(1 - history[i] / history[i-1]) < conv_thresh:
                    stop += 1
                else:
                    stop = 0

            # Print iteration info
            if debug:
                print(f'Iteration {i}: loss={loss} @ learning_rate={learning_rate}')
                if stop > 0:
                    print(f'Converging: {stop}/{conv_steps}')

        results = {'coefs': coefs,
                   'b': b,
                   'loss': loss,
                   'history': history,
                   'converged': stop > conv_steps,
                   'at_step': i}

        self.results = results

        return self

    def predict_proba(self, x):
        return super().predict(x)

    def predict(self, x):
        proba = self.predict_proba(x)
        return (proba > np.mean(proba)).astype(int)


if __name__ == '__main__':

    X = np.array([[1, 1], [2, 2], [2, 1], [0, 1]])
    Y = np.array([0, 1, 1, 0])

    mod = LogReg(learning_rate=0.1,
                 max_its=10000,
                 reg=None,
                 norm=True)
    mod.fit(X, Y,
            debug=True)
    mod.plot_history()

    y_pred = mod.predict(X)

    print(mod)

    accuracy(Y, y_pred)

    from sklearn.model_selection import train_test_split
    from sklearn.datasets import make_blobs
    X, Y = data = make_blobs(n_samples=600,
                             n_features=30,
                             centers=2)
    X = pd.DataFrame(X, columns=['x_'+str(x) for x in range(X.shape[1])])
    Y = pd.DataFrame(Y)

    x_train, x_test, y_train, y_test = train_test_split(X, Y,
                                                        test_size=0.2,
                                                        random_state=48)

    mod = LogReg(learning_rate=0.01,
                 max_its=100000,
                 norm=True,
                 learning_rate_decay=0.005,
                 reg='l1')
    mod = mod.fit(x_train, y_train)

    print(mod)
    y_pred_train = mod.predict(x_train)
    y_pred_test = mod.predict(x_test)

    accuracy(y_train.values, y_pred_train)
    accuracy(y_test.values, y_pred_test)
