import numpy as np
import pandas as pd

from typing import Dict, List, Any, Union

import matplotlib.pyplot as plt

from manual_ml.base import ParametricModel
from manual_ml.helpers.metrics import mse
from manual_ml.helpers.scaling import feature_scale, standard_score


class LinReg(ParametricModel):
    def __init__(self,
                 learning_rate: int =1,
                 max_its: int=100,
                 reg: str='l2',
                 a: float=0.1,
                 norm: bool=False,
                 lr_decay=0,
                 conv_thresh=0.00001,
                 conv_steps=3):
        """
        Linear regression using gradient descent.

        :param learning_rate: Initial learning rate.
        :param max_its: Maximum number of iterations - can be inf.
        :param reg: Regularisation type, either 'l1' or 'l2'.
        :param a: Regularisation strength.
        :param norm: Apply normalisation on current data before fitting or predicting. Doesn't learn from training set,
                     so not ideal, but convenient.
        :param lr_decay: Learning rate decay, %.
        :param conv_thresh: Minimum reduction in error rate before stopping.
        :param conv_steps: How many iterations with error rate reduction < conv_thresh before stopping.
        """

        self.params = {'learning_rate': learning_rate,
                       'max_its': max_its,
                       'reg': reg,
                       'lambda': a,
                       'norm': norm,
                       'lr_decay': lr_decay,  # PC on each step
                       'conv_thresh': conv_thresh,  # PC
                       'conv_steps': conv_steps}

        self.feature_names: List[str] = []
        self.results: Dict[str, Any]

    def fit(self, x: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.DataFrame],
            debug: bool=False) -> None:
        """
        Fit model.

        :param x: Features.
        :param y: Labels.
        :param debug: If true, print information on each iteration.
        """

        max_its = self.params['max_its']
        learning_rate = self.params['learning_rate']
        conv_thresh = self.params['conv_thresh']
        conv_steps = self.params['conv_steps']
        lr_decay = self.params['lr_decay']

        # Set feature names
        self.set_names(x)

        # Convert to mats if not
        x = self.strip_df(x)
        y = self.strip_df(y)

        # Norm?
        if self.params['norm']:
            x = standard_score(x)

        # Initialise coeffs
        b = 0
        coefs = np.random.normal(0, 1,
                                  size=(x.shape[1]))
        n = x.shape[0]

        stop = 0
        i = -1
        history = []
        while (stop < conv_steps) and (i < (max_its-1)):
            i += 1

            # Update learning_rate
            learning_rate = learning_rate - learning_rate * lr_decay

            # Make predictions and get loss
            h = np.matmul(x, coefs) + b
            # Get reg value
            reg = self.reg(coefs, n)
            loss = mse(y, h) + reg
            history.append(loss)

            # Calculate gradients
            m_grad = learning_rate * 1 / n * (np.matmul(np.transpose(x), y - h) + self.params['lambda'] / n * coefs)
            b_grad = learning_rate * 1 / n * np.sum((y - h))

            # Update gradients
            coefs += m_grad
            b += b_grad

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


if __name__ == '__main__':

    x = np.array([[1, 1], [2, 2], [2, 1], [1, 2], [3, 3], [3, 2]])
    y = 2 * x[:, 0] + 3 * x[:, 1] + 1

    mod = LinReg(learning_rate=0.01,
                 max_its=1000,
                 a=1,
                 reg='l1')
    mod.fit(x, y,
            debug=True)
    mod.plot_history(log=True)
    print(mod.results['coefs'])

    y_pred = mod.predict(x)
    plt.scatter(y, y_pred)
    plt.xlabel('y_pred')
    plt.ylabel('y')
    plt.show()

    print(mod.results['coefs'], mod.results['b'])

    print(mod.feature_importance())
