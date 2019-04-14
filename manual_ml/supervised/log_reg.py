import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from manual_ml.base import ParametricModel

from manual_ml.helpers.metrics import log_loss
from manual_ml.helpers.functions import Fs
from manual_ml.helpers.regularization import Regs
from manual_ml.helpers.scaling import standard_score


class LogReg(ParametricModel, Regs, Fs):
    def __init__(self,
                 learning_rate=1,
                 max_its=100,
                 reg='l2',
                 a=0.1,
                 norm=False,
                 learning_rate_decay=0,
                 conv_thresh=0.00001,
                 conv_steps=12):

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
        self = self.set_names(x)

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
            loss = log_loss(y, h)
            history.append(loss)

            # Apply regularisation
            reg = self.reg(coefs)

            # Calculate gradients
            m_grad = learning_rate * (np.matmul(np.transpose(x), y - h))
            b_grad = learning_rate * np.sum(b * (y - h))

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
                   'atStep': i}

        self.results = results
        return self

    def predict(self, x):
        return self.sigmoid(super().predict(x))


if __name__== '__main__':

    X = np.array([[1, 1], [2, 2], [2, 1]])
    Y = np.array([0, 1, 1])

    mod = LogReg(learning_rate=0.01,
                 max_its=10000,
                 reg='L1',
                 norm=True)
    mod = mod.fit(X, Y, debug=True)
    mod.plot_history()

    y_pred = mod.predict(X)

    # print(mod.results['coeffs'], mod.results['b'])
    mod.print()
    accuracy(Y, y_pred, 1)

#%% Test

if __name__== '__main__':
    from sklearn.datasets import load_boston

    boston = load_boston()
    X = boston['data']
    Y = boston['target']

    mod = linReg(LR=0.5, maxIts=100000, norm='FS', LRDecay=0.00005, reg='L1')
    mod = mod.fit(X, Y, debug=True)
    mod.plot_history()

    yPred = mod.predict(X)

    plt.scatter(Y,yPred)


#%% Banknote
if __name__ == '__main__':
    from sklearn.model_selection import train_test_split as tts


    data = pd.read_csv('data_banknote_authentication.txt', header=None)
    data.columns = ['x'+str(x) for x in range(4)]+list('Y')
    data.head()

    X = data.loc[:,['x'+str(x) for x in range(4)]]
    Y = data.loc[:,'Y']
    XTrain, XValid, YTrain, YValid = tts(
        X, Y, test_size=0.25, random_state=512)

    mod = LogReg(learning_rate=0.01, max_its=100000, norm='FS', learning_rate_decay=0.005, reg='L2')
    mod = mod.fit(XTrain, YTrain, debug=True)
    mod.print()
    yPredTrain = mod.predict(XTrain)
    yPredValid = mod.predict(XValid)

    mod.accuracy(yPredTrain, YTrain, p=0)
    mod.accuracy(yPredValid, YValid, p=0)

#%% Generated

if __name__ == '__main__':

    nF = 30
    X,Y = data = mk(n_samples=600,
              n_features=nF,
              n_informative=20,
              n_redundant=5,
              n_repeated=0,
              n_classes=2)
    X = pd.DataFrame(X, columns=['x'+str(x) for x in range(nF)])
    Y = pd.DataFrame(Y)

    XTrain, XValid, YTrain, YValid = tts(
            X, Y, test_size=0.2, random_state=48)

    mod = LogReg(learning_rate=0.01, max_its=100000, norm='FS', learning_rate_decay=0.005, reg='L2')
    mod = mod.fit(XTrain, YTrain)

    mod.print()
    yPredTrain = mod.predict(XTrain)
    yPredValid = mod.predict(XValid)

    mod.accuracy(YTrain, yPredTrain)
    mod.accuracy(YValid, yPredValid)


#%% Titanic
if __name__ == "__main__":

    train = pd.read_csv('titanic_train_PPed.csv')
    test = pd.read_csv('titanic_test_PPed.csv')

    Y = train.Survived
    X = train.loc[:,['Pclass', 'Sex', 'Adult', 'Title']]
    XTest = test.loc[:,['Pclass', 'Sex', 'Adult', 'Title']]

    XTrain, XValid, YTrain, YValid = tts(
        X, Y, test_size=0.33, random_state=47)

    mod = LogReg(learning_rate=0.00001, max_its=100000, norm='SS', learning_rate_decay=0.00001, reg='L2')
    mod = mod.fit(XTrain, YTrain, debug=True)

    yPredTrain = mod.predict(XTrain)
    yPredValid = mod.predict(XValid)
    yPredTest = mod.predict(XTest)

    mod.accuracy(YTrain, yPredTrain, 1)
    mod.accuracy(YValid, yPredValid, 1)

    sub = pd.DataFrame()
    sub['PassengerId'] = test['PassengerId']
    sub['Survived'] = np.int32(yPredTest)
    sub.to_csv('manualLogReg.csv', index=False)

