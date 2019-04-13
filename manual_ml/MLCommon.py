# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 19:53:49 2017

@author: garet
"""


#%% Imports

import numpy as np
import pandas as pd


#%%

class Fs():
    """
    General helpers
    """
    @staticmethod
    def sigmoid(x):
        return 1/(1+np.exp(-x))
    
    @staticmethod
    def dSigmoid(x):
        return Fs.sigmoid(x) * (1.0 - Fs.sigmoid(x))   
    
    @staticmethod
    def softmax(x):
        return (np.argmax(x, axis=1))
    
class Ks():
    """
    Kernels
    """
    def __init__(self, ty='linear', **kwargs):
        self.params = dict([(key,value) for key, value in kwargs.items()])
        self.kernelType = ty.lower()
        
    def __call__(self, x1, x2):
        if self.kernelType.lower() == 'linear':
            res = self. linear(x1, x2)
        elif self.kernelType.lower() == 'polynomial':
            res = self.polynomial(x1, x2)
        return res
        
    def linear(self, x1, x2):
        return np.dot(x1, x2)
    
    def polynomial(self, x1, x2):
        
        p = self.params.get('p', 2)
         
        return (1+np.dot(x1, x2)) ** p
    
    def gaussian(self, x1, x2):
        sig = self.params.get('sigma', 3)
        
        return np.exp(-np.linalg.norm(x1-x2, axis=0)**2 / (2*(sig**2)))
    
    def RBF(self, x1, x2, gamma=1):
        return np.exp(-gamma*np.abs(x1-x2)**2).squeeze()
    
    
class Scales():
    """
    Normalisation methods
    """
    def norm(self, X):
        """
        Assuming self here is ML object. ie. inherited in to ML object and 
        called from there.
        """
        if (self.params['norm']=='FS') or (self.params['norm']==True):
            X = self.featureScale(X)
        elif self.params['norm']=='SS':
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
        

class Losses():
    """
    Cost helpers
    """
    @staticmethod
    def mse(Y, YPred):
        """
        Mean squared error
        """
        return 1/len(Y) * np.sum((Y-YPred)**2)
    
    @staticmethod
    def logLoss(Y, YPred):
        """
        Logloss for logistic regression
        -log P(yt|yp) = -(yt log(yp) + (1 - yt) log(1 - yp))
        """
        return 1/len(Y) * np.sum(-Y*np.log(YPred)-(1-Y)*np.log(1-YPred))
    
    @staticmethod
    def accuracy(Y, YPred, p=3):
        """
        Calcualte accuracy after round to number of decimals specified in p
        """
        
        # Handle weird behaviour of pandas input (.values has singleton dims)
        if (type(Y)==pd.Series) or (type(Y)==pd.DataFrame):
            Y = Y.values.squeeze()
        if type(YPred) == pd.Series:
            YPred = YPred.values
        
        # Round
        Y = np.round(Y, p)
        YPred = np.round(YPred, p)
        
        acc = np.sum(Y==YPred)/len(Y)
        print('Accuracy: ', acc)
        return acc

class Regs():
    """
    Regularisation methods
    """
    def reg(self, coeffs):
        """
        Self here assumed to be ML object
        """
        # Regularisation
        if (self.params['reg'].lower=='l1') | \
            (self.params['reg'].lower=='lasso'):
                reg = regs.LASSO(coeffs, self.params['lambda'])
        elif (self.params['reg'].lower=='l2') | \
            (self.params['reg'].lower=='ridge'):
                reg = regs.ridge(coeffs, self.params['lambda'])
        else:
            reg = 1
            
        return reg
    
    @staticmethod
    def ridge(coeffs, a=1):
        """
        Ridge / L2 regularization,
        Objective = RSS + α * (sum of square of coefficients)
        """
        return a * np.sum(coeffs**2)
    
    @staticmethod
    def LASSO(coeffs, a=1):
        """
        LASSO / L1 reg
        LASSO stands for Least Absolute Shrinkage and Selection Operator. 
        Objective = RSS + α * (sum of absolute value of coefficients)
        """
        return a * np.sum(coeffs)


class MLHelpers():
    def setNames(self, X):
        """
        Assuming self here is ML object - set featureNames property
        """
        if type(X) == pd.DataFrame:
            # Remember feature names as strings
            self.featureNames = [str(s) for s in X.columns]
        else:
            self.featureNames = ['x'+str(s) for s in range(X.shape[1])]

        return self

    @staticmethod
    def stripDF(df):
        """
        Avoids need to type below if statement...
        """
        if (type(df)==pd.DataFrame) or (type(df)==pd.Series):
            df = df.values.squeeze()
            
        return df.astype(np.float32)
    
    @staticmethod
    def binClass01(Y):
        for ci, c in enumerate(np.sort(np.unique(Y))):
            Y[Y==c]=ci
        return Y
    
    @staticmethod
    def binClass1Minus1(Y):
        ci = -3
        for c in np.sort(np.unique(Y)):
            ci+=2
            Y[Y==c]=ci
            
        return Y
  
    
#%% Test losses
if __name__ == "__main__":
    
    from sklearn.metrics import log_loss
    from sklearn.metrics import mean_squared_error as mse
    
    X = np.array([[1, 1], [2, 2], [2, 1]])
    Y = np.array([0, 1, 1])
    m = np.array([1, 2])
    h = Fs.sigmoid(np.matmul(X,m))
    
    print('logloss')
    print('Losses.logLoss:', np.sum(Losses.logLoss(Y, h)))
    print('Scikit:', log_loss(Y, h))
    
    
    X = np.array([[1, 1], [2, 2], [2, 1]])
    Y = 2*X[:,0] + 3*X[:,1] + 1
    m = np.array([1, 2])
    h = np.matmul(X,m)
    
    print('mse')
    print('Losses.mse:', np.sum(Losses.mse(Y, h)))
    print('Scikit:', mse(Y, h))


#%% Test kernels

if __name__ == "__main__":
    
    X = np.array([[1, 5],
                  [2, 6],
                  [3, 7],
                  [4, 8],
                  [5, 9],
                  [6, 10],
                  [7, 11]])
    wt = np.array([-0.5, 0.5])
    
    print(Ks.linear(X, wt))
    print(Ks.polynomial(X, wt))
    print(Ks.gaussian(X, wt))
    print(Ks.RBF(X, wt))
    