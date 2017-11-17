# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 12:38:49 2017

@author: garet
"""

#%% Imports

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import importlib as il
import MLCommon
il.reload(MLCommon)

from MLCommon import MLHelpers, Scales, Losses, Regs, Fs


#%% Linear regression
       
class LogReg(MLHelpers, Scales, Regs, Fs, Losses):
    def __init__(self, LR=1, maxIts=100, reg='L2', a=0.1, norm=False,
                 LRDecay=0, convThresh=0.00001, convSteps=12):
        self.params = {'LR': LR,
                       'maxIts': maxIts,
                       'reg': reg,
                       'lambda': a, 
                       'norm': norm,
                       'LRDecay': LRDecay, # PC on each step
                       'convThresh': convThresh, # PC
                       'convSteps': convSteps
                       }
        self.featureNames = []
        
        
    def fit(self, X, Y, debug=False):
        
        maxIts = self.params['maxIts']
        LR = self.params['LR']
        convThresh = self.params['convThresh']
        convSteps = self.params['convSteps']
        LRDecay = self.params['LRDecay']
        
        # Set feature names
        self = self.setNames(X)
        
        # Convert to mats if not
        X = self.stripDF(X)
        Y = self.stripDF(Y)
        
        # Norm?
        X = self.norm(X)
        
        # Initialise coeffs
        b = 0
        coeffs = np.random.normal(0,1, size=(X.shape[1]))
        n = X.shape[0]
        
        history = np.zeros(shape=(maxIts))
        
        stop = 0
        i=-1
        while (stop<convSteps) and (i<(maxIts-1)):
            i+=1
            
            # Recalc LR
            LR = LR - LR*LRDecay
            
            h = self.sigmoid(np.matmul(X, coeffs) + b)
            loss = self.logLoss(Y, h)
            history[i] = loss
            
            # Apply regularisation
            reg = self.reg(coeffs)
            # reg=1
            
            # Calculate gradients   
            mGrad = LR * (np.matmul(np.transpose(X), Y-h))
            bGrad = LR * np.sum(b*(Y-h))
            
            # Update gradients
            coeffs = coeffs + mGrad*reg
            b = b+bGrad
            
            # Check for convergence
            if i>0:
                if np.abs(1-history[i]/history[i-1]) < convThresh:
                    stop+=1
                else:
                    stop=0
                    
            # Print iteration info        
            if debug:
                print('Iteration:', i, 'loss='+str(loss), '@ LR='+str(LR),
                      'acc@0.5=', str(self.accuracy(Y,h,1)))
                if stop > 0:
                    print('Converging:', 
                          str(stop)+'/'+str(convSteps))
                
        results = {'coeffs': coeffs,
                   'b': b,
                   'loss': loss,
                   'history': history[0:i],
                   'converged': stop>convSteps,
                   'atStep': i}
        
        self.results = results
        return self
    
        
    def predict(self, X):
        
        X = self.stripDF(X)
            
        m = self.results['coeffs']
        b = self.results['b']
        
        return self.sigmoid(np.matmul(X, m) + b)
    
    def plotHistory(self):
        plt.plot(self.results['history'])
        plt.ylabel('loss')
        plt.xlabel('Iteration')
        plt.show()
  
    def print(self):
        print('Intercept:', str(self.results['b']))
        [print(fn,str(c)) for fn,c in zip(self.featureNames, 
                                          self.results['coeffs'])]
        
    
#%%     
   
if __name__== '__main__':
    
    X = np.array([[1, 1], [2, 2], [2, 1]])
    Y = np.array([0, 1, 1])
    
    mod = LogReg(LR=0.01, maxIts=10000, reg='L1', norm=True)
    mod = mod.fit(X, Y, debug=True)
    mod.plotHistory()
    
    yPred = mod.predict(X)
    
    # print(mod.results['coeffs'], mod.results['b'])
    mod.print()
    mod.accuracy(Y, YPred, 1)
    
#%% Test

if __name__== '__main__':
    from sklearn.datasets import load_boston
    
    
    boston = load_boston()
    X = boston['data']
    Y = boston['target']
    
    mod = linReg(LR=0.5, maxIts=100000, norm='FS', LRDecay=0.00005, reg='L1')
    mod = mod.fit(X, Y, debug=True)
    mod.plotHistory()

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
    
    mod = LogReg(LR=0.01, maxIts=100000, norm='FS', LRDecay=0.005, reg='L2')
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
        
    mod = LogReg(LR=0.01, maxIts=100000, norm='FS', LRDecay=0.005, reg='L2')
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
    
    mod = LogReg(LR=0.00001, maxIts=100000, norm='SS', LRDecay=0.00001, reg='L2')
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
    
