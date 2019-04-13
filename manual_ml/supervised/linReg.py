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

from MLCommon import MLHelpers, Scales, Losses, Regs


#%% Linear regression
       
class LinReg(MLHelpers, Scales, Regs, Losses):
    def __init__(self, LR=1, maxIts=100, reg='L2', a=0.1, norm=False,
                 LRDecay=0, convThresh=0.00001, convSteps=3):
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
            
            h = np.matmul(X, coeffs) + b
            loss = self.mse(Y, h)
            history[i] = loss
            
            # Apply regularisation
            reg = self.reg(coeffs)
            # reg=1
            
            # Calculate gradients   
            mGrad = LR * 2/n * (np.matmul(np.transpose(X), Y-h))
            bGrad = LR * 2/n * np.sum((Y-h))
            
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
                print('Iteration:', i, 'loss='+str(loss), '@ LR='+str(LR))
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
        
        if type(X) == pd.DataFrame:
            X = X.values
            
        m = self.results['coeffs']
        b = self.results['b']
        
        return np.matmul(X, m) + b
    
    
    def plotHistory(self):
        plt.plot(self.results['history'])
        plt.ylabel('loss')
        plt.xlabel('Iteration')
        plt.show()
  
    
#%%     
   
if __name__== '__main__':
    
    # y = 2*x0 + 3*x1 + 1
    
    X = np.array([[1, 1], [2, 2], [2, 1]])
    Y = 2*X[:,0] + 3*X[:,1] + 1
    
    mod = LinReg(LR=0.1, maxIts=10000, reg='L2')
    mod = mod.fit(X, Y, debug=True)
    mod.plotHistory()
    
    yPred = mod.predict(X)
    
    print(mod.results['coeffs'], mod.results['b'])
    
    
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
    