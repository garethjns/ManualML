# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 19:25:05 2017

@author: Gareth
"""

#%% Imports

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

import importlib as il
import MLCommon
il.reload(MLCommon)
from MLCommon import Fs, MLHelpers, Losses

from sklearn.preprocessing import OneHotEncoder as OH
from numpy import transpose as t

from sklearn.model_selection import train_test_split as tts


#%% Classes

class Layer(Fs):
    def __init__(self, inUnits=2, outUnits=2, name='', ipt=False,
          act=Fs.sigmoid, dAct=Fs.dSigmoid):
        
        if ipt==True:
            ty = 'Input'
            w = np.random.randn(outUnits, inUnits)
            m = np.zeros(shape=(1, inUnits))
            dAct = lambda x: x
            act = lambda x: x
        elif outUnits==0 or outUnits==None:
            ty = 'Output'
            w = None
            m = np.zeros(shape=(1, inUnits))
        else:
            ty = 'Hidden'
            w = np.random.randn(outUnits, inUnits+1)
            m = np.zeros(shape=(1, inUnits+1))
            
        self.name = name
        self.type = ty # Input, output or hidden
        self.w = w # Weights
        self.actF = act
        self.dActF = dAct
        self.act = m # Last activation values
        self.d = m # Last delta values
        self.u = m # Last update
        self.next = []
        
    def connect(self, nxt):
        self.next = nxt
        
        return self
    
    
class Network(MLHelpers, Losses):
    def __init__(self, layerList):
        
        # for li in np.arange(len(layerList)-1, 0, -1):
        #    layerList[li-1] = layerList[li].connect(layerList[li-1])
    
        self.net = layerList
        self.nLayers = len(layerList)
        
    def predict(self, x):
        """
        Forward prop
        """
        # Set input layer
        self.net[0].act = t(x)
        
        for li in range(1, self.nLayers-1):
            print(self.net[li].name, 'to', self.net[li+1].name)
            
            # Mat mul and activateF
            act = self.net[li+1].actF(np.matmul(self.net[li].w, 
                                    self.net[li].act))
                
            # Append bias on hidden layers
            if self.net[li+1].type == 'hidden':
                self.net[li+1].act = np.append(
                                         np.ones(shape=(1, act.shape[1])), 
                                         act, axis=0)
        return self
    

    
    def fit(self, x, y, its=100):
        
        n = x.shape[0]
        
        for it in range(its):
            print('It:', it)
            
            # Forward prop
            # Run hidden layers
            self = self.predict(x)
        
            # Calculate error
            #cost = 1/n * np.sum(0.5*((y-t(self.net[self.nLayers-1].act))**2))
            cost = 1/X.shape[0] * np.sum(0.5*((y.squeeze()-t(self.net[self.nLayers-1].act).squeeze())**2))
            print('Iteration: '+str(it)+'/'+str(its), 'Cost:', cost)
        
            # Run backprop
            # Accumulate updates for all n
            for r in range(n):
                # Set error for output
                self.net[self.nLayers].d = self.net[self.nLayers].act - \
                            np.expand_dims(
                                    self.net[self.nLayers][:,r]-y[r,:], axis=1)
                            
                # For remaining layers
                for li in np.arange(self.nLayers, 1, -1):
                
                    d = np.matmul(t(self.net[li].w), 
                                  self.net[li+1].act)
                    u = d * self.net[li].dActF(self.net[li].act[:,r])
                    
                    self.net[li].u += u
                    self.net[li].d += d
            
            
        # Update
        for li in range(0,self.nLayers):
            self.net[li].w -= self.LR*self.net[li].u
            
            
        return self
                    
    
    def predictProba():
        pass
    
    
#%% Tests 
if __name__ == '__main__':
    # Generated
    from sklearn.datasets import make_classification as mk
    
    nF = 4
    X,Y = data = mk(n_samples=100, 
              n_features=nF, 
              n_informative=3, 
              n_redundant=0,
              n_repeated=0, 
              n_classes=2,
              n_clusters_per_class=1,
              scale=1,
              shift=0)
    
    XTrain, XValid, YTrain, YValid = tts(
        X, Y, test_size=0.33, random_state=47)
    
    XTest = XValid
    
    
    ipt = Layer(inUnits=2, outUnits=3, name='input', ipt=True)
    hidden = Layer(inUnits=3, outUnits=4, name='hidden1')
    hidden2 = Layer(inUnits=4, outUnits=1, name='hidden2')
    output = Layer(inUnits=1, outUnits=0, name='output')
    
    mod = Network([ipt, hidden, hidden2, output])
    
    mod.fit(XTrain, YTrain)