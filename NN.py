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
            m = np.zeros(shape=(outUnits, inUnits))
            dAct = lambda x: x
            act = lambda x: x
        elif outUnits==0 or outUnits==None:
            ty = 'Output'
            w = None
            m = np.zeros(shape=(1, inUnits))
        else:
            ty = 'Hidden'
            w = np.random.randn(outUnits, inUnits+1)
            m = np.zeros(shape=(outUnits, inUnits+1))
            
        self.name = name
        self.type = ty # Input, output or hidden
        self.w = w # Weights
        self.actF = act
        self.dActF = dAct
        self.act = m # Last activation values
        self.d = m # Last delta values
        self.u = m # Last update
        self.next = []
        
    def forward():
        """
        Forward pass
        """
        pass
    
    def back():
        """
        Backwards pass
        """
        pass
        
    def connect(self, nxt):
        self.next = nxt
        
        return self
    
    
class Network(MLHelpers, Losses):
    def __init__(self, layerList):
        
        for li in np.arange(len(layerList)-1, 0, -1):
            layerList[li-1] = layerList[li].connect(layerList[li-1])
    
        self.net = layerList
        
    def fit():
        pass
    
    def predict():
        pass
    
    def predictProba():
        pass
    
    
#%% Tests 

ipt = Layer(inUnits=2, outUnits=3, name='input', ipt=True)
hidden = Layer(inUnits=3, outUnits=4, name='hidden1')
hidden2 = Layer(inUnits=4, outUnits=1, name='hidden1')
output = Layer(inUnits=1, outUnits=0, name='output')

mod = Network([ipt, hidden, hidden2, output])