# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 14:03:01 2017

@author: Gareth
"""

#%% 

import numpy as np
from numpy import transpose as t

import string

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder as LE
from sklearn.preprocessing import OneHotEncoder as OH


#%% Random text
# http://randomtextgenerator.com/

# PP
text = []
fn = 'RandomText.txt'
fn = 'PrideAndPred.txt'
lim = 300
with open(fn, 'r') as f:
    i = -1
    for line in f.readlines():
        i+=1
        # print(i)
        if i>=lim:
            break
        line = line.lower()
        line = line.replace('.', '')
        line = line.replace(',', '')
        line = line.replace('?', '')
        line = line.replace('!', '')
        line = line.replace(';', '')
        line = line.replace('"', '')
        line = line.replace('\'', '')
        line = line.replace('_', '')
        line = line.replace('“', '')
        line = line.replace('”', '')
        line = line.replace('\n', '')

        if line !='':
            text = text + line.split(' ')

# Generate a training set
le = LE()
le.fit(text)
tokens = le.transform(text)

# Train one hot encoder
oh = OH()
oh.fit(tokens.reshape(-1, 1))
textOH = oh.transform(tokens.reshape(-1, 1)).todense()
plt.imshow(textOH)
plt.show()


#%% One word -> next word

# Prep
X = textOH[0,:]
Y = textOH[1,:]

inDims = textOH.shape[1]
hiddenDims = 100
outDims = inDims

# Input -> hidden weights
w1 = np.random.randn(inDims, hiddenDims)
# Hidden - > output weights
w2 = np.random.randn(hiddenDims, outDims)    

# Forward prop
# Input -> hidden, linear activation
hAc = np.matmul(X, w1)
# Hidden -> output
outAc = np.matmul(hAc, w2)
# Softmax activation
output =  oh.transform(np.argmax(outAc)).todense()

# Error
err = Y-output

# Backprop
deltaOut = err
deltaHidden = np.matmul(deltaOut, t(w2))
deltaIn = np.matmul(deltaHidden, t(w1))

# Calc updates
w2Updates = t(deltaHidden)*deltaOut
w1Updates = t(deltaIn)*deltaHidden

# Update
LR = 0.01
w1 = w1 - LR*w1Updates
w2 = w2 - LR*w2Updates


#%% Context -> word

# Prep
# Context is 4 words
X = textOH[0:4,:]
# Target is 1 word
Y = textOH[5,:]

inDims = textOH.shape[1]
hiddenDims = 100
outDims = inDims

# Input -> hidden weights
w1 = np.random.randn(inDims, hiddenDims)
# Hidden - > output weights
w2 = np.random.randn(hiddenDims, outDims)    

# Forward prop
# Input -> hidden, linear activation
# Average of input words
hAc = np.zeros(shape=(1,hiddenDims))
for w in range(4):
    hAc += np.matmul(X[w,:], w1)
hAc /=4
# Hidden -> output
outAc = np.matmul(hAc, w2)
# Softmax activation
output =  oh.transform(np.argmax(outAc)).todense()

# Error
err = Y-output

# Backprop
deltaOut = err
deltaHidden = np.matmul(deltaOut, t(w2))
deltaIn = np.matmul(deltaHidden, t(w1))

# Calc updates
w2Updates = t(deltaHidden)*deltaOut
w1Updates = t(deltaIn)*deltaHidden

# Update
LR = 0.01
w1 = w1 - LR*w1Updates
w2 = w2 - LR*w2Updates


#%% Generate offset training data
# Split up for CBOW Training
# X=[-2, -1, +1, +2] -> Y=[0]

# Prepare
n = tokens.shape[0]-4
X = np.empty(shape=(n,4))
Y = np.empty(shape=(n,1))
for r in range(n):
    X[r,0:2] = tokens[r:r+2]
    Y[r] = tokens[r+2]
    X[r,2:5] = tokens[r+3:r+5]
    
inDims = textOH.shape[1]
hiddenDims = 120
outDims = inDims

# Input -> hidden weights
w1 = np.random.randn(inDims, hiddenDims)
# Hidden - > output weights
w2 = np.random.randn(hiddenDims, outDims)    

# Run iterations...
for it in range(2000):
    # Forward prop all inputs over first layer
    # One at a time here, but vecotrised for hidden layer onwards
    # Also need to one-hot as tokens at the mo - inefficient
    # Input -> hidden, linear activation
    # Average of input words
    interX = np.empty(shape=(X.shape[0], hiddenDims))
    for ri, r in enumerate(X):
        hAc = np.zeros(shape=(1,hiddenDims))
        for w in range(4):
            hAc += np.matmul(oh.transform(r[w]).toarray(), w1)
        hAc /=4
        
        interX[ri,:] = hAc
    
    # Hidden -> output
    outAc = np.matmul(interX, w2)
    # Softmax activation
    output = np.argmax(outAc, axis=1)
    
    # Backprop for every training example
    # Error
    err = np.sqrt((oh.transform(Y).toarray()-outAc)**2)
    cost = 1/X.shape[0] * np.sum(0.5*((oh.transform(Y).toarray()-outAc.squeeze())**2))
    # err = (np.sqrt((outAc-oh.transform(Y).toarray())**2))
    
    print('It: ', it, ':',  cost)
    w1Updates = 0
    w2Updates = 0
    
    for r in range(n):
        # Backprop
        deltaOut = np.expand_dims(err[r,:],0)
        deltaHidden = np.matmul(deltaOut, t(w2))
        deltaIn = np.matmul(deltaHidden, t(w1))
        
        # Calc updates
        w2Updates += t(deltaHidden)*deltaOut
        w1Updates += t(deltaIn)*deltaHidden
    
    # Update
    LR = 0.0000000005
    w1 -= LR*w1Updates
    w2 -= LR*w2Updates

    
#%% Test predictions

out = []
targets = []
for r in range(100, 200):
    x = X[r:r+1,:]
    y = Y[r:r+1]
    
    targets = targets+[le.inverse_transform(int(y[0,0]))]
    
    interX = np.empty(shape=(x.shape[0], hiddenDims))
    for ri, r in enumerate(x):
        hAc = np.zeros(shape=(1,hiddenDims))
        for w in range(4):
            hAc += np.matmul(oh.transform(r[w]).toarray(), w1)
        hAc /=4
        
        interX[ri,:] = hAc
    
    # Hidden -> output
    outAc = np.matmul(interX, w2)
    # Softmax activation
    output = np.argmax(outAc, axis=1)
    
    # print([le.inverse_transform(int(xi)) for xi in x.squeeze()])
    # print(le.inverse_transform(int(y)))
    # print(le.inverse_transform(output))
    
    out = out +[le.inverse_transform(output)[0]]
    
_ = [print(t,p) for t,p in zip(targets, out)]
