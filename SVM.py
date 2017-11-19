# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 14:19:27 2017

@author: garet
"""

#%% Imports

import numpy as np

from tqdm import tqdm

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split as tts
from sklearn.datasets import make_classification as mk


#%% SVM class
# Linear, classification
class SVM():
    def __init__(self, costThresh=0):
        self.results = {'mag': np.nan,
                       'w' : np.nan,
                       'b': np.nan}
        self.params = {'costThresh':costThresh}
    
    def predict(self, x):
        if np.isnan(self.results['mag']):
            print('Fit first')
            return
            
        return np.sign(self.predictProba(x))

    def predictProba(self, x):
        if np.isnan(self.results['mag']):
            print('Fit first')
            return
            
        return np.dot(x,self.results['w'])+self.results['b']
    
    def hyperplane(self, xRange, v, d1=0, d2=1):
        # V = 1 = positive support vector
        # V = 0 = decision boundary
        # V = -1 = negative support vector
        
        if np.isnan(self.results['mag']):
            print('Fit first')
            return
            
        y = (-self.results['w'][d1]*xRange-self.results['b']+v) \
                /self.results['w'][d2]    
        return y 
            
    def plotDecision(self, x, y=[]):
        
        if np.isnan(self.results['mag']):
            print('Fit first')
            return
            
        if y==[]:
            y = self.predict(x)
            
        data = np.array([x.min()*0.9, x.max()*1.1])
    
        hp = self.hyperplane(data, -1)
        plt.plot(data, hp, label='Neg supp.', c='k')
        plt.scatter(x[y==-1,0], x[y==-1,1], c='b')
        
        hp = self.hyperplane(data, 0)
        plt.plot(data, hp, 'y--', label='Decision boundary.')
        hp = self.hyperplane(data, +1)
        plt.plot(data, hp, label='Pos supp.', c='k')
        
        plt.scatter(x[y==1,0], x[y==1,1], c='r')
        plt.scatter(x[y==-1,0], x[y==-1,1], c='b')
        plt.xlabel('x1')
        plt.ylabel('x2')
    
        plt.show()
    
    def fit(self, x, y, 
               wStart=10, wStep=1, 
               bStart=50, bStep=0.1, 
               rSteps=3, debug=False,
               rStep=0, magBest=99999,
               trans=np.array([[1,1], [-1,1], [1,-1], [-1,-1]])):
        """
        Fit using recursion
        w range reduces with each call
        b range static
        
        Matulmul not loop over x data
        Early stopping on transformations
        B loop runs fully
        Early stopping on w loop if mag starts increasing
        """
        
        if (rStep+1)>rSteps:
            print('Done')
            return self
                    
        else:
            print('R step:', str(rStep+1)+'/'+str(rSteps))
            if debug: print('R step:', str(rStep+1)+'/'+str(rSteps))
        
        wRange = np.arange(wStart, -wStep, -wStep)
        bRange = np.arange(bStart, 
                           -bStart-bStep,
                           -bStep)
    
        # If ||w|| starts increasing again, want to stop 
        incCount = 0
        candidateFound = False
        bestCost=1
        # For all ws
        for wr in tqdm(wRange):
            w = np.array([wr, wr])
            thisBest = 99999
            # Check all values of b
            for b in bRange:
                if debug:
                    print('*'*5)
                    print(w, b)
                # At each transformation
                for wt in trans:
                    test = False
                    
                    res = y*(np.dot(x,w*wt)+b)
                    if self.params['costThresh']==0:
                        test = np.all(res>=1)
                        cost = 0
                    else:
                        cost = 1-np.sum(res>=1)/len(res) 
                        test= cost<self.params['costThresh']
                    
                    # If a test succedes, test against best
                    if test:
                    # print('w:', w*wt, 'b', b, '=', test, np.all(test>0))                
                        if debug: print('Candidate:', w*wt, test)
                        candidateFound = True
                        wMag = np.linalg.norm(w)+cost
                        if wMag<thisBest:
                            thisBest = wMag
                            thisWBest = w*wt
                            thisBBest = b
                            
                            if debug: print('New best', thisBest, 
                                            thisWBest, thisBBest)
                            
                        break # No need to check other transformations
                    else:
                        if debug: print('Fail at :', w*wt, test)
           
            # b loop done
            # Check best from b loop against overall best
            if thisBest<magBest:
                self.results['mag'] = thisBest
                self.results['w'] = thisWBest
                self.results['b'] = thisBBest
            else:
                # If this b loop was worse, probably going up other side of
                # function
                incCount+=1
                
                if debug: 
                    print('Passed optimum point', incCount)
            
            # Only use inc count if overlap is none
            # If there's overlap, caon't be sure if mag is increasing due
            # to overlap or other side of function
            # (mag is curretely ||w|| + cost)
            if (incCount>2) & (cost==0):
                stepsSaved = len(wRange)-np.argmax(wr==wRange)
                print('Saved:', stepsSaved, 'w steps, (',
                      stepsSaved*len(bRange), 'its )')
                # Stop looping over w, go to return
                break
        
        if not candidateFound:
            # Failed to converge entirely
            print('Failed')
            return self            
        else:
            # Increase w range resoltuion, run again
           return self.fit(x, y, wStart=self.results['w'][0]*1.025, wStep=wStep/10,
                      bStart=bStart, bStep=bStep,
                      rSteps=rSteps, rStep=rStep+1, 
                      debug=False)
           
           
#%% Basic test
if __name__ == "__main__":
    
    x = np.array([[1, 7],
              [2, 8],
              [3, 8],
              [5, 1],
              [6, -1],
              [7, 3]])

    y = np.array([-1, -1, -1, 1, 1, 1])
    
    mod = SVM()
    mod = mod.fit(x, y)
    mod.plotDecision(x, y)
    mod.plotDecision(x)
    
    
#%% Overlapping clusters
# Works with simple cost added, requires correct setting of cost thresh
if __name__ == "__main__":
    
    x = np.array([
              [1, 7],
              [2, 8],
              [3, 8],
              [5, 1],
              [3, 6],
              [3, 7],
              [4, 7],
              [2, 7],
              [3, 8],
              [9, 1],
              [6, -1],
              [7, 3],
              [5, -1],
              [7, -1],
              [7, 3],
              [5, 2],
              [6, -1],
              [5, -3]
              ])

    y = np.array([-1, -1, -1, -1, -1, -1, -1, -1, -1, 
                  1, 1, 1, 1, 1, 1, 1, 1, 1])
    
    mod = SVM(costThresh=0.3)
    mod = mod.fit(x, y, wStart=100, wStep=1)
    mod.plotDecision(x, y)
    mod.plotDecision(x)
    
#%% Generated

if __name__ == '__main__':
    
    nF = 2
    X,Y = data = mk(n_samples=600, 
              n_features=nF, 
              n_informative=2, 
              n_redundant=0,
              n_repeated=0, 
              n_classes=2)
    Y[Y==0]=-1
    # X = pd.DataFrame(X, columns=['x'+str(x) for x in range(nF)])
    # Y = pd.DataFrame(Y)
    
    XTrain, XValid, YTrain, YValid = tts(
            X, Y, test_size=0.2, random_state=48)
        
    mod = SVM(costThresh=0.4)
    mod = mod.fit(XTrain, YTrain)
    mod.plotDecision(x, y)
    mod.plotDecision(x)
    
    yPredTrain = mod.predict(XTrain)
    yPredValid = mod.predict(XValid)
    
   
#%%
    