import numpy as np
import pandas as pd

from tqdm import tqdm
import matplotlib.pyplot as plt
from itertools import product

from sklearn.model_selection import train_test_split as tts
from sklearn.datasets import make_classification as mk

from manual_ml.base import MLHelpers
from manual_ml.helpers.kernels import Ks
from manual_ml.helpers.metrics import accuracy


class SVM(MLHelpers):
    """
    SVM Class.

    Linear, classification.
    """

    def __init__(self, kernel=Ks('Linear'), **kwargs):
        self.results = {'mag': np.nan,
                       'w' : np.nan,
                       'b': np.nan}
        self.params = dict([(key,value) for key, value in kwargs.items()])
        self.params = dict(kwargs)
        self.featureNames = []
        self.kernel = kernel

    def predict(self, X, m1=True):
        x = self.stripDF(X)
        
        if np.isnan(self.results['mag']):
            print('Fit first')
            return
            
        y = np.sign(self.predictProba(x))
        
        if m1: 
            return y
        else:
            return self.binClass01(y)

    def predictProba(self, X):
        x = self.stripDF(X)
        
        if np.isnan(self.results['mag']):
            print('Fit first')
            return
            
        return self.kernel(x,self.results['w'])+self.results['b']
    
    def hyperplane(self, xRange, v, d1=0, d2=1):
        # V = 1 = positive support vector
        # V = 0 = decision boundary
        # V = -1 = negative support vector
        
        if np.isnan(self.results['mag']):
            print('Fit first')
            return
            
        # y = (-self.results['w'][d1]*xRange-self.results['b']+v) \
        #         /self.results['w'][d2]    
        y = (-self.kernel(self.results['w'][d1],xRange)-self.results['b']+v) \
                 / self.results['w'][d2]   
        return y 
            
    def plotDecision(self, X, Y=None, d1=0, d2=1):
        
        if np.isnan(self.results['mag']):
            print('Fit first')
            return
        
        x = self.stripDF(X)
        if Y is None:
            y = self.predict(x)
        else:
            y = self.stripDF(Y)
            # Make sure classes are -1,1
            y = self.binClass1Minus1(y)
            
        
        data = np.arange(x[:,(0,1)].min()*0.9, x[:,(0,1)].max()*1.1,0.1)
    
        hp = self.hyperplane(data, -1, d1=d1, d2=d2)
        plt.plot(data, hp, label='Neg supp.', c='k')
        plt.scatter(x[y==-1,d1], x[y==-1,d2], c='b')
        
        hp = self.hyperplane(data, 0, d1=d1, d2=d2)
        plt.plot(data, hp, 'y--', label='Decision boundary.')
        hp = self.hyperplane(data, +1, d1=d1, d2=d2)
        plt.plot(data, hp, label='Pos supp.', c='k')
        
        plt.scatter(x[y==1,d1], x[y==1,d2], c='r')
        plt.scatter(x[y==-1,d1], x[y==-1,d2], c='b')
        plt.xlabel('x1')
        plt.ylabel('x2')
    
        plt.show()
    
    def fit(self, X, Y, **kwargs):
        if True:
            # Linear SVM, recursive
            
            # Check and set default params
            self.params['costThresh'] = self.params.get('costThresh', 0)
            self.params['convVerSteps'] = self.params.get('convVerSteps', 6)
            self.params['C'] = self.params.get('C', 0.5)
            self.params['slack'] = self.params.get('slack', 0)
            
            self.fitLinearRec(X, Y, **kwargs)
        elif self.mode=='KernelRec':
            # Kernel SVM, recursive
            self.fitV2(X, Y, **kwargs)
        
        return self
              
    def fitLinearRec(self, X, Y, 
               wStart=5, wStep=1, 
               bStart=50, bStep=1, 
               rSteps=3, debug=False,
               rStep=0, magBest=np.inf):
        """
        Fit using recursion
        w range reduces with each call
        b range static
        
        Matmul not loop over x data
        Early stopping on transformations
        B loop runs fully
        Early stopping on w loop if mag starts increasing
        """
        # Set feature names
        self = self.setNames(X)
        
        # Convert to mats if not
        x = self.stripDF(X)
        y = self.stripDF(Y)
        
        # Make sure classes are -1,1
        y = self.binClass1Minus1(y)
        
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
        # Create transformations to test
        trans = np.array(list(product([1, -1], repeat=x.shape[1])))
        
        C = self.params['C']
        slack = self.params['slack']
        
        # If ||w|| starts increasing again, want to stop 
        incCount = 0
        
        anyCandidateFound = False
        # For all ws
        for wr in tqdm(wRange):
            # Candidate found at this w?
            candidateFound = False
            
            # Assign same w for every dimension
            w = np.array([wr for i in range(x.shape[1])])
            thisBest = np.inf
            thisWBest = np.inf
            thisBBest = np.inf
            thisCost = 0 
            # Check all values of b
            for b in bRange:
                if debug:
                    print('*'*5)
                    print(w, b)
                # At each transformation
                for wt in trans:
                    test = False
                    
                    # res = y*(np.dot(x,w*wt)+b)
                    res = y*(self.kernel(x,w*wt)+b)
                    test = np.all(res>=(1-slack)) 

                    # If a test succedes, test against best
                    if test:
                    # print('w:', w*wt, 'b', b, '=', test, np.all(test>0))                
                        if debug: pass
                        print('Candidate:', w*wt, test)
                        candidateFound = True
                        anyCandidateFound = True
                        # Calculate slack used to get here. Eg if slack is 10,
                        # count amount used for each point in x
                        # Include negative values?
                        # slackUsed = np.sum(res-(np.zeros(shape=(1,len(res)))+10))
                        # wMag = np.linalg.norm(w)+C*slack*x.shape[0]
                        # Or not?
                        slackUsed = np.abs(np.sum(res[res<1]-1))
                        if debug: print('Used slack:', slackUsed, 
                                        'with weight', C*slackUsed)
                        # print(wMag, thisBest)
                        # print(w, wMag, thisBest)
                        wMag = np.linalg.norm(w)+C*slackUsed
                        if wMag<thisBest:
                            thisBest = wMag
                            thisWBest = w*wt
                            thisBBest = b
                            thisCost = C*slackUsed
                            
                            if debug: print('New best', thisBest, 
                                            thisWBest, thisBBest)
                            
                        break # No need to check other transformations
                    else:
                        pass#if debug: print('Fail at :', w*wt, test)
                
                if candidateFound:
                    # Candidate found at this w/b. No need to search other bs.
                    if debug: print('breaking b loop')
                    break
           
            # b loop done
            # Check best from b loop against overall best
            # print(thisBest, magBest)
            if candidateFound & (thisBest<magBest):
                if debug: print('Condition 1:', thisBest, magBest)
                self.results['mag'] = thisBest
                self.results['w'] = thisWBest
                self.results['b'] = thisBBest
                self.results['slackCost'] = thisCost
            elif candidateFound & (thisBest!=np.inf) & (thisBest==magBest):
                if debug: print('Condition 2:', thisBest, magBest)
                pass
            elif candidateFound &(thisBest!=np.inf):
                if debug: print('Condition 3:', thisBest, magBest)
                # If this b loop was worse, probably going up other side of
                # function
                # print(w, wMag, thisBest)
                incCount+=1
                
                if debug: 
                    print('Passed optimum point', incCount)
            
            # Only use inc count if slack is 0
            # (mag is curretely ||w|| + cost)
            if (incCount>self.params['convVerSteps']) & (slack==0) & \
            anyCandidateFound:
                stepsSaved = len(wRange)-np.argmax(wr==wRange)
                print('Saved:', stepsSaved, 'w steps, (',
                      stepsSaved*len(bRange), 'its )')
                # Stop looping over w, go to return
                break
        
        if not anyCandidateFound:
            # Failed to converge entirely
            print('Failed...')
            return self            
        else:
            # Increase w range resoltuion, run again
           print('Mag best: ', self.results['mag'])
           return self.fit(x, y, 
                           wStart=self.results['w'][0]*1.025, wStep=wStep/10,
                           bStart=bStart, bStep=bStep,
                           rSteps=rSteps, rStep=rStep+1, 
                           debug=False)
           
    def plotContour(self, x, y, d1=0, d2=1):
        
        X1, X2 = np.meshgrid(np.linspace(x.ravel().min(),x.ravel().max(),50), 
                             np.linspace(x.ravel().min(),x.ravel().max(),50))
        X = np.array([[x1, x2] for x1, x2 in zip(np.ravel(X1), np.ravel(X2))])
        Z = mod.predictProba(X).reshape(X1.shape)
        plt.contour(X1, X2, Z-1, 0, colours='b')
        plt.contour(X1, X2, Z, 0, colours='k')
        plt.contour(X1, X2, Z+1, 0, colours='b')
        plt.scatter(x[y==1,d1], x[y==1,d2], c='r')
        plt.scatter(x[y==-1,d1], x[y==-1,d2], c='b')
        plt.show()
           
        
#%% Basic 2D test
if __name__ == "__main__":
    
    x = np.array([[1, 7],
              [2, 8],
              [3, 8],
              [5, 1],
              [6, -1],
              [7, 3]])

    y = np.array([-1, -1, -1, 1, 1, 1])
    
    mod = SVM(kernel=Ks('linear'), slack=0, C=0)
    mod = mod.fit(x, y, debug=True, wStart=2, wStep=0.1, bStep=1)
    mod.plotDecision(x, y)
    mod.plotDecision(x)
    
    yPred = mod.predict(x)
    accuracy(y, yPred)
    
    
#%% Basic 2D test
if __name__ == "__main__":
    
    x = np.array([[1, 7],
              [2, 8],
              [3, 8],
              [5, 1],
              [6, -1],
              [7, 3]])

    y = np.array([-1, -1, -1, 1, 1, 1])
    
    mod = SVM(kernel=Ks('polynomial', p=3), costThresh=0)
    mod = mod.fit(x, y, debug=True, bStart=50, bStep=0.1, wStart=1, 
                  wStep=0.1)
    mod.plotDecision(x, y)
    mod.plotDecision(x)
    
    yPred = mod.predict(x)
    accuracy(y, yPred)
    
    mod.plotContour(x,y)
    
    
#%% Basic 3D test  
if __name__ == "__main__":
    
    x = np.array([[1, 7, 1],
              [2, 8, 2],
              [3, 8, 1],
              [5, 1, 6],
              [6, -1, 7],
              [7, 3, 6]])

    y = np.array([-1, -1, -1, 1, 1, 1])
    
    mod = SVM(kernel=Ks('polynomial', p=8), costThresh=0)
    mod = mod.fit(x, y, wStart=10, wStep=0.1, bStart=50)
    mod.plotDecision(x, y, d1=0, d2=1)
    mod.plotDecision(x, y, d1=1, d2=2)
    mod.plotDecision(x, y, d1=0, d2=2)
    mod.plotDecision(x)
    
    yPred = mod.predict(x)
    accuracy(y, yPred)
        
    mod.plotContour(x,y)
    
    
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
    
    mod = SVM(kernel=Ks('linear'), slack=1, C=1)
    mod = mod.fit(x, y, wStart=1, wStep=0.01, debug=False)
    mod.plotDecision(x, y)
    mod.plotDecision(x)
    
    yPred = mod.predict(x)
    mod.accuracy(y, yPred)
    
    mod.plotContour(x,y)
    
    
#%% Generated

if __name__ == '__main__':
    
    nF = 2
    X,Y = data = mk(n_samples=100, 
              n_features=nF, 
              n_informative=2, 
              n_redundant=0,
              n_repeated=0, 
              n_classes=2,
              n_clusters_per_class=1,
              scale=1,
              shift=10)
    X = pd.DataFrame(X, columns=['x'+str(x) for x in range(nF)])
    Y = pd.DataFrame(Y)
    
    XTrain, XValid, YTrain, YValid = tts(
            X, Y, test_size=0.2, random_state=48)
        
    mod = SVM(kernel=Ks('linear'), costThresh=0.3)
    mod = mod.fit(XTrain, YTrain, wStart=40, wStep=0.01, bStart=50)
    mod.plotDecision(X, Y, d1=0, d2=1)
    mod.plotDecision(X, Y, d1=1, d2=2)
    mod.plotDecision(X, Y, d1=0, d2=2)
    mod.plotDecision(X, d1=0, d2=1)
    mod.plotDecision(X, d1=1, d2=2)
    mod.plotDecision(X, d1=0, d2=2)
    
    yPredTrain = mod.predict(XTrain, m1=False)
    yPredValid = mod.predict(XValid, m1=False)
    
    print('Train acc:', accuracy(YTrain, yPredTrain))
    print('Valid acc:', accuracy(YValid, yPredValid))
