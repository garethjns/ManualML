# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 16:18:55 2017

@author: Gareth
"""

#%%

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split as tts
from sklearn.datasets import make_classification as mk

import importlib as il
import MLCommon
il.reload(MLCommon)

from MLCommon import MLHelpers


#%%

class tree(MLHelpers, Losses):
    """
    Binary classification tree
    """
    def __init__(self, minData=2, maxDepth=3,
               dynamicBias=True, bias=0.5):
        self.params = {'maxDepth': maxDepth,
                       'minData': minData,
                       'dynamicBias': dynamicBias,
                       'bias': bias}
        self.featureNames = []
    
        
    def print(self, nodes=[], root=True):
        
        if root==True:
            nodes = self.tree
            
        tree.printNode(nodes)    
        
        if nodes['Class']==-1:
            self.print(nodes=nodes['lNode'], root=False)
            self.print(nodes=nodes['rNode'], root=False)
            
            
    @staticmethod
    def printNode(node):
        bS =' '*node['depth']*2
        bS2 = '-'*2
        print(bS+'|'+bS2+'*'*50)
        print(bS+'|'+bS+node['nodeStr'])
        print(bS+'|'+bS+'Depth:', node['depth'])
        print(bS+'|'+bS+'n samples:', node['n'])
        if node['terminal']==False:
            print(bS+'|'+bS+'Name: '+node['name'])
            print(bS+'|'+bS+'Split Value:', node['splitVal'])
            print(bS+'|'+bS+'Gini coeff:', node['gini'])
            print(bS+'|'+bS+'Class prop requirement:', node['bias'], 
                  '('+node['biasMode']+')')
            print(bS+'|'+bS+'Prop L:', node['propL'])
            print(bS+'|'+bS+'Prop R:', node['propR'])
            
        else:
            print(bS+'|'+bS+'Leaf')
            print(bS+'|'+bS+node['note'])

    def fit(self, X, Y, debug=False):
        """
        Convert data to matrix if dataframe
        Recursively create nodes using tree.buildNodes()
        """
        
        # Set feature names
        self = self.setNames(X)
        
        # Convert to mats if not
        X = self.stripDF(X)
        Y = self.stripDF(Y)
            
        self.tree = tree.buildNodes(X, Y, 
                                    maxDepth=self.params['maxDepth'],
                                    minData=self.params['minData'],
                                    dynamicBias=self.params['dynamicBias'],
                                    debug=debug, names=self.featureNames)
        
        return self
        
    @staticmethod
    def buildNodes(X, Y, names, maxDepth=2, minData=2, dynamicBias=False, 
                   depth=0, nomClass=-777, bias=0.5, dStr='Root', 
                   debug=False):

        if dynamicBias==True:
            bias = tree.prop(Y)
            dS = 'dynamic'
        else:
            if bias == '':
                dS = 'highest'
            else:
                dS = 'static'

        # Add terminal checks here
        # If a terminal node, return a node (dict) containing just the class label
        # This label is set by highest represented label in subset
        if depth > maxDepth:
            # Too deep
            cla =  tree.highClass(Y, bias)
            node = {'Class': cla,
                    'depth': depth,
                    'note': 'Max depth reached, class is: '+ str(cla),
                    'terminal': True,
                    'n': len(X),
                    'nodeStr' : dStr}
        elif X.shape[0]<minData:
            if X.shape[0]==0:
                # Too few data points
                cla = nomClass
            else:
                cla = tree.highClass(Y, bias)
                
            node = {'Class': cla,
                    'depth': depth,
                    'note': 'Too few data points, class is: '+ str(cla),
                    'terminal': True,
                    'n': len(X),
                    'nodeStr' : dStr}
            # In this case, Y will be empty
            # So use nominal class that will be the opposite of the other side node
        elif X.shape[1]<1:
            # Too few features
            cla = tree.highClass(Y, bias)
            node = {'Class' : cla,
                    'depth' : depth,
                    'note' : 'No features remaining, class is: '+ str(cla),
                    'terminal': True,
                    'n': len(X),
                    'nodeStr' : dStr}
        elif len(np.unique(Y)) == 1:
            # Only one class
            cla = tree.highClass(Y, bias)
            node = {'Class' : cla,
                    'depth' : depth,
                    'note': 'One class at depth, class is: '+ str(cla),
                    'terminal': True,
                    'n': len(X),
                    'nodeStr' : dStr}
        else:
            # First find best split to run
            colIdx, bestX, gini = tree.getBestSplitAll(X, Y)
            # Convert integer index to logical
            # logIdx = np.array(range(X.shape[1]))==2
            
            # Split in to left and right subsets
            lIdx = (X[:,colIdx]<bestX).squeeze()
            rIdx = (X[:,colIdx]>=bestX).squeeze()
            
            nomClassL = -999
            nomClassR = -999
            if np.sum(lIdx)==0:
                nomClassL = np.int8(not tree.highClass(Y[rIdx], bias))
            if np.sum(rIdx)==0:
                nomClassR = np.int8(not tree.highClass(Y[lIdx], bias))
            
            # Build next node, leaving out used feaure and data not in this split
            lNode = tree.buildNodes(X[lIdx][:,~colIdx], Y[lIdx], 
                               maxDepth=maxDepth, 
                               minData=minData, 
                               depth=depth+1,
                               nomClass=nomClassL, 
                               dynamicBias=dynamicBias,
                               bias=bias, 
                               dStr=dStr+'->L',
                               names = [n for ni,n in enumerate(names) if ni != np.argmax(colIdx)])
            rNode = tree.buildNodes(X[rIdx][:,~colIdx], Y[rIdx], 
                               maxDepth=maxDepth, 
                               minData=minData, 
                               depth=depth+1,
                               nomClass=nomClassR, 
                               dynamicBias=dynamicBias,
                               bias=bias, 
                               dStr=dStr+'->R',
                               names = [n for ni,n in enumerate(names) if ni != np.argmax(colIdx)])
            
            # Return a full node containing meta/debug data
            # As this isn't a leaf/terminal node, set class to -1
            node = {'name' : names[np.argmax(colIdx)],
                    'n': len(X),
                    'nL': np.sum(lIdx),
                    'nR': np.sum(rIdx),
                    'lIdx': lIdx,
                    'rIdx': rIdx,
                    'splitVal': bestX.squeeze(),
                    'gini': gini.squeeze(),
                    'depth': depth,
                    'lNode': lNode,
                    'rNode': rNode,
                    'Class': -1,
                    'propL': tree.prop(Y[lIdx]),
                    'propR': tree.prop(Y[rIdx]),
                    'biasMode': dS,
                    'bias': bias,
                    'nodeStr' : dStr,
                    'terminal': False}
    
        if debug:
            tree.printNode(node)
            
        return node
    
    
    def predict(self, X):
        
        yPred = X.apply(tree.predictStatic, args=(self.tree,), axis=1)
        
        return yPred
    
    @staticmethod
    def predictStatic(x, mod):

        # If this is a leaf node, return class
        if mod['Class']>-1:
            return mod['Class']
        
        # If this isn't a leaf node, check X against split value
        # and follow tree
        if x.loc[mod['name']] < mod['splitVal']:
            # If less than split val, go left
            yPred = tree.predictStatic(x, mod['lNode'])
        else:
            # If greater than split val, go right
            yPred = tree.predictStatic(x, mod['rNode'])
            
        return yPred
    
    @staticmethod
    def gi(groups, classes):
        groups = np.array(groups)
        classes = np.array(classes)
        
        # For each group
        sumP = 0.0
        for g in np.unique(groups):
            # print('G:',g)
            gIdx = groups == g
            # Calculate and sum class proportions
            P = 0.0
            # For each class
            for c in np.unique(classes):
                # print('C:',c)
                cIdx = classes[gIdx] == c
                # Get proportions and square
                # And sum across classes
                P += (np.sum(cIdx)/np.sum(gIdx))**2
                # print('P:',P)
            
            # Weight by size of group
            # And sum across groups
            sumP += (1-P) * sum(gIdx)/len(gIdx)
    
        return sumP

    @staticmethod
    def split(X, Y, splitVal):
        groups = np.int8(X<splitVal)
        
        return tree.gi(groups, Y)
    
    @staticmethod
    def getBestSplitAll(X, Y):
        """
        This function calculates all splits on all columns
        Returns the column index with best split and the values to use
        """
        m = X.shape[1]
        colBestGin = np.ones(shape=(m))
        colBestVal = np.ones(shape=(m))
        for c in range(m):
            best = 1
            bestX = 0
            for x in np.unique(X[:,c]):
                gini = tree.split(X[:,c], Y, x)
                if gini < best:
                    best = gini
                    bestX = x
                colBestGin[c] = best
                colBestVal[c] = bestX
        
        # Select best feature to split on
        colIdx = np.argmin(colBestGin)
        # Convert to bool index
        colIdx = np.array(range(X.shape[1]))==colIdx
        
        return colIdx, colBestVal[colIdx], colBestGin[colIdx]

    @staticmethod
    def prop(Y):
        if np.sum(Y)>0:
            return Y.sum()/Y.shape[0]
        else:
            return 0
        
    @staticmethod
    def highClass(Y, bias=''):
    
        if bias == '':
            # Just return highest class
            return np.argmax(Y.value_counts())
        else:
            # Return logical of class prop>bias
            if len(Y)>0:
                return np.int8(tree.prop(Y)>bias)
            else: 
                return 0
            

        
            
#%% Small
if __name__ == '__main__':
    print('Test')
    
    data = pd.DataFrame({'X1': [2.77, 1.73, 3.68, 3.96, 2.99, 7.50, 9.00, 7.44, 
                            10.12, 6.64],
                    'X2': [1.78, 1.17, 2.81, 2.62, 2.21, 3.16, 3.34, 0.48, 
                           3.23, 3.32],
                    'Y': [1, 0, 0, 0, 0, 1, 1, 1, 0, 0]})
    Y = data.Y
    X = data.loc[:,['X1', 'X2']]
    
    mod = tree(maxDepth=2, minData=2, dynamicBias=False)        
    mod = mod.fit(X, Y)
    
    mod.print()
    
    yPred = mod.predict(X)
    mod.accuracy(Y, yPred)
    
    plt.scatter(data.X1[data.Y==0], data.X2[data.Y==0])
    plt.scatter(data.X1[data.Y==1], data.X2[data.Y==1])
    plt.show()
    plt.scatter(data.X1[yPred==0], data.X2[yPred==0])
    plt.scatter(data.X1[yPred==1], data.X2[yPred==1])
    plt.show()
   
    
#%% Banknote
if __name__ == '__main__':
    data = pd.read_csv('data_banknote_authentication.txt', header=None)
    data.columns = ['x'+str(x) for x in range(4)]+list('Y')
    data.head()
    
    X = data.loc[:,['x'+str(x) for x in range(4)]]
    Y = data.loc[:,'Y']
    XTrain, XValid, YTrain, YValid = tts(
        X, Y, test_size=0.25, random_state=512)
    
    mod = tree(minData=10, maxDepth=6, 
               dynamicBias=False, bias=0.5)
    mod = mod.fit(XTrain, YTrain)
    mod.print()
    yPredTrain = mod.predict(XTrain)
    yPredValid = mod.predict(XValid)
    
    mod.accuracy(yPredTrain, YTrain)
    mod.accuracy(yPredValid, YValid)


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
        
    mod = tree(minData=10, maxDepth=10, 
               dynamicBias=False, bias=0.5)
    mod = mod.fit(XTrain, YTrain)
    mod.print()
    yPredTrain = mod.predict(XTrain)
    yPredValid = mod.predict(XValid)
    
    mod.accuracy(YTrain, yPredTrain)
    mod.accuracy(YValid, yPredValid)
