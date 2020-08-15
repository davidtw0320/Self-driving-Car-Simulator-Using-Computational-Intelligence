import numpy as np

class Particle():
     def __init__(self, **kw):
        self.xVector = np.hstack((np.random.uniform(-1, 1, 1+kw['hiddenLayerNeuronsNumber']+kw['inputDim']*kw['hiddenLayerNeuronsNumber']),np.random.uniform(0,1,kw['hiddenLayerNeuronsNumber'])))
        self.pVector = np.hstack((np.random.uniform(-1, 1, 1+kw['hiddenLayerNeuronsNumber']+kw['inputDim']*kw['hiddenLayerNeuronsNumber']),np.random.uniform(0,1,kw['hiddenLayerNeuronsNumber'])))
        self.vVector = np.hstack((np.random.uniform(-1, 1, 1+kw['hiddenLayerNeuronsNumber']+kw['inputDim']*kw['hiddenLayerNeuronsNumber']),np.random.uniform(0,1,kw['hiddenLayerNeuronsNumber'])))
        self.xFitness = 0.0
        self.pFitness = 0.0
    
    