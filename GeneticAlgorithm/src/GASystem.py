import numpy as np
import random

class GASystem():
    def __init__(self, **kw):
        self.geneNumber = kw["geneNumber"]
        self.crossoverProbability = kw["crossoverProbability"]
        self.mutationProbability = kw["mutationProbability"]
        self.inputDim = kw["inputDim"]
        self.hiddenLayerNeuronsNumber = kw["hiddenLayerNeuronsNumber"]
        # copy
        self.competeGroupSize = 2
        # crossover
        self.sigma = 0.1
        # mutation
        self.s = 0.1
    
    def getInitializeGeneticVectors(self):
        self.geneticVectors = []
        for index in range(self.geneNumber):
            self.geneticVectors.append(np.hstack((np.random.uniform(-1, 1, 1+self.hiddenLayerNeuronsNumber+self.inputDim*self.hiddenLayerNeuronsNumber),np.random.uniform(0,1,self.hiddenLayerNeuronsNumber))))
        return self.geneticVectors
        
    def getNewGeneticVectors(self):
        return self.geneticVectors
    
    def copy(self, Efunctions):
        accurracySet = []
        for geneticVectorIndex in range(len(self.geneticVectors)):
            accurracySet.append(1-Efunctions[geneticVectorIndex]/sum(Efunctions))
        total = sum(accurracySet)
        for geneticVectorIndex in range(len(self.geneticVectors)):
            accurracySet[geneticVectorIndex] = accurracySet[geneticVectorIndex]/total
        
        """
        #(a) 輪盤式
        copyGeneticVectors = []
        for time in range(len(self.geneticVectors)):
            pointer = random.random()
            accurracySum = 0
            for geneticIndex in range(len(accurracySet)):
                accurracySum += accurracySet[geneticIndex]
                if pointer <= accurracySum:
                    #print(geneticIndex)
                    copyGeneticVectors.append(self.geneticVectors[geneticIndex])
                    break
        """
        
        # (b) 競爭式
        index = 0
        copyGeneticVectors = []
        copyIndexes = random.sample(list(range(len(self.geneticVectors))), len(self.geneticVectors))
        for time in range(int(len(copyIndexes)/self.competeGroupSize)):
            competeGroupAccurracy = []
            for competeGroupIndex in range(self.competeGroupSize):
                competeGroupAccurracy.append(accurracySet[copyIndexes[index+competeGroupIndex]])
            for competeGroupIndex in range(self.competeGroupSize):
                copyGeneticVectors.append(self.geneticVectors[copyIndexes[index + competeGroupAccurracy.index(max(competeGroupAccurracy))]])
            index += self.competeGroupSize

        self.geneticVectors = copyGeneticVectors
        
    def crossover(self):
        crossoverIndexes = random.sample(list(range(len(self.geneticVectors))), round(len(self.geneticVectors)*self.crossoverProbability))
        index = 0
        for time in range(int(len(crossoverIndexes)/2)):
            tempVector1 = self.geneticVectors[crossoverIndexes[index]]
            tempVector2 = self.geneticVectors[crossoverIndexes[index+1]]
            self.geneticVectors[crossoverIndexes[index]] = tempVector1 + self.sigma *(tempVector1-tempVector2)
            self.geneticVectors[crossoverIndexes[index+1]] = tempVector2 - self.sigma *(tempVector1-tempVector2)
            index += 2

    def mutation(self):
        mutationIndexes = random.sample(list(range(len(self.geneticVectors))), round(len(self.geneticVectors)*self.mutationProbability))
        for mutationIndex in mutationIndexes:
            self.geneticVectors[mutationIndex] += self.s * np.hstack((np.random.uniform(-1, 1, 1+self.hiddenLayerNeuronsNumber+self.inputDim*self.hiddenLayerNeuronsNumber),np.random.uniform(0,1,self.hiddenLayerNeuronsNumber)))        