import numpy as np
import sys
from GASystem import GASystem
import random
import math
import time
import matplotlib.pyplot as plt
import time

class RBFN():
    def __init__(self, **kw):
        self.loopTimes = kw["loopTimes"]
        self.inputDim, self.dataNumber, self.trainingData = self.getTrainingData(kw["trainingDataPath"])
        self.hiddenLayerNeuronsNumber = kw["hiddenLayerNeuronsNumber"]
        self.GASystem = GASystem(geneNumber=kw["geneNumber"], mutationProbability=kw["mutationProbability"], crossoverProbability=kw["crossoverProbability"], inputDim=self.inputDim, hiddenLayerNeuronsNumber=self.hiddenLayerNeuronsNumber)
        self.batchSize = kw["batchSize"]

    def getTrainingData(self, trainingDataPath):
        with open(trainingDataPath, "r") as f:
            lines = f.readlines()
            dim = 0
            dataNumber = 0
            for line in lines:
                dim = len(line.split())
                dataNumber += 1 
            trainingData = np.zeros((dataNumber, dim))
            lineIndex = 0
            for line in lines:
                for dimIndex in range(dim):
                    if dimIndex == dim-1:
                        trainingData[lineIndex][dimIndex] = float(line.split()[dimIndex]) / 40
                    else:
                        trainingData[lineIndex][dimIndex] = (float(line.split()[dimIndex])-40)/40
                lineIndex += 1
            return dim-1, dataNumber, trainingData
    
    def train(self):
        minList, avgList, loopTimeList = [], [], []
        startTime = time.time()
        for loopTime in range(self.loopTimes):
            # get genetic vectors
            if loopTime == 0:
                geneticVectors = self.GASystem.getInitializeGeneticVectors()
            else:
                geneticVectors = self.GASystem.getNewGeneticVectors()
            # forwarding    
            Efunctions = []
            batchTrainingData = self.trainingData[random.sample(list(range(self.dataNumber)), self.batchSize)]
            for geneticVector in geneticVectors:
                # updata RBFN weight
                self.updataHiddenLayer(geneticVector)
                Efunctions.append(self.getEFunction(self.feedforwarding(batchTrainingData[:, :-1], geneticVector[0]), batchTrainingData[:, -1]))
            # copy
            self.GASystem.copy(Efunctions)
            # crossover
            self.GASystem.crossover()
            # mutation
            self.GASystem.mutation()
            # show result
            min, avg = self.getMinimumErrorRates()

            loopTimeList.append(loopTime)
            minList.append(min)
            avgList.append(avg)
            print("#",loopTime)
            print("min:",min)
            print("avg:",avg)
        endTime = time.time()
        print("duration:", endTime-startTime)
        plt.plot(loopTimeList, minList, label='minimum')
        plt.plot(loopTimeList, avgList, label='averageg')
        plt.title('Relationship between Epoch and Average Error Rate')
        plt.xlabel('Epoch')
        plt.ylabel('Average Error Rate')
        plt.legend()
        plt.show()
    
    def getMVector(self, geneticVector):
        for J in range(self.hiddenLayerNeuronsNumber):
            if J == 0:
                temVector = geneticVector[1+self.hiddenLayerNeuronsNumber: 1+self.hiddenLayerNeuronsNumber+self.inputDim]
            else:
                temVector = np.vstack((temVector,geneticVector[1+self.hiddenLayerNeuronsNumber + self.inputDim*J: 1+self.hiddenLayerNeuronsNumber + self.inputDim*(J+1)]))
        return temVector

    def updataHiddenLayer(self, geneticVector):
        self.mVector = self.getMVector(geneticVector)
        self.sigmaVector = geneticVector[-self.hiddenLayerNeuronsNumber::]
        self.wVector = np.reshape(geneticVector[1:1+self.hiddenLayerNeuronsNumber], (self.hiddenLayerNeuronsNumber,1))

    def feedforwarding(self, xVector, theta):
        self.thetaVector = np.full((np.size(xVector,0), 1), theta)
        hiddenLayerOutput = np.zeros((np.size(xVector,0), self.hiddenLayerNeuronsNumber))
        for batchIndex in range(np.size(xVector,0)):
            for jIndex in range(self.hiddenLayerNeuronsNumber):
                hiddenLayerOutput[batchIndex][jIndex] = math.exp((-1 * pow(np.linalg.norm(xVector[batchIndex,:]-self.mVector[jIndex,:]),2))/(2 * pow(self.sigmaVector[jIndex],2)))
        return (np.dot(hiddenLayerOutput, self.wVector) + self.thetaVector).flatten()
    
    def getEFunction(self, F, yVector):
        return 0.5 * np.sum(np.power(F-yVector, 2))

    def getError(self, F, yVector):
        return np.sum(abs(F-yVector))/self.dataNumber
    
    def getMinimumErrorRates(self):
        geneticVectors = self.GASystem.getNewGeneticVectors()
        errorRates = []
        for geneticVector in geneticVectors:
            self.updataHiddenLayer(geneticVector)
            errorRates.append(self.getError(self.feedforwarding(self.trainingData[:, :-1], geneticVector[0]), self.trainingData[:, -1]))
        self.minErrorRatesIndex = errorRates.index(min(errorRates))
        return min(errorRates), sum(errorRates) / len(errorRates)
    
    def writeData(self, path):
        geneticVectors = self.GASystem.getNewGeneticVectors()
        minErrorGeneticVectors = geneticVectors[self.minErrorRatesIndex]
        with open(path.name, "w") as f:
            f.writelines(str(minErrorGeneticVectors[0])+"\n")
            for jIndex in range(self.hiddenLayerNeuronsNumber):
                f.writelines(str(minErrorGeneticVectors[1+jIndex])+" ")
                for m in minErrorGeneticVectors[1+self.hiddenLayerNeuronsNumber+jIndex*self.hiddenLayerNeuronsNumber:1+self.hiddenLayerNeuronsNumber+(jIndex+1)*self.hiddenLayerNeuronsNumber]:
                    f.writelines(str(m)+" ")
                f.writelines(str(minErrorGeneticVectors[-jIndex-2])+"\n")
    
    def calculate(self, carPosition, distance):
        if self.inputDim == 3:
            xVector = (np.array([[distance[1], distance[2], distance[0]]])-40)/40
        else:
            xVector = (np.array([[carPosition[0], carPosition[1], distance[1], distance[2], distance[0]]])-40)/40
        geneticVectors = self.GASystem.getNewGeneticVectors()
        minErrorGeneticVectors = geneticVectors[self.minErrorRatesIndex]
        self.updataHiddenLayer(minErrorGeneticVectors)
        return (self.feedforwarding(xVector, minErrorGeneticVectors[0])[0])*40