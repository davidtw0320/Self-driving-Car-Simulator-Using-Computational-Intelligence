import numpy as np
import sys
import random
import math
import matplotlib.pyplot as plt
import time

from PSO import PSO

class RBFN():
    def __init__(self, **kw):
        self.loopTimes = kw["loopTimes"]
        self.inputDim, self.dataNumber, self.trainingData = self.getTrainingData(kw["trainingDataPath"])
        self.hiddenLayerNeuronsNumber = kw["hiddenLayerNeuronsNumber"]
        self.PSO = PSO(particleNumber=kw["particleNumber"], learningRate1=kw["learningRate1"], learningRate2=kw["learningRate2"], inputDim=self.inputDim, hiddenLayerNeuronsNumber=self.hiddenLayerNeuronsNumber)
        self.batchSize = kw["batchSize"]
        self.minErrorRate = 1.0

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
        self.PSO.getInitialParticles()
        for loopTime in range(self.loopTimes):
            # get genetic vectors
            if loopTime == 0:
                particles = self.PSO.getInitialParticles()
            else:
                particles = self.PSO.getNewParticles()
            # forwarding 
            batchTrainingData = self.trainingData[random.sample(list(range(self.dataNumber)), self.batchSize)]
            for particle in particles:
                # updata RBFN weight
                self.updataHiddenLayer(particle.xVector)
                particle.xFitness = self.getEFunction(self.feedforwarding(batchTrainingData[:, :-1], particle.xVector[0]), batchTrainingData[:, -1])
                if loopTime == 0:
                    particle.pFitness = particle.xFitness
                    particle.pVector = particle.pVector
            # learn
            self.PSO.particlesMove()
            # show result
            min, avg,  allMin= self.getErrorRateData()
            # data log
            loopTimeList.append(loopTime)
            minList.append(min)
            avgList.append(avg)
            print("#",loopTime)
            print("min (a loop):", min)
            print("avg (a loop):", avg)
            print("min (all loops):", allMin)
        # plot history data
        endTime = time.time()
        print("duration:", endTime-startTime)
        plt.plot(loopTimeList, minList, label='minimum')
        plt.plot(loopTimeList, avgList, label='averageg')
        plt.title('Relationship between Epoch and Average Error Rate')
        plt.xlabel('Epoch')
        plt.ylabel('Average Error Rate')
        plt.legend()
        plt.show()
    
    def getMVector(self, xVector):
        for J in range(self.hiddenLayerNeuronsNumber):
            if J == 0:
                temVector = xVector[1+self.hiddenLayerNeuronsNumber: 1+self.hiddenLayerNeuronsNumber+self.inputDim]
            else:
                temVector = np.vstack((temVector,xVector[1+self.hiddenLayerNeuronsNumber + self.inputDim*J: 1+self.hiddenLayerNeuronsNumber + self.inputDim*(J+1)]))
        return temVector

    def updataHiddenLayer(self, xVector):
        self.mVector = self.getMVector(xVector)
        self.sigmaVector = xVector[-self.hiddenLayerNeuronsNumber::]
        self.wVector = np.reshape(xVector[1:1+self.hiddenLayerNeuronsNumber], (self.hiddenLayerNeuronsNumber,1))

    def feedforwarding(self, xVector, theta):
        self.thetaVector = np.full((np.size(xVector,0), 1), theta)
        hiddenLayerOutput = np.zeros((np.size(xVector,0), self.hiddenLayerNeuronsNumber))
        for batchIndex in range(np.size(xVector,0)):
            for jIndex in range(self.hiddenLayerNeuronsNumber):
                hiddenLayerOutput[batchIndex][jIndex] = math.exp((-1 * pow(np.linalg.norm(xVector[batchIndex,:]-self.mVector[jIndex,:]),2))/(2 * pow(self.sigmaVector[jIndex],2)))
        return (np.dot(hiddenLayerOutput, self.wVector) + self.thetaVector).flatten()
    
    def getEFunction(self, F, yVector):
        return -0.5 * np.sum(np.power(yVector-F, 2))

    def getErrorRate(self, F, yVector):
        return np.sum(abs(yVector-F))/self.dataNumber
    
    def getErrorRateData(self):
        particles = self.PSO.getNewParticles()
        errorRates = []
        for particle in particles:
            self.updataHiddenLayer(particle.xVector)
            errorRates.append(self.getErrorRate(self.feedforwarding(self.trainingData[:, :-1], particle.xVector[0]), self.trainingData[:, -1]))
        if  min(errorRates) < self.minErrorRate:
            self.minErrorRate = min(errorRates)
            minErrorRatesIndex = errorRates.index(min(errorRates))
            self.minErrorRatesXVector = particles[minErrorRatesIndex].xVector
        return min(errorRates), sum(errorRates) / len(errorRates), self.minErrorRate
    
    def writeData(self, path):
        with open(path.name, "w") as f:
            f.writelines(str(self.minErrorRatesXVector[0])+"\n")
            for jIndex in range(self.hiddenLayerNeuronsNumber):
                f.writelines(str(self.minErrorRatesXVector[1+jIndex])+" ")
                for m in self.minErrorRatesXVector[1+self.hiddenLayerNeuronsNumber+jIndex*self.inputDim:1+self.hiddenLayerNeuronsNumber+(jIndex+1)*self.inputDim]:
                    f.writelines(str(m)+" ")
                f.writelines(str(self.minErrorRatesXVector[-1 * (self.hiddenLayerNeuronsNumber -jIndex)])+"\n")
    
    def calculate(self, carPosition, distance):
        if self.inputDim == 3:
            xVector = (np.array([[distance[1], distance[2], distance[0]]])-40)/40
        else:
            xVector = (np.array([[carPosition[0], carPosition[1], distance[1], distance[2], distance[0]]])-40)/40
        self.updataHiddenLayer(self.minErrorRatesXVector)
        return (self.feedforwarding(xVector,  self.minErrorRatesXVector[0])[0])*40