import numpy as np
import random

from Particle import Particle

class PSO():
    def __init__(self, **kw):
        self.particleNumber = kw["particleNumber"]
        self.learningRate1 = kw["learningRate1"]
        self.learningRate2 = kw["learningRate2"]
        self.inputDim = kw["inputDim"]
        self.hiddenLayerNeuronsNumber = kw["hiddenLayerNeuronsNumber"]

    def getInitialParticles(self):
        self.particles = []
        for index in range(self.particleNumber):
            self.particles.append(Particle(hiddenLayerNeuronsNumber=self.hiddenLayerNeuronsNumber, inputDim=self.inputDim))
        return self.particles
        
    def getNewParticles(self):
        return self.particles
    
    def particlesMove(self):
        for particle in self.particles:
            if particle.xFitness > particle.pFitness:
                particle.pVector = particle.xVector
                particle.pFitness = particle.xFitness
            NeighborPerfectParticle = particle
            for NeighborParticle in self.particles:
                if NeighborParticle.pFitness > NeighborPerfectParticle.pFitness:
                    NeighborPerfectParticle = NeighborParticle
            particle.vVector += self.learningRate1*(particle.pVector - particle.xVector) + self.learningRate2*(NeighborPerfectParticle.pVector - particle.xVector)
            self.vVectorWithinRestriction(particle.vVector)
            particle.xVector += particle.vVector
            self.xVectorWithinRestriction(particle.xVector)

    def vVectorWithinRestriction(self, vVector):
        restriction = 0.01
        vMax = restriction * np.ones(1+self.hiddenLayerNeuronsNumber+self.inputDim*self.hiddenLayerNeuronsNumber+self.hiddenLayerNeuronsNumber)
        vMin = -restriction * np.ones(1+self.hiddenLayerNeuronsNumber+self.inputDim*self.hiddenLayerNeuronsNumber+self.hiddenLayerNeuronsNumber)
        for d in range(len(vVector)):
            if vVector[d] > vMax[d]:
                vVector[d] = vMax[d]
            elif vVector[d] < vMin[d]:
                vVector[d] = vMin[d]

    def xVectorWithinRestriction(self, xVector):
        xMax = np.hstack((np.ones(1+self.hiddenLayerNeuronsNumber+self.inputDim*self.hiddenLayerNeuronsNumber),np.ones(self.hiddenLayerNeuronsNumber)))
        xMin = np.hstack((-1 * np.ones(1+self.hiddenLayerNeuronsNumber+self.inputDim*self.hiddenLayerNeuronsNumber),np.zeros(self.hiddenLayerNeuronsNumber)))
        for d in range(len(xVector)):
            if xVector[d] > xMax[d]:
                xVector[d] = xMax[d]
            elif xVector[d] < xMin[d]:
                xVector[d] = xMin[d]
