from MathLibrary import MathLibrary
import numpy as np

class FuzzySystem():
    def __init__(self, inferenceVarName):
        self.MathLibrary = MathLibrary()
        self.outputBound = [-40,40]
        self.membershipFunc = self.initialMembershipFunc(inferenceVarName)
        self.fuzzyRuleBase = self.initialFuzzyRuleBase()

    # membership Function
    def initialMembershipFunc(self, inferenceVarName):
        membershipFunc = {}
        for key in inferenceVarName:
            temp = {key: None}
            membershipFunc.update(temp)
        for key in membershipFunc:
            if key == "d0" or key =="d2":
                temp = {key: self.membershipFunc0()} 
            elif key == "d1":
                temp = {key: self.membershipFunc1()}
            membershipFunc.update(temp)
        return membershipFunc

    def membershipFunc0(self):
        small = np.array([ [0,1], [5,1], [10,0] ], dtype='f')
        medium = np.array([ [5,0], [10,1], [16,0] ], dtype='f')
        large = np.array([ [12,0],[20,1],[50,1] ], dtype='f')
        superLarge = np.array([ [20,0],[60,1],[100,1] ], dtype='f')
        return self.creatTermSet(small = small, medium = medium, large = large, superLarge=superLarge)

    def membershipFunc1(self):
        small = np.array([ [0,1], [5,1], [10,0] ], dtype='f')
        medium = np.array([ [5,0], [20,1], [16,0] ], dtype='f')
        large = np.array([ [12,0], [16,1], [50,1] ], dtype='f')
        return self.creatTermSet(small = small, medium= medium, large = large)
    
    def creatTermSet(self, **TermPointSet):
        TermSet = {}
        for key in TermPointSet:
            temp = {key:[]}
            TermSet.update(temp)
        for key in TermPointSet:
            for index in range(len(TermPointSet[key])-1):
                TermSet[key].append(self.MathLibrary.getSegment(TermPointSet[key][index], TermPointSet[key][index+1]))
        return TermSet

    # fuzzyRuleBase
    def initialFuzzyRuleBase(self):
        fuzzyRuleBase = [
        [{'d0':'superLarge'}, np.array([ [-40,1], [-10, 1], [0,0.8] ], dtype='f')],
        [{'d2':'superLarge'}, np.array([ [0,0.8], [10, 1], [40,1] ], dtype='f')],
        [{'d0':'large'}, np.array([ [-40,1], [-30, 1], [0,0] ], dtype='f')],
        [{'d2':'large'}, np.array([ [0,0], [30, 1], [40,1] ], dtype='f')],
        [{'d0':'medium'}, np.array([ [-40,1], [-35,0] ], dtype='f')],
        [{'d2':'medium'}, np.array([ [35,0], [40,1] ], dtype='f')],
        [{'d1':'large', 'd0':'small'}, np.array([ [0,1], [20,1], [40,0] ], dtype='f')],
        [{'d1':'large', 'd2':'small'}, np.array([ [-40,0], [-20,1], [0,1] ], dtype='f')],
        [{'d1':'medium', 'd0':'small'}, np.array([ [0,0], [5,1], [40,1] ], dtype='f')],
        [{'d1':'medium', 'd2':'small'}, np.array([ [-40,1], [-5,1], [0,0] ], dtype='f')],
        [{'d1':'small', 'd0':'small'}, np.array([ [0,0], [10,1], [40,1] ], dtype='f')],
        [{'d1':'small', 'd2':'small'}, np.array([ [-40,1], [-10,1], [0,0] ], dtype='f')],
        [{'d1':'large', 'd2':'large', 'd0':'large'}, np.array([ [-40,1], [-10,1], [0,0.8] ], dtype='f')],
        ]
        return fuzzyRuleBase

    #inferenceEngine
    def getOutputSet(self, alpha, thetaFuzzySubset):
        alphaRay = self.MathLibrary.getRay(np.array([self.outputBound[0] - 20, alpha], dtype='f'), np.array([self.outputBound[0] -10, alpha], dtype='f'))
        outputPointSet = []
        for segment in thetaFuzzySubset:
            answer = self.MathLibrary.getMeetingPoint(alphaRay, segment)
            pointL, pointR = segment['segment'][0], segment['segment'][1]
            if answer is None:
                if pointL[1] >= alpha:
                    pointList = [ np.array([pointL[0], alpha], dtype='f'), np.array([pointR[0], alpha], dtype='f') ]
                else:
                    pointList = [pointL, pointR]
            else:
                if pointL[1] == 0 and pointR[1] > alpha: 
                    pointList = [ pointL, answer, np.array([pointR[0], alpha], dtype='f') ] 
                elif pointL[1] > alpha and pointR[1] == 0: 
                    pointList = [ np.array([pointL[0], alpha], dtype='f'), answer, pointR ]
                # point0[1] == alpha or point1[1] == alpha:
                else:
                    pointList = [pointL, pointR]
            for point in pointList:
                outputPointSet.append(point)
        return self.getFuzzySet(outputPointSet)
    
    def getAlpha(self, TermSet, inferenceVar):
        distance = []
        for segment in TermSet:
            distance.append(self.MathLibrary.distancePtoS(np.array([inferenceVar, 0], dtype='f'), segment))
        for value in distance:
            if value is not None:
                return value
        return 0

    def getFuzzySet(self, pointSet):
        fuzzySet = [] 
        for index in range(len(pointSet)-1):
            if np.linalg.norm(pointSet[index] - pointSet[index+1]) >= 0.1:
                fuzzySet.append(self.MathLibrary.getSegment(pointSet[index], pointSet[index+1]))
        return fuzzySet

    def inferenceEngine(self, **inferenceVarDict):
        outputSet = []
        for fuzzyRule in self.fuzzyRuleBase:
            antecedent = fuzzyRule[0]
            thetaFuzzySubset = self.getFuzzySet(fuzzyRule[1])
            alpha = []
            for key in antecedent:
                alpha.append(self.getAlpha(self.membershipFunc[key][antecedent[key]], inferenceVarDict[key]))
            if np.min(alpha) != 0:
                outputSet.append(self.getOutputSet(np.min(alpha), thetaFuzzySubset))
        return outputSet
    
    # defuzzier
    def getFuzzSetValue(self, outputSet, theta):
        value = []
        for outputSubset in outputSet:
            for segment in outputSubset:
                if segment['segment'][0][0]<= theta <= segment['segment'][1][0]:
                    value.append(self.MathLibrary.getValue(segment, theta))
        if value ==[]:
            return 0
        else:
            return np.max(value) 

    def defuzzier(self, outputSet):
        valueSum = 0
        thetaSum = 0
        for theta in np.arange(self.outputBound[0], self.outputBound[1], 0.1):
            fuzzySetValue = self.getFuzzSetValue(outputSet, theta)
            valueSum += fuzzySetValue * theta
            thetaSum += fuzzySetValue
        Theta = valueSum / thetaSum
        return Theta

    # calculate
    def calculate(self, inferenceVar):
        outputSet = self.inferenceEngine(d0=inferenceVar[0], d1=inferenceVar[1], d2=inferenceVar[2])
        theta = self.defuzzier(outputSet)
        return theta