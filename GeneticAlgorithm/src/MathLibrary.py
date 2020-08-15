from sympy import *
import numpy as np

class MathLibrary():
    def __init__(self):
        self.a, self.b ,self.x, self.y = symbols('a b x y')
    
    def SympyToNumpy(self, dict):
        for key in dict:
            dict[key] = np.float(dict[key])

    # initial method
    def getSegment(self, point1, point2):
        answer = solve([self.a * point1[0] -point1[1] + self.b, self.a * point2[0] -point2[1] + self.b],[self.a, self.b])
        self.SympyToNumpy(answer)
        #a -> 無限大
        if len(answer) == 0:
            answer = {self.x: point1[0]} 
        segment = {'segment': [point1, point2]}
        answer.update(segment)
        return answer
    
    def getRay(self, point1, point2): 
        #point1 -> point2
        answer = solve([self.a*point1[0] -point1[1] + self.b, self.a*point2[0] -point2[1] + self.b],[self.a, self.b])
        self.SympyToNumpy(answer)
        #unlimit
        if len(answer) == 0:
            answer = {self.x: point1[0]}
        ray = {'ray': [point1, point2]} 
        answer.update(ray)
        return answer

    # bump method
    def distancePtoS(self, point, segment): 
        if self.x in segment:
            projectionPoint = np.array([segment[self.x], point[1]], dtype='f')
            if self.pointOnSegment(projectionPoint, segment):
                return abs(segment[self.x] - point[0])
            else:
                return None
        else:
            a ,b, x0, y0= segment[self.a], segment[self.b], point[0], point[1]
            projectionPoint = np.array([x0 - a * (a*x0 - y0 + b) / (a**2 + 1), y0 + 1  * (a*x0 - y0 + b) / (a**2 + 1)], dtype='f')
            if self.pointOnSegment(projectionPoint, segment):
                return round(np.abs(a * x0 - y0 + b)/np.sqrt(a**2 + 1), 3)
            else:
                return None
    
    def pointOnSegment(self, point, segment):
        dot = np.dot((segment['segment'][0] - point), (segment['segment'][1] - point))
        if dot <= 0:
            return True
        else:
            return False

    # detect method
    def pointOnRay(self, point, ray):
        dot = np.dot((ray['ray'][1] - ray['ray'][0]), (point - ray['ray'][0]))
        if dot >= 0:
            return True
        else:
            return False

    def getMeetingPoint(self, ray, segment):
        answer = solve([self.getFunction(ray), self.getFunction(segment)],[self.x, self.y])
        self.SympyToNumpy(answer)
        #無解
        if len(answer) == 0:
            return None
        #無限多解
        elif len(answer) == 1:
            if np.dot(ray['ray'][1] - ray['ray'][0], segment['segment'][0] - ray['ray'][0]) >= 0:
                length0 = np.linalg.norm(segment['segment'][0] - ray['ray'][1]) 
                length1 = np.linalg.norm(segment['segment'][1] - ray['ray'][1])
                return segment['segment'][np.argmin([length0, length1])]
            else:
                return None
        # 有一解 
        else:
            meetingPoint = np.array([answer[self.x], answer[self.y]], dtype='f')
            if self.pointOnSegment(meetingPoint, segment) and self.pointOnRay(meetingPoint, ray):
                return meetingPoint

    def getFunction(self, func):
        if self.x in func:
            return str(self.x) + "-" + str(func[self.x])
        else:
            return str(func[self.a]) + "*" + str(self.x) + "-" + str(self.y) + "+" +str(func[self.b])

    def getMinimumDistance(self, meetingPointSet, MainPoint):
        distance = []
        for meetingPoint in meetingPointSet:
            if meetingPoint is not None:
                distance.append(np.linalg.norm(meetingPoint - MainPoint))
        return min(distance)
    
    # defuzzier
    def getValue(self, segment ,x):
        return segment[self.a] * x + segment[self.b]