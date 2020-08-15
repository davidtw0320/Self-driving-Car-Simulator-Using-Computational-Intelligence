from MathLibrary import MathLibrary
import numpy as np

class Car():
    def __init__(self, carData, endSegmentData, roadData):
        self.carPosition = np.array(carData[:2], dtype='f')
        self.phi = carData[2]
        self.radius = 3
        self.MathLibrary = MathLibrary()
        #左 前 右
        self.sensorPosition = self.initialSensorPosition(carData[2])
        # line
        self.EndSegment, self.RoadSegment = self.initialRoadSegment(np.array(endSegmentData, dtype='f'), np.array(roadData, dtype='f'))
        self.detectRay = self.initialDetectRay()

    # initial method
    def initialRoadSegment(self, endSegmentData, roadData):
        EndSegment = self.MathLibrary.getSegment(endSegmentData[0], endSegmentData[1])
        RoadSegment = []
        for index in range(len(roadData)-1):
            RoadSegment.append(self.MathLibrary.getSegment(roadData[index],roadData[index+1]))
        return EndSegment, RoadSegment

    def initialDetectRay(self):
        detectRay = []
        for position in self.sensorPosition:
            detectRay.append(self.MathLibrary.getRay(self.carPosition, position))
        return detectRay
    
    def initialSensorPosition(self, phi):
        x = self.carPosition[0]
        y = self.carPosition[1]
        sensorPosition = []
        for i in range(0, 3):
            degrees = phi + 45 - i*45 
            sensorPosition.append(np.array([x+ self.radius * np.cos(np.deg2rad(degrees)), y + self.radius * np.sin(np.deg2rad(degrees))], dtype='f'))
        return sensorPosition

    # bump method
    def bump(self):
        result = False
        distance = []
        for roadSegment in self.RoadSegment:
            distance.append(self.MathLibrary.distancePtoS(self.carPosition, roadSegment))
        for value in distance:
            if value is not None and value <= self.radius:
                result = True
        return result
    
    # arriveTarget method
    def arriveTarget(self):
        answer = self.MathLibrary.distancePtoS(self.carPosition, self.EndSegment)
        if answer is not None and answer <= self.radius:
            return True
        else:
            return False

    # detect method
    def detect(self, distance):
        index = 0 
        for detectRay in self.detectRay:
            meetingPointSet = []
            for roadSegment in self.RoadSegment:
                meetingPointSet.append(self.MathLibrary.getMeetingPoint(detectRay, roadSegment))
            distance[index] = self.MathLibrary.getMinimumDistance(meetingPointSet, self.carPosition)
            index +=1 
    
    # move
    def updateSensorPosition(self):
        x = self.carPosition[0]
        y = self.carPosition[1]
        for i in range(0, 3):
            degrees = self.phi + 45 - i*45 
            self.sensorPosition[i] = np.array([x+ self.radius * np.cos(np.deg2rad(degrees)), y + self.radius * np.sin(np.deg2rad(degrees))], dtype='f')
    
    def updateDetectRay(self):
        self.updateSensorPosition()
        for index in range(len(self.detectRay)):
            self.detectRay[index] = self.MathLibrary.getRay(self.carPosition, self.sensorPosition[index])
    
    def arrangePhi(self):
        # 確保 self.phi 在 (-90, 270]
        self.phi %= 360
        if self.phi >= 270:
            return -(360-self.phi)
        elif self.phi < -90:
            return 360 + self.phi
        else:
            return self.phi

    def move(self, theta):
        # update x, y, phi
        self.carPosition[0] += np.cos(np.deg2rad(self.phi + theta)) + np.sin(np.deg2rad(theta)) * np.sin(np.deg2rad(self.phi))
        self.carPosition[1] += np.sin(np.deg2rad(self.phi + theta)) - np.sin(np.deg2rad(theta)) * np.cos(np.deg2rad(self.phi))
        self.phi -= np.rad2deg(np.arcsin(2 * np.sin(np.deg2rad(theta))/ (2*self.radius) ))
        self.arrangePhi()
        # update DetectRay
        self.updateDetectRay()