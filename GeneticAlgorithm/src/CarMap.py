from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.patches import Circle
import matplotlib.pyplot as plt
from Car import Car

class CarMap():
    def __init__(self, window):
        self.window = window
        self.plt = plt
        self.fig = self.plt.Figure()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.window)
        self.ax = self.fig.add_subplot(111)
        self.train4D = []
        self.train6D = []

    def grid(self, row, column, columnspan):
        self.canvas.get_tk_widget().grid(row=row, column=column, columnspan=columnspan) 
    
    # road
    def setRoad(self, endSegmentData, roadData):
        # end line
        x=[endSegmentData[0][0],endSegmentData[1][0]]
        y=[endSegmentData[0][1],endSegmentData[1][1]]
        self.ax.plot(x,y,color='red')
        # road 
        for index in range(len(roadData)-1):
            x=[roadData[index][0],roadData[index+1][0]]
            y=[roadData[index][1],roadData[index+1][1]]
            self.ax.plot(x,y,color='black')
        self.canvas.draw()
    
    # car
    def setCar(self, carData, endSegmentData, roadData):
        self.car = Car(carData, endSegmentData, roadData)
    
    def drawCar(self):
        position = [self.car.carPosition[0], self.car.carPosition[1]]
        self.carRecord = Circle(xy = position, radius=self.car.radius, alpha= 0.5)
        self.ax.add_patch(self.carRecord)
        self.ax.plot()
        self.canvas.draw()
    #log
    def log(self, distance, theta):
        data = [distance[1], distance[2], distance[0], theta]
        data = ['%6f '%value for value in data ]
        data.append("\n")
        self.train4D.extend(data)

        data = [self.car.carPosition[0], self.car.carPosition[1], distance[1], distance[2], distance[0], theta]
        data = ['%6f '%value for value in data ]
        data.append("\n")
        self.train6D.extend(data)
    
    def writeData(self):
        #前方距離、右方距離、左方距離、方向盤得出角度
        with open("logs/train4D.txt", "w") as f:
            f.writelines(self.train4D)
        #X座標、Y座標、前方距離、右方距離、左方距離、方向盤得出角度
        with open("logs/train6D.txt", "w") as f:
            f.writelines(self.train6D)
