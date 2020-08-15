import tkinter as tk
from tkinter.filedialog import askopenfilename
from CarMap import CarMap
from FuzzySystem import FuzzySystem

class Window():
	def __init__(self):
		self.window = tk.Tk()
		self.window.title("HW1")
		self.window.config(bg="#323232")
		self.path = tk.StringVar()
		self.filename = tk.StringVar()
		# element
		self.bar = tk.Label(self.window, bg="#323232", fg="white", text="選擇檔案:")
		self.entry = tk.Entry(self.window, width=20, bg="#323232", fg="white", textvariable = self.filename)
		self.data_btn = tk.Button(self.window, bg="#323232", fg="black", text = "檔案選擇", command = self.selectFile)
		self.check_btn = tk.Button(self.window, bg="#323232", fg="black", text = "確認地圖", command = self.showInitialMap)
		self.run_btn = tk.Button(self.window, bg="#323232", fg="black", text = "執行", command = self.mainLoop)
		self.element = [self.bar, self.entry, self.data_btn, self.check_btn, self.run_btn]
		self.setElement()
		# carMap
		self.carMap = CarMap(self.window)
		self.carMap.grid(row=0,column=0,columnspan=len(self.element))
		# mainloop
		self.window.mainloop()

	def selectFile(self):
		self.path = askopenfilename()
		self.filename.set(self.path.split("/")[-1])
	
	def setElement(self):
		i = 0
		for element in self.element:
			element.grid(row=1, column=i)
			i += 1
	
	def getData(self):
		data = []
		f = open(self.path, "r")
		for i in f.read().split("\n"):
			data.append(i.split(","))
		# roadData
		roadData = []
		[roadData.append(list(map(int, i))) for i in data[1:]]
		# endSegmentData
		endSegmentData = [roadData.pop(0), roadData.pop(0)]
		return list(map(int, data[0])), endSegmentData, roadData
	
	def showInitialMap(self):
		# clear
		self.carMap.ax.clear()
		self.carMap.train4D = []
		self.carMap.train6D = []
		# get Data
		carData, endSegmentData, roadData = self.getData()
		# initialize car
		self.carMap.setCar(carData, endSegmentData, roadData)
		self.carMap.drawCar()
		# initialize road 
		self.carMap.setRoad(endSegmentData, roadData)

	def mainLoop(self):
		# initialize distance list
		self.distance = [0,0,0]
		#initialize a fuzzysystem
		fuzzySystem = FuzzySystem(["d0", "d1", "d2"])
		#main loop
		count = 0 
		#and not self.carMap.car.bump()
		while (not self.carMap.car.arriveTarget()):
			self.carMap.car.detect(self.distance)
			theta = fuzzySystem.calculate(self.distance)
			self.carMap.car.move(theta) #update x, y, phi, detectRay
			self.carMap.drawCar()
			self.carMap.log(self.distance, theta)
			count += 1
			print("step:", count)
		self.carMap.writeData()
		
#####################  main  ####################
if __name__ == "__main__":
	window = Window()