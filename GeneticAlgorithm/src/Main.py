import tkinter as tk
from tkinter.filedialog import askopenfilename
from CarMap import CarMap
from RBFN import RBFN

from tkinter.filedialog import asksaveasfile 

class Window():
	def __init__(self):
		self.window = tk.Tk()
		self.window.title("HW2")
		self.window.config(bg="#323232")
		# column0
		self.loopTimes = tk.IntVar()
		self.geneNumber = tk.IntVar()
		self.mutationProbability = tk.DoubleVar()
		self.crossoverProbability = tk.DoubleVar()
		# column2
		self.J = tk.IntVar()
		self.batchSize = tk.IntVar()
		self.trainingDataPath = tk.StringVar()
		self.trainingDataFileName = tk.StringVar()
		self.trainingStatus = tk.StringVar()
		# column6
		self.mapFileName = tk.StringVar()
		self.d0 = tk.DoubleVar()
		self.d1 = tk.DoubleVar()
		self.d2 = tk.DoubleVar()

		elements ={
		0:[{"labelName":"-基因演算法參數-"}, {"labelName":"迭代次數:","textvarible":self.loopTimes},{"labelName":"族群數量:","textvarible":self.geneNumber},{"labelName":"交配機率:","textvarible":self.crossoverProbability},{"labelName":"突變機率:","textvarible":self.mutationProbability}],
		2:[{"labelName":"-RBFN訓練-"},{"labelName":"網路J值:","textvarible":self.J}, {"labelName":"批量大小:","textvarible":self.batchSize}, {"labelName":"選擇訓練資料集:", "textvarible":self.trainingDataFileName, "buttonName1":"檔案選擇", "buttonCommand1":self.selectTrainingData, "buttonName2":"開始訓練", "buttonCommand2":self.training},{"labelName":"訓練狀態:", "textvarible":self.trainingStatus}, {"buttonName1":"儲存訓練參數","buttonCommand1":self.saveFile}],
		6:[{"labelName":"-訓練結果執行-"}, {"labelName":"選擇地圖:", "textvarible":self.mapFileName, "buttonName1":"檔案選擇", "buttonCommand1":self.selectMapFile, "buttonName2":"確認地圖", "buttonCommand2":self.showInitialMap},{"buttonName2":"執行", "buttonCommand2":self.mainLoop}, {"labelName":"d0:","textvarible":self.d0},{"labelName":"d1:","textvarible":self.d1},{"labelName":"d2:","textvarible":self.d2}]
		} 
		self.setElements(elements)

		# carMap
		self.carMap = CarMap(self.window)
		self.carMap.grid(row=0,column=0,columnspan=10)
		# mainloop
		self.window.mainloop()

	def setElements(self, kw):
		for column, columnElements in kw.items():
			for elementIndex in range(len(columnElements)):
				self.getLabelEntryButtons(elementIndex+1, column, columnElements[elementIndex])

	def getLabelEntryButtons(self, row, column, kw):
		columnIndex = column
		if "labelName" in kw:
			label = tk.Label(self.window, bg="#323232", fg="white", text=kw["labelName"])
			label.grid(row=row, column=columnIndex,  sticky=tk.W)
			columnIndex += 1
		if "textvarible" in kw:
			entry = tk.Entry(self.window, width=10, bg="#323232", fg="white", textvariable =kw["textvarible"])
			entry.grid(row=row, column=columnIndex,  sticky=tk.W)
			columnIndex += 1
		if "buttonName1" in kw and "buttonCommand1" in kw:
			button1 = tk.Button(self.window, bg="#323232", fg="black", text = kw["buttonName1"], command = kw["buttonCommand1"], width = 10)
			button1.grid(row=row, column=columnIndex,  sticky=tk.W)
			columnIndex += 1
		if "buttonName2" in kw and "buttonCommand2" in kw:
			button2 = tk.Button(self.window, bg="#323232", fg="black", text = kw["buttonName2"], command = kw["buttonCommand2"], width = 10)
			button2.grid(row=row, column=columnIndex,  sticky=tk.W)
			columnIndex += 1
	
	# column2
	def selectTrainingData(self):
		self.trainingDataPath = askopenfilename()
		self.trainingDataFileName.set(self.trainingDataPath.split("/")[-1])

	def saveFile(self):
		if self.trainingStatus.get() == "訓練結束":
			files = [ ('Text Document', '*.txt')] 
			path = asksaveasfile(filetypes = files, defaultextension = files)
			self.RBFN.writeData(path)
		else:
			self.trainingStatus.set("尚未訓練")

	def training(self):
		if self.loopTimes.get()==0 or self.geneNumber.get()==0 or self.J.get()== 0 or self.batchSize.get()== 0 or self.trainingDataFileName.get() == "":
			self.trainingStatus.set("不合法")
		else:
			self.trainingStatus.set("訓練中")
			self.RBFN = RBFN(loopTimes=self.loopTimes.get(), geneNumber=self.geneNumber.get(), mutationProbability=self.mutationProbability.get(), crossoverProbability=self.crossoverProbability.get(), trainingDataPath=self.trainingDataPath, hiddenLayerNeuronsNumber= self.J.get(), batchSize = self.batchSize.get())
			self.RBFN.train()
			self.trainingStatus.set("訓練結束")
	
	# column6
	def selectMapFile(self):
		self.path = askopenfilename()
		self.mapFileName.set(self.path.split("/")[-1])
	
	def loadFile(self):
		self.path = askopenfilename()
		self.loadFileName.set(self.path.split("/")[-1]) 
	
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
	
	def updateDistanceGUI(self):
		self.d0.set(self.distance[0])
		self.d1.set(self.distance[1])
		self.d2.set(self.distance[2])

	def mainLoop(self):
		if self.trainingStatus.get() == "訓練結束":
			# initialize distance list
			self.distance = [0,0,0]
			#main loop
			count = 0 
			while (not self.carMap.car.arriveTarget()):
				self.carMap.car.detect(self.distance)
				theta = self.RBFN.calculate(self.carMap.car.carPosition, self.distance)
				self.carMap.car.move(theta) #update x, y, phi, detectRay
				self.carMap.drawCar()
				self.carMap.log(self.distance, theta)
				count += 1
				print("step:", count)
				self.updateDistanceGUI()
			self.carMap.writeData()
		else:
			self.trainingStatus.set("尚未訓練")
		
#####################  main  ####################
if __name__ == "__main__":
	window = Window()