import torch
import math
import random
import numpy as np
import csv
import ast
import sys

csv.field_size_limit(2**20)

# credit to the pytorch tutorials on LSTMs and GANs
	
def oneHot(inputs,vectorSize):
	oneHotInputs=[]
	for inVal in inputs:
		oh=[0]*vectorSize
		oh[inVal % vectorSize]=1
		oneHotInputs.append(oh)
	return oneHotInputs

def getIndexesOfSortedList(listToSort):
	newList=[]
	for i,item in enumerate(listToSort):
		newList.append([i,item])
	newList.sort(key=lambda x: float(x[1]), reverse=True)
	return [item[0] for item in newList]
	
def topKAverage(realOutputs,predictedOutputDist):
	sumK=0
	for i,realOutput in enumerate(realOutputs):
		sumK+=getIndexesOfSortedList(predictedOutputDist[i]).index(realOutput)
	return sumK/(i+1)
	
def top3Accuracy(realOutputs,predictedOutputDist):
	inTop3=0
	totalOuts=len(realOutputs)
	for i,realOutput in enumerate(realOutputs):
		if getIndexesOfSortedList(predictedOutputDist[i]).index(realOutput) < 3:
			inTop3+=1
	return inTop3/totalOuts

class LSTM_LogSoftMax_RNN(torch.nn.Module):

	def __init__(self,inDim,hiddenDim,outDim,hiddenLayers,device):
		super(LSTM_LogSoftMax_RNN, self).__init__()
		
		self.hiddenDim=hiddenDim
		self.hiddenLayers=hiddenLayers
		
		self.lstm=torch.nn.LSTM(input_size=inDim,hidden_size=hiddenDim,num_layers=hiddenLayers)
		self.h2o=torch.nn.Linear(hiddenDim,outDim)
		
		self.hidden = self.initHidden()
		
		self.device=device
	
	def initHidden(self):
		return (torch.zeros(self.hiddenLayers, 1, self.hiddenDim),
				torch.zeros(self.hiddenLayers, 1, self.hiddenDim))
	
	def forward(self, netInput):
		formattedNetInp=netInput.view(len(netInput),1,-1).float().to(self.device)
		self.hidden=self.hidden[0].to(self.device),self.hidden[1].to(self.device)
		lstmOut,self.hidden=self.lstm(formattedNetInp,self.hidden)
		h2oOut=self.h2o(lstmOut.view(len(netInput),-1))
		netOut=torch.nn.functional.log_softmax(h2oOut,dim=1)
		return netOut
		
class NoteDiscriminator(torch.nn.Module):
	def __init__(self,numNotesIn,hiddenDim,numHiddenLayers,device):
		super(NoteDiscriminator,self).__init__()
		self.numNotesIn=numNotesIn
		layers=[torch.nn.Linear(numNotesIn*12,hiddenDim),torch.nn.ELU()]
		for i in range(numHiddenLayers):
			layers.extend([torch.nn.Linear(hiddenDim,hiddenDim),torch.nn.ELU()])
		layers.extend([torch.nn.Linear(hiddenDim,1),torch.nn.Sigmoid()])
		self.device=device
		self.main=torch.nn.Sequential(*layers)

	def forward(self,inNotes):
		catInps=inNotes[0]
		for i in range(1,self.numNotesIn):
			catInps=torch.cat((catInps,inNotes[i]),dim=1)
		netInp=catInps.float().to(self.device)
		return self.main(netInp)
		
class NoteGenerator(torch.nn.Module):
	def __init__(self,noiseDim,hiddenDim,numHiddenLayers,numNotesOut,temp,device):
		super(NoteGenerator,self).__init__()
		self.noiseDim=noiseDim
		self.numNotesOut=numNotesOut
		layers=[torch.nn.Linear(noiseDim,hiddenDim),torch.nn.ELU()]
		for i in range(numHiddenLayers):
			layers.extend([torch.nn.Linear(hiddenDim,hiddenDim),torch.nn.ELU()])
		layers.append(torch.nn.Linear(hiddenDim,numNotesOut*12))
		self.temp=temp
		self.device=device
		self.main=torch.nn.Sequential(*layers)
		
	def forward(self,noise):
		batchSize=len(noise)
		netInp=noise.to(self.device)
		main=self.main(netInp)
		xs=torch.chunk(main,self.numNotesOut,dim=1)
		xs=tuple([torch.nn.functional.log_softmax(x.view(batchSize,-1),dim=1) for x in xs])
		xs=tuple([torch.nn.functional.gumbel_softmax(x.view(batchSize,-1), tau=self.temp, hard=True) for x in xs])
		return xs

class VanillaNeuralNet:
	def __init__(self,layerSizes,learningRate=0.001,useMomentum=True,momentumGamma=0.5,useL2Reg=True,l2Lambda=0.001):
		self.layerCount=1
		self.layerSizes=layerSizes
		self.learningRate=learningRate
		self.useMomentum=useMomentum
		self.momentumGamma=momentumGamma
		self.useL2Reg=useL2Reg
		self.l2Lambda=l2Lambda
		self.outputs=[]
		self.weightedSums=[]
		self.weights=[]
		self.weightDerivatives=[]
		self.weightDeltas=[]
		self.biases=[]
		self.biasDerivatives=[]
		self.biasDeltas=[]
		self.errorDerivatives=np.array([])
		prev=layerSizes[0]
		for s in layerSizes[1:]:
			self.weights.append(np.random.randn(prev,s)*np.sqrt(2/prev))
			self.weightDerivatives.append(np.zeros([prev,s]))
			self.weightDeltas.append(np.zeros([prev,s])+learningRate)
			self.biases.append(np.array([0]*s))
			self.biasDerivatives.append(np.array([0]*s))
			self.biasDeltas.append(np.array([learningRate]*s))			
			prev=s
			self.layerCount+=1
		self.correctOutput=[]
		self.error=0
	
	@staticmethod
	def leakyReLU(x):
		 return np.where(x > 0, x, 0.01*x)
	
	@staticmethod
	def leakyReLUDerivative(x):
		return np.where(x > 0, 1, 0.01)
	
	@staticmethod
	def softmax(x):
		e_x = np.exp(x - np.max(x))
		return e_x / e_x.sum()
	
	@staticmethod
	def crossEntropy(trueValues,predictedValues):
		for i,target in enumerate(trueValues):
			if target:
				if predictedValues[i] == 0:
					return 750
				else:
					return -math.log(predictedValues[i])
		
	def forwardPropagation(self,nninp,correctOutput):
		self.outputs=[np.array(nninp)]
		self.weightedSums=[np.array(nninp)]
		for i in range(0,self.layerCount-2):
			weightedSum=np.matmul(self.outputs[-1],self.weights[i])+self.biases[i]
			self.weightedSums.append(weightedSum)
			self.outputs.append(self.leakyReLU(weightedSum))
		weightedSum=np.matmul(self.outputs[-1],self.weights[self.layerCount-2])+self.biases[self.layerCount-2]
		self.weightedSums.append(weightedSum)
		self.outputs.append(self.softmax(weightedSum))
		self.correctOutput=correctOutput
		self.error=self.crossEntropy(correctOutput,self.outputs[-1])
	
	def backPropagation(self):
		currentLayerDerivatives=self.outputs[-1]-np.array(self.correctOutput)
		self.errorDerivatives=currentLayerDerivatives
		for i in range(self.layerCount-2,-1,-1):
			self.biasDerivatives[i]=currentLayerDerivatives
			self.weightDerivatives[i]=np.matmul(self.outputs[i][np.newaxis].T,currentLayerDerivatives[np.newaxis])
			currentLayerDerivatives=self.leakyReLUDerivative(self.weightedSums[i])*np.matmul(self.weights[i],currentLayerDerivatives)
	
	def gradientDescent(self):
		for i in range(0,self.layerCount-1):
			regWeights=0
			regBiases=0
			if self.useL2Reg:
				regWeights=2*self.l2Lambda*self.weights[i]
				regBiases=2*self.l2Lambda*self.biases[i]
			if self.useMomentum:
				self.weightDeltas[i]=self.momentumGamma*self.weightDeltas[i]+self.learningRate*(self.weightDerivatives[i]+regWeights)
				self.biasDeltas[i]=self.momentumGamma*self.biasDeltas[i]+self.learningRate*(self.biasDerivatives[i]+regBiases)
				self.weights[i]=self.weights[i]-self.weightDeltas[i]
				self.biases[i]=self.biases[i]-self.biasDeltas[i]
			else:
				self.weights[i]=self.weights[i]-self.weightDeltas[i]*(self.weightDerivatives[i]+regWeights)
				self.biases[i]=self.biases[i]-self.biasDeltas[i]*(self.biasDerivatives[i]+regBiases)
	
	def saveTocsv(self,fileName):
		spamWriter=csv.writer(open(fileName,'w'))
		spamWriter.writerow([str(self.layerCount)])
		spamWriter.writerow([str(self.layerSizes)])
		spamWriter.writerow([str(self.learningRate)])
		spamWriter.writerow([str(self.useMomentum)])
		spamWriter.writerow([str(self.momentumGamma)])
		spamWriter.writerow([str(self.useL2Reg)])
		spamWriter.writerow([str(self.l2Lambda)])
		for layerW in self.weights:
			spamWriter.writerow([str(layerW.tolist())])
		for layerWDeriv in self.weightDerivatives:
			spamWriter.writerow([str(layerWDeriv.tolist())])
		for layerWDelt in self.weightDeltas:
			spamWriter.writerow([str(layerWDelt.tolist())])
		for layerB in self.biases:
			spamWriter.writerow([str(layerB.tolist())])
		for layerBDeriv in self.biasDerivatives:
			spamWriter.writerow([str(layerBDeriv.tolist())])
		for layerBDelt in self.biasDeltas:
			spamWriter.writerow([str(layerBDelt.tolist())])
		spamWriter.writerow([str(self.errorDerivatives.tolist())])
		spamWriter.writerow([str(self.correctOutput)])
		spamWriter.writerow([str(self.error)])
	
	def readFromcsv(self,fileName):
		spamReader=csv.reader(open(fileName))
		self.layerCount=int(next(spamReader)[0])
		next(spamReader)
		self.layerSizes=ast.literal_eval(next(spamReader)[0])
		next(spamReader)
		self.learningRate=float(next(spamReader)[0])
		next(spamReader)
		self.useMomentum=bool(next(spamReader)[0])
		next(spamReader)
		self.momentumGamma=float(next(spamReader)[0])
		next(spamReader)
		self.useL2Reg=bool(next(spamReader)[0])
		next(spamReader)
		self.l2Lambda=float(next(spamReader)[0])
		next(spamReader)
		for i in range(0,self.layerCount-1):
			self.weights[i]=np.array(ast.literal_eval(next(spamReader)[0]))
			next(spamReader)
		for i in range(0,self.layerCount-1):
			self.weightDerivatives[i]=np.array(ast.literal_eval(next(spamReader)[0]))
			next(spamReader)
		for i in range(0,self.layerCount-1):
			self.weightDeltas[i]=np.array(ast.literal_eval(next(spamReader)[0]))
			next(spamReader)
		for i in range(0,self.layerCount-1):
			self.biases[i]=np.array(ast.literal_eval(next(spamReader)[0]))
			next(spamReader)
		for i in range(0,self.layerCount-1):
			self.biasDerivatives[i]=np.array(ast.literal_eval(next(spamReader)[0]))
			next(spamReader)
		for i in range(0,self.layerCount-1):
			self.biasDeltas[i]=np.array(ast.literal_eval(next(spamReader)[0]))
			next(spamReader)
		self.errorDerivatives=ast.literal_eval(next(spamReader)[0])
		next(spamReader)
		self.correctOutput=ast.literal_eval(next(spamReader)[0])
		next(spamReader)
		self.error=float(next(spamReader)[0])
	
	