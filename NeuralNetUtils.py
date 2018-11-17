import torch

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
	
	