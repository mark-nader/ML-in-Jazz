import torch

# credit to the pytorch tutorial on LSTMs

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
	
def oneHot(inputs,vectorSize):
	oneHotInputs=[]
	for inVal in inputs:
		oh=[0]*vectorSize
		oh[inVal % vectorSize]=1
		oneHotInputs.append(oh)
	return oneHotInputs