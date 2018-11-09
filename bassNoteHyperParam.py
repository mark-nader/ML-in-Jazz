import torch
import NeuralNetUtils as nnu
import MidiUtils as mu
import random

batchSize, dimIn, dimHidden, dimOut, hiddenLayers = 1, 12, 128, 12, 2

trainSize, testSize = 1, 1

learningRate, weightDecay = 0.01, 0.001

numEpochs=1
testEvery=1

cuda=torch.device("cuda:0") ######### GPU
print(device)				######### GPU

model=nnu.LSTM_LogSoftMax_RNN(dimIn,dimHidden,dimOut,hiddenLayers)
model=model.to(cuda)		######### GPU
lossFunction=torch.nn.NLLLoss()
optimizer=torch.optim.Adamax(model.parameters(), weight_decay=weightDecay)

songList=mu.getSongList("projectMidiTraining")
random.shuffle(songList)
trainList=songList[:trainSize]
testList=songList[-testSize:]
allTrainSongBassLines=[]
allTestSongBassLines=[]
for i,songName in enumerate(trainList,1):
	print("loading training song: {} -- {}/{}".format(songName,i,trainSize))
	allTrainSongBassLines.extend(mu.getInstrumentFromSong("projectMidiTraining/{}".format(songName),32,39))
for i,songName in enumerate(testList,1):
	print("loading testing song: {} -- {}/{}".format(songName,i,testSize))
	allTestSongBassLines.extend(mu.getInstrumentFromSong("projectMidiTraining/{}".format(songName),32,39))
	
epoch=1
testingIteration=False
printedTestLoss=False
while epoch <= numEpochs:
	basslineList=allTrainSongBassLines
	if testingIteration:
		basslineList=allTestSongBassLines
	
	numBassLines=(len(basslineList))
	
	printLoss=0
	for i,bassline in enumerate(basslineList):
	
		model.zero_grad()
		model.hidden = model.initHidden()
		loss=0

		noteBuffer=mu.getBassNoteIntervalsFromBassline(bassline)
					
		if len(noteBuffer) > 1:
			netInputSeq=torch.tensor(nnu.oneHot(noteBuffer[:-1],12))
			netTargetSeq=torch.tensor([interval % 12 for interval in noteBuffer[1:]])
			netInputSeq=netInputSeq.to(cuda)	######### GPU
			netTargetSeq=netTargetSeq.to(cuda)	######### GPU
			if testingIteration:
				with torch.no_grad():
					predictedInterval=model(netInputSeq)
					loss=lossFunction(predictedInterval,netTargetSeq)
					printLoss+=loss
			else:
				predictedInterval=model(netInputSeq)
				loss=lossFunction(predictedInterval,netTargetSeq)
				printLoss+=loss
				loss.backward()
				optimizer.step()
	if testingIteration:
		print("testing set loss: {} -- epoch {}/{} [|TEST DATA|]".format(printLoss/numBassLines,epoch,numEpochs))
		printedTestLoss=True
	else:
		print("training set loss: {} -- epoch {}/{}".format(printLoss/numBassLines,epoch,numEpochs))
		

	testingIteration=(epoch % testEvery == 0 and not printedTestLoss)
	
	if not testingIteration:
		epoch+=1
		printedTestLoss=False


torch.save(model.state_dict(), "bassNotes.pt")