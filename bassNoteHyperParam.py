import torch
import NeuralNetUtils as nnu
import MidiUtils as mu
import random

dimIn, dimHidden, dimOut, hiddenLayers = 12, 64, 12, 6

trainSize, testSize = 5, 2

learningRate, weightDecay = 0.001, 0.001

repeatWeight, fifthWeight = 0.5, 0.5

sequenceLength = 12

numEpochs=10
testEvery=2

optimisationAlg="SGD" #change this below also

cuda=torch.device("cuda:0" if torch.cuda.is_available() else "cpu") ######### GPU
print(cuda)				######### GPU

model=nnu.LSTM_LogSoftMax_RNN(dimIn,dimHidden,dimOut,hiddenLayers,cuda)
model=model.to(cuda)		######### GPU
lossFunction=torch.nn.NLLLoss(weight=torch.tensor([repeatWeight,1,1,1,1,1,1,fifthWeight,1,1,1,1]))
optimizer=torch.optim.SGD(model.parameters(), lr=learningRate, weight_decay=weightDecay)

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

	printTopKAvg=0
	j=0
	basslineSectionsTrained=0
	for i,bassline in enumerate(basslineList):
	
		basslineToTrain=mu.getBassNoteIntervalsFromBassline(bassline)
		basslineToTrainLen=len(basslineToTrain)
		while basslineToTrainLen >= sequenceLength:
			
			basslineSection=basslineToTrain[:sequenceLength]
			basslineToTrain=basslineToTrain[1:]
			basslineToTrainLen-=1
	
			model.zero_grad()
			model.hidden = model.initHidden()
						
			netInputSeq=torch.tensor(nnu.oneHot(basslineSection[:-1],12))
			netTargetSeq=torch.tensor([interval % 12 for interval in basslineSection[1:]])
			netInputSeq=netInputSeq.to(cuda)	######### GPU
			netTargetSeq=netTargetSeq.to(cuda)	######### GPU
			if testingIteration:
				with torch.no_grad():
					predictedInterval=model(netInputSeq)
					basslineTopKAvg=nnu.topKAverage(netTargetSeq.tolist(),predictedInterval)
					printTopKAvg+=basslineTopKAvg
					basslineSectionsTrained+=1
			else:
				predictedInterval=model(netInputSeq)
				loss=lossFunction(predictedInterval,netTargetSeq)
				basslineTopKAvg=nnu.topKAverage(netTargetSeq.tolist(),predictedInterval)
				printTopKAvg+=basslineTopKAvg
				basslineSectionsTrained+=1
				loss.backward()
				optimizer.step()
					
					
	if testingIteration:
		print("testing set loss: {} -- epoch {}/{} [|TEST DATA|]".format(printTopKAvg/basslineSectionsTrained,epoch,numEpochs))
		printedTestLoss=True
	elif epoch % testEvery == 0:
		print("training set loss: {} -- epoch {}/{}".format(printTopKAvg/basslineSectionsTrained,epoch,numEpochs))
		

	testingIteration=(epoch % testEvery == 0 and not printedTestLoss)
	
	if not testingIteration:
		epoch+=1
		printedTestLoss=False

torch.save(model.state_dict(), "bassNotes.pt")

print("################################################")
print("Input dimension = {}, Hidden dimension ={}, Output dimension = {}, Hidden layers = {}".format(dimIn,dimHidden,dimOut,hiddenLayers))
print("Train set size = {}, Test set size = {}".format(trainSize,testSize))
print("Learning Rate = {}, Weight decay = {}".format(learningRate,weightDecay))
print("Repeat note weight = {}, Fith note weight = {}".format(repeatWeight,fifthWeight))
print("Sequence length = {}".format(sequenceLength))
print("Epochs = {}".format(numEpochs))
print("Optimisation algorithm = {}".format(optimisationAlg))
