import NeuralNetUtils as nnu
import MidiUtils as mu
import random

netNum=6
resumeTraining=False
netStructure = [108, 108, 12]
learningRate, weightDecay = 0.0005, 0.001
numEpochs=51
testEvery=10
optimisationAlg="SGD"

songList=mu.getSongList("projectMidiTraining")
numSongs=len(songList)
songNetworkPairs=mu.readSongNetworksFromcsv("trained networks/songCategorise_v{}_smallBatch100.csv".format(netNum),numSongs)
neuralNet=nnu.VanillaNeuralNet(netStructure,learningRate=learningRate,useMomentum=False,momentumGamma=0.1,useL2Reg=True,l2Lambda=weightDecay)

prevEpochs=0
if resumeTraining:
	neuralNet.readFromcsv("v{}Class100.csv".format(netNum))

trainSize=0
songList=[]
for songNetPair in songNetworkPairs:
	if int(songNetPair[1]) == 1:
		songList.append(songNetPair[0])
		trainSize+=1

trainBasslines=[]
trainBassIntervals=[]
for i,songName in enumerate(songList,1):
	print("loading training song: {} -- {}/{}".format(songName,i,trainSize))
	trainBasslines.extend(mu.getInstrumentFromSong("projectMidiTraining/{}".format(songName),32,39))
for bassline in trainBasslines:
	trainBassIntervals.append(mu.getBassNoteIntervalsFromBassline(bassline))
inputNoteCount=neuralNet.layerSizes[0]//12
epoch=0
for epoch in range(numEpochs):
	printTopKAvg=0
	printTop3Acc=0
	basslineSectionsTrained=0
	for bassline in trainBassIntervals:
		
		basslineLen=len(bassline)
		if basslineLen > inputNoteCount:
			netInp=[]
			for i in range(inputNoteCount):
				oneHotInp=[0]*12
				oneHotInp[bassline[i] % 12]=1
				netInp.extend(oneHotInp)
			for i in range(inputNoteCount,basslineLen-inputNoteCount):
				target=[0]*12
				target[bassline[i] % 12]=1
				neuralNet.forwardPropagation(netInp,target)
				neuralNet.backPropagation()
				neuralNet.gradientDescent()
				netInp=netInp[12:]
				netInp.extend(target)
				predictedRank=nnu.getIndexesOfSortedList(neuralNet.outputs[-1]).index(bassline[i] % 12)
				printTopKAvg+=predictedRank
				if predictedRank < 3:
					printTop3Acc+=1
				basslineSectionsTrained+=1
		
	if epoch % testEvery == 0:
		print("training average: {} | top 3 accuracy: {} -- epoch {}/{}".format(printTopKAvg/basslineSectionsTrained,printTop3Acc/basslineSectionsTrained,epoch,numEpochs))
		
neuralNet.saveTocsv("v{}Class100.csv".format(netNum))

print("################################################")
print("Network Structure = {}".format(netStructure))
print("Train set size = {}".format(trainSize))
print("Learning Rate = {}, Weight decay = {}".format(learningRate,weightDecay))
print("Total Epochs = {}".format(prevEpochs+numEpochs))
print("Optimisation algorithm = {}".format(optimisationAlg))
