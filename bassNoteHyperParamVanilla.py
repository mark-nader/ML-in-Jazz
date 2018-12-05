import NeuralNetUtils as nnu
import MidiUtils as mu
import random

resumeTraining=False
nameToSave="v1.csv"
netStructure = [72,72,72,72,24,12]
trainSize, testSize = 800, 200
learningRate, weightDecay = 0.0005, 0.001
numEpochs=100
testEvery=10
optimisationAlg="SGD"

neuralNet=nnu.VanillaNeuralNet(netStructure,learningRate=learningRate,useMomentum=False,momentumGamma=0.1,useL2Reg=True,l2Lambda=weightDecay)

prevEpochs=0
if resumeTraining:
	neuralNet.readFromcsv(nameToSave)

songList=mu.getSongList("projectMidiTraining")
random.shuffle(songList)
trainList=songList[:trainSize]
testList=songList[-testSize:]
trainBasslines=[]
trainBassIntervals=[]
testBasslines=[]
testBassIntervals=[]
for i,songName in enumerate(trainList,1):
	print("loading training song: {} -- {}/{}".format(songName,i,trainSize))
	trainBasslines.extend(mu.getInstrumentFromSong("projectMidiTraining/{}".format(songName),32,39))
for bassline in trainBasslines:
	trainBassIntervals.append(mu.getBassNoteIntervalsFromBassline(bassline))
for i,songName in enumerate(testList,1):
	print("loading testing song: {} -- {}/{}".format(songName,i,testSize))
	testBasslines.extend(mu.getInstrumentFromSong("projectMidiTraining/{}".format(songName),32,39))
for bassline in testBasslines:
	testBassIntervals.append(mu.getBassNoteIntervalsFromBassline(bassline))

inputNoteCount=neuralNet.layerSizes[0]//12
epoch=0
testingIteration=True
printedTestLoss=False
while epoch <= numEpochs:
	basslineIntervals=trainBassIntervals
	if testingIteration:
		basslineIntervals=testBassIntervals

	printTopKAvg=0
	printTop3Acc=0
	basslineSectionsTrained=0
	for bassline in basslineIntervals:
		
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
				if testingIteration:
					neuralNet.forwardPropagation(netInp,target)
				else:
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
		
	if testingIteration:
		print("testing average: {} | top 3 accuracy: {} -- epoch {}/{} [|TEST DATA|]".format(printTopKAvg/basslineSectionsTrained,printTop3Acc/basslineSectionsTrained,epoch,numEpochs))
		printedTestLoss=True
	elif epoch % testEvery == 0:
		print("training average: {} | top 3 accuracy: {} -- epoch {}/{}".format(printTopKAvg/basslineSectionsTrained,printTop3Acc/basslineSectionsTrained,epoch,numEpochs))
		

	testingIteration=(epoch % testEvery == 0 and not printedTestLoss)
	
	if not testingIteration:
		epoch+=1
		printedTestLoss=False

neuralNet.saveTocsv(nameToSave)

print("################################################")
print("Network Structure = {}".format(netStructure))
print("Train set size = {}, Test set size = {}".format(trainSize,testSize))
print("Learning Rate = {}, Weight decay = {}".format(learningRate,weightDecay))
print("Total Epochs = {}".format(prevEpochs+numEpochs))
print("Optimisation algorithm = {}".format(optimisationAlg))
