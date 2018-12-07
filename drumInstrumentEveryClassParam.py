import MidiUtils as mu
import NeuralNetUtils as nnu
import MusicTheoryUtils as mtu
import random
import csv
import math

songList=mu.getSongList("projectMidiTraining")
numSongs=len(songList)
random.shuffle(songList)

neuralNet=nnu.VanillaNeuralNet([44,44,11],learningRate=0.001,useMomentum=False,momentumGamma=0.2,useL2Reg=True,l2Lambda=0.001)
trainSongCount=800
testSongCount=200
epochCount=50
testEvery=5


allSongTrainDrumbeats=[]
allSongTestDrumbeats=[]
for i,songName in enumerate(songList[:trainSongCount]):
	print("loading training song {}".format(i))
	songDir="projectMidiTraining/{}".format(songName)
	allSongTrainDrumbeats.append([mu.getDrumsFromSong(songDir),mu.getTicksPerBeat(songDir)])
for i,songName in enumerate(songList[-testSongCount:]):
	print("loading testing song {}".format(i))
	songDir="projectMidiTraining/{}".format(songName)
	allSongTestDrumbeats.append([mu.getDrumsFromSong(songDir),mu.getTicksPerBeat(songDir)])
	
epoch=0
testingIteration=True
printedTestLoss=False
while epoch <= epochCount:
	drumBeatList=allSongTrainDrumbeats
	if testingIteration:
		drumBeatList=allSongTestDrumbeats
	forwardPasses=0
	inTop3=0
	topKSum=0
	for drumBeat in drumBeatList:
		ticksPerBeat=drumBeat[1]
		currentTime=0
		drumHitBuffer=[]
		pendingHits=[0]*11
		for drumHit in drumBeat[0]:		
			if drumHit[0] == currentTime:
				drumType=mtu.lookUpDrumType(drumHit[1])
				if not drumType == "NOT FOUND":
					pendingHits[drumType[0]]=1
			else:
				currentTime=drumHit[0]
				if not pendingHits == [0]*11:
					drumHitBuffer.extend(pendingHits)
				pendingHits=[0]*11
				drumType=mtu.lookUpDrumType(drumHit[1])
				if not drumType == "NOT FOUND":
					pendingHits[drumType[0]]=1
			if len(drumHitBuffer) > 44:
				for i in range(0,11):
					if drumHitBuffer[-11:][i] == 1:
						correctOutput=[0]*11
						correctOutput[i]=1
						neuralNet.forwardPropagation(drumHitBuffer[:44],correctOutput)						
						if i in nnu.getIndexesOfSortedList(neuralNet.outputs[-1])[:3]:
							inTop3+=1
						topKSum+=nnu.getIndexesOfSortedList(neuralNet.outputs[-1]).index(i)
						forwardPasses+=1
						if not testingIteration:
							neuralNet.backPropagation()
							neuralNet.gradientDescent()
				drumHitBuffer=drumHitBuffer[11:]
	
	if testingIteration:
		print("testing average: {} | top 3 accuracy: {} -- epoch {}/{} [|TEST DATA|]".format(topKSum/forwardPasses,inTop3/forwardPasses,epoch,epochCount))
		printedTestLoss=True
	elif epoch % testEvery == 0:
		print("training average: {} | top 3 accuracy: {} -- epoch {}/{}".format(topKSum/forwardPasses,inTop3/forwardPasses,epoch,epochCount))
		

	testingIteration=(epoch % testEvery == 0 and not printedTestLoss)
	
	if not testingIteration:
		epoch+=1
		printedTestLoss=False
				
neuralNet.saveTocsv("drumBeatEveryInstrument.csv")
				
				