import MidiUtils as mu
import NeuralNetUtils as nnu
import MusicTheoryUtils as mtu
import random
import csv
import math

songList=mu.getSongList("projectMidiTraining")
numSongs=len(songList)
random.shuffle(songList)

neuralNet=nnu.NeuralNet([44,44,44,22,11,11],learningRate=0.001,useMomentum=False,momentumGamma=0.2,useL2Reg=True,l2Lambda=0.001)
trainSongCount=1000
epochCount=100
allSongDrumbeats=[]
sumError=0
errorCount=0

for i,songName in enumerate(songList[:trainSongCount]):
	print("loading song {}".format(i))
	songDir="projectMidiTraining/{}".format(songName)
	allSongDrumbeats.append([mu.getDrumsFromSong(songDir),mu.getTicksPerBeat(songDir)])
for epochNum in range(0,epochCount):
	for songIndex in range(0,trainSongCount):
		print("({}) training on song: {} -- epoch {}/{}".format(songIndex,songList[songIndex],epochNum+1,epochCount))
		ticksPerBeat=allSongDrumbeats[songIndex][1]
		currentTime=0
		drumHitBuffer=[]
		pendingHits=[0]*11
		for drumHit in allSongDrumbeats[songIndex][0]:		
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
						#[1,0] for no hit, [0,1] for hit
						neuralNet.forwardPropagation(drumHitBuffer[:44],correctOutput)
						neuralNet.backPropagation()
						neuralNet.gradientDescent()
				drumHitBuffer=drumHitBuffer[11:]
				if epochNum+1 == epochCount:
					sumError+=neuralNet.error
					errorCount+=1
neuralNet.saveTocsv("drumBeatEveryInstrument.csv")
print("average error: {}".format(sumError/errorCount))
				
				