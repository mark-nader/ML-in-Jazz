import MidiUtils as mu
import NeuralNetUtils as nnu
import random
import csv

# 0 undecided, 1 agrees with to network

netName="v1_smallBatch100.csv"
netStructure = [72, 72, 72, 36, 24, 12]

mu.createNewSongNetworks("trained networks/songCategorise_{}".format(netName))
songList=mu.getSongList("projectMidiTraining")
numSongs=len(songList)
	
songNetworkPairs=mu.readSongNetworksFromcsv("trained networks/songCategorise_{}".format(netName),numSongs)

neuralNet=nnu.VanillaNeuralNet(netStructure)
neuralNet.readFromcsv("trained networks/{}".format(netName))

inputNoteCount=neuralNet.layerSizes[0]//12
songsAgree=0
for j,songNetPair in enumerate(songNetworkPairs):
	print("Checking song {}".format(j+1))
	songBasslines=mu.getInstrumentFromSong("projectMidiTraining/{}".format(songNetPair[0]),32,39)
	bassIntervals=[]
	for bassline in songBasslines:
		bassIntervals.append(mu.getBassNoteIntervalsFromBassline(bassline))
	printTopKAvg=0
	printTop3Acc=0
	basslineSectionsTested=0
	for bassline in bassIntervals:
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
				netInp=netInp[12:]
				netInp.extend(target)
				predictedRank=nnu.getIndexesOfSortedList(neuralNet.outputs[-1]).index(bassline[i] % 12)
				printTopKAvg+=predictedRank
				if predictedRank < 3:
					printTop3Acc+=1
				basslineSectionsTested+=1
	if not basslineSectionsTested == 0:
		if printTopKAvg/basslineSectionsTested < 2.75 and printTop3Acc/basslineSectionsTested > 0.625:
			songNetworkPairs[j][1]=1
			songsAgree+=1
	
mu.saveSongNetworksTocsv("trained networks/songCategorise_{}".format(netName),songNetworkPairs)
print("{} close match songs found".format(songsAgree))
