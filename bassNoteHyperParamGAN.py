import torch
import NeuralNetUtils as nnu
import MidiUtils as mu
import random
import numpy as np
import math

resumeTraining=True
nameToSaveG="bassNotesGAN_Generator.pt"
nameToSaveD="bassNotesGAN_Discriminator.pt"
sequenceLength = 8
dimInG, dimHiddenG, numHiddenG = 5, 6*12*2, 4
dimHiddenD, numHiddenD = 6*12*2, 4
trainSize = 5
learningRate, adamBetas, gumbelTemp = 0.0002, (0.5, 0.999), 0.1
numEpochs=10
testEvery=1

optimisationAlg="Adam" #change this below also

cuda=torch.device("cuda:0" if torch.cuda.is_available() else "cpu") ######### GPU
print(cuda)				######### GPU

modelD=nnu.NoteDiscriminator(sequenceLength,dimHiddenD,numHiddenD,cuda)
modelD=modelD.to(cuda)
modelG=nnu.NoteGenerator(dimInG,dimHiddenG,numHiddenG,sequenceLength,gumbelTemp,cuda)
modelG=modelG.to(cuda)

optimizerD=torch.optim.Adam(modelD.parameters(),lr=learningRate,betas=adamBetas)
optimizerG=torch.optim.Adam(modelG.parameters(),lr=learningRate,betas=adamBetas)

prevEpochs=0
if resumeTraining:
	checkpoint = torch.load(nameToSaveG)
	modelG.load_state_dict(checkpoint['model_state_dict'])
	optimizerG.load_state_dict(checkpoint['optimizer_state_dict'])
	prevEpochs = checkpoint['epoch']
	modelG.train()
	
	checkpoint = torch.load(nameToSaveD)
	modelD.load_state_dict(checkpoint['model_state_dict'])
	optimizerD.load_state_dict(checkpoint['optimizer_state_dict'])
	modelD.train()
	

criterion=torch.nn.BCELoss()

songList=mu.getSongList("projectMidiTraining")
random.shuffle(songList)
trainList=songList[:trainSize]
allTrainSongBassLines=[]
for i,songName in enumerate(trainList,1):
	print("loading training song: {} -- {}/{}".format(songName,i,trainSize))
	allTrainSongBassLines.extend(mu.getInstrumentFromSong("projectMidiTraining/{}".format(songName),32,39))
basslineTrainIntervalsOneHot=[]
numBasslines=len(allTrainSongBassLines)
for i,bassline in enumerate(allTrainSongBassLines):
	print("converting basslines into one hot intervals -- {}/{}".format(i,numBasslines))
	singleOctaveBassNoteIntervals=np.array(mu.getBassNoteIntervalsFromBassline(bassline)) % 12
	basslineTrainIntervalsOneHot.append(nnu.oneHot(singleOctaveBassNoteIntervals,12))
	
for epoch in range(numEpochs+1):
	epochErrD=0
	epochErrG=0
	for i,bassline in enumerate(basslineTrainIntervalsOneHot):
		basslineLen=len(bassline)
		if basslineLen >= sequenceLength:
			
			realInputD=[]
			batchSize=basslineLen-sequenceLength+1
			
			for j in range(sequenceLength):
				realInputD.append(torch.tensor(bassline[j:j+batchSize]).to(cuda))
				
			realLabels=torch.tensor([[1]]*batchSize).float().to(cuda)
			fakeLabels=torch.tensor([[0]]*batchSize).float().to(cuda)
			
			# train D

			# first with real batch

			modelD.zero_grad()
			output=modelD(realInputD)
			errRealD=criterion(output,realLabels)
			errRealD.backward()
			
			# then with fake batch

			noise=torch.randn(batchSize,dimInG).to(cuda)
			fakeInputD=modelG(noise)
			output=modelD([fI.detach() for fI in fakeInputD])
			errFakeD=criterion(output,fakeLabels)
			errFakeD.backward()
			epochErrD+=(errRealD+errFakeD)/2
			optimizerD.step()
			
			# train G
	
			modelG.zero_grad()
			output=modelD(fakeInputD)
			errG=criterion(output,realLabels)
			errG.backward()
			epochErrG+=errG
			optimizerG.step()
					
	if epoch % testEvery == 0:
		print("Discriminator Accuracy: {} | Generator Accuracy: {} -- epoch {}/{}".format(math.exp(-epochErrD/numBasslines),math.exp(-epochErrG/numBasslines),epoch+1,numEpochs))

torch.save({
            'epoch': prevEpochs+numEpochs,
            'model_state_dict': modelG.state_dict(),
            'optimizer_state_dict': optimizerG.state_dict(),
            }, nameToSaveG)
torch.save({
            'model_state_dict': modelD.state_dict(),
            'optimizer_state_dict': optimizerD.state_dict(),
            }, nameToSaveD)

print("################################################")
print("Sequence length = {}".format(sequenceLength))
print("Generator: Input dimension = {}, Hidden dimension = {}, Hidden Depth = {}".format(dimInG,dimHiddenG,numHiddenG))
print("Discriminator: Hidden dimension = {}, Hidden Depth = {}".format(dimHiddenD,numHiddenD))
print("Train set size = {}".format(trainSize))
print("Learning Rate = {}, Adam Betas = {}, Gumbel Temperature = {}".format(learningRate,adamBetas,gumbelTemp))
print("Epochs = {}".format(numEpochs))
print("Optimisation algorithm = {}".format(optimisationAlg))


noise=torch.randn(1,dimInG)
fakeInputD=modelG(noise)
output=modelD(fakeInputD)
errG=criterion(output,torch.tensor([[1]]).float())
print("=====")
print("=====")
print(fakeInputD)
print(output)
print(errG)
