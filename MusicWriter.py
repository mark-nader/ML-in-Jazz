import mido
import torch
import NeuralNetUtils as nnu
import MusicTheoryUtils as mtu
import random

ticksPerBeat=480
cuda=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def getIndexesOfSortedList(listToSort):
	newList=[]
	for i,item in enumerate(listToSort):
		newList.append([i,item])
	newList.sort(key=lambda x: float(x[1]), reverse=True)
	return [item[0] for item in newList]
			
class WeightedNumberPicker:
	def __init__(self,lowerBounds,upperBounds,probabilities):
		self.lowerBounds=lowerBounds
		self.upperBounds=upperBounds
		self.probabilities=probabilities
	
	def getValue(self):
		currentPercent=0
		choice=random.randint(1,100)
		for i,prob in enumerate(self.probabilities):
			currentPercent+=prob*100
			if choice <= currentPercent:
				return random.randint(self.lowerBounds[i],self.upperBounds[i])

class InstrumentSection:
	def __init__(self,sectionName):
		self.sectionName=sectionName
		self.numOfBars=1
		self.numOfRepetitions=1
		self.creationLog=[]
		self.bars=[]
		self.beatsInBars=[]
		self.subDivisionsPerBeats=[]
		self.sectionEndTimeDelta=0
		self.formattedBars=[]
		self.msgs=[]
		self.chosenRhythm=[]
		self.chosenMelody=[]
		self.chosenVelocities=[]
		self.chosenDurations=[]

	def setNumOfRepetitions(self,numOfRepetitions):
		self.numOfRepetitions=numOfRepetitions
	
	def formatBars(self,noteOverlapFlag,chromaticFrequencyFactor,chromaticSizeRange,slideFrequencyFactor,slideSizeRange,nextNote):
	#making nextNote negative will mean it is ignored, useful for the end of a piece
		flattenBars=[]
		barBeats=0
		for i,bar in enumerate(self.bars):
			for j, nE in enumerate(bar):
				flattenBars.append([barBeats+bar[j][0],bar[j][1],bar[j][2],bar[j][3]])
			barBeats+=self.beatsInBars[i]
		if nextNote >= 0:
			flattenBars.append([barBeats,nextNote,0,0])
		
		if not noteOverlapFlag:
			for i,nE in enumerate(flattenBars[:-1]):
				if flattenBars[i][0]+flattenBars[i][3] >= flattenBars[i+1][0]:
					flattenBars[i][3]=flattenBars[i+1][0]-flattenBars[i][0]
		
		sliding=False
		for i,nE in enumerate(flattenBars[:-1]):
			if sliding:
				sliding=False
			else:
				noteDifference=abs(flattenBars[i+1][1]-flattenBars[i][1])
				travelDirection=1
				if flattenBars[i][1] > flattenBars[i+1][1]:
					travelDirection=-1
				if not noteDifference == 0:
					timeInterval=(flattenBars[i+1][0]-flattenBars[i][0])/noteDifference
				if noteDifference in chromaticSizeRange and chromaticFrequencyFactor.getValue() == 1:
					for j in range(0,noteDifference):
						self.formattedBars.append([0,flattenBars[i][0]+j*timeInterval,flattenBars[i][1]+travelDirection*j,flattenBars[i][2],flattenBars[i][3]/noteDifference])
				elif noteDifference in slideSizeRange and slideFrequencyFactor.getValue() == 1:
					self.formattedBars.append([1,flattenBars[i][0],flattenBars[i][1],flattenBars[i+1][2],travelDirection,noteDifference,timeInterval,flattenBars[i+1][3]])
					sliding=True
				else:
					self.formattedBars.append([0]+flattenBars[i])
			
		#[NoteOrSlideFlag(=0),BeatsSinceSectionStart,Note,Velocity,Duration]
		#[NoteOrSlideFlag(=1),BeatsSinceSectionStart,StartNote,Velocity,TravelDirection,SlideSize,TimePerSemi,FinishDuration]
						
		
	def convertToMidi(self,startTime,channelNum,instrumentSound,ticksPerInstrumentBeat):
		toConvert=[]
		time=startTime
		finalBarStart=0
		roundedTime=0
		for i,note in enumerate(self.formattedBars):
			if note[0] == 0:
				toConvert.append([1,int(ticksPerInstrumentBeat*note[1]),note[2],note[3]])
				toConvert.append([0,int(ticksPerInstrumentBeat*(note[1]+note[4])),note[2],0])
			else:
				toConvert.append([2,int(ticksPerInstrumentBeat*note[1])]+note[2:6]+[int(ticksPerInstrumentBeat*note[6]),int(ticksPerInstrumentBeat*note[7])])
		toConvert.sort(key=lambda x: x[1])
		
		prevTime=-startTime
		self.msgs=[mido.Message('program_change', program=instrumentSound, channel=channelNum, time=0)]
		for i,event in enumerate(toConvert):
			types=["note_off","note_on"]
			deltaTime=max(0,event[1]-prevTime)
			if event[0] == 2:
				self.msgs.append(mido.Message("note_on",time=deltaTime,channel=channelNum,note=event[2],velocity=event[3]))
				self.msgs.append(mido.Message("control_change",time=0,channel=channelNum,control=101,value=0))
				self.msgs.append(mido.Message("control_change",time=0,channel=channelNum,control=100,value=0))
				self.msgs.append(mido.Message("control_change",time=0,channel=channelNum,control=6,value=event[5]))
				for j in range(1,event[5]+1):
					self.msgs.append(mido.Message("pitchwheel",time=int(event[6]/2),channel=channelNum,pitch=round(event[4]*(j-0.5)*(8191.5/event[5])-0.5)))
					self.msgs.append(mido.Message("pitchwheel",time=int(event[6]/2),channel=channelNum,pitch=round(event[4]*j*(8191.5/event[5])-0.5)))
				self.msgs.append(mido.Message("note_off",time=event[7],channel=channelNum,note=event[2],velocity=0))
				self.msgs.append(mido.Message("pitchwheel",time=0,channel=channelNum,pitch=0))
				prevTime=event[1]+event[5]*event[6]+event[7]
			else:
				self.msgs.append(mido.Message(types[event[0]],time=deltaTime,channel=channelNum,note=event[2],velocity=event[3]))
				prevTime=event[1]
		self.sectionEndTimeDelta=sum(self.beatsInBars)*ticksPerInstrumentBeat-prevTime
		
	def saveMidi(self,saveName):
		mid = mido.MidiFile()
		track = mido.MidiTrack()
		mid.tracks.append(track)
		track.append(mido.Message('program_change', program=32, channel=0, time=0))
		
		for msg in self.msgs:
			track.append(msg)

		mid.save(saveName)

class MelodyInstrumentSection(InstrumentSection):
	
	def __init__(self,sectionName,minNote,maxNote,melodyNet):
		InstrumentSection.__init__(self,sectionName)
		self.melodyNet=melodyNet
		self.minNote=minNote
		self.maxNote=maxNote
		
	def addBar(self,chord,scale,beatsInBar,subDivisionsPerBeat,
	division1Of4Factor,division3Of4Factor,division2Or4Of4Factor,
	division1Of3Factor,division2Of3Factor,division3Of3Factor,
	noteSeed,melodyChoiceTypeFactor,melodyChoiceFactor,
	octaveRangeFactor,directionChangeFactor,velocityRange,durationRange):
		self.numOfBars+=1
		self.beatsInBars.append(beatsInBar)
		self.subDivisionsPerBeats.append(subDivisionsPerBeat)
		notesAdded=0
		self.chosenRhythm=[]
		self.chosenMelody=[]
		self.chosenVelocities=[]
		self.chosenDurations=[]
		
		for i in range(0,beatsInBar):
			self.chosenRhythm.append([])
			subDivisionRemainingCount=subDivisionsPerBeat
			if subDivisionRemainingCount == 1:
				self.chosenRhythm[-1].append(division1Of4Factor.getValue())
				notesAdded+=self.chosenRhythm[-1][-1]
			else:
				smallestChunkSize=2
				if subDivisionsPerBeat == 3 or subDivisionsPerBeat == 4:
					smallestChunkSize=subDivisionsPerBeat
				while subDivisionRemainingCount > 0:
					divisionChunkSize=random.randint(smallestChunkSize,min(4,subDivisionRemainingCount))
					if subDivisionRemainingCount == divisionChunkSize+1:
						if subDivisionRemainingCount == 5:
							divisionChunkSize=random.randint(2,3)
						else:
							divisionChunkSize=subDivisionRemainingCount
					subDivisionRemainingCount-=divisionChunkSize
					if divisionChunkSize == 4:
						self.chosenRhythm[-1].append(division1Of4Factor.getValue())
						notesAdded+=self.chosenRhythm[-1][-1]
						self.chosenRhythm[-1].append(division2Or4Of4Factor.getValue())
						notesAdded+=self.chosenRhythm[-1][-1]
						self.chosenRhythm[-1].append(division3Of4Factor.getValue())
						notesAdded+=self.chosenRhythm[-1][-1]
						self.chosenRhythm[-1].append(division2Or4Of4Factor.getValue())
						notesAdded+=self.chosenRhythm[-1][-1]
					elif divisionChunkSize == 3:
						self.chosenRhythm[-1].append(division1Of3Factor.getValue())
						notesAdded+=self.chosenRhythm[-1][-1]
						self.chosenRhythm[-1].append(division2Of3Factor.getValue())
						notesAdded+=self.chosenRhythm[-1][-1]
						self.chosenRhythm[-1].append(division3Of3Factor.getValue())
						notesAdded+=self.chosenRhythm[-1][-1]
					else:
						self.chosenRhythm[-1].append(division1Of4Factor.getValue())
						notesAdded+=self.chosenRhythm[-1][-1]
						self.chosenRhythm[-1].append(division3Of4Factor.getValue())
						notesAdded+=self.chosenRhythm[-1][-1]
				
		with torch.no_grad():
			self.melodyNet.hidden=self.melodyNet.initHidden()
			netInputSeq=torch.tensor(nnu.oneHot(noteSeed,12))
			predictedInterval=self.melodyNet(netInputSeq)
		lastNote=noteSeed[-1]
		direction=random.randint(0,1)
		for i in range(0,notesAdded):
			rankedNotes=getIndexesOfSortedList(predictedInterval.tolist()[-1])
			noteFound=False
			attempts=0
			chosenCategory=""
			while not noteFound and attempts < 100:
				maxIndex=melodyChoiceFactor.getValue()
				melodyChoiceType=melodyChoiceTypeFactor.getValue()
				if melodyChoiceType == 0:
					chosenNote=chord.rootNote+random.choice(scale)
					self.creationLog.append("Note From Chord Attempted")
					chosenCategory="== Note From Chord Accepted =="
				elif melodyChoiceType == 1:
					chosenNote=chord.rootNote+random.choice(scale)
					self.creationLog.append("Note From Scale Attempted")
					chosenCategory="== Note From Scale Accepted =="
				elif melodyChoiceType == 2:
					chosenNote=chord.rootNote+random.choice(list(range(0,12)))
					self.creationLog.append("Random Note Attempted")
					chosenCategory="== Random Note Accepted =="
				noteFound=((chosenNote-lastNote) % 12 in rankedNotes[:maxIndex])	
				attempts+=1
			if noteFound:
				self.creationLog.append(chosenCategory)
			else:
				chosenNote=lastNote+rankedNotes[0]
				self.creationLog.append("Couldn't find suitable note, adding least risk note")
			with torch.no_grad():
				netInputSeq=torch.tensor(nnu.oneHot([chosenNote-lastNote],12))
				predictedInterval=self.melodyNet(netInputSeq)
			chosenNote+=6*direction*(2*octaveRangeFactor.getValue()+1)-6
			while chosenNote < self.minNote:
				chosenNote+=12
			while chosenNote > self.maxNote:
				chosenNote-=12
			self.chosenMelody.append(chosenNote)
			lastNote=chosenNote
			if directionChangeFactor.getValue() == 1:
				direction=direction-2*direction
			
			self.chosenVelocities.append(random.randint(*velocityRange))
			self.chosenDurations.append(random.randint(*durationRange))
			
		bar=[]
		startTime=0
		noteNum=0
		subDivisionTimeDelta=1/subDivisionsPerBeat
		for rhythmDivision in self.chosenRhythm:
			for rhythmOnOff in rhythmDivision:
				if rhythmOnOff:
					bar.append([startTime,self.chosenMelody[noteNum],self.chosenVelocities[noteNum],self.chosenDurations[noteNum]*subDivisionTimeDelta])
					noteNum+=1
				startTime+=subDivisionTimeDelta		
		self.bars.append(bar)

class PercussionInstrumentSection(InstrumentSection):
	
	def __init__(self,sectionName,melodyNet):
		InstrumentSection.__init__(self,sectionName)
		self.melodyNet=melodyNet
		self.sectionSkeleton=[]
	
	def getModifiedSkeleton(self,skeleton,removeChance,addChance,addPercussionGroups,melodyChoiceFactor,velocity):
		newSkeleton=skeleton
		numDivs=skeleton[0]*skeleton[1]
		for i in range(numDivs):
			newSkeleton[2][i]=[elem for elem in skeleton[2][i] if random.randint(1,100) > removeChance]
		netInp=[]
		emptyCount=0
		while len(netInp) < self.melodyNet.layerSizes[0] and emptyCount < numDivs:
			emptyCount=0
			for subDiv in newSkeleton[2]:
				if subDiv:
					drumHits=[0]*self.melodyNet.layerSizes[-1]
					for beat in subDiv:
						drumHits[mtu.lookUpDrumType(beat[0])[0]]=1
					netInp.extend(drumHits)
				else:
					emptyCount+=1
		if emptyCount < numDivs:
			netInp=netInp[-self.melodyNet.layerSizes[0]:]
			for i in range(numDivs):
				if random.randint(1,100) <= addChance:
					self.melodyNet.forwardPropagation(netInp,[0]*self.melodyNet.layerSizes[-1])
					possibleGroups=[group for group in getIndexesOfSortedList(self.melodyNet.outputs[-1])[:melodyChoiceFactor.getValue()] if group in addPercussionGroups]
					addGroup=random.choice(addPercussionGroups)
					if possibleGroups:
						addGroup=random.choice(possibleGroups)
					hitNote=random.choice(mtu.lookUpDrumCategory(addGroup))[1]
					newSkeleton[2][i].append([hitNote,velocity])
				drumHits=[0]*self.melodyNet.layerSizes[-1]
				for beat in newSkeleton[2][i]:
					drumHits[mtu.lookUpDrumType(beat[0])[0]]=1
				netInp=netInp[self.melodyNet.layerSizes[-1]:]
				netInp.extend(drumHits)
		return newSkeleton
	
	def setSectionSkeleton(self,skeleton,removeChance,addChance,addPercussionGroups,melodyChoiceFactor,velocity):
		self.sectionSkeleton=self.getModifiedSkeleton(skeleton,removeChance,addChance,addPercussionGroups,melodyChoiceFactor,velocity)
	
	def addBar(self,removeChance,addChance,addPercussionGroups,melodyChoiceFactor,velocity,fillSkeleton=[]):
		self.numOfBars+=1
		newBeats=self.getModifiedSkeleton(self.sectionSkeleton,removeChance,addChance,addPercussionGroups,melodyChoiceFactor,velocity)
		if fillSkeleton:
			newBeats=self.getModifiedSkeleton(fillSkeleton,removeChance,addChance,addPercussionGroups,melodyChoiceFactor,velocity)
		beatsInBar=newBeats[0]
		subDivsPerBeat=newBeats[1]
		self.beatsInBars.append(beatsInBar)
		self.subDivisionsPerBeats.append(subDivsPerBeat)
		subDivTimeDelta=1/subDivsPerBeat
		bar=[]
		for i in range(beatsInBar):
			for j in range(subDivsPerBeat):
				hitStartTime=i+j*subDivTimeDelta
				for beat in newBeats[2][i*subDivsPerBeat+j]:
					bar.append([hitStartTime,*beat,1])
		self.bars.append(bar)

class AccompanyInstrumentSection(InstrumentSection):
	def __init__(self,sectionName):
		InstrumentSection.__init__(self,sectionName)
	
	def addBar(self,chord,beatsInBar,subDivisionsPerBeat,velocity):
		self.numOfBars+=1
		self.beatsInBars.append(beatsInBar)
		self.subDivisionsPerBeats.append(subDivisionsPerBeat)
		bar=[]
		for note in chord.chordIntervals:
			bar.append([0,note+chord.rootNote,velocity,beatsInBar])
		self.bars.append(bar)
				
def combineMidi(midiTracks,saveName):
	mid = mido.MidiFile()
	for midiTrack in midiTracks:
		track = mido.MidiTrack()
		mid.tracks.append(track)
		for msg in midiTrack:
			track.append(msg)
	mid.save(saveName)

cuda=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

melodyModel=nnu.LSTM_LogSoftMax_RNN(12,64,12,4,cuda)
checkpoint = torch.load("trained networks/6174.pt",map_location='cpu')
melodyModel.load_state_dict(checkpoint['model_state_dict'])
melodyModel.eval()

drumsNet=nnu.VanillaNeuralNet([44,44,44,22,11,11])
drumsNet.readFromcsv('drumBeatEveryInstrument_Good.csv')