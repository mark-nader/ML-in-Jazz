import mido
import torch
import NeuralNetUtils as nnu
import MusicTheoryUtils as mtu
import random

ticksPerBeat=480
			
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
	
def percentPicker(numOfNotes):
	if numOfNotes == 1 or numOfNotes == 2:
		return 1
	elif numOfNotes == 3:
		return 2/3
	elif numOfNotes == 4 or numOfNotes == 5:
		return (numOfNotes-2+random.randint(0,1))/numOfNotes
	elif numOfNotes == 6:
		return 4/6
	else:
		return 0.70

class InstrumentSection:
	def __init__(self,sectionName):
		self.sectionName=sectionName
		self.numOfBars=0
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
	
class MelodyInstrumentSection(InstrumentSection):
	
	def __init__(self,sectionName,minNote,maxNote,melodyNet):
		InstrumentSection.__init__(self,sectionName)
		self.melodyNet=melodyNet
		self.minNote=minNote
		self.maxNote=maxNote
		
	def addProgressionBars(self,chords,scales,beatsInBar,subDivsPerBeat,
	div1Of4Factor,div3Of4Factor,div2Or4Of4Factor,
	div1Of3Factor,div2Of3Factor,div3Of3Factor,
	octaveRangeFactor,directionChangeFactor,
	durationRange,velocityRange):
		numOfBars=len(chords)
		self.beatsInBars.extend([beatsInBar]*numOfBars)
		noteCount=0
		notesPerBar=[]
		chosenRhythm=[]
		chosenMelody=[]
		chosenDurations=[]
		chosenVelocities=[]
		
		def addNoteToRhythmChance(noteCount,currentTime,factor,subDivs):
			if factor.getValue() == 1:
				chosenRhythm.append(currentTime)
				noteCount+=1
			currentTime+=1/subDivs
			return noteCount, currentTime
		
		for i in range(numOfBars):
			currentTime=0
			prevBarNoteCount=noteCount
			for j in range(beatsInBar):
				subDivisionRemainingCount=subDivsPerBeat
				if subDivisionRemainingCount == 1:
					noteCount, currentTime =addNoteToRhythmChance(noteCount,currentTime,div1Of4Factor,subDivsPerBeat)
				else:
					smallestChunkSize=2
					if subDivsPerBeat == 3 or subDivsPerBeat == 4:
						smallestChunkSize=subDivsPerBeat
					while subDivisionRemainingCount > 0:
						divChunkSize=random.randint(smallestChunkSize,min(4,subDivisionRemainingCount))
						if subDivisionRemainingCount == divChunkSize+1:
							if subDivisionRemainingCount == 5:
								divChunkSize=random.randint(2,3)
							else:
								divChunkSize=subDivisionRemainingCount
						subDivisionRemainingCount-=divChunkSize
						if divChunkSize == 4:
							noteCount, currentTime = addNoteToRhythmChance(noteCount,currentTime,div1Of4Factor,subDivsPerBeat)
							noteCount, currentTime = addNoteToRhythmChance(noteCount,currentTime,div2Or4Of4Factor,subDivsPerBeat)
							noteCount, currentTime = addNoteToRhythmChance(noteCount,currentTime,div3Of4Factor,subDivsPerBeat)
							noteCount, currentTime = addNoteToRhythmChance(noteCount,currentTime,div2Or4Of4Factor,subDivsPerBeat)
						elif divChunkSize == 3:
							noteCount, currentTime = addNoteToRhythmChance(noteCount,currentTime,div1Of3Factor,subDivsPerBeat)
							noteCount, currentTime = addNoteToRhythmChance(noteCount,currentTime,div2Of3Factor,subDivsPerBeat)
							noteCount, currentTime = addNoteToRhythmChance(noteCount,currentTime,div3Of3Factor,subDivsPerBeat)
						else:
							noteCount, currentTime = addNoteToRhythmChance(noteCount,currentTime,div1Of4Factor,subDivsPerBeat)
							noteCount, currentTime = addNoteToRhythmChance(noteCount,currentTime,div3Of4Factor,subDivsPerBeat)
			notesPerBar.append(noteCount-prevBarNoteCount)
		if noteCount > 0:
			scaleNotes=[]
			for i in range(numOfBars):
				scaleNotes.append([(interval+chords[i].rootNote) % 12 for interval in scales[i]])
			blendNoteDepth=0
			potentialIntervals=[]
			for i in range(numOfBars):
				# print("writing bar {}".format(i))
				attemptsPerBlendDepth=25
				suitableNoteChoice=False
				while not suitableNoteChoice:
					while len(potentialIntervals) < notesPerBar[i]:
						attempts=0
						blendSearching=True
						while blendSearching:
							if blendNoteDepth > 0 and attempts == attemptsPerBlendDepth:
								attempts=0
								blendNoteDepth-=1
							attempts+=1
							genOut=self.melodyNet(torch.randn(1,self.melodyNet.noiseDim))
							generatedNotes=[]
							for oneHotNote in genOut:
								generatedNotes.append(oneHotNote[0].tolist().index(1))
							# print("generatedNotes: {}".format(generatedNotes))
							# print("generatedNotes[:blendNoteDepth] {}".format(generatedNotes[:blendNoteDepth]))
							# print("potentialIntervals {}".format(potentialIntervals))
							# print("potentialIntervals[-blendNoteDepth:] {}".format(potentialIntervals[-blendNoteDepth:]))
							# input()
							blendSearching=not (blendNoteDepth == 0 or generatedNotes[:blendNoteDepth] == potentialIntervals[-blendNoteDepth:])
						potentialIntervals.extend(generatedNotes[blendNoteDepth:])
						# print("potentialIntervals {}".format(potentialIntervals))
					suitableNotes=0
					potentialNotes=[]
					currentNote=chords[0].rootNote
					if chosenMelody:
						currentNote=chosenMelody[-1]
					direction=random.randint(0,1)
					# print(potentialIntervals[:notesPerBar[i]])
					for interval in potentialIntervals[:notesPerBar[i]]:
						if (currentNote+interval) % 12 in scaleNotes[i]:
							suitableNotes+=1
						currentNote+=interval+6*direction*(2*octaveRangeFactor.getValue()+1)-6
						while currentNote < self.minNote:
							currentNote+=12
						while currentNote > self.maxNote:
							currentNote-=12
						# print(currentNote)
						# input()
						potentialNotes.append(currentNote)
						if directionChangeFactor.getValue() == 1:
							direction=direction-2*direction
					# print("suitable notes {}".format(suitableNotes))
					if suitableNotes == 0:
						suitableNoteChoice=(notesPerBar[i] == 0)
					else:
						suitableNoteChoice=(suitableNotes/notesPerBar[i] >= percentPicker(notesPerBar[i]))
					# print("suitableNoteChoice {}".format(suitableNoteChoice))
					if not suitableNoteChoice:
						potentialIntervals=[]
						blendNoteDepth=0
						# print("potentialIntervals {}".format(potentialIntervals))
						# input()
				chosenMelody.extend(potentialNotes[:])
				# print("chosenMelody {}".format(chosenMelody))
				potentialIntervals=potentialIntervals[notesPerBar[i]:]
				blendNoteDepth=len(potentialIntervals)	
				
			
			for i in range(noteCount):
				chosenVelocities.append(random.randint(*velocityRange))
				chosenDurations.append(random.randint(*durationRange)/subDivsPerBeat)
		
		prevNoteCount=0
		for nPB in notesPerBar:
			bar=[]
			for i in range(prevNoteCount,prevNoteCount+nPB):
				bar.append([chosenRhythm[i],chosenMelody[i],chosenVelocities[i],chosenDurations[i]])
			prevNoteCount+=nPB
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
					possibleGroups=[group for group in nnu.getIndexesOfSortedList(self.melodyNet.outputs[-1])[:melodyChoiceFactor.getValue()] if group in addPercussionGroups]
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

melodyModel_5304 = nnu.NoteGenerator(5,144,4,8,0.1,cuda)
melodyModel_5304.load_state_dict(torch.load("trained networks/5304.pt",map_location='cpu'))
melodyModel_5304.eval()

melodyModel_5305 = nnu.NoteGenerator(5,144,4,7,0.1,cuda)
melodyModel_5305.load_state_dict(torch.load("trained networks/5305.pt",map_location='cpu'))
melodyModel_5305.eval()

melodyModel_5306 = nnu.NoteGenerator(5,144,4,9,0.1,cuda)
melodyModel_5306.load_state_dict(torch.load("trained networks/5306.pt",map_location='cpu'))
melodyModel_5306.eval()

melodyModel_6276=nnu.NoteGenerator(5,144,1,8,0.1,cuda)
checkpoint = torch.load("trained networks/6276.pt",map_location='cpu')
melodyModel_6276.load_state_dict(checkpoint['model_state_dict'])
melodyModel_6276.eval()

drumsNet=nnu.VanillaNeuralNet([44,44,44,22,11,11])
drumsNet.readFromcsv('trained networks/d3.csv')


