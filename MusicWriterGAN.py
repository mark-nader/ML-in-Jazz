import mido
import torch
import NeuralNetUtils as nnu
import MusicTheoryUtils as mtu
import random
import time

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
	
def percentPicker(numOfNotes):
	if numOfNotes == 1 or numOfNotes == 2:
		return 1
	elif numOfNotes == 3 or numOfNotes == 4:
		return (numOfNotes-1)/numOfNotes
	else:
		return 0.8

class InstrumentSection:
	def __init__(self,sectionName,melodyNet):
		self.sectionName=sectionName
		self.noteEvents=[]
		self.numOfBars=0
		self.beatsInBars=[]
		self.noteCount=0
		self.noteTimeIntervals=[]
		self.msgs=[]
		self.melodyNet=melodyNet
		self.sectionEndTimeDelta=0
	
class MelodyInstrumentSection(InstrumentSection):
	
	def __init__(self,sectionName,minNote,maxNote,melodyNet):
		InstrumentSection.__init__(self,sectionName,melodyNet)
		self.formattedEvents=[]
		self.minNote=minNote
		self.maxNote=maxNote
		
	def addProgressionBars(self,chords,scales,beatsInBar,subDivsPerBeat,
	div1Of4Factor,div3Of4Factor,div2Or4Of4Factor,
	div1Of3Factor,div2Of3Factor,div3Of3Factor,
	octaveRangeFactor,directionChangeFactor,
	durationRange,velocityRange):
		numOfBars=len(chords)
		self.numOfBars+=numOfBars
		self.beatsInBars.append(beatsInBar)
		noteCount=0
		notesPerBar=[]
		chosenRhythm=[]
		chosenMelody=[]
		chosenDurations=[]
		chosenVelocities=[]
		
		currentTime=0
		def addNoteToRhythmChance(noteCount,currentTime,factor,subDivs):
			currentTime+=1/subDivs
			if factor.getValue() == 1:
				chosenRhythm.append(currentTime)
				noteCount+=1
			return noteCount, currentTime
		
		for i in range(numOfBars):
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
							noteCount, currentTime =addNoteToRhythmChance(noteCount,currentTime,div1Of4Factor,subDivsPerBeat)
							noteCount, currentTime =addNoteToRhythmChance(noteCount,currentTime,div2Or4Of4Factor,subDivsPerBeat)
							noteCount, currentTime =addNoteToRhythmChance(noteCount,currentTime,div3Of4Factor,subDivsPerBeat)
							noteCount, currentTime =addNoteToRhythmChance(noteCount,currentTime,div2Or4Of4Factor,subDivsPerBeat)
						elif divChunkSize == 3:
							noteCount, currentTime =addNoteToRhythmChance(noteCount,currentTime,div1Of3Factor,subDivsPerBeat)
							noteCount, currentTime =addNoteToRhythmChance(noteCount,currentTime,div2Of3Factor,subDivsPerBeat)
							noteCount, currentTime =addNoteToRhythmChance(noteCount,currentTime,div3Of3Factor,subDivsPerBeat)
						else:
							noteCount, currentTime =addNoteToRhythmChance(noteCount,currentTime,div1Of4Factor,subDivsPerBeat)
							noteCount, currentTime =addNoteToRhythmChance(noteCount,currentTime,div3Of4Factor,subDivsPerBeat)
			notesPerBar.append(noteCount-prevBarNoteCount)
		self.noteCount+=noteCount
		if noteCount > 0:
			scaleNotes=[]
			for i in range(numOfBars):
				print("notes in bar {}: {}".format(i,notesPerBar[i]))
				scaleNotes.append([(interval+chords[i].rootNote) % 12 for interval in scales[i]])
			blendNoteDepth=0
			potentialNotes=[]
			for i in range(numOfBars):
				barNotesDecided=0
				attemptsPerBlendDepth=25
				suitableNoteChoice=False
				while not suitableNoteChoice:
					while barNotesDecided < notesPerBar[i]:
						print("finding possible notes for bar {}".format(i))
						attempts=0
						blendSearching=True
						while blendSearching:
							if blendNoteDepth > 0 and attempts == attemptsPerBlendDepth:
								attempts=0
								blendNoteDepth-=1
							attempts+=1
							genOut=self.melodyNet(torch.randn(1,self.melodyNet.noiseDim))
							generatedNotes=[]
							currentNote=chords[i].rootNote
							for oneHotNote in genOut:
								currentNote=(currentNote+oneHotNote[0].tolist().index(1)) % 12
								generatedNotes.append(currentNote)
							print("generatedNotes: {}".format(generatedNotes))
							print("generatedNotes[:blendNoteDepth] {}".format(generatedNotes[:blendNoteDepth]))
							print("potentialNotes {}".format(potentialNotes))
							print("potentialNotes[-blendNoteDepth:] {}".format(potentialNotes[-blendNoteDepth:]))
							input()
							blendSearching=not (generatedNotes[:blendNoteDepth] == potentialNotes[-blendNoteDepth:])
						potentialNotes.extend(generatedNotes[blendNoteDepth:])
						print("potentialNotes {}".format(potentialNotes))
						barNotesDecided=len(potentialNotes)	
						print("barNotesDecided {}".format(barNotesDecided))
					suitableNotes=0
					for note in potentialNotes[:notesPerBar[i]]:
						if note in scaleNotes[i]:
							suitableNotes+=1
					suitableNoteChoice=(suitableNotes/notesPerBar[i] >= percentPicker(notesPerBar[i]))
					print("suitableNoteChoice {}".format(suitableNoteChoice))
					if not suitableNoteChoice:
						barNotesDecided=0
						potentialNotes=[]
				chosenMelody.extend(potentialNotes[:notesPerBar[i]])
				print("chosenMelody {}".format(chosenMelody))
				potentialNotes=potentialNotes[notesPerBar[i]:]
				blendNoteDepth=len(potentialNotes)
	
			print("NOTES DECIDED FOR PROGRESSION")
	
			direction=1-2*random.randint(0,1)
			for i in range(noteCount):
				if directionChangeFactor.getValue() == 1:
					direction=direction-2*direction
			chosenMelody[i]+=octaveRangeFactor.getValue()*direction*12
			while chosenMelody[i] < self.minNote:
				chosenMelody[i]+=12
			while chosenMelody[i] > self.maxNote:
				chosenMelody[i]-=12		
			
			for i in range(noteCount):
				chosenVelocities.append(random.randint(*velocityRange))
				chosenDurations.append(random.randint(*durationRange)/subDivsPerBeat)
				
		for i in range(noteCount):
			noteEvent=[chosenRhythm[i],chosenMelody[i],chosenVelocities[i],chosenDurations[i]]
			self.noteEvents.extend(noteEvent)
			
	def formatBars(self,noteOverlapFlag,chromaticFrequencyFactor,chromaticSizeRange,slideFrequencyFactor,slideSizeRange,nextNote):
	#making nextNote negative will mean it is ignored, useful for the end of a piece
	#next note does not have to be the actual note to follow the melody being made here but gives the last note an
	#opportunity to do chromatic steps or slide
		eventsToFormat=self.noteEvents
		noteCount=self.noteCount
		if not noteOverlapFlag:
			for i in range(noteCount-1):
				if eventsToFormat[i][0]+eventsToFormat[i][3] > eventsToFormat[i+1][0]:
					eventsToFormat[i][3]=eventsToFormat[i+1][0]
			if eventsToFormat[noteCount-1][0]+eventsToFormat[noteCount-1][3] > sum(self.beatsInBars):
				eventsToFormat[noteCount-1][3]=sum(self.beatsInBars)
			
		if nextNote > 0:
			eventsToFormat.append([self.numOfBars,nextNote,0,0])
			noteCount-=1
		sliding=False
		for i in range(noteCount):
			if sliding:
				sliding=False
			else:
				noteDifference=abs(eventsToFormat[i+1][1]-eventsToFormat[i][1])
				travelDirection=1
				if eventsToFormat[i][1] > eventsToFormat[i+1][1]:
					travelDirection=-1
				if not noteDifference == 0:
					timeInterval=(eventsToFormat[i+1][0]-eventsToFormat[i][0])/noteDifference
				if noteDifference in chromaticSizeRange and chromaticFrequencyFactor.getValue() == 1:
					for j in range(0,noteDifference):
						self.formattedEvents.append([0,eventsToFormat[i][0]+j*timeInterval,eventsToFormat[i][1]+travelDirection*j,eventsToFormat[i][2],eventsToFormat[i][3]/noteDifference])
				elif noteDifference in slideSizeRange and slideFrequencyFactor.getValue() == 1:
					self.formattedEvents.append([1,eventsToFormat[i][0],eventsToFormat[i][1],eventsToFormat[i+1][2],travelDirection,noteDifference,timeInterval,eventsToFormat[i+1][3]])
					sliding=True
				else:
					self.formattedEvents.append([0]+eventsToFormat[i])
			
		#[SlideFlag(=0),BeatsSinceSectionStart,Note,Velocity,Duration]
		#[SlideFlag(=1),BeatsSinceSectionStart,StartNote,Velocity,TravelDirection,SlideSize,TimePerSemi,FinishDuration]
						
		
	def convertToMidi(self,startTime,channelNum,ticksPerBeat):
		toConvert=[]
		time=startTime
		finalBarStart=0
		roundedTime=0
		for i,note in enumerate(self.formattedEvents):
			if note[0] == 0:
				toConvert.append([1,int(ticksPerBeat*note[1]),note[2],note[3]])
				toConvert.append([0,int(ticksPerBeat*(note[1]+note[4])),note[2],0])
			else:
				toConvert.append([2,int(ticksPerBeat*note[1])]+note[2:6]+[int(ticksPerBeat*note[6]),int(ticksPerBeat*note[7])])
		
		prevTime=-startTime
		self.msgs=[]
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
		self.sectionEndTimeDelta=sum(self.beatsInBars)*ticksPerBeat-prevTime
		
	def saveMidi(self):
		mid = mido.MidiFile()
		track = mido.MidiTrack()
		mid.tracks.append(track)
		track.append(mido.Message('program_change', program=32, channel=0, time=0))
		
		for msg in self.msgs:
			track.append(msg)

		mid.save('GANmelodytest.mid')

cuda=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
melodyModel=nnu.NoteGenerator(5,144,4,7,0.1,cuda)
melodyModel.load_state_dict(torch.load("trained networks/5305.pt",map_location='cpu'))
melodyModel.eval()

testBassSection=MelodyInstrumentSection("First Test",28,52,melodyModel)

scaleChoiceFactor=WeightedNumberPicker([0,0],[3,5],[0.65,0.35])
beatsInBar=4
subDivsPerBeat=3
div1O4=WeightedNumberPicker([0,1],[0,1],[0,1])
div3O4=WeightedNumberPicker([0,1],[0,1],[1,0])
div24O4=WeightedNumberPicker([0,1],[0,1],[1,0])
div1O3=WeightedNumberPicker([0,1],[0,1],[0.10,0.90])
div2O3=WeightedNumberPicker([0,1],[0,1],[0.90,0.10])
div3O3=WeightedNumberPicker([0,1],[0,1],[0.80,0.20])
velocitySeed=[]
durationSeed=[]
oRF=WeightedNumberPicker([0,1],[0,1],[0.85,0.15])#positive only now because of directionChangeFactor
dCF=WeightedNumberPicker([0,1],[0,1],[0.65,0.35])
durationRange=[1,6]
velocityRange=[64,127]

noteOverlapFlag=False
cFF=WeightedNumberPicker([0,1],[0,1],[0.7,0.3])
cSR=list(range(3,5))
sFF=WeightedNumberPicker([0,1],[0,1],[0.9,0.1])
sSR=list(range(2,5))
nextNote=62

Dm7=mtu.Chord(62,11)
G7=mtu.Chord(67,9)
CMaj7=mtu.Chord(60,10)

chords=[Dm7,G7,CMaj7,CMaj7]

scales=[]
scales.append(Dm7.rankedScalesWithScore[scaleChoiceFactor.getValue()][2])
scales.append(G7.rankedScalesWithScore[scaleChoiceFactor.getValue()][2])
scales.append(CMaj7.rankedScalesWithScore[scaleChoiceFactor.getValue()][2])
scales.append(CMaj7.rankedScalesWithScore[scaleChoiceFactor.getValue()][2])

testBassSection.addProgressionBars(chords,scales,beatsInBar,subDivsPerBeat,div1O4,div3O4,div24O4,div1O3,div2O3,div3O3,
oRF,dCF,durationRange,velocityRange)

testBassSection.formatBars(noteOverlapFlag,cFF,cSR,sFF,sSR,nextNote)

testBassSection.convertToMidi(480,0,480)

testBassSection.saveMidi()

