import os
import mido

def getSongList(folderName):
	return [name for name in os.listdir("{}/".format(folderName))]

def getTicksPerBeat(songName):
	mid = mido.MidiFile(songName)
	return mid.ticks_per_beat

# [Start_Time, Note, Velocity, Duration, Pitch_Bend_Flag(=0)]
# [Start_Time, Semitone_Bend, NIL, NIL, Pitch_Bend_Flag(=1)]
def getInstrumentFromSong(songName,programLowerBound,programUpperBound):
	mid = mido.MidiFile(songName)
	completeInsrumentMelodies=[]
	activeChannels=[False]*16
	activeInstrumentMelodies=[[] for _ in range(0,16)]
	activeInstrumentMelodyLengths=[0]*16
	releaseIndexes=[0]*16
	currentTime=0
	
	def releaseNote(channel,note,releaseIndex):
		allNotesReleased=True
		for i in range(releaseIndex,activeInstrumentMelodyLengths[channel]):
			if activeInstrumentMelodies[channel][i][3] == -1:
				allNotesReleased=False
				if activeInstrumentMelodies[channel][i][1] == note:
					activeInstrumentMelodies[channel][i][3]=currentTime-activeInstrumentMelodies[channel][i][0]
			elif allNotesReleased:
				releaseIndex=i
		return releaseIndex
	
	for track in mid.tracks:
		activeChannels=[False]*16
		activeInstrumentMelodies=[[] for _ in range(0,16)]
		activeInstrumentMelodyLengths=[0]*16
		releaseIndexes=[0]*16
		controlChangeInputs=[[] for _ in range(0,16)]
		pitchBendRanges=[2]*16 #2 semitones up, 2 down
		currentTime=0
		for msg in track:
			currentTime+=msg.time
			if msg.type == "program_change":
				channel=msg.channel
				activeChannels[channel]=(msg.program <= programUpperBound and msg.program >= programLowerBound and not channel == 9)
				if activeInstrumentMelodies[channel]:
					completeInsrumentMelodies.append(activeInstrumentMelodies[channel])
					activeInstrumentMelodies[channel]=[]
					activeInstrumentMelodyLengths[channel]=0
					releaseIndexes[channel]=0
			elif msg.type == "note_on":
				channel=msg.channel
				if activeChannels[channel]:
					note=msg.note
					velocity=msg.velocity
					if velocity == 0:
						releaseIndexes[channel]=releaseNote(channel,note,releaseIndexes[channel])
					else:				
						activeInstrumentMelodies[channel].append([currentTime,note,velocity,-1,0])
						activeInstrumentMelodyLengths[channel]+=1
			elif msg.type == "note_off":
				channel=msg.channel
				if activeChannels[channel]:
					releaseIndexes[channel]=releaseNote(channel,msg.note,releaseIndexes[channel])
			elif msg.type == "pitchwheel":
				channel=msg.channel
				if activeChannels[channel]:
					bend=int(round(msg.pitch*pitchBendRanges[channel]/8192))
					activeInstrumentMelodies[channel].append([currentTime,bend,0,0,1])
					activeInstrumentMelodyLengths[channel]+=1
					# 8192=0   [currentTime,note,velocity,-1,0]
					# 0 to 16,383  or -8,192 to 8,192
					# control_change channel=1 control=101 value=0 time=0
					# control_change channel=1 control=100 value=0 time=0
					# control_change channel=1 control=6 value=12 time=0   (value is the bend range)
			elif msg.type == "control_change":
				channel=msg.channel
				if activeChannels[channel]:
					control=msg.control
					value=msg.value
					if len(controlChangeInputs[channel]) == 2:
						if control == 6:
							pitchBendRanges[channel]=value
						controlChangeInputs[channel]=[]
					elif (control == 100 or control == 101) and value == 0:
						if not control in controlChangeInputs[channel]:
							controlChangeInputs[channel].append(control)
		for channel in range(0,16):
			releaseIndexes[channel]=releaseNote(channel,any,releaseIndexes[channel])
		for aim in activeInstrumentMelodies:
			if aim:
				completeInsrumentMelodies.append(aim)
	return completeInsrumentMelodies

def getBassNoteIntervalsFromBassline(bassline):
	basslineLength=len(bassline)
	noteBuffer=[]
	pitchBend=0
	prevNote=0
	prevNoteBend=0
	prevNoteEndTime=0
	i=0
	firstNoteFound=False
	while i < basslineLength and not firstNoteFound:
		bassNote=bassline[i]
		if bassNote[4] == 0:
			prevNote=bassNote[1]
			prevNoteBend=pitchBend
			prevNoteEndTime=bassNote[0]+bassNote[3]
			firstNoteFound=True
		else:
			pitchBend=bassNote[1]
			i+=1
	bendingNote=False
	for j in range(i,basslineLength):
		bassNote=bassline[j]
		if bassNote[4] == 0:
			if bendingNote:
				noteBuffer.append(pitchBend-prevNoteBend)
				noteBuffer.append(bassNote[1]-prevNote)
			else:
				noteBuffer.append(bassNote[1]+pitchBend-prevNote-prevNoteBend)
			prevNote=bassNote[1]
			prevNoteBend=pitchBend
			prevNoteEndTime=bassNote[0]+bassNote[3]
			bendingNote=False
		else:
			if bassNote[0] < prevNoteEndTime:
				pitchBend=bassNote[1]
				bendingNote=True
			elif bendingNote:
				noteBuffer.append(pitchBend-prevNoteBend)
				prevNoteBend=pitchBend
				pitchBend=bassNote[1]
				bendingNote=False
			else:
				pitchBend=bassNote[1]
	return noteBuffer
	
# [Start_Time, Note(drumType), Velocity, Duration]
def getDrumsFromSong(songName):
	mid = mido.MidiFile(songName)
	currentTime=0
	drumsActive=False
	allQuietIndex=0
	currentDrumTrack=[]
	currentDrumTrackLength=0
	completeDrumTracks=[]
	
	def drumFade(note,allQuietIndex,endTrackFlag):
		allQuiet=True
		for i in range(allQuietIndex,currentDrumTrackLength):
			if currentDrumTrack[i][3] == -1:
				allQuiet=False
				if currentDrumTrack[i][1] == note or endTrackFlag:
					currentDrumTrack[i][3]=currentTime-currentDrumTrack[i][0]
			elif allQuiet:
				allQuietIndex=i
		return allQuietIndex
	
	for track in mid.tracks:
		for msg in track:
			currentTime+=msg.time
			if not msg.is_meta and not msg.type == "sysex":
				if msg.channel == 9:
					if msg.type == "note_on":
						if msg.velocity == 0:
							allQuietIndex=drumFade(msg.note,allQuietIndex,False)
						else:
							currentDrumTrack.append([currentTime,msg.note,msg.velocity,-1])
							currentDrumTrackLength+=1
					elif msg.type == "note_off":
						allQuietIndex=drumFade(msg.note,allQuietIndex,False)
		if currentDrumTrack:
			drumFade(0,allQuietIndex,True)
			completeDrumTracks.extend(currentDrumTrack)
			currentDrumTrack=[]
			currentDrumTrackLength=0
			allQuietIndex=0
		drumsActive=False
		currentTime=0
	completeDrumTracks.sort(key=lambda x:(x[0]))
	return completeDrumTracks
