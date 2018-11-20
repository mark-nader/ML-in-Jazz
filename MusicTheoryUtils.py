import numpy as np

ScaleBank=[
#Name								Intervals
["Ionian Mode (Major Scale)",		[0,2,4,5,7,9,11]],	#0
["Dorian Mode",						[0,2,3,5,7,9,10]],	#1
["Phrygian Mode",					[0,1,3,5,7,8,10]],	#2
["Lydian Mode",						[0,2,4,6,7,9,11]],	#3
["Mixolydian Mode",					[0,2,4,5,7,9,10]],	#4
["Aeolian Mode (Minor Scale)",		[0,2,3,5,7,8,10]],	#5
["Locrian Mode",					[0,1,3,5,6,8,10]],	#6
["Minor pentatonic",				[0,3,5,7,10]],		#7
["Major pentatonic",				[0,2,4,7,9]],		#8
["Suspended (Egyptian) pentatonic",	[0,2,5,7,10]],		#9
["Blues Minor pentatonic",			[0,3,5,7,10]],		#10
["Blues Major pentatonic",			[0,2,5,7,9]],		#11
["Blues",							[0,3,5,6,7,10]]		#12
]
	
ChordBank=[
#Name			Intervals
["Major",		[0,4,7]],				#r 3 5				| 0
["Minor",		[0,3,7]],				#r b3 5				| 1
["Diminished",	[0,3,6]],				#r b3 b5			| 2
["Sus2",		[0,2,7]],				#r 2 5				| 3
["Sus4",		[0,5,7]],				#r 4 5				| 4
["Augmented",	[0,4,8]],				#r 3 #5				| 5
["5",			[0,7]],					#r 5				| 6
["6",			[0,4,7,9]],				#r 3 5 6			| 7
["Minor6",		[0,3,7,9]],				#r b3 5 6			| 8
["7",			[0,4,7,10]],			#r 3 5 m7			| 9
["Major7",		[0,4,7,11]],			#r 3 5 7			| 10
["Minor7",		[0,3,7,10]],			#r b3 5 m7			| 11
["7b5",			[0,4,6,10]],			#r 3 b5 m7			| 12
["7#5",			[0,4,8,10]],			#r 3 #5 m7			| 13
["9",			[0,4,7,10,14]],			#r 3 5 m7 9			| 14
["Major9",		[0,4,7,10,14]],			#r 3 5 7 9			| 15
["Minor9",		[0,3,7,10,14]],			#r b3 5 m7 9		| 16
["11",			[0,4,7,10,14,17]],		#r 3 5 m7 9 11		| 17
["Minor11",		[0,3,7,10,14,17]],		#r b3 5 m7 9 11		| 18
["13",			[0,4,7,10,14,17,21]],	#r 3 5 m7 9 11 13	| 19
["Add2",		[0,2,4,7]],				#r 2 3 5			| 20
["Add4",		[0,4,5,7]],				#r 3 4 5			| 21
["Add9",		[0,4,7,14]],			#r 3 5 9			| 22
["Add11",		[0,4,7,17]],			#r 3 5 11			| 23
["6Add9",		[0,4,7,9,14]],			#r 3 5 6 9			| 24
]

class Chord:
	def __init__(self,rootNote,chordBankIndex):
		self.rootNote=rootNote
		self.chordName=ChordBank[chordBankIndex][0]
		self.chordIntervals=(np.array(ChordBank[chordBankIndex][1]) % 12).tolist()
		self.rankedScalesWithScore=[]
		for scale in ScaleBank:
			for scaleRoot in range(0,12):
				keyedScale=((np.array(scale[1])+scaleRoot) % 12).tolist()
				foundNotes=0
				numNotesInScale=len(scale[1])
				for interval in self.chordIntervals:
					if interval in keyedScale:
						foundNotes+=1
				self.rankedScalesWithScore.append([scaleRoot,scale[0],keyedScale,foundNotes,foundNotes/numNotesInScale])
		self.rankedScalesWithScore.sort(key=lambda x:(x[4],x[3]),reverse=True)

DrumSoundBank=[
#category number | midi program number | drum name
[0,35,"Bass Drum 2"],
[0,36,"Bass Drum 1"],

[1,38 ,"Snare Drum 1"],
[1,40,"Snare Drum 2"],

[2,43,"Low Tom 1"],
[2,41,"Low Tom 2"],
[2,47,"Mid Tom 1"],
[2,45,"Mid Tom 2"],
[2,50,"High Tom 1"],
[2,48,"High Tom 2"],

[3,46,"Open Hi-hat"],
[3,42,"Closed Hi-hat"],
[3,44,"Pedal Hi-hat"],

[4,49,"Crash Cymbal 1"],
[4,57,"Crash Cymbal 2"],
[4,51,"Ride Cymbal 1"],
[4,59,"Ride Cymbal 2"],
[4,53,"Ride Bell"],
[4,55,"Splash Cymbal"],
[4,52,"Chinese Cymbal"],

[5,65,"High Timbale"],
[5,66,"Low Timbale"],

[6,37,"Side Stick/Rimshot"],
[6,75,"Claves"],
[6,56,"Cowbell"],
[6,54,"Tambourine"],
[6,39,"Hand Clap"],
[6,58,"Vibra Slap"],

[7,60,"High Bongo"],
[7,61,"Low Bongo"],
[7,62,"Mute High Conga"],
[7,63,"Open High Conga"],
[7,64,"Low Conga"],
[7,78,"Mute Cuíca"],
[7,79,"Open Cuíca"],

[8,67,"High Agogô"],
[8,68,"Low Agogô"],
[8,73,"Short Güiro"],
[8,74,"Long Güiro"],
[8,69,"Cabasa"],
[8,70,"Maracas"],

[9,76,"High Wood Block"],
[9,77,"Low Wood Block"],
[9,80,"Mute Triangle"],
[9,81,"Open Triangle"],

[10,71,"Short Whistle"],
[10,72,"Long Whistle"]]
	
DrumSkeletons=[
#number of beats | number of subdivisions per beat | pattern
[3,[3,3,3],[[[[59,127,16],[35,127,16],[40,127,16]],[],[[35,127,16],[40,127,16]]],
			[[[59,127,16],[44,127,16]],[],[[59,127,16]]],
			[[[59,127,16],[44,127,16],[35,127,16],[40,127,16]],[],[]]]],
			
[3,[3,3,3],[[[[59,127,16],[44,127,16]],[],[[35,127,16],[40,127,16]]],
			[[[59,127,16],[35,127,16],[40,127,16]],[],[[59,127,16],[44,127,16],[35,127,16],[40,127,16]]],
			[[[59,127,16],[35,127,16],[40,127,16]],[],[]]]],
			
[3,[3,3,3],[[[[59,127,16]],[],[[44,127,16]]],
			[[[59,127,16]],[],[[59,127,16],[35,127,16],[40,127,16]]],
			[[[59,127,16],[44,127,16]],[[35,127,16],[40,127,16]],[]]]],

# [3,[3,3,3],[[[],[],[]],
			# [[],[],[]],
			# [[],[],[]]]],

[4,[3,3,3,3],[[[[59,127,16],[35,127,16]],[],[]],
			  [[[59,127,16],[35,127,16],[44,127,16]],[],[[59,127,16]]],
			  [[[59,127,16],[36,127,16]],[],[]],
			  [[[59,127,16],[35,127,16],[44,127,16]],[],[[59,127,16]]]
			 ]],
				
[4,[3,3,3,3], [[[[59,127,16],[40,127,16],[36,127,16]],[],[[40,127,16]]],
			   [[[59,127,16],[40,127,16],[36,127,16],[44,127,16]],[],[[59,127,16],[40,127,16]]],
			   [[[59,127,16],[40,127,16],[36,127,16]],[],[[40,127,16]]],
			   [[[59,127,16],[40,127,16],[36,127,16],[44,127,16]],[],[[59,127,16],[40,127,16]]]
			  ]],
			  
# [4,[4,4,4,4],[[[[36,127,16],[53,127,16],[75,127,16]],[],[[44,127,16],[53,127,16]],[]],
			  # [[],[],[],[]],
			  # [[],[],[],[]],
			  # [[],[],[],[]]
			 # ]],
			  
# [4,[4,4,4,4],[[[],[],[],[]],
			  # [[],[],[],[]],
			  # [[],[],[],[]],
			  # [[],[],[],[]]
			 # ]],
			  
# [4,[3,3,3,3], [[],[],[],
			   # [],[],[],
			   # [],[],[],
			   # [],[],[]
			  # ]],

]

def lookUpDrumType(noteNum):
	for drumSound in DrumSoundBank:
		if drumSound[1] == noteNum:
			return drumSound
	return "NOT FOUND"
	
def lookUpDrumCategory(categoryNum):
	drumTypes=[]
	for drumSound in DrumSoundBank:
		if drumSound[0] == categoryNum:
			drumTypes.append(drumSound)
	return drumTypes

	

# import mido
# import time
# import random
# outport = mido.open_output()

# skeletonIndex=2
# sleepTime=0.4
# for l in range(0,2):
	# for skeletonIndex in range(0,5):
		# for j in range(0,12):
			# for i,drumBeats in enumerate(DrumSkeletons[skeletonIndex][2]):
				# print(drumBeats)
				# for drumDiv in drumBeats:
					# for drumType in drumDiv:
						# outport.send(mido.Message('note_on', channel=9, note=drumType[0], time=0, velocity=drumType[1]))
					# time.sleep(sleepTime/DrumSkeletons[skeletonIndex][1][i])

# noteNum=38
# outport.send(mido.Message('note_on', channel=9, note=noteNum, time=0, velocity=127))	
# time.sleep(0.5)
# outport.send(mido.Message('note_on', channel=9, note=noteNum, time=0, velocity=127))	
# time.sleep(0.5)
# outport.send(mido.Message('note_on', channel=9, note=noteNum, time=0, velocity=127))	
# time.sleep(0.5)
# outport.send(mido.Message('note_on', channel=9, note=noteNum, time=0, velocity=127))	
# time.sleep(0.5)
# outport.send(mido.Message('note_on', channel=9, note=noteNum, time=0, velocity=127))		
	
	# # # input()

