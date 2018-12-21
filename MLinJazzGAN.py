import mido
import torch
import NeuralNetUtils as nnu
import MusicTheoryUtils as mtu
import random
import MusicWriterGAN as mw

numLoops=4
beatsInBar=4
subDivsPerBeat=3
ticksPerBeat=240

AbMaj7=mtu.Chord(56,10)
Abm=mtu.Chord(56,1)
Abm7=mtu.Chord(56,11)
A7=mtu.Chord(57,9)
Am7=mtu.Chord(57,11)
BbMaj7=mtu.Chord(58,10)
Bbm7=mtu.Chord(58,11)
BMaj=mtu.Chord(59,0)
CMaj7=mtu.Chord(60,10)
Cm7=mtu.Chord(60,11)
Db7=mtu.Chord(61,9)
Db9=mtu.Chord(61,14)
Dbm9=mtu.Chord(61,16)
Dbdim7=mtu.Chord(61,25)
D7=mtu.Chord(62,9)
D9=mtu.Chord(62,14)
Dm7=mtu.Chord(62,11)
Eb7=mtu.Chord(63,9)
Eb9=mtu.Chord(63,14)
Ebdim7=mtu.Chord(63,25)
EMaj7=mtu.Chord(64,10)
E9=mtu.Chord(64,14)
Em7=mtu.Chord(64,11)
Fm7=mtu.Chord(65,11)
F7=mtu.Chord(65,9)
F9=mtu.Chord(65,14)	
GbMaj=mtu.Chord(66,0)
Gb7=mtu.Chord(66,9)
Gb9=mtu.Chord(66,14)
GMaj=mtu.Chord(67,0)
G7=mtu.Chord(67,9)

bassSection=mw.MelodyInstrumentSection("Test Bass",28,52,mw.melodyModel_6276)

scaleChoiceFactor=mw.WeightedNumberPicker([0,0],[3,5],[0.65,0.35])
div1O4=mw.WeightedNumberPicker([0,1],[0,1],[0,1])
div3O4=mw.WeightedNumberPicker([0,1],[0,1],[1,0])
div24O4=mw.WeightedNumberPicker([0,1],[0,1],[1,0])
div1O3=mw.WeightedNumberPicker([0,1],[0,1],[0.50,0.50])
div2O3=mw.WeightedNumberPicker([0,1],[0,1],[0.95,0.05])
div3O3=mw.WeightedNumberPicker([0,1],[0,1],[0.75,0.25])
velocitySeed=[]
durationSeed=[]
oRF=mw.WeightedNumberPicker([0,1],[0,1],[0.85,0.15])#positive only now because of directionChangeFactor
dCF=mw.WeightedNumberPicker([0,1],[0,1],[0.65,0.35])
durationRange=[2,6]
velocityRange=[64,127]

noteOverlapFlag=False
cFF=mw.WeightedNumberPicker([0,1],[0,1],[0.7,0.3])
cSR=list(range(3,5))
sFF=mw.WeightedNumberPicker([0,1],[0,1],[0.9,0.1])
sSR=list(range(2,5))
nextNote=62

verseChords=[BMaj,BMaj,Abm,Abm,GMaj,GMaj,GbMaj,GbMaj,BMaj,BMaj,Abm,Abm,GMaj,GMaj,GbMaj,F7]
bridgeChords=[E9,E9,Eb9,Eb9,D9,D9,Db9,Db9,D9,D9,Eb9,Eb9,E9,E9,E9,E9,E9,E9,Eb9,Eb9,D9,D9,Db9,Db9,D9,Db9,E9,F9,Gb9,Gb7,Gb7,Gb7]
chorusChords=[BMaj,BMaj,Fm7,Fm7,EMaj7,EMaj7,Dbm9,GbMaj,BMaj,BMaj,Fm7,Fm7,EMaj7,EMaj7,Dbm9,GbMaj]
interludeChords=[BMaj,BMaj,BMaj,BMaj,BMaj,BMaj,BMaj,BMaj,BMaj,BMaj,BMaj,BMaj,BMaj,BMaj,BMaj,Gb7]
verseBeatsPerBar=4
bridgeBeatsPerBar=1
chorusBeatsPerBar=4
interludeBeatsPerBar=4

verseScales=[]
bridgeScales=[]
chorusScales=[]
interludeScales=[]
for chrd in verseChords:
	verseScales.append(chrd.rankedScalesWithScore[scaleChoiceFactor.getValue()][2])
for chrd in bridgeChords:
	bridgeScales.append(chrd.rankedScalesWithScore[scaleChoiceFactor.getValue()][2])
for chrd in chorusChords:
	chorusScales.append(chrd.rankedScalesWithScore[scaleChoiceFactor.getValue()][2])
for chrd in interludeChords:
	interludeScales.append(chrd.rankedScalesWithScore[scaleChoiceFactor.getValue()][2])

drumsSection=mw.PercussionInstrumentSection("Test Drums",mw.drumsNet)

drumsMCT=mw.WeightedNumberPicker([2,4],[3,5],[0.60,0.40])
zeroFactor=mw.WeightedNumberPicker([0],[0],[1])

verseSkeleton=random.choice([skel for skel in mtu.DrumSkeletons if skel[0] == 4 and skel[1] == subDivsPerBeat])
bridgeSkeleton=random.choice([skel for skel in mtu.DrumSkeletons if skel[0] == 2 and skel[1] == subDivsPerBeat])
chorusSkeleton=random.choice([skel for skel in mtu.DrumSkeletons if skel[0] == 4 and skel[1] == subDivsPerBeat])
interludeSkeleton=random.choice([skel for skel in mtu.DrumSkeletons if skel[0] == 4 and skel[1] == subDivsPerBeat])

pianoSection=mw.AccompanyInstrumentSection("Test Piano")

bassSection.addProgressionBars(verseChords,verseScales,verseBeatsPerBar,subDivsPerBeat,div1O4,div3O4,div24O4,div1O3,div2O3,div3O3,
oRF,dCF,durationRange,velocityRange)

bassSection.addProgressionBars(bridgeChords,bridgeScales,bridgeBeatsPerBar,1,div1O4,div3O4,div24O4,div1O3,div2O3,div3O3,
oRF,dCF,durationRange,velocityRange)

bassSection.addProgressionBars(chorusChords,chorusScales,chorusBeatsPerBar,subDivsPerBeat,div1O4,div3O4,div24O4,div1O3,div2O3,div3O3,
oRF,dCF,durationRange,velocityRange)

bassSection.addProgressionBars(interludeChords,interludeScales,chorusBeatsPerBar,subDivsPerBeat,div1O4,div3O4,div24O4,div1O3,div2O3,div3O3,
oRF,dCF,durationRange,velocityRange)

drumsSection.setSectionSkeleton(verseSkeleton,10,15,[0,1,2,3,4,6],drumsMCT,64)
drumsSection.addBar(10,15,[1,2,3,6,7],drumsMCT,64)
drumsSection.addBar(10,15,[1,2,3,6,7],drumsMCT,64)
drumsSection.addBar(10,15,[1,2,3,6,7],drumsMCT,64)
drumsSection.addBar(10,15,[1,2,3,6,7],drumsMCT,64,random.choice([skel for skel in mtu.DrumFills if skel[0] == 4 and skel[1] == subDivsPerBeat]))
drumsSection.addBar(10,15,[1,2,3,6,7],drumsMCT,64)
drumsSection.addBar(10,15,[1,2,3,6,7],drumsMCT,64)
drumsSection.addBar(10,15,[1,2,3,6,7],drumsMCT,64)
drumsSection.addBar(10,15,[1,2,3,6,7],drumsMCT,64,random.choice([skel for skel in mtu.DrumFills if skel[0] == 4 and skel[1] == subDivsPerBeat]))

drumsSection.setSectionSkeleton(bridgeSkeleton,10,15,[0,1,2,3,4,6],drumsMCT,64)
drumsSection.addBar(10,15,[1,2,3,6,7],drumsMCT,64)
drumsSection.addBar(10,15,[1,2,3,6,7],drumsMCT,64)
drumsSection.addBar(10,15,[1,2,3,6,7],drumsMCT,64)
drumsSection.addBar(10,15,[1,2,3,6,7],drumsMCT,64)
drumsSection.addBar(10,15,[1,2,3,6,7],drumsMCT,64)
drumsSection.addBar(10,15,[1,2,3,6,7],drumsMCT,64)
drumsSection.addBar(10,15,[1,2,3,6,7],drumsMCT,64)
drumsSection.addBar(10,15,[1,2,3,6,7],drumsMCT,64,random.choice([skel for skel in mtu.DrumFills if skel[0] == 2 and skel[1] == subDivsPerBeat]))

drumsSection.setSectionSkeleton(chorusSkeleton,10,15,[0,1,2,3,4,6],drumsMCT,64)
drumsSection.addBar(10,15,[1,2,3,6,7],drumsMCT,64)
drumsSection.addBar(10,15,[1,2,3,6,7],drumsMCT,64)
drumsSection.addBar(10,15,[1,2,3,6,7],drumsMCT,64)
drumsSection.addBar(10,15,[1,2,3,6,7],drumsMCT,64,random.choice([skel for skel in mtu.DrumFills if skel[0] == 4 and skel[1] == subDivsPerBeat]))
drumsSection.addBar(10,15,[1,2,3,6,7],drumsMCT,64)
drumsSection.addBar(10,15,[1,2,3,6,7],drumsMCT,64)
drumsSection.addBar(10,15,[1,2,3,6,7],drumsMCT,64)
drumsSection.addBar(10,15,[1,2,3,6,7],drumsMCT,64,random.choice([skel for skel in mtu.DrumFills if skel[0] == 4 and skel[1] == subDivsPerBeat]))

drumsSection.setSectionSkeleton(interludeSkeleton,10,15,[0,1,2,3,4,6],drumsMCT,64)
drumsSection.addBar(10,15,[1,2,3,6,7],drumsMCT,64)
drumsSection.addBar(10,15,[1,2,3,6,7],drumsMCT,64)
drumsSection.addBar(10,15,[1,2,3,6,7],drumsMCT,64)
drumsSection.addBar(10,15,[1,2,3,6,7],drumsMCT,64)
drumsSection.addBar(10,15,[1,2,3,6,7],drumsMCT,64)
drumsSection.addBar(10,15,[1,2,3,6,7],drumsMCT,64)
drumsSection.addBar(10,15,[1,2,3,6,7],drumsMCT,64)
drumsSection.addBar(10,15,[1,2,3,6,7],drumsMCT,64,random.choice([skel for skel in mtu.DrumFills if skel[0] == 4 and skel[1] == subDivsPerBeat]))

for chrd in verseChords:
	pianoSection.addBar(chrd,verseBeatsPerBar,subDivsPerBeat,64)
for chrd in bridgeChords:
	pianoSection.addBar(chrd,bridgeBeatsPerBar,subDivsPerBeat,64)
for chrd in chorusChords:
	pianoSection.addBar(chrd,chorusBeatsPerBar,subDivsPerBeat,64)
for chrd in interludeChords:
	pianoSection.addBar(chrd,interludeBeatsPerBar,subDivsPerBeat,64)

	
bassSection.formatBars(noteOverlapFlag,cFF,cSR,sFF,sSR,nextNote)
drumsSection.formatBars(True,zeroFactor,[],zeroFactor,[],-1)
pianoSection.formatBars(True,zeroFactor,[],zeroFactor,[],-1)

bassSection.convertToMidi(240,0,32,ticksPerBeat)
drumsSection.convertToMidi(240,9,0,ticksPerBeat*2)
pianoSection.convertToMidi(240,1,0,ticksPerBeat)

bassLooped=[]
drumsLooped=[]
pianoLooped=[]
for i in range(numLoops):
	bassLooped.extend(bassSection.msgs)
for i in range(numLoops):
	drumsLooped.extend(drumsSection.msgs)
for i in range(numLoops):
	pianoLooped.extend(pianoSection.msgs)

mw.combineMidi([bassLooped,drumsLooped,pianoLooped],'GAN5305 -sir duke.mid')
# for noteLog in bassSection.creationLog:
	# print(noteLog)
