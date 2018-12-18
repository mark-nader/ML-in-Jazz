import mido
import torch
import NeuralNetUtils as nnu
import MusicTheoryUtils as mtu
import random
import MusicWriterVanilla as mw

numLoops=1#20
beatsInBar=6
subDivsPerBeat=3
ticksPerBeat=480

bassSection=mw.MelodyInstrumentSection("Test Bass",28,52,mw.melodyModel)

scaleChoiceFactor=mw.WeightedNumberPicker([0,0],[3,5],[0.65,0.35])
div1O4=mw.WeightedNumberPicker([0,1],[0,1],[0,1])
div3O4=mw.WeightedNumberPicker([0,1],[0,1],[1,0])
div24O4=mw.WeightedNumberPicker([0,1],[0,1],[1,0])
div1O3=mw.WeightedNumberPicker([0,1],[0,1],[0.00,1.00])
div2O3=mw.WeightedNumberPicker([0,1],[0,1],[0.90,0.10])
div3O3=mw.WeightedNumberPicker([0,1],[0,1],[0.75,0.25])
# noteSeed=[71,67,67,60,62,64,65,60,65]
noteSeed=[71,67,60,62,64,65,60,65]
# noteSeed=[71,67,60,62,65,60,65]
mCTF=mw.WeightedNumberPicker([0,1,2],[0,1,2],[0.25,0.65,0.10])
mCF=mw.WeightedNumberPicker([2,3,4,5],[2,3,4,5],[0.15,0.30,0.35,0.20])
oRF=mw.WeightedNumberPicker([0,1,2],[0,1,2],[0.55,0.30,0.15])
dCF=mw.WeightedNumberPicker([0,1],[0,1],[0.65,0.35])
velocityRange=[96,127]
durationRange=[2,3]

noteOverlapFlag=False
cFF=mw.WeightedNumberPicker([0,1],[0,1],[0.75,0.25])
cSR=list(range(3,5))
sFF=mw.WeightedNumberPicker([0,1],[0,1],[0.95,0.05])
sSR=list(range(2,5))
nextNote=60

AbMaj7=mtu.Chord(56,10)
Abm7=mtu.Chord(56,11)
A7=mtu.Chord(57,9)
Am7=mtu.Chord(57,11)
BbMaj7=mtu.Chord(58,10)
Bbm7=mtu.Chord(58,11)
CMaj7=mtu.Chord(60,10)
Cm7=mtu.Chord(60,11)
Db7=mtu.Chord(61,9)
Dbdim7=mtu.Chord(61,25)
D7=mtu.Chord(62,9)
Dm7=mtu.Chord(62,11)
Eb7=mtu.Chord(63,9)
Ebdim7=mtu.Chord(63,25)
Em7=mtu.Chord(64,11)
F7=mtu.Chord(65,9)
G7=mtu.Chord(67,9)

noteList=noteSeed

drumsSection=mw.PercussionInstrumentSection("Test Drums",mw.drumsNet)

drumsMCT=mw.WeightedNumberPicker([2,4],[3,5],[0.60,0.40])
zeroFactor=mw.WeightedNumberPicker([0],[0],[1])

drumsSection.setSectionSkeleton(random.choice([skel for skel in mtu.DrumSkeletons if skel[0] == beatsInBar and skel[1] == subDivsPerBeat]),10,15,[0,1,2,3,4,6],drumsMCT,64)

pianoSection=mw.AccompanyInstrumentSection("Test Piano")

for i in range(numLoops):

	chosenScale=CMaj7.rankedScalesWithScore[scaleChoiceFactor.getValue()]
	bassSection.addBar(CMaj7,chosenScale[2],beatsInBar,subDivsPerBeat,div1O4,div3O4,div24O4,div1O3,div2O3,div3O3,
	noteList,mCTF,mCF,oRF,dCF,velocityRange,durationRange)
	noteList+=bassSection.chosenMelody[:]
	noteList=noteList[-8:]

	chosenScale=CMaj7.rankedScalesWithScore[scaleChoiceFactor.getValue()]
	bassSection.addBar(CMaj7,chosenScale[2],beatsInBar,subDivsPerBeat,div1O4,div3O4,div24O4,div1O3,div2O3,div3O3,
	noteList,mCTF,mCF,oRF,dCF,velocityRange,durationRange)
	noteList+=bassSection.chosenMelody[:]
	noteList=noteList[-8:]

	chosenScale=D7.rankedScalesWithScore[scaleChoiceFactor.getValue()]
	bassSection.addBar(D7,chosenScale[2],beatsInBar,subDivsPerBeat,div1O4,div3O4,div24O4,div1O3,div2O3,div3O3,
	noteList,mCTF,mCF,oRF,dCF,velocityRange,durationRange)
	noteList+=bassSection.chosenMelody[:]
	noteList=noteList[-8:]

	chosenScale=D7.rankedScalesWithScore[scaleChoiceFactor.getValue()]
	bassSection.addBar(D7,chosenScale[2],beatsInBar,subDivsPerBeat,div1O4,div3O4,div24O4,div1O3,div2O3,div3O3,
	noteList,mCTF,mCF,oRF,dCF,velocityRange,durationRange)
	noteList+=bassSection.chosenMelody[:]
	noteList=noteList[-8:]
	
	chosenScale=Dm7.rankedScalesWithScore[scaleChoiceFactor.getValue()]
	bassSection.addBar(Dm7,chosenScale[2],beatsInBar,subDivsPerBeat,div1O4,div3O4,div24O4,div1O3,div2O3,div3O3,
	noteList,mCTF,mCF,oRF,dCF,velocityRange,durationRange)
	noteList+=bassSection.chosenMelody[:]
	noteList=noteList[-8:]
	
	chosenScale=G7.rankedScalesWithScore[scaleChoiceFactor.getValue()]
	bassSection.addBar(G7,chosenScale[2],beatsInBar,subDivsPerBeat,div1O4,div3O4,div24O4,div1O3,div2O3,div3O3,
	noteList,mCTF,mCF,oRF,dCF,velocityRange,durationRange)
	noteList+=bassSection.chosenMelody[:]
	noteList=noteList[-8:]
	
	chosenScale=CMaj7.rankedScalesWithScore[scaleChoiceFactor.getValue()]
	bassSection.addBar(CMaj7,chosenScale[2],beatsInBar,subDivsPerBeat,div1O4,div3O4,div24O4,div1O3,div2O3,div3O3,
	noteList,mCTF,mCF,oRF,dCF,velocityRange,durationRange)
	noteList+=bassSection.chosenMelody[:]
	noteList=noteList[-8:]
	
	chosenScale=CMaj7.rankedScalesWithScore[scaleChoiceFactor.getValue()]
	bassSection.addBar(CMaj7,chosenScale[2],beatsInBar,subDivsPerBeat,div1O4,div3O4,div24O4,div1O3,div2O3,div3O3,
	noteList,mCTF,mCF,oRF,dCF,velocityRange,durationRange)
	noteList+=bassSection.chosenMelody[:]
	noteList=noteList[-8:]
	
	drumsSection.addBar(10,15,[1,2,3,6,7],drumsMCT,64)
	drumsSection.addBar(10,15,[1,2,3,6,7],drumsMCT,64)
	drumsSection.addBar(10,15,[1,2,3,6,7],drumsMCT,64)
	drumsSection.addBar(10,15,[1,2,3,6,7],drumsMCT,64,random.choice([skel for skel in mtu.DrumFills if skel[0] == beatsInBar and skel[1] == subDivsPerBeat]))
	drumsSection.addBar(10,15,[1,2,3,6,7],drumsMCT,64)
	drumsSection.addBar(10,15,[1,2,3,6,7],drumsMCT,64)
	drumsSection.addBar(10,15,[1,2,3,6,7],drumsMCT,64)
	drumsSection.addBar(10,15,[1,2,3,6,7],drumsMCT,64,random.choice([skel for skel in mtu.DrumFills if skel[0] == beatsInBar and skel[1] == subDivsPerBeat]))

	pianoSection.addBar(CMaj7,beatsInBar//2,subDivsPerBeat,64)
	pianoSection.addBar(CMaj7,beatsInBar//2,subDivsPerBeat,64)
	pianoSection.addBar(CMaj7,beatsInBar//2,subDivsPerBeat,64)
	pianoSection.addBar(CMaj7,beatsInBar//2,subDivsPerBeat,64)
	pianoSection.addBar(D7,beatsInBar//2,subDivsPerBeat,64)
	pianoSection.addBar(D7,beatsInBar//2,subDivsPerBeat,64)
	pianoSection.addBar(D7,beatsInBar//2,subDivsPerBeat,64)
	pianoSection.addBar(D7,beatsInBar//2,subDivsPerBeat,64)
	pianoSection.addBar(Dm7,beatsInBar//2,subDivsPerBeat,64)
	pianoSection.addBar(Dm7,beatsInBar//2,subDivsPerBeat,64)
	pianoSection.addBar(G7,beatsInBar//2,subDivsPerBeat,64)
	pianoSection.addBar(G7,beatsInBar//2,subDivsPerBeat,64)
	pianoSection.addBar(CMaj7,beatsInBar//2,subDivsPerBeat,64)
	pianoSection.addBar(CMaj7,beatsInBar//2,subDivsPerBeat,64)
	pianoSection.addBar(CMaj7,beatsInBar//2,subDivsPerBeat,64)
	pianoSection.addBar(CMaj7,beatsInBar//2,subDivsPerBeat,64)

bassSection.formatBars(noteOverlapFlag,cFF,cSR,sFF,sSR,nextNote)
drumsSection.formatBars(True,zeroFactor,[],zeroFactor,[],-1)
pianoSection.formatBars(True,zeroFactor,[],zeroFactor,[],-1)

bassSection.convertToMidi(480,0,32,ticksPerBeat)
drumsSection.convertToMidi(480,9,0,ticksPerBeat)
pianoSection.convertToMidi(480,1,0,ticksPerBeat)

mw.combineMidi([bassSection.msgs,drumsSection.msgs,pianoSection.msgs],'v4Class100 -song 6.mid')
# for noteLog in bassSection.creationLog:
	# print(noteLog)
