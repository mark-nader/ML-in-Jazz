import mido
import torch
import NeuralNetUtils as nnu
import MusicTheoryUtils as mtu
import random
import MusicWriterVanilla as mw

numLoops=1#20
beatsInBar=4
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
noteSeed=[71,67,67,60,62,64,65,60,65]
mCTF=mw.WeightedNumberPicker([0,1,2],[0,1,2],[0.35,0.55,0.10])
mCF=mw.WeightedNumberPicker([2,3,4,5],[2,3,4,5],[0.15,0.35,0.35,0.15])
oRF=mw.WeightedNumberPicker([0,1,2],[0,1,2],[0.60,0.25,0.15])
dCF=mw.WeightedNumberPicker([0,1],[0,1],[0.65,0.35])
velocityRange=[96,127]
durationRange=[2,3]

noteOverlapFlag=False
cFF=mw.WeightedNumberPicker([0,1],[0,1],[0.75,0.25])
cSR=list(range(3,5))
sFF=mw.WeightedNumberPicker([0,1],[0,1],[0.95,0.05])
sSR=list(range(2,5))
nextNote=60

A7=mtu.Chord(57,9)
Am7=mtu.Chord(57,11)
CMaj7=mtu.Chord(60,10)
Dm7=mtu.Chord(62,11)
Em7=mtu.Chord(64,11)
G7=mtu.Chord(67,9)

noteList=noteSeed

drumsSection=mw.PercussionInstrumentSection("Test Drums",mw.drumsNet)

drumsMCT=mw.WeightedNumberPicker([2,4],[3,5],[0.60,0.40])
zeroFactor=mw.WeightedNumberPicker([0],[0],[1])

drumsSection.setSectionSkeleton(random.choice([skel for skel in mtu.DrumSkeletons if skel[0] == beatsInBar and skel[1] == subDivsPerBeat]),10,15,[0,1,2,3,4,6],drumsMCT,64)

pianoSection=mw.AccompanyInstrumentSection("Test Piano")

for i in range(numLoops):

	# chosenScale=Dm7.rankedScalesWithScore[scaleChoiceFactor.getValue()]
	# bassSection.addBar(Dm7,chosenScale[2],beatsInBar,subDivsPerBeat,div1O4,div3O4,div24O4,div1O3,div2O3,div3O3,
	# noteList,mCTF,mCF,oRF,dCF,velocityRange,durationRange)
	# noteList+=bassSection.chosenMelody[:]
	# noteList=noteList[-9:]
	
	# chosenScale=G7.rankedScalesWithScore[scaleChoiceFactor.getValue()]
	# bassSection.addBar(G7,chosenScale[2],beatsInBar,subDivsPerBeat,div1O4,div3O4,div24O4,div1O3,div2O3,div3O3,
	# noteList,mCTF,mCF,oRF,dCF,velocityRange,durationRange)
	# noteList+=bassSection.chosenMelody[:]
	# noteList=noteList[-9:]
	
	# chosenScale=CMaj7.rankedScalesWithScore[scaleChoiceFactor.getValue()]
	# bassSection.addBar(CMaj7,chosenScale[2],beatsInBar,subDivsPerBeat,div1O4,div3O4,div24O4,div1O3,div2O3,div3O3,
	# noteList,mCTF,mCF,oRF,dCF,velocityRange,durationRange)
	# noteList+=bassSection.chosenMelody[:]
	# noteList=noteList[-9:]
	
	# chosenScale=CMaj7.rankedScalesWithScore[scaleChoiceFactor.getValue()]
	# bassSection.addBar(CMaj7,chosenScale[2],beatsInBar,subDivsPerBeat,div1O4,div3O4,div24O4,div1O3,div2O3,div3O3,
	# noteList,mCTF,mCF,oRF,dCF,velocityRange,durationRange)
	# noteList+=bassSection.chosenMelody[:]
	# noteList=noteList[-9:]
	

	# chosenScale=CMaj7.rankedScalesWithScore[scaleChoiceFactor.getValue()]
	# bassSection.addBar(CMaj7,chosenScale[2],beatsInBar//2,subDivsPerBeat,div1O4,div3O4,div24O4,div1O3,div2O3,div3O3,
	# noteList,mCTF,mCF,oRF,dCF,velocityRange,durationRange)
	# noteList+=bassSection.chosenMelody[:]
	# noteList=noteList[-6:]

	# chosenScale=Am7.rankedScalesWithScore[scaleChoiceFactor.getValue()]
	# bassSection.addBar(Am7,chosenScale[2],beatsInBar//2,subDivsPerBeat,div1O4,div3O4,div24O4,div1O3,div2O3,div3O3,
	# noteList,mCTF,mCF,oRF,dCF,velocityRange,durationRange)
	# noteList+=bassSection.chosenMelody[:]
	# noteList=noteList[-6:]

	# chosenScale=Dm7.rankedScalesWithScore[scaleChoiceFactor.getValue()]
	# bassSection.addBar(Dm7,chosenScale[2],beatsInBar//2,subDivsPerBeat,div1O4,div3O4,div24O4,div1O3,div2O3,div3O3,
	# noteList,mCTF,mCF,oRF,dCF,velocityRange,durationRange)
	# noteList+=bassSection.chosenMelody[:]
	# noteList=noteList[-6:]

	# chosenScale=G7.rankedScalesWithScore[scaleChoiceFactor.getValue()]
	# bassSection.addBar(G7,chosenScale[2],beatsInBar//2,subDivsPerBeat,div1O4,div3O4,div24O4,div1O3,div2O3,div3O3,
	# noteList,mCTF,mCF,oRF,dCF,velocityRange,durationRange)
	# noteList+=bassSection.chosenMelody[:]
	# noteList=noteList[-6:]
	
	# chosenScale=Em7.rankedScalesWithScore[scaleChoiceFactor.getValue()]
	# bassSection.addBar(Em7,chosenScale[2],beatsInBar//2,subDivsPerBeat,div1O4,div3O4,div24O4,div1O3,div2O3,div3O3,
	# noteList,mCTF,mCF,oRF,dCF,velocityRange,durationRange)
	# noteList+=bassSection.chosenMelody[:]
	# noteList=noteList[-6:]
	
	# chosenScale=A7.rankedScalesWithScore[scaleChoiceFactor.getValue()]
	# bassSection.addBar(A7,chosenScale[2],beatsInBar//2,subDivsPerBeat,div1O4,div3O4,div24O4,div1O3,div2O3,div3O3,
	# noteList,mCTF,mCF,oRF,dCF,velocityRange,durationRange)
	# noteList+=bassSection.chosenMelody[:]
	# noteList=noteList[-6:]
	
	# chosenScale=Dm7.rankedScalesWithScore[scaleChoiceFactor.getValue()]
	# bassSection.addBar(Dm7,chosenScale[2],beatsInBar//2,subDivsPerBeat,div1O4,div3O4,div24O4,div1O3,div2O3,div3O3,
	# noteList,mCTF,mCF,oRF,dCF,velocityRange,durationRange)
	# noteList+=bassSection.chosenMelody[:]
	# noteList=noteList[-6:]
	
	# chosenScale=G7.rankedScalesWithScore[scaleChoiceFactor.getValue()]
	# bassSection.addBar(G7,chosenScale[2],beatsInBar//2,subDivsPerBeat,div1O4,div3O4,div24O4,div1O3,div2O3,div3O3,
	# noteList,mCTF,mCF,oRF,dCF,velocityRange,durationRange)
	# noteList+=bassSection.chosenMelody[:]
	# noteList=noteList[-6:]

	drumsSection.addBar(10,15,[1,2,3,6,7],drumsMCT,64)
	drumsSection.addBar(10,15,[1,2,3,6,7],drumsMCT,64)
	drumsSection.addBar(10,15,[1,2,3,6,7],drumsMCT,64)
	drumsSection.addBar(10,15,[1,2,3,6,7],drumsMCT,64,random.choice([skel for skel in mtu.DrumFills if skel[0] == beatsInBar and skel[1] == subDivsPerBeat]))

	# pianoSection.addBar(CMaj7,beatsInBar//2,subDivsPerBeat,64)
	# pianoSection.addBar(Am7,beatsInBar//2,subDivsPerBeat,64)
	# pianoSection.addBar(Dm7,beatsInBar//2,subDivsPerBeat,64)
	# pianoSection.addBar(G7,beatsInBar//2,subDivsPerBeat,64)
	# pianoSection.addBar(Em7,beatsInBar//2,subDivsPerBeat,64)
	# pianoSection.addBar(A7,beatsInBar//2,subDivsPerBeat,64)
	# pianoSection.addBar(Dm7,beatsInBar//2,subDivsPerBeat,64)
	# pianoSection.addBar(G7,beatsInBar//2,subDivsPerBeat,64)

bassSection.formatBars(noteOverlapFlag,cFF,cSR,sFF,sSR,nextNote)
drumsSection.formatBars(True,zeroFactor,[],zeroFactor,[],-1)
pianoSection.formatBars(True,zeroFactor,[],zeroFactor,[],-1)

bassSection.convertToMidi(480,0,32,ticksPerBeat)
drumsSection.convertToMidi(480,9,0,ticksPerBeat)
pianoSection.convertToMidi(480,1,0,ticksPerBeat)

mw.combineMidi([bassSection.msgs,drumsSection.msgs,pianoSection.msgs],'d4 -phrase 3.mid')
