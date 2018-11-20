import mido
import torch
import NeuralNetUtils as nnu
import MusicTheoryUtils as mtu
import random
import time

	
cuda=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
melodyModel=nnu.NoteGenerator(5,144,4,6,0.1,cuda)
melodyModel.load_state_dict(torch.load("trained networks/5337.pt",map_location='cpu'))
melodyModel.eval()
genOut=melodyModel(torch.randn(1,5).to(cuda))
print(genOut)
print("====")

rootNote=48
mid = mido.MidiFile()
track = mido.MidiTrack()
mid.tracks.append(track)
track.append(mido.Message('program_change', program=32, channel=0, time=0))
currentNote=rootNote
for oneHotNote in genOut:
	print(oneHotNote[0].tolist().index(1))
	currentNote+=(random.randint(-1,0)*12)+oneHotNote[0].tolist().index(1)
	if currentNote < 36:
		currentNote+=12
	elif currentNote > 60:
		currentNote-=12
	track.append(mido.Message("note_on", channel=0, note=currentNote, time=0))
	track.append(mido.Message("note_off", channel=0, note=currentNote, time=480))
mid.save('rawOutputGAN.mid')
