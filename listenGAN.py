import mido
import torch
import NeuralNetUtils as nnu
import MusicTheoryUtils as mtu
import random
import time

	
cuda=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
melodyModel=nnu.NoteGenerator(5,12*6,4,6,0.1,cuda)
melodyModel.load_state_dict(torch.load("bassNotesGAN.pt"))
melodyModel.eval()
genOut=melodyModel(torch.randn(1,5).to(cuda))
print(genOut)
print("====")

rootNote=48
mid = mido.MidiFile()
track = mido.MidiTrack()
mid.tracks.append(track)
track.append(mido.Message('program_change', program=32, channel=0, time=0))
for oneHotNote in genOut:
	print(oneHotNote[0].tolist().index(1))
	oneOctaveNote=rootNote+(random.randint(-1,0)*12)+oneHotNote[0].tolist().index(1)
	track.append(mido.Message("note_on", channel=0, note=oneOctaveNote, time=0))
	track.append(mido.Message("note_off", channel=0, note=oneOctaveNote, time=480))
mid.save('rawOutputGAN.mid')
