import torch
import MidiUtils as mu
import NeuralNetUtils as nnu
import random

model=nnu.LSTM_LogSoftMax_RNN(12,64,12,6,torch.device("cpu"))
model.load_state_dict(torch.load("bassNotes.pt"))
model.eval()

# for i in range(20):
	# netInput=torch.zeros(1,1,12)
	# netInput[0][0][random.randint(0,11)]=1
	# print(model(netInput))

basslines=mu.getInstrumentFromSong("projectMidiTraining/Four Brothers.mid",32,39)
with torch.no_grad():
	for bassline in basslines:
		
		model.hidden = model.initHidden()
		loss=0

		noteBuffer=mu.getBassNoteIntervalsFromBassline(bassline)
					
		nonEmptyBassline=(len(noteBuffer) > 1)
		
		while len(noteBuffer) > 1:
			netInput=torch.zeros(1,1,12)
			netInput[0][0][noteBuffer[0] % 12]=1
			correctOutput=torch.tensor([noteBuffer[1] % 12])
			noteBuffer=noteBuffer[1:]
			predictedInterval=model(netInput)
			for i in range(12):
				print("prediction {}: {}".format(i,predictedInterval[0][i]))
			print("correct output: {}".format(correctOutput))
			input()