import mido
import os
import time

outport = mido.open_output()
for msg in mido.MidiFile("projectMidiTraining/ST_LOUIS.mid"):
	time.sleep(msg.time)
	if msg.type == "note_on" or msg.type == "pitchwheel" or msg.type == "program_change" or msg.type == "control_change":
		if msg.channel == 9:
			print(msg)
			outport.send(msg)