# Author: Jacob Dawson
#
# In this script, we'll take our trained model and run it until we get a song!

import numpy as np
import tensorflow as tf
import pretty_midi
import os
from random import choice
from arch import *
from midiReader import numpyFromFile, midisDirectory

# this determines the length of the song made:
numPredictions = 128

predictionFile = "prediction.mid"

# make model
print("Retrieving model")
model = attentionModel((timestepsPerBatch, 128))

# and compile... this doesn't actually matter because we aren't training
model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"],
)

# need to set built as true
model.built = True

# and finally load our weights
print("Retrieving weights")
model.load_weights(modelSaveLocation)

# predict!
print("Making predictions")
predictions = []
# "Seed crystal" to start on. Zeroes apparently doesn't work all that well, it
# appears that the network will just keep guessing zeroes--fair enough, I guess.
# lastPrediction = np.zeros((timestepsPerBatch, 128), dtype=np.float16)
#
# as our "seed", we'll choose a random midi from our set, and then only the
# first chunk
if not os.path.isdir(midisDirectory):
    raise Exception(f"No directory found at {midisDirectory}")

list_of_files = []
for dirpath, _, filenames in os.walk(midisDirectory):
    for filename in filenames:
        if filename.endswith(".mid"):
            list_of_files.append(os.sep.join([dirpath, filename]))
randomMidi = choice(list_of_files)
print(f"Seeding prediction with first chunk of {randomMidi}")
lastPrediction = numpyFromFile(randomMidi)
lastPrediction = np.swapaxes(lastPrediction, axis1=0, axis2=1)
lastPrediction = lastPrediction[0:timestepsPerBatch, :]
lastPrediction = lastPrediction / 256
lastPrediction = lastPrediction.astype(np.float16)
for i in range(numPredictions):
    # print("prediction", i)
    prediction = model(tf.expand_dims(lastPrediction, axis=0))[0]
    predictions.append(prediction)
    # print(f"lastPrediction shape: {lastPrediction.shape}, prediction shape: {prediction.shape}")
    # print(f"lastPrediction min, max: {np.min(lastPrediction)}, {np.max(lastPrediction)}; prediction min, max: {np.min(prediction)}, {np.max(prediction)}")
    lastPrediction = prediction

print("Postprocessing predictions")
predictions = np.concatenate(predictions, axis=0)
predictions = predictions * 256
predictions = predictions.astype(np.uint16)
# apparently our piano_roll_to_pretty_midi function takes an object of the
# shape [128, length]; we need to reorder our data:
predictions = np.swapaxes(predictions, axis1=1, axis2=0)

print("Writing predictions to", predictionFile)
# this is my first attempt, but it's pretty bad at reflecting how timings work
"""outputMidi = pretty_midi.PrettyMIDI()
piano_program = pretty_midi.instrument_name_to_program("Acoustic Grand Piano")
piano = pretty_midi.Instrument(program=piano_program)
timestep = 0
for prediction in predictions:
    i = 0
    for note in prediction:
        if note > 1:
            note = pretty_midi.Note(
                velocity=note, pitch=i, start=timestep, end=timestep + 1
            )
            piano.notes.append(note)
        i += 1
    timestep += 1
outputMidi.instruments.append(piano)
outputMidi.write(predictionFile)"""


# here's some code I found in the pretty_midi repo that ought to work better.
# originally found here:
# github.com/craffel/pretty-midi/blob/main/examples/reverse_pianoroll.py
def piano_roll_to_pretty_midi(piano_roll, fs=100, program=0):
    """Convert a Piano Roll array into a PrettyMidi object
     with a single instrument.

    Parameters
    ----------
    piano_roll : np.ndarray, shape=(128,frames), dtype=int
        Piano roll of one instrument
    fs : int
        Sampling frequency of the columns, i.e. each column is spaced apart
        by ``1./fs`` seconds.
    program : int
        The program number of the instrument.

    Returns
    -------
    midi_object : pretty_midi.PrettyMIDI
        A pretty_midi.PrettyMIDI class instance describing
        the piano roll.

    """
    notes, frames = piano_roll.shape
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=program)

    # pad 1 column of zeros so we can acknowledge inital and ending events
    piano_roll = np.pad(piano_roll, [(0, 0), (1, 1)], "constant")

    # use changes in velocities to find note on / note off events
    velocity_changes = np.nonzero(np.diff(piano_roll).T)

    # keep track on velocities and note on times
    prev_velocities = np.zeros(notes, dtype=int)
    note_on_time = np.zeros(notes)

    for time, note in zip(*velocity_changes):
        # use time + 1 because of padding above
        velocity = piano_roll[note, time + 1]
        time = time / fs
        if velocity > 0:
            if prev_velocities[note] == 0:
                note_on_time[note] = time
                prev_velocities[note] = velocity
        else:
            pm_note = pretty_midi.Note(
                velocity=prev_velocities[note],
                pitch=note,
                start=note_on_time[note],
                end=time,
            )
            instrument.notes.append(pm_note)
            prev_velocities[note] = 0
    pm.instruments.append(instrument)
    return pm


outputMidi = piano_roll_to_pretty_midi(predictions)
outputMidi.write(predictionFile)
