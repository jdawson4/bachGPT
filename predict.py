# Author: Jacob Dawson
#
# In this script, we'll take our trained model and run it until we get a song!

import numpy as np
import tensorflow as tf
import pretty_midi
from arch import *

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
lastPrediction = np.zeros((timestepsPerBatch, 128), dtype=np.float16)
for i in range(numPredictions):
    # print("prediction", i)
    prediction = model(tf.expand_dims(lastPrediction, axis=0))[0]
    predictions.append(prediction)
    # print(f"lastPrediction shape: {lastPrediction.shape}, prediction shape: {prediction.shape}")
    lastPrediction = prediction

print("Postprocessing predictions")
predictions = np.concatenate(predictions, axis=0)
predictions = predictions * 256
predictions = predictions.astype(np.uint16)

print("Writing predictions to", predictionFile)
# probably need to do more work here to get the timings correct, realizing now
# that I don't entirely know how they were encoded.
# TODO: determine how to turn a piano roll back into midi better
outputMidi = pretty_midi.PrettyMIDI()
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
outputMidi.write(predictionFile)
