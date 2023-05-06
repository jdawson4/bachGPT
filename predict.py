# Author: Jacob Dawson
#
# In this script, we'll take our trained model and run it until we get a song!

import numpy as np
import tensorflow as tf
from arch import *

# this determines the length of the song made:
numPredictions = 128

# make model
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
model.load_weights(modelSaveLocation)

# predict!
predictions = []
lastPrediction = np.zeros((timestepsPerBatch, 128), dtype=np.float16)
for i in range(numPredictions):
    # print("prediction", i)
    prediction = model(tf.expand_dims(lastPrediction, axis=0))[0]
    predictions.append(prediction)
    # print(f"lastPrediction shape: {lastPrediction.shape}, prediction shape: {prediction.shape}")
    lastPrediction = prediction
predictions = np.concatenate(predictions, axis=0)
predictions = predictions * 256
predictions = predictions.astype(np.uint16)
