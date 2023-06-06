# Author: Jacob Dawson
#
# This file contains the architecture for our model.
# I believe that the final shape of our inputs will have 128 notes in [BATCH
# SIZE HERE] timesteps. With this in mind, the model should take the shape of
# the input should be [batchSize, timestepsPerBatch, notes (128)].

import tensorflow as tf
from tensorflow import keras

# import keras_nlp

seed = 7
timestepsPerBatch = 256
batchSize = 16
learnRate = 0.005
momentum = 0.9
epochInterval = 5
epochs = 25
modelSaveLocation = "modelWeights"


def attnLayer(
    input,
    kernelInit,
    layerCounter,
    heads=4,
    kDim=16,
    out_shape=64,
    residual=True,
    dropout=0.25,
):
    output = keras.layers.LayerNormalization()(input)
    output = keras.layers.MultiHeadAttention(
        num_heads=heads,
        key_dim=kDim,
        output_shape=out_shape,
        kernel_initializer=kernelInit,
    )(output, output)
    output = keras.layers.Dropout(dropout)(output)
    if residual:
        output = keras.layers.Add()([input, output])

    output = keras.layers.LayerNormalization()(output)
    output = tf.keras.layers.Conv1D(
        out_shape,
        1,
        strides=1,
        kernel_initializer=kernelInit,
    )(output)
    output = keras.layers.Activation("selu")(output)
    output = keras.layers.Dropout(dropout)(output)

    return keras.Model(inputs=input, outputs=output, name=f"block{layerCounter}")(input)


def attentionModel(inputShape):
    layerCounter = 0
    init = keras.initializers.RandomNormal(seed=seed)

    input = keras.layers.Input(shape=inputShape, dtype=tf.float16)
    # embed = keras_nlp.layers.PositionEmbedding(timestepsPerBatch, initializer=init)(
    #    input
    # )

    layerCounter += 1
    output = attnLayer(input=input, layerCounter=layerCounter, kernelInit=init, residual=False)
    for _ in range(49):
        layerCounter += 1
        output = attnLayer(input=output, layerCounter=layerCounter, kernelInit=init)

    output = keras.layers.Dense(
        inputShape[1], activation=None, kernel_initializer=init
    )(output)

    return keras.Model(inputs=input, outputs=output, name="attentionModel")


if __name__ == "__main__":
    model = attentionModel((timestepsPerBatch, 128))
    model.summary()
