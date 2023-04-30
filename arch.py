# Author: Jacob Dawson
#
# This file contains the architecture for our model.
# I believe that the final shape of our inputs will have 128 notes in [BATCH
# SIZE HERE] timesteps. With this in mind, the model should take the shape of
# the input should be [batchSize, timestepsPerBatch, notes (128)].

import tensorflow as tf
from tensorflow import keras

seed = 7
batchSize = 1024


def attnLayer(
    input,
    kernelInit,
    layerCounter,
    heads=8,
    kDim=32,
    out_shape=128,
    residual=True,
    dropout=0.33,
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

    input = keras.layers.Input(shape=inputShape, dtype=tf.float32)

    layerCounter += 1
    a1 = attnLayer(
        input=input, layerCounter=layerCounter, kernelInit=init, residual=False
    )
    layerCounter += 1
    a2 = attnLayer(input=a1, layerCounter=layerCounter, kernelInit=init, residual=False)
    layerCounter += 1
    a3 = attnLayer(input=a2, layerCounter=layerCounter, kernelInit=init)
    layerCounter += 1
    a4 = attnLayer(input=a3, layerCounter=layerCounter, kernelInit=init)
    layerCounter += 1
    a5 = attnLayer(input=a4, layerCounter=layerCounter, kernelInit=init)
    layerCounter += 1
    a6 = attnLayer(input=a5, layerCounter=layerCounter, kernelInit=init)
    layerCounter += 1
    a7 = attnLayer(input=a6, layerCounter=layerCounter, kernelInit=init)
    layerCounter += 1
    a8 = attnLayer(input=a7, layerCounter=layerCounter, kernelInit=init)
    layerCounter += 1
    a9 = attnLayer(input=a8, layerCounter=layerCounter, kernelInit=init)
    layerCounter += 1
    a10 = attnLayer(input=a9, layerCounter=layerCounter, kernelInit=init)

    output = keras.layers.Dense(
        units=inputShape[1], activation=None, kernel_initializer=init
    )(a10)

    return keras.Model(inputs=input, outputs=output, name="attentionModel")


if __name__ == "__main__":
    model = attentionModel((batchSize, 128))
    model.summary()
