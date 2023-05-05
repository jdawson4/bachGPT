# Author: Jacob Dawson
#
# This file contains the main train loop, using the data pulled from our midis
# in pullArray.py and the architecture/constants declared in arch.py. Here, we
# specify how the keras model will be compiled, how the data will be packaged
# into train and val, and the main .fit() function.

from arch import *
from midiReader import *
import gc

keras.mixed_precision.set_global_policy("mixed_float16")


def trainLoop():
    returnSignature = tf.TensorSpec(shape=[2, timestepsPerBatch, 128], dtype=tf.float16)
    dataset = (
        tf.data.Dataset.from_generator(
            getNextMusicChunk, output_signature=returnSignature
        )
        .apply(tf.data.experimental.assert_cardinality(numberOfBatches))
        .prefetch(batchSize * 2)
    )

    model = attentionModel((timestepsPerBatch, 128))
    model.summary()

    model.compile(
        keras.optimizers.Adam(learning_rate=learnRate, beta_1=momentum),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=["mae"],
    )

    ckptsDir = "ckpts"
    if not os.path.isdir(ckptsDir):
        os.mkdir(ckptsDir)

    class EveryKCallback(keras.callbacks.Callback):
        def __init__(self, epoch_interval=epochInterval):
            self.epoch_interval = epoch_interval

        def on_epoch_begin(self, epoch, logs=None):
            if (epoch % self.epoch_interval) == 0:
                self.model.save_weights(
                    ckptsDir + "/ckpt" + str(epoch), overwrite=True, save_format="h5"
                )
                # self.model.save("network", overwrite=True)

    gc.collect()

    model.fit(
        x=dataset,
        batch_size=batchSize,
        epochs=epochs,
        callbacks=EveryKCallback(),
        validation_split=0.3,
        shuffle=True,
    )


if __name__ == "__main__":
    trainLoop()
