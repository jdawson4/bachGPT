# Author: Jacob Dawson
#
# This file contains the main train loop, using the data pulled from our midis
# in pullArray.py and the architecture/constants declared in arch.py. Here, we
# specify how the keras model will be compiled, how the data will be packaged
# into train and val, and the main .fit() function.

from arch import *
from pullArray import *
import gc

keras.mixed_precision.set_global_policy("mixed_float16")


def loadData():
    # load data from file, or walk:
    if os.path.isfile(storedDataDir):
        with np.load(storedDataDir) as data:
            musicData = data["a"]
    else:
        musicData = walk()

    # split data up into batches
    batchedData = []
    for i in range(
        0,
        timestepsPerBatch * (musicData.shape[0] // timestepsPerBatch),
        timestepsPerBatch,
    ):
        batchedData.append(musicData[i : i + timestepsPerBatch, :])
    batchedData = np.array(batchedData, dtype=np.float16)
    # scale the values to between [-1,1]
    maxVal = np.max(batchedData)
    gc.collect()
    batchedData = (batchedData / (maxVal / 2)) - 1
    batchedData = batchedData.astype(np.float16)
    # print(musicData.shape)
    # print(batchedData.shape)
    print(f"original size: {musicData.size}, batched size: {batchedData.size}")

    # now that data is batched, split into train and val:
    train = batchedData[0 : batchedData.shape[0] - 1 : 1, :, :]
    val = batchedData[1 : batchedData.shape[0] : 1, :, :]

    print(f"train shape: {train.shape}, val size: {val.shape}")

    gc.collect()

    return train, val, maxVal


def trainLoop():
    train, val, maxVal = loadData()

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
        x=train,
        y=val,
        batch_size=batchSize,
        epochs=epochs,
        callbacks=EveryKCallback(),
        validation_split=0.3,
        shuffle=True,
    )


if __name__ == "__main__":
    trainLoop()
