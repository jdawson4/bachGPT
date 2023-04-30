# Author: Jacob Dawson
#
# This file contains the main train loop, using the data pulled from our midis
# in pullArray.py and the architecture/constants declared in arch.py. Here, we
# specify how the keras model will be compiled, how the data will be packaged
# into train and val, and the main .fit() function.

from arch import *
from pullArray import *


def loadData():
    # load data from file, or walk:
    if os.path.isfile(storedDataDir):
        with np.load(storedDataDir) as data:
            musicData = data["a"]
    else:
        musicData = walk()

    # split data up into batches
    batchedData = []
    for i in range(0, batchSize * (musicData.shape[0] // batchSize), batchSize):
        batchedData.append(musicData[i : i + batchSize, :])
    batchedData = np.array(batchedData, dtype=musicData.dtype)
    # print(musicData.shape)
    # print(batchedData.shape)
    print(f"original size: {musicData.size}, batched size: {batchedData.size}")

    # now that data is batched, split into train and val:
    train = batchedData[0 : batchedData.shape[0] - 1 : 1, :, :]
    val = batchedData[1 : batchedData.shape[0] : 1, :, :]

    print(f"train shape: {train.shape}, val size: {val.shape}")

    return train, val


def trainLoop():
    train, val = loadData()


if __name__ == "__main__":
    trainLoop()
