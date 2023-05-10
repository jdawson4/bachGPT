# Author: Jacob Dawson
#
# This file is dedicated to reading our midi files into a format
# (a numpy array) for training our machine learning model. Because of our
# limitations, I've found that it's best to use tensorflow's Dataset utilities
# to load and preprocess our midis
#
# In any case, I've poked around the internet and I've copied some people's
# code, but I was unsure of permissions, so I've removed that. In the end,
# I've decided to use the library pretty_midi, which has a few benefits:
# 1. used by lots of people for this purpose
# 2. its get_piano_roll function, which returns numpy data
# 3. MIT license, so I can use it without worry!

# import mido # not needed but a cool library!
import numpy as np
import os
import pretty_midi
import mido
import tensorflow as tf

from arch import *

# where we've got our midis
midisDirectory = "midis"
valMidisDirectory = "val_midis"

# these can be determined using the (maybe commented out) script in "main".
# I've done some experimenting and apparently you MUST set the cardinality
# of your datasets--frustrating, because I'd rather not hardcode this based
# on our dataset. Oh well.
datasetSize = 44898
valDatasetSize = 1683
midiStandardDeviation = 13.4
midiMean = 2.05


def numpyFromFile(filename):
    """
    Given a filename, return the piano roll for that midi file, in numpy format
    """
    pm = pretty_midi.PrettyMIDI(filename)
    pr = pm.get_piano_roll()
    return pr


def getNextMusicChunk(directory=midisDirectory):
    """
    This is our generator. Using this, one can declare a dataset to iterate
    through our data without having to load the whole thing in RAM. Tradeoff:
    disk speed.
    """

    if not os.path.isdir(directory):
        raise Exception(f"No directory found at {directory}")

    list_of_files = {}
    for dirpath, _, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith(".mid"):
                list_of_files[filename] = os.sep.join([dirpath, filename])

    for _, v in list_of_files.items():
        try:
            pr = numpyFromFile(v)
        except EOFError:
            # print("EOFError in " + v)
            continue
        except mido.KeySignatureError:
            # print("KeySignatureError in " + v)
            continue

        # need some light preprocessing:
        # we want this organized in the shape [length, 128]
        pr = np.swapaxes(pr, axis1=0, axis2=1)
        # to scale, we'll demean and divide by standard deviation:
        pr = (pr - midiMean) / midiStandardDeviation
        # let's also return as float16s
        pr = pr.astype(np.float16)

        # we want to return x and y of a certain size, and offset from one
        # another. We use yield for this, a function I was unfamiliar with!
        for i in range(
            0,
            (timestepsPerBatch * (pr.shape[0] // timestepsPerBatch))
            - timestepsPerBatch,
            timestepsPerBatch,
        ):
            # and here's where the dataset gets its values from!
            yield pr[i : i + timestepsPerBatch, :], pr[
                i + timestepsPerBatch : i + (2 * timestepsPerBatch), :
            ]


def determineSetCharacteristics(directory=midisDirectory):
    """
    this is a copy-paste of our generator, but instead of yielding, we
    determine the size of our training set
    """

    size = 0
    maxNote = 0.0
    means = []
    stdevs = []

    if not os.path.isdir(directory):
        raise Exception(f"No directory found at {directory}")

    list_of_files = {}
    for dirpath, _, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith(".mid"):
                list_of_files[filename] = os.sep.join([dirpath, filename])

    for _, v in list_of_files.items():
        try:
            pr = numpyFromFile(v)
        except EOFError:
            # print("EOFError in " + v)
            continue
        except mido.KeySignatureError:
            # print("KeySignatureError in " + v)
            continue
        
        # record characteristics
        if np.max(pr) > maxNote:
            maxNote = np.max(pr)
        means.append(np.mean(pr))
        stdevs.append(np.std(pr))

        # need some light preprocessing:
        # we want this organized in the shape [length, 128]
        pr = np.swapaxes(pr, axis1=0, axis2=1)
        # to scale, we'll demean and divide by standard deviation:
        pr = (pr - midiMean) / midiStandardDeviation
        # let's also return as float16s
        pr = pr.astype(np.float16)

        # we want to return x and y of a certain size, and offset from one
        # another. We use yield for this, a function I was unfamiliar with!
        for i in range(
            0,
            (timestepsPerBatch * (pr.shape[0] // timestepsPerBatch))
            - timestepsPerBatch,
            timestepsPerBatch,
        ):
            size += 1

    # these aren't actually the mean and standard deviation but they're close
    mean = np.mean(np.array(means))
    stdev = np.mean(np.array(stdevs))
    print(f"size of dataset: {size}")
    print(f"max of dataset: {maxNote}")
    print(f"mean: {mean}, standard deviation: {stdev}\n")


if __name__ == "__main__":
    returnSignature = tf.TensorSpec(shape=(timestepsPerBatch, 128), dtype=tf.float16)
    dataset = (
        tf.data.Dataset.from_generator(
            getNextMusicChunk, output_signature=(returnSignature, returnSignature)
        )
        .apply(tf.data.experimental.assert_cardinality(datasetSize))
        .prefetch(batchSize * 2)
    )
    valDataset = (
        tf.data.Dataset.from_generator(
            lambda: getNextMusicChunk(valMidisDirectory),
            output_signature=(returnSignature, returnSignature),
        )
        .apply(tf.data.experimental.assert_cardinality(valDatasetSize))
        .prefetch(batchSize * 2)
    )

    # get some characteristics of our dataset
    for x, y in dataset.take(1):
        # determine shape returned
        print(f"x shape: {x.shape}, y shape: {y.shape}")
        allData = np.concatenate((x, y), -1)
        # and range:
        print(f"max: {np.max(allData)}, min: {np.min(allData)}")
        # if taking more than one, might be better to see range above threshold:
        # if np.max(allData) > 1:
        #    print("bigger than 1")
    for x, y in valDataset.take(1):
        allData = np.concatenate((x, y), -1)
        print(f"val dataset returns: {allData.shape}")

    # get some stats about our data:
    determineSetCharacteristics()
    determineSetCharacteristics(valMidisDirectory)
    # size of train dataset: 44898
    # size of validation dataset: 1683
    # seems like the max value for midis is 564, which seems strange
    # the approximate mean is 2.05, standard deviation is around 13.4
