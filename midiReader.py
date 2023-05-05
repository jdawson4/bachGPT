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
        # not sure if this is the right scaling factor; do we want our inputs
        # [-1,1] or [0,1]?
        pr = pr / 256
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


def determineSizeOfSet(directory=midisDirectory):
    """
    this is a copy-paste of our generator, but instead of yielding, we
    determine the size of our training set
    """

    size = 0

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
        # not sure if this is the right scaling factor; do we want our inputs
        # [-1,1] or [0,1]?
        pr = pr / 256
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
    return size


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

    # see the whole size (note: only do this when determineSizeOfSet is updated)
    # print(f"size of train dataset: {determineSizeOfSet(midisDirectory)}")
    # size of whole dataset: 44898

    # print(f"size of validation dataset: {determineSizeOfSet(valMidisDirectory)}")
    # size of validation dataset: 1683
