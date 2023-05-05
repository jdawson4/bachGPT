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

if not os.path.isdir(midisDirectory):
    raise Exception(f"No directory found at {midisDirectory}")

list_of_files = {}
for dirpath, _, filenames in os.walk(midisDirectory):
    for filename in filenames:
        if filename.endswith(".mid"):
            list_of_files[filename] = os.sep.join([dirpath, filename])


def numpyFromFile(filename):
    """
    Given a filename, return the piano roll for that midi file, in numpy format
    """
    pm = pretty_midi.PrettyMIDI(filename)
    pr = pm.get_piano_roll()
    return pr


def getNextMusicChunk():
    for _, v in list_of_files.items():
        try:
            pr = numpyFromFile(v)
        except EOFError:
            print("EOFError in " + v)
            continue
        except mido.KeySignatureError:
            print("KeySignatureError in " + v)
            continue
        pr = np.swapaxes(pr, axis1=0, axis2=1)
        pr = (pr / 256)
        pr = pr.astype(np.float16)
        for i in range(
            0,
            (timestepsPerBatch * (pr.shape[0] // timestepsPerBatch))
            - timestepsPerBatch,
            timestepsPerBatch,
        ):
            yield pr[i : i + timestepsPerBatch, :], pr[
                i + timestepsPerBatch : i + (2 * timestepsPerBatch), :
            ]


if __name__ == "__main__":
    returnSignature = tf.TensorSpec(shape=[2, timestepsPerBatch, 128], dtype=tf.float16)
    dataset = (
        tf.data.Dataset.from_generator(
            getNextMusicChunk, output_signature=returnSignature
        )
        .apply(tf.data.experimental.assert_cardinality(numberOfBatches))
        .prefetch(batchSize * 2)
    )

    for x,y in dataset.take(111):
        print(f"x shape: {x.shape}, y shape: {y.shape}")
        allData = np.concatenate((x,y), - 1)
        print(f"max: {np.max(allData)}, min: {np.min(allData)}")
        #if np.max(allData) > 1:
        #    print("bigger than 1")
