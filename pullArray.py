# Author: Jacob Dawson
#
# This file is dedicated to reading our midi files into a format
# (a numpy array) for training our machine learning model. Depending on how
# fast we can do this, it might be better to save this read data to its own
# file, or perhaps to just return it/a generator for training the model, I
# haven't quite decided yet.
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
import gc
import pretty_midi
import mido

# where we've got our midis
midisDirectory = "midis"
storedDataDir = "allMusic.npz"


def numpyFromFile(filename):
    """
    Given a filename, return the piano roll for that midi file, in numpy format
    """
    pm = pretty_midi.PrettyMIDI(filename)
    pr = pm.get_piano_roll()
    return pr


def walk():
    """
    Walk through our midi files and return a big numpy array of all their data
    """

    if not os.path.isdir(midisDirectory):
        raise Exception(f"No directory found at {midisDirectory}")

    list_of_files = {}
    for dirpath, _, filenames in os.walk(midisDirectory):
        for filename in filenames:
            if filename.endswith(".mid"):
                list_of_files[filename] = os.sep.join([dirpath, filename])

    allMusic = []
    lengths = []
    for _, v in list_of_files.items():
        try:
            pr = numpyFromFile(v)
        except (EOFError):
            print("EOFError in " + v)
            continue
        except (mido.KeySignatureError):
            print("KeySignatureError in " + v)
            continue
        allMusic.append(pr.astype(np.uint16))
        lengths.append(pr.shape[1])
        # print(pr.shape)
    gc.collect()
    allMusic = np.concatenate(allMusic, axis=1)

    # for some reason, the music gets returned in the format (128, LENGTH).
    # I believe we actually want that in the format (LENGTH, 128).
    musicArr = [allMusic[:, i] for i in range(allMusic.shape[1])]
    musicArr = np.array(musicArr).astype(np.uint16)
    gc.collect()

    print("Music data returned:")
    print(f"type: {musicArr.dtype}")
    print(f"min: {np.min(musicArr)}, max: {np.max(musicArr)}")
    sum = 0
    for length in lengths:
        sum+=length
    avLength = sum / len(lengths)
    print(f"Average length of piano roll: {avLength}")

    return musicArr


if __name__ == "__main__":
    allMusic = walk()
    np.savez_compressed(storedDataDir, a=allMusic)
    print(f"all music: {allMusic.shape}")
