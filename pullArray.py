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
import pretty_midi


if __name__ == "__main__":
    list_of_files = {}
    for dirpath, dirnames, filenames in os.walk("midis"):
        for filename in filenames:
            if filename.endswith(".mid"):
                list_of_files[filename] = os.sep.join([dirpath, filename])

    allMusic = []
    for k, v in list_of_files.items():
        pm = pretty_midi.PrettyMIDI(v)
        pr = pm.get_piano_roll()
        allMusic.append(pr)
        print(pr.shape)
    allMusic = np.concatenate(allMusic, axis=1)
    np.savez_compressed("allMusic.npz", allMusic)
    print(f"all music: {allMusic.shape}")
