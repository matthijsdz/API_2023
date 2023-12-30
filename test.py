import numpy as np
import pandas as pd
import scipy as sp
import librosa
import munotes as mn


note = librosa.hz_to_note(20000)
split = note.split('-')
note_name = ""
for n in split:
    note_name = note_name+n


print(mn.notes.Note(note_name))
