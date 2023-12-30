import numpy as np
import pandas as pd
import scipy as sp
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import munotes as mn


from sklearn.ensemble import RandomForestClassifier

from joblib import dump, load

from CreateFeatures import Create_Features
from utils import butter_bandpass

name_to_id = {"S" : 1, "A" : 2, "T" : 3, "B" : 4}
id_to_name = {1 : "S", 2 : "A", 3 : "T", 4 : "B"}


frame_length = 2024
hop_length = 512

#open audio file
def Open_Audiofile(path,filename,sr):
    original_audio, sr = librosa.load('{}/{}'.format(path,filename),sr=sr)
    return original_audio, sr

class VoiceSeparator2:

    def __init__(self):
        self.original_audio = None
        self.segmented_audio = None
        self.Seperate_Voices = None
        self.classifier = None

        self.sample_rate = 44100
        self.names = ['S','A','T','B']
        pass

    def __call__(self,path,filename):

        self.original_audio, _ = Open_Audiofile(path,filename,self.sample_rate)

        print(" ------ Segmenting ------------ ")
        self.segmented_audio = self.Segment_Frames()

        print(" ------ Features --------------- ")
        fcr = Create_Features(self.segmented_audio,self.sample_rate)
        self.features = fcr()

        print(" ------ Create new frames ------ ")
        self.Frames = self.split_frames()

        print(" ------ Separate --------------- ")
        separation = self.seperate()
        All_Voices = self.DeSegment_Frames(separation)

        print(" ------ Save Audio ------------- ")
        self.Save_Separate_Voices(All_Voices)

        pass

    def Segment_Frames(self):
        num_frames = 1 + int((len(self.original_audio) - frame_length) / hop_length)
        segmented_audio = np.zeros((num_frames, frame_length))

        for i in range(num_frames):
            start = i * hop_length
            end = start + frame_length
            segmented_audio[i, :] = self.original_audio[start:end]

        return segmented_audio

    def DeSegment_Frames(self,separation):
        All_Voices = {}

        for rule in ['rule1','rule2','rule3']:
            frames = separation[rule]
            Voices = {}
            for name in self.names:
                segments = frames[name]
                new_audio = np.zeros((len(segments) * hop_length + frame_length))

                for i in range(len(segments)):
                    start = i * hop_length
                    end = start + frame_length
                    new_audio[start:end] = 0.5 * np.array(segments[i])

                Voices[name] = new_audio

            All_Voices[rule] = Voices

        return All_Voices

    def split_frames(self):
        Frames = []
        for idx in range(len(self.segmented_audio)):
            frames = self.split_one_frame(idx)
            Frames.append(frames)

        return Frames

    def split_one_frame(self,idx):

        frame = self.segmented_audio[idx]
        stft = librosa.stft(frame)
        manitude_spectrum = np.abs(stft)

        pitches = self.features['pitch'][idx]
        pitches = np.sort(pitches)
        split_frames = []

        freq = [librosa.note_to_hz('C1'),librosa.note_to_hz('C8')]
        freq[1:1] = pitches

        cuts = [freq[i:i+3] for i in range(len(freq)-2)]

        for cut in cuts:
            low_cut = cut[1] - abs(cut[0] - cut[1]) / 2
            high_cut = cut[1] + abs(cut[1] - cut[2]) / 2
            split_frames.append(butter_bandpass(frame, low_cut,high_cut,self.sample_rate, order = 4))

        return split_frames

    def seperate(self):
        separation = {}
        separation['rule1'] = self.rule1()
        separation['rule2'] = self.rule2()
        separation['rule3'] = self.rule3()
        return separation

    def rule1(self):

        range_S = [librosa.note_to_hz('C4'),librosa.note_to_hz('C6')]
        range_A = [librosa.note_to_hz('F3'),librosa.note_to_hz('F5')]
        range_T = [librosa.note_to_hz('C3'),librosa.note_to_hz('G5')]
        range_B = [librosa.note_to_hz('E2'),librosa.note_to_hz('E4')]

        S = [list(np.zeros(len(self.segmented_audio[0]))) for _ in range(len(self.segmented_audio))]
        A = [list(np.zeros(len(self.segmented_audio[0]))) for _ in range(len(self.segmented_audio))]
        T = [list(np.zeros(len(self.segmented_audio[0]))) for _ in range(len(self.segmented_audio))]
        B = [list(np.zeros(len(self.segmented_audio[0]))) for _ in range(len(self.segmented_audio))]

        for idx,frames,pitches in zip(np.arange(len(self.Frames)),self.Frames,self.features['pitch']):
            for frame,pitch in zip(frames,pitches):
                if pitch > range_S[0] and pitch < range_S[1]:
                    S[idx] = list(frame)
                if pitch > range_A[0] and pitch < range_A[1]:
                    A[idx] = list(frame)
                if pitch > range_T[0] and pitch < range_T[1]:
                    T[idx] = list(frame)
                if pitch > range_B[0] and pitch < range_B[1]:
                    B[idx] = list(frame)

        return {'S' : S, 'A': A, 'T' : T, 'B' : B}

    def rule2(self):

        S = [list(np.zeros(len(self.segmented_audio[0]))) for _ in range(len(self.segmented_audio))]
        A = [list(np.zeros(len(self.segmented_audio[0]))) for _ in range(len(self.segmented_audio))]
        T = [list(np.zeros(len(self.segmented_audio[0]))) for _ in range(len(self.segmented_audio))]
        B = [list(np.zeros(len(self.segmented_audio[0]))) for _ in range(len(self.segmented_audio))]


        for idx,frames,pitches in zip(np.arange(len(self.Frames)),self.Frames,self.features['pitch']):
            sorted_pitches = np.sort(self.features['pitch'][idx])
            for frame,pitch in zip(frames,pitches):
                j = np.where(sorted_pitches == pitch)[0][0]
                sorted_pitches[j] = -1
                if j == 0:
                    S[idx] = list(frame)
                if j == 1:
                    A[idx] = list(frame)
                if j == 2:
                    T[idx] = list(frame)
                if j == 3:
                    B[idx] = list(frame)

        return {'S' : S, 'A': A, 'T' : T, 'B' : B}

    def rule3(self):
        th_S = [0.03,0.05]
        th_A = [0.05,0.08]
        th_T = [0.08,1.5]

        S = [list(np.zeros(len(self.segmented_audio[0]))) for _ in range(len(self.segmented_audio))]
        A = [list(np.zeros(len(self.segmented_audio[0]))) for _ in range(len(self.segmented_audio))]
        T = [list(np.zeros(len(self.segmented_audio[0]))) for _ in range(len(self.segmented_audio))]
        B = [list(np.zeros(len(self.segmented_audio[0]))) for _ in range(len(self.segmented_audio))]

        for idx,frames,harmonic_energy_ratios in zip(np.arange(len(self.Frames)),self.Frames,self.features['harmonic_energy_ratios']):
            for frame,harmonic_energy_ratio in zip(frames,harmonic_energy_ratios):
                if harmonic_energy_ratio > th_S[0] and harmonic_energy_ratio < th_S[1]:
                    S[idx] = list(frame)
                if harmonic_energy_ratio > th_A[0] and harmonic_energy_ratio < th_A[1]:
                    A[idx] = list(frame)
                if harmonic_energy_ratio > th_T[0] and harmonic_energy_ratio < th_T[1]:
                    T[idx] = list(frame)
                else:
                    B[idx] = list(frame)

        return {'S' : S, 'A': A, 'T' : T, 'B' : B}



    def Save_Separate_Voices(self,All_Voices):
        for rule in ['rule1','rule2','rule3']:
            Voices = All_Voices[rule]
            for name in self.names:
                audio = Voices[name]
                plt.plot(audio)
                plt.savefig('{}_{}.png'.format(rule,name))
                plt.show()
                sf.write('{}_{}.wav'.format(rule,name), audio, self.sample_rate)
        pass


if __name__ == "__main__":

    dir = 'Audio'
    filename = 'mixture_1.wav'

    Separator = VoiceSeparator2()
    Separator(dir, filename)
