"""
@file mfcc-mel.py
@author Bagus Tris Atmaja (bagus@ep.its.ac.id)
Returns:
    plot MFCC, MEL, and LOG MEL SPECTROGRAM
"""

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
# import pylustrator
# pylustrator.start()


file = '../sound/Ses01F_impro01_F008.wav'

y, sr = librosa.load(file, sr=None)
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_fft=400, hop_length=160, 
                            n_mfcc=13)
mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64, n_fft=400,          
                                     hop_length=160)
logmel = librosa.power_to_db(mel, ref=np.max)
# mfcc = librosa.feature.mfcc(S=logmel)

plt.figure(figsize=(4, 5))
plt.subplot(311)
librosa.display.specshow(mfcc, x_axis='time', sr=sr, hop_length=160)
plt.subplot(312)
librosa.display.specshow(mel, x_axis='time',
                         y_axis='mel', sr=sr, hop_length=160,
                         fmax=8000)
plt.subplot(313)
librosa.display.specshow(logmel, x_axis='time',
                         y_axis='mel', sr=sr, hop_length=160,
                         fmax=8000)
plt.show()
# plt.savefig('../fig/mfcc-mel.svg')
