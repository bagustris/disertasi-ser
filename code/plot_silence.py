# -*- coding: utf-8 -*-
"""
================
Viterbi decoding
================

This notebook demonstrates how to use Viterbi decoding to impose temporal
smoothing on frame-wise state predictions.

Our working example will be the problem of silence/non-silence detection.
"""

# Code source: Brian McFee
# License: ISC

##################
# Standard imports
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import librosa

import librosa.display

#############################################
# Load an example signal
# y, sr = librosa.load('sir_duke_slow.mp3')
y, sr = librosa.load('../sound/Ses01F_impro01_F008.wav')

# And compute the spectrogram magnitude and phase
S_full, phase = librosa.magphase(librosa.stft(y))

###########################################################
# As you can see, there are periods of silence and
# non-silence throughout this recording.
#

# As a first step, we can plot the root-mean-square (RMS) curve
rms = librosa.feature.rms(y=y)[0]
times_t = np.linspace(0, len(y)/sr, num=len(y))
times = librosa.frames_to_time(np.arange(len(rms)))

# plot 
plt.figure(figsize=(12, 10))
plt.subplot(4, 1, 1)
plt.plot(times_t, y)
plt.ylim(-1.05, 1.05)
# plt.title('(a) Waveform', y=-0.02)
plt.xlabel('(a) Waveform')

plt.subplot(4, 1, 2)
# plt.figure(figsize=(12, 4))
plt.plot(times, rms)
plt.axhline(0.065, color='r', alpha=0.5)
plt.xlabel('(b) Silence detection by RMS energy')
plt.ylabel('RMS')
plt.axis('tight')
plt.tight_layout()

# The red line at 0.065 indicates a reasonable threshold for silence detection.
# However, the RMS curve occasionally dips below the threshold momentarily,
# and we would prefer the detector to not count these brief dips as silence.

#####################################################
# As a first step, we will convert the raw RMS score
# into a likelihood (probability) by logistic mapping
#
#   :math:`P[V=1 | x] = \frac{\exp(x - \tau)}{1 + \exp(x - \tau)}`
#
# where :math:`x` denotes the RMS value and :math:`\tau=0.02` is our threshold.
# The variable :math:`V` indicates whether the signal is non-silent (1) or silent (0).
#
# We'll normalize the RMS by its standard deviation to expand the
# range of the probability vector

r_normalized = (rms - 0.065) / np.std(rms)
p = np.exp(r_normalized) / (1 + np.exp(r_normalized))

# We can plot the probability curve over time:
# plt.figure(figsize=(12, 4))
plt.subplot(4, 1, 3)
plt.plot(times, p, label='P[V=1|x]')
plt.axhline(0.51, color='r', alpha=0.5, label='Descision threshold')
plt.xlabel('(c) Silence detection by probability mapping')
plt.axis('tight')
plt.legend()
plt.tight_layout()

#######################################################################
# which looks much like the first plot, but with the decision threshold
# shifted to 0.5.  A simple silence detector would classify each frame
# independently of its neighbors, which would result in the following plot:


# plt.figure(figsize=(12, 6))
# ax = plt.subplot()
# librosa.display.specshow(librosa.amplitude_to_db(S_full, ref=np.max),
#                          y_axis='log', x_axis='time', sr=sr)
plt.subplot(4, 1, 4)
plt.step(times, p>=0.51, label='Non-silent')
plt.xlabel('(d) Silence/non-silence detection in binary value')
plt.axis('tight')
plt.ylim([0, 1.05])
plt.legend()
plt.tight_layout()

plt.show()
