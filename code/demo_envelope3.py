# demo_envelope3.py
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt

def getEnvelope (inputSignal):
    # Taking the absolute value
    
    # absoluteSignal = []
    # for sample in inputSignal:
    #     absoluteSignal.append (abs (sample))

    absoluteSignal = np.abs(inputSignal)
    # Peak detection
    
    intLen = 60 # Interval length, experiment with this number, it depends on your sample frequency and highest "whistle" frequency
    
    outputSignal = []
    # bI = base index
    # lI = lookback index
    for bI in range(intLen, len(absoluteSignal)):
        maximum = 0
        for lI in range(intLen):
            maximum = max(absoluteSignal[bI - lI], maximum)
        outputSignal.append(maximum)
        # outputSignal = [max(absoluteSignal[bI-lI], 0) for lI in range(intLen)]

    return outputSignal

s = '../sound/Ses01F_impro01_F008.wav'
fs, w = wavfile.read(s)
w = w/max(abs(w))
env = getEnvelope(w)

# calcluate f_o
import crepe
time, frequency, confidence, activation = crepe.predict(w, fs, viterbi=True)  


# obtaine time axis
time_t = np.linspace(0, len(w) / fs, num=len(w))
time_f = np.linspace(0, len(frequency)/100, num=len(frequency))

plt.subplot(211)
# plt.xticks(" ")
plt.plot(w, label='Waveform')
plt.ylim(-1.05, 1.05)
plt.plot(env, label='ENV')
plt.subplot(212)
plt.plot(time_f, frequency, color='r', label='$f_o$')
plt.ylabel('Frequency (Hz)')
plt.show()
