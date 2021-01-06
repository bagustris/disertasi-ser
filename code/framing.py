import matplotlib.pyplot as plt
import numpy as np
from scipy.io.wavfile import read

(fs, x) = read('../sound/Ses01F_impro01_F008.wav')
M = 512     # frame/window length
H = 160     # hop length
start = int(.62*fs)

plt.figure(1)
x0 = x[start:start+3*M]/float(max(x))
plt.plot(x0)

offset = 1.5
x1 = np.zeros(3*M)+offset
x1[0:M] += (x0[0:M] * np.hamming(M))

offset = 2.5
x2 = np.zeros(3*M)+offset
x2[H:M+H] += (x0[H:M+H] * np.hamming(M))

offset = 3.5
x3 = np.zeros(3*M)+offset
x3[H*2:M+H*2] += (x0[2*H:M+H*2] * np.hamming(M))

offset = 4.5
x4 = np.zeros(3*M)+offset
x4[H*3:M+H*3] += (x0[3*H:M+H*3] * np.hamming(M))

offset = 5.5
x5 = np.zeros(3*M)+offset
x5[H*4:M+H*4] += (x0[4*H:M+H*4] * np.hamming(M))


# plot
plt.axis([0, 1200, max(x0)+5.5, min(x0)])
plt.plot(x5,'b')
plt.plot(x4,'b')
plt.plot(x3,'b')
plt.plot(x2,'b')
plt.plot(x1,'b')


plt.tight_layout()
plt.savefig('../fig/framing.svg')
plt.show()
