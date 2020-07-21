import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft, ifft

N = 256     # N-FFT
M = 63      # window size
f0 = 1000
fs = 10000
A0 = .8 
hN = N//2 
hM = (M+1)//2
fftbuffer = np.zeros(N)
X1 = np.zeros(N, dtype='complex')
X2 = np.zeros(N, dtype='complex')

x = A0 * np.cos(2*np.pi*f0/fs*np.arange(-hM+1,hM))
w = np.hamming(M)

# plot demo windowing
plt.figure(figsize=(9,3))
plt.subplot(131)
plt.plot(x)
plt.subplot(132)
plt.plot(np.arange(-hM+1, hM), w, 'b', lw=1.5)
plt.axis([-hM+1, hM, 0, 1])
plt.subplot(133)
xw = x*w
plt.plot(np.arange(-hM+1, hM), xw, 'b', lw=1.5)
plt.axis([-hM+1, hM, -1, 1])

plt.tight_layout()
plt.savefig('../fig/windowing_demo.pdf')
plt.show()
