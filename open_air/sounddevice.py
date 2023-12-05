# From Zihan Wei
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import scipy.signal as sg
import scipy.io as scio
from scipy.io import wavfile
 
def normalize_signal(signal):
    max = np.max(np.absolute(signal))
    return signal/max
    
if __name__ == "__main__":
    channels = 10
    fs = 96000
    duration = 0.5
    t = np.arange(0, int(duration*fs))/fs

    sd.default.device = "ASIO MADIface USB"
    sd.default.channels = channels
    sd.default.samplerate = fs

    time_domain = []
    for i in range(channels): 
        fs,data =  wavfile.read("ofdm.wav")
        #sig = np.zeros((int(fs*duration), channels))
        sig = np.zeros((len(data), channels))
        #sig[:,i] = 0.01*normalize_signal(sg.chirp(t, f0 = 5e3, f1 = 15e3, t1 = duration))
        sig[:,i] = 0.08*normalize_signal(data)
        #play back an array and record at the same time
        received = sd.playrec(sig, samplerate=fs, blocking=True, channels=channels)
        #estimate power spectral density, returns f = array of sample freq, pxx = power spectral density
        f, pxx = sg.welch(received[:, i], fs=fs)
        plt.plot(f/1e3, 10*np.log10(pxx))
        #time_domain.append(received[:,i])