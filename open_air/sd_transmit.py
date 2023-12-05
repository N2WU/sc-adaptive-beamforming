# From Zihan Wei
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import scipy.signal as sg
import scipy.io as scio
from scipy.io import wavfile

class sd_transmit():
    device = "ASIO MADIface USB"
    channels = 10
    fs = 96000

    def __init__(self, device,channels,fs,s) -> None:
        self.device = device
        self.channels = channels
        self.fs = fs
        self.s = s
        pass
        
    def transmit(self):
        channels = 10
        fs = 96000

        sd.default.device = "ASIO MADIface USB"
        sd.default.channels = channels
        sd.default.samplerate = fs

        time_domain = []
        sig = np.zeros((np.length(self.s),self.channels))

        for i in range(channels): 
            sig[:,i] = 0.08*self.normalize_signal(self.s)
            #play back an array and record at the same time
            received = sd.playrec(sig, samplerate=fs, blocking=True, channels=self.channels)
            #estimate power spectral density, returns f = array of sample freq, pxx = power spectral density
            f, pxx = sg.welch(received[:, i], fs=fs)
            plt.plot(f/1e3, 10*np.log10(pxx))
            #time_domain.append(received[:,i])
        
    def normalize_signal(signal):
        max = np.max(np.absolute(signal))
        return signal/max