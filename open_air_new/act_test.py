import numpy as np
import sounddevice as sd   

fs = 96000
duration = 1
channels = 8

output_channels = range(3)
input_channels = range(channels)

sd.default.device = "ASIO MADIface USB"
sd.default.channels = channels
sd.default.samplerate = fs

t = np.linspace(0, duration, num=duration*fs)
y = np.cos(2 * np.pi * 10000 * t)
tx = np.zeros((len(t), channels))

for ch in output_channels:
    tx[:, ch] = y

print(np.shape(tx))

rx = np.squeeze(sd.playrec(tx * 0.01, blocking=True))

print(np.shape(rx))
