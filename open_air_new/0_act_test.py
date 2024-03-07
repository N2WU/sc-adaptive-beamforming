import numpy as np
import sounddevice as sd   

def testbed(s_tx,n_tx,n_rx,Fs):
    # testbed constants
    sd.default.device = "ASIO MADIface USB"
    sd.default.channels = 20 
    sd.default.samplerate = Fs

    # devices [0-15] are for array
    # devices [16-20] are for users
    tx = np.zeros((len(s_tx[:,0]),20))
    if n_tx < n_rx:
        # implies uplink
        s_tx = s_tx.flatten()
        for ch in range(16,16+n_tx):
            tx[:,ch] = s_tx
        print("Transmitting...")
        rx_raw = sd.playrec(tx * 0.05, samplerate=Fs, blocking=True)
        rx = rx_raw[:,:n_rx]
        print("Received")
    else:
        # downlink, meaning s_tx is already weighted array
        for ch in range(n_tx):
            tx[:,ch] = s_tx[:,ch]
        print("Transmitting...")
        rx_raw = sd.playrec(tx * 0.05, samplerate=Fs, blocking=True)
        rx = rx_raw[:,16:16+n_rx] #testbed specific
        print("Received")
    return rx

if __name__ == "__main__":
    fs = 96000
    duration = 2

    n_tx = 12
    n_rx = 1

    t = np.linspace(0, duration, num=duration*fs)
    y = np.cos(2 * np.pi * 5e3 * t)

    if n_tx == 1:
        y = y.reshape(-1,1)
    else:
        y = np.tile(y,(n_tx,1))
        y = y.T
    rx = testbed(y,n_tx,n_rx,fs)
