import numpy as np
import scipy.signal as sg
import matplotlib.pyplot as plt

if __name__ == "__main__":
    bits = 7
    rep = 16
    R = 3000
    fs = 48000
    ns = fs / R
    fc = 15e3

    # generate the signal to be transmitted
    d = np.array(sg.max_len_seq(bits)[0]) * 2 - 1.0
    tbits = np.array(sg.max_len_seq(bits)[0])
    u = np.tile(d, rep)
    u = sg.resample_poly(u, ns, 1)
    s = np.real(u * np.exp(2j * np.pi * fc * np.arange(len(u)) / fs))
    s /= np.max(np.abs(s))
    s = np.concatenate((np.zeros((int(fs*0.1),)), s, np.zeros((int(fs*0.1),))))

    # pass through the channel
    r = np.copy(s)
    r = r + 0.3 * np.roll(r, 300)
    r += 0.01 * np.random.randn(s.shape[0])

    # xcorr
    v = r * np.exp(-2j * np.pi * fc * np.arange(len(r)) / fs)
    xcor = sg.fftconvolve(v, sg.resample_poly(d[::-1].conj(), ns, 1))

    print(len(v))
    print(len(sg.resample_poly(d[::-1].conj(), ns, 1)))
    # plot
    plt.plot(np.abs(xcor))
    plt.xlim([np.argmax(xcor) - 100, np.argmax(xcor) + 500])
    plt.xlabel('Delay [samples]')
    plt.ylabel('Cross-correlation')
    plt.show()
