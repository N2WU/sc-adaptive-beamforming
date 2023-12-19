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
    n_rx = 12
    reflection = np.array([1, 0.5])
    SNR_db = 10
    SNR = np.power(10, SNR_db/10)
    x_tx = np.array([5])
    y_tx = np.array([20])

    # generate the signal to be transmitted
    d = np.array(sg.max_len_seq(bits)[0]) * 2 - 1.0
    tbits = np.array(sg.max_len_seq(bits)[0])
    u = np.tile(d, rep)
    u = sg.resample_poly(u, ns, 1)
    s = np.real(u * np.exp(2j * np.pi * fc * np.arange(len(u)) / fs))
    s /= np.max(np.abs(s))

    s = np.concatenate((np.zeros((int(fs*0.1),)), s, np.zeros((int(fs*0.1),))))
    # pass through the channel
    def simulate(el_spacing):
        x_rx = np.arange(0, el_spacing*n_rx, el_spacing)
        y_rx = np.zeros_like(x_rx)
        dx, dy = x_rx - x_tx, y_rx - y_tx
        r = np.random.randn(s.shape[0],) / SNR
        for i in range(len(reflection)):
            reflect = reflection[i]
            for j in range(n_tx):
                r[delay[j]:delay[j]+len(s)] += s_multi[j,:] * reflect

        # xcorr
        v = r * np.exp(-2j * np.pi * fc * np.arange(len(r)) / fs)
        xcor = sg.fftconvolve(v,sg.resample_poly(d[::-1].conj(),ns,1),'full')

        mse = 10 * np.log10(
            np.mean(np.abs(d_rls[n_training + 1 : -1] - d_hat[n_training + 1 : -1]) ** 2)) 
    # plot
    plt.plot(np.abs(xcor))
    plt.xlim([np.argmax(xcor) - 100, np.argmax(xcor) + 500])
    plt.xlabel('Delay [samples]')
    plt.ylabel('Cross-correlation')
    plt.show()