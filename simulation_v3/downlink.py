import numpy as np
import scipy.signal as sg
from commpy.modulation import QAMModem
import matplotlib.pyplot as plt
import sounddevice as sd
from tqdm import tqdm

class downlink():

    fc = 6.5e3
    M0 = 16 # Type of QAM
    duration = 5
    Fs = 96000
    n_repeat = 20
    R = 3000
    Ns = 4
    channels = 1
    M = 1  # 12 - the channels to be use
    num_tx = 4
    d0 = 0.05
    c = 343
    n_path = 1
    n_sim = 1
    feedforward_taps = 1
    feedbackward_taps = 1
    alpha_rls = 0.9999

    theta_start = -90
    theta_end = 90

    reflection_list = np.array([1]) #elminimate reflections for now
    x_tx_list = np.arange(0, d0*channels, d0)
    y_tx_list = np.zeros_like(x_tx_list)

    snr_list = [15] # np.arange(0, 15, 1) #changed from 15

    def __init__(self, fc, n_path, n_sim, theta, wk, apply_bf = True) -> None:
        self.fc = fc
        self.n_path = n_path
        self.n_sim = n_sim
        self.theta = theta
        self.wk = wk
        self.apply_bf = apply_bf

        self.MSE_SNR = np.zeros((len(self.snr_list), n_sim))
        self.ERR_SNR = np.zeros_like(self.MSE_SNR)

        self.preamble = np.array(sg.max_len_seq(bits)[0]) * 2 - 1.0

        #self.return_symbols = 
        self.u = np.tile(self.preamble, self.n_repeat)
        self.u = sg.resample_poly(self.u, self.ns, 1)
        self.s = np.real(self.u * np.exp(2j * np.pi * self.fc * np.arange(len(self.u)) / self.fs))
        self.steering_vec = self.calc_steering_vec(theta, wk)
        self.s = np.dot(np.reshape(self.steering_vec,[-1,1]),np.reshape(self.s,[1,-1]))
        self.s /= np.max(np.abs(self.s))
        pass

    def calc_steering_vec(self, theta, wk):
        if self.apply_bf == True:
            steering_vec = np.exp(-1j*2*np.pi*np.sin(theta)*np.arange(self.num_tx)*self.d0)
            steering_vec = wk
        elif self.apply_bf == False:
            steering_vec = np.ones(self.num_tx)/self.num_tx
        return steering_vec

    def simulation(self):
        x_rx = np.array([0])
        y_rx = np.array([1])
        rng = np.random.RandomState(2021)
        for i_snr, snr in enumerate(self.snr_list):
            SNR = np.power(10, snr/10)
            print(f"SNR={snr}dB")
            for i_sim in tqdm(range(self.n_sim)):
                r_singlechannel = rng.randn(self.duration * self.Fs) / SNR
                r_singlechannel = r_singlechannel.astype('complex128')
                # receive the signal
                for i in range(self.n_path):
                    x_tx, y_tx = self.x_tx_list[i], self.y_tx_list[i]
                    reflection = self.reflection_list[i]
                    dx, dy = x_rx - x_tx, y_rx - y_tx
                    d_rx_tx = np.sqrt(dx**2 + dy**2)
                    delta_tau = d_rx_tx / self.c
                    delay = np.round(delta_tau * self.Fs).astype(int)
                    for j, delay_j in enumerate(delay):
                        # print(delay_j/4)
                        r_singlechannel[delay_j:delay_j+len(self.s_tx[:,1])] += self.s_tx[:,j] * reflection #(eliminated for now)

                # get S(theta) -- the angle of the source
                # phase correction/precombining cannot work
                r = r_singlechannel
                r_fft = np.fft.fft(r, axis=0)
                freqs = np.fft.fftfreq(len(r), 1/self.Fs)

                # processing the signal we get from bf (perhaps not, though y adds additional channel effects?)
                #r_singlechannel_1 = y
                r_singlechannel_1 = r_singlechannel #it's delay corrected...
                #r_singlechannel_1 = np.copy(self.s)
                K_list = [self.n_path]
                K = self.n_path
                if K == 1:
                    r_singlechannel_1 = r_singlechannel_1.reshape(len(r_singlechannel_1), 1)
                v_singlechannel = [] #gah
                peaks_rx = 0
                for i in range(K): #this does it twice, one for reflect and one for los
                    r = np.squeeze(r_singlechannel_1[:, i])
                    v = r * np.exp(-2 * np.pi * 1j * self.fc * np.arange(len(r))/self.Fs)
                    g = sg.decimate(v, self.df)
                    fractional_spacing = self.nsps/self.df
                    _, rc_rx = self.rrcosfilter(16 * int(self.fs / self.R), 0.5, 1 / self.R, self.fs)
                    v = np.convolve(g, rc_rx, "full")
                    # v /= np.sqrt(np.var(v))
                    if np.var(v) > 1e-6:
                        v /= np.sqrt(np.var(v))
                    else:
                        v = np.zeros_like(v)
                    if i == 0:
                        xcorr_for_peaks = np.abs(sg.fftconvolve(sg.decimate(v, self.Ns), self.preamble[::-1].conj()))
                        xcorr_for_peaks /= xcorr_for_peaks.max()
                        time_axis_xcorr = np.arange(0, len(xcorr_for_peaks)) / self.R * 1e3  # ms
                        peaks_rx, _ = sg.find_peaks(
                            xcorr_for_peaks, height=0.2, distance=len(self.preamble) - 100
                        )
                        # plt.figure()
                        # plt.plot(np.abs(xcorr_for_peaks))
                        # plt.show()
                    v = v[peaks_rx[1] * self.Ns :]
                    v_singlechannel.append(v)
                d = np.tile(self.preamble, 3)

                v_rls = np.zeros((K, len(v_singlechannel[0])), dtype=complex)
                for channel in range(K):
                    v = v_singlechannel[channel]
                    v_rls[channel, 0:len(v)] = v

                for K_channels in K_list:
                    d_hat, mse, n_err, n_training = self.processing(v_rls, d, K_channels, bf=True)
                    self.MSE_SNR[i_snr, i_sim] = mse
                    self.ERR_SNR[i_snr, i_sim] = n_err
                    self.return_symbols[i_snr, i_sim, :] = d_hat

        self.mean_symbols = np.mean(self.return_symbols, axis=1)    
        self.mean_mse = np.mean(self.MSE_SNR, axis=1)
        self.mean_err = np.mean(self.ERR_SNR, axis=1)
        self.mean_v = np.mean(v_rls, axis=0)

    def processing(self, v_rls, d, K=1, bf=True):
        channel_list = {1:[0], 2:[0,11], 3:[0,6,11], 4:[0,4,7,11], 8:[0,1,3,4,6,7,9,11], 6:[0,2,4,6,8,10]}
        if bf == False:
            v_rls = v_rls[channel_list[K], :]
        # v_rls = v_rls[random.sample(range(0, 12), K), :]
        delta = 0.001
        Nplus = 3
        n_training = int(4 * (self.feedforward_taps + self.feedbackward_taps) / self.Ns)
        fractional_spacing = 4
        K_1 = 0.0000
        K_2 = K_1 / 10

        # v_rls = np.tile(v, (K, 1))
        d_rls = d
        x = np.zeros((K,self.feedforward_taps), dtype=complex)
        a = np.zeros((K*self.feedforward_taps,), dtype=complex)
        b = np.zeros((self.feedbackward_taps,), dtype=complex)
        c = np.concatenate((a, -b))
        theta = np.zeros((len(d_rls),K), dtype=float)
        d_hat = np.zeros((len(d_rls),), dtype=complex)
        d_tilde = np.zeros((len(d_rls),), dtype=complex)
        d_backward = np.zeros((self.feedbackward_taps,), dtype=complex)
        e = np.zeros((len(d_rls),), dtype=complex)
        Q = np.eye(K*self.feedforward_taps + self.feedbackward_taps) / delta
        pk = np.zeros((K,), dtype=complex)

        cumsum_phi = 0.0
        sse = 0.0
        for i in np.arange(len(d_rls) - 1, dtype=int):
            nb = (i) * self.Ns + (Nplus - 1) * self.Ns - 1
            xn = v_rls[
                :, int(nb + np.ceil(self.Ns / fractional_spacing / 2)) : int(nb + self.Ns)
                + 1 : int(self.Ns / fractional_spacing)
            ]
            for j in range(K):
                xn[j,:] *= np.exp(-1j * theta[i,j])
            x_buf = np.concatenate((xn, x), axis=1)
            x = x_buf[:,:self.feedforward_taps]

            # p = np.inner(x, a.conj()) * np.exp(-1j * theta[i])
            for j in range(K):
                pk[j] = np.inner(x[j,:], a[j*self.feedforward_taps:(j+1)*self.feedforward_taps].conj())
            p = pk.sum()

            q = np.inner(d_backward, b.conj())
            d_hat[i] = p - q

            if i > n_training:
                # d_tilde[i] = slicing(d_hat[i], M)
                d_tilde[i] = self.modem.modulate(np.array(self.modem.demodulate(np.array(d_hat[i], ndmin=1), 'hard'), dtype=int))
            else:
                d_tilde[i] = d_rls[i]
            e[i] = d_tilde[i] - d_hat[i]
            sse += np.abs(e[i] ** 2)

            # y = np.concatenate((x * np.exp(-1j * theta[i]), d_backward))
            y = np.concatenate((x.reshape(K*self.feedforward_taps), d_backward))

            # PLL
            phi = np.imag(p * np.conj(d_tilde[i] + q))
            cumsum_phi += phi
            theta[i + 1] = theta[i] + K_1 * phi + K_2 * cumsum_phi

            # RLS
            k = (
                np.matmul(Q, y.T)
                / self.alpha_rls
                / (1 + np.matmul(np.matmul(y.conj(), Q), y.T) / self.alpha_rls)
            )
            c += k.T * e[i].conj()
            Q = Q / self.alpha_rls - np.matmul(np.outer(k, y.conj()), Q) / self.alpha_rls

            a = c[:K*self.feedforward_taps]
            b = -c[-self.feedbackward_taps:]

            d_backward_buf = np.insert(d_backward, 0, d_tilde[i])
            d_backward = d_backward_buf[:self.feedbackward_taps]

        err = d_rls[n_training + 1 : -1] - d_tilde[n_training + 1 : -1]
        mse = 10 * np.log10(
            np.mean(np.abs(d_rls[n_training + 1 : -1] - d_hat[n_training + 1 : -1]) ** 2)
        )
        n_err = np.sum(np.abs(err) > 0.01)

        # plt.figure()
        # plt.plot(np.abs(a))
        # plt.show()

        return d_hat, mse, n_err, n_training