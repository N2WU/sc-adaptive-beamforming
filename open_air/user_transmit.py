import numpy as np
import scipy.signal as sg
from commpy.modulation import QAMModem
import matplotlib.pyplot as plt
import sounddevice as sd
from tqdm import tqdm

# this code is identical to bf_multichannel but introduces get_constellation to plot the constellation diagram

class bf_multichannel_const():

    fc = 6.5e3
    M0 = 16
    duration = 5
    Fs = 96000
    n_repeat = 20
    R = 3000
    Ns = 4
    channels = 16
    M = 12  # the channels to be use
    d0 = 0.05
    c = 343
    n_path = 1
    n_sim = 1
    feedforward_taps = 30
    feedbackward_taps = 2
    alpha_rls = 0.9999

    theta_start = -90
    theta_end = 90

    reflection_list = np.array([1])
    x_tx_list = np.array([0])
    y_tx_list = np.array([1])

    snr_list = [30] #np.arange(10, 15, 1) #changed from 15

    def __init__(self, fc, n_path, n_sim, n_UE) -> None:
        self.fc = fc
        self.n_path = n_path
        self.n_sim = n_sim
        self.n_UE = n_UE

        self.MSE_SNR = np.zeros((len(self.snr_list), n_sim))
        self.ERR_SNR = np.zeros_like(self.MSE_SNR)

        self.modem = QAMModem(self.M0)
        self.preamble = self.generate_preamble(11, self.modem)
        self.preambles = np.tile(self.preamble, self.n_repeat)

        self.return_symbols = np.zeros((len(self.snr_list), n_sim,3*len(self.preamble)), dtype=complex)
        #self.mean_symbols = np.zeros((len(self.snr_list),len(self.preamble)), dtype=complex)

        self.seq0 = self.generate_gold_sequence(7, index=0) * 2 - 1
        self.seq1 = self.generate_gold_sequence(7, index=1) * 2 - 1

        self.fs = self.Ns * self.R
        self.nsps = int(self.Fs / self.R)
        self.uf = int(self.Fs / self.R)
        self.df = int(self.uf / self.Ns)
        
        self.time_idx, self.rc_tx = self.rrcosfilter(16 * int(1 / self.R * self.Fs), 0.5, 1 / self.R, self.Fs)

        self.preamble_upsampled = self.upsample(self.preambles, self.nsps)
        self.preamble_rc = sg.convolve(self.preamble_upsampled, self.rc_tx, mode="same")
        self.preamble_xcorr = sg.convolve(self.upsample(self.preamble, self.nsps), self.rc_tx, mode="same")

        self.s = np.real(self.preamble_rc * np.exp(2 * np.pi * 1j * self.fc * np.arange(len(self.preamble_rc))/self.Fs))
        self.s /= self.s.max()
        pass

    # functions for preamble generation
    def gold_preferred_poly(self, nBits):
        return {
            5: (np.array([5, 2, 0]), np.array([5, 4, 3, 2, 0])),
            6: (np.array([6, 1, 0]), np.array([6, 5, 2, 1, 0])),
            7: (np.array([7, 3, 0]), np.array([7, 3, 2, 1, 0])),
            9: (np.array([9, 4, 0]), np.array([9, 6, 4, 3, 0])),
            10: (np.array([10, 3, 0]), np.array([10, 8, 3, 2, 0])),
            11: (np.array([11, 2, 0]), np.array([11, 8, 5, 2, 0])),
        }[nBits]

    def generate_gold_sequence(
            self, nBits, index=0, poly1=None, poly2=None, state1=None, state2=None
    ):
        from scipy.signal import max_len_seq as mls

        if (poly1 and poly2) is None:
            poly1, poly2 = self.gold_preferred_poly(nBits)
        return np.bitwise_xor(
            mls(nBits, taps=poly1, state=state1)[0],
            np.roll(mls(nBits, taps=poly2, state=state2)[0], -index),
        )

    def generate_preamble(self, nBits, modem):
        sequence = np.append(self.generate_gold_sequence(nBits, index=0), 0)
        sequence_mod = modem.modulate(sequence)
        return sequence_mod

    def upsample(self, s, n, phase=0):
        return np.roll(np.kron(s, np.r_[1, np.zeros(n - 1)]), phase)

    def rrcosfilter(self, N, alpha, Ts, Fs):
        T_delta = 1 / float(Fs)
        time_idx = ((np.arange(N) - N / 2)) * T_delta
        sample_num = np.arange(N)
        h_rrc = np.zeros(N, dtype=float)

        for x in sample_num:
            t = (x - N / 2) * T_delta
            if t == 0.0:
                h_rrc[x] = 1.0 - alpha + (4 * alpha / np.pi)
            elif alpha != 0 and t == Ts / (4 * alpha):
                h_rrc[x] = (alpha / np.sqrt(2)) * (
                    ((1 + 2 / np.pi) * (np.sin(np.pi / (4 * alpha))))
                    + ((1 - 2 / np.pi) * (np.cos(np.pi / (4 * alpha))))
                )
            elif alpha != 0 and t == -Ts / (4 * alpha):
                h_rrc[x] = (alpha / np.sqrt(2)) * (
                    ((1 + 2 / np.pi) * (np.sin(np.pi / (4 * alpha))))
                    + ((1 - 2 / np.pi) * (np.cos(np.pi / (4 * alpha))))
                )
            else:
                h_rrc[x] = (
                    np.sin(np.pi * t * (1 - alpha) / Ts)
                    + 4 * alpha * (t / Ts) * np.cos(np.pi * t * (1 + alpha) / Ts)
                ) / (np.pi * t * (1 - (4 * alpha * t / Ts) * (4 * alpha * t / Ts)) / Ts)

        return time_idx, h_rrc

    def simulation(self):
        x_rx = np.arange(0, self.d0*self.channels, self.d0)
        y_rx = np.zeros_like(x_rx)
        for i_snr, snr in enumerate(self.snr_list):
            SNR = np.power(10, snr/10)
            print(f"SNR={snr}dB")
            for i_sim in tqdm(range(self.n_sim)):
                r_multichannel = np.random.randn(self.duration * self.Fs, self.channels) / SNR
                # receive the signal
                for i in range(self.n_path):
                    x_tx, y_tx = self.x_tx_list[i], self.y_tx_list[i]
                    reflection = self.reflection_list[i] #delay ans sum not scale
                    dx, dy = x_rx - x_tx, y_rx - y_tx
                    d_rx_tx = np.sqrt(dx**2 + dy**2)
                    delta_tau = d_rx_tx / self.c
                    delay = np.round(delta_tau * self.Fs).astype(int)
                    for j, delay_j in enumerate(delay):
                        # print(delay_j/4)
                        r_multichannel[delay_j:delay_j+len(self.s), j] += reflection * self.s

                # get S(theta) -- the angle of the source
                r = r_multichannel[:,:self.M]
                r_fft = np.fft.fft(r, axis=0)
                freqs = np.fft.fftfreq(len(r[:, 0]), 1/self.Fs)

                index = np.where((freqs >= self.fc-self.R/2) & (freqs < self.fc+self.R/2))[0]
                N = len(index)
                fk = freqs[index]       
                yk = r_fft[index, :]    # N*M

                S_theta = np.arange(self.theta_start, self.theta_end, 1,dtype="complex128")
                N_theta = len(S_theta)
                theta_start = np.deg2rad(self.theta_start)
                theta_end = np.deg2rad(self.theta_end)
                theta_list = np.linspace(theta_start, theta_end, N_theta)

                for n_theta, theta in enumerate(theta_list):
                    d_tau = np.sin(theta) * self.d0/self.c
                    S_M = np.exp(-2j * np.pi * d_tau * np.dot(fk.reshape(N, 1), np.arange(self.M).reshape(1, self.M)))    # N*M
                    SMxYk = np.einsum('ij,ji->i', S_M.conj(), yk.T,dtype="complex128")
                    S_theta[n_theta] = np.real(np.vdot(SMxYk, SMxYk))

                S_theta_peaks_idx, _ = sg.find_peaks(S_theta, height=0)
                S_theta_peaks = S_theta[S_theta_peaks_idx] # plot and see
                theta_m_idx = np.argsort(S_theta_peaks)
                theta_m = theta_list[S_theta_peaks_idx[theta_m_idx[-self.n_path:]]]

                # do beamforming
                y_tilde = np.zeros((N,self.n_path), dtype=complex)
                for k in range(N):
                    d_tau_m = np.sin(theta_m) * self.d0/self.c
                    Sk = np.exp(-2j * np.pi * fk[k] * np.arange(self.M).reshape(self.M, 1) @ d_tau_m.reshape(1, self.n_path))
                    for i in range(self.n_path):
                        e_pu = np.zeros((self.n_path, 1))
                        e_pu[i, 0] = 1
                        wk = Sk @ np.linalg.inv(Sk.conj().T @ Sk) @ e_pu
                        y_tilde[k, i] = wk.conj().T @ yk[k, :].T

                y_fft = np.zeros((len(r[:, 0]), self.n_path), complex)
                y_fft[index, :] = y_tilde
                y = np.fft.ifft(y_fft, axis=0)

                # processing the signal we get from bf
                r_multichannel_1 = y

                K_list = [self.n_path]
                K = self.n_path
                if K == 1:
                    r_multichannel_1 = r_multichannel_1.reshape(len(r_multichannel_1), 1)
                v_multichannel = []
                peaks_rx = 0
                for i in range(K):
                    r = np.squeeze(r_multichannel_1[:, i])
                    v = r * np.exp(-2 * np.pi * 1j * self.fc * np.arange(len(r))/self.Fs)
                    v = sg.decimate(v, self.df)
                    fractional_spacing = self.nsps/self.df
                    _, rc_rx = self.rrcosfilter(16 * int(self.fs / self.R), 0.5, 1 / self.R, self.fs)
                    v = np.convolve(v, rc_rx, "full")
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
                    v_multichannel.append(v)
                d = np.tile(self.preamble, 3)

                v_rls = np.zeros((K, len(v_multichannel[0])), dtype=complex)
                for channel in range(K):
                    v = v_multichannel[channel]
                    v_rls[channel, 0:len(v)] = v

                for K_channels in K_list:
                    d_hat, mse, n_err, n_training = self.processing(v_rls, d, K_channels, bf=True)
                    self.MSE_SNR[i_snr, i_sim] = mse
                    self.ERR_SNR[i_snr, i_sim] = n_err
                    self.return_symbols[i_snr, i_sim, :] = d_hat

                # now transmit based on theta
                theta = theta_m[0:self.n_UE] 
                #A = np.exp(-1j*2*np.pi*np.sind(theta).T *self.d0*np.arange(0,self.M-1)).T 
                # data portion -> received signal? or just fabricate
                #F = v_rls # ?
                #X = A*F
                # transmit(X)
        self.mean_symbols = np.mean(self.return_symbols, axis=1)
        self.mean_mse = np.mean(self.MSE_SNR, axis=1)
        self.mean_err = np.mean(self.ERR_SNR, axis=1)
        self.mean_v = np.mean(v_rls, axis=0)
        return theta, wk, S_theta

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