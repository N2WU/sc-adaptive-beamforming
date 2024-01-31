import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import cm
import scipy.signal as sg
import sounddevice as sd
from commpy.modulation import QAMModem

def gold_preferred_poly(nBits):
    return {
        5: (np.array([5, 2, 0]), np.array([5, 4, 3, 2, 0])),
        6: (np.array([6, 1, 0]), np.array([6, 5, 2, 1, 0])),
        7: (np.array([7, 3, 0]), np.array([7, 3, 2, 1, 0])),
        9: (np.array([9, 4, 0]), np.array([9, 6, 4, 3, 0])),
        10: (np.array([10, 3, 0]), np.array([10, 8, 3, 2, 0])),
        11: (np.array([11, 2, 0]), np.array([11, 8, 5, 2, 0])),
    }[nBits]


def generate_gold_sequence(
    nBits, index=0, poly1=None, poly2=None, state1=None, state2=None
):
    from scipy.signal import max_len_seq as mls

    if (poly1 and poly2) is None:
        poly1, poly2 = gold_preferred_poly(nBits)
    return np.bitwise_xor(
        mls(nBits, taps=poly1, state=state1)[0],
        np.roll(mls(nBits, taps=poly2, state=state2)[0], -index),
    )


def generate_preamble(nBits, modem):
    sequence = np.append(generate_gold_sequence(nBits, index=0), 0)
    sequence_mod = modem.modulate(sequence)
    return sequence_mod

def upsample(s, n, phase=0):
    return np.roll(np.kron(s, np.r_[1, np.zeros(n - 1)]), phase)

def rrcosfilter(N, alpha, Ts, Fs):
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

def receive_signal(fc=6.5e3, tx_channel=16):
    sd.default.device = "ASIO MADIface USB"
    Fs = 96000
    sd.default.samplerate = Fs

    duration = 5

    M = 16
    n_repeat = 20
    # modem = mod.QAMModem(M, gray_map=True, bin_input=True, bin_output=True, soft_decision=False)
    modem = QAMModem(M)
    preamble = generate_preamble(11, modem)
    preambles = np.tile(preamble, n_repeat)
    np.save('data/preamble.npy', preamble)

    seq0 = generate_gold_sequence(7, index=0) * 2 - 1
    seq1 = generate_gold_sequence(7, index=1) * 2 - 1

    R = 3000
    Ns = 4
    fs = Ns * R
    nsps = int(Fs/R)
    uf = int(Fs / R)
    df = int(uf / Ns)
    time_idx, rc_tx = rrcosfilter(16 * int(1 / R * Fs), 0.5, 1 / R, Fs)

    preamble_upsampled = upsample(preambles, nsps)
    preamble_rc = sg.convolve(preamble_upsampled, rc_tx, mode="same")
    preamble_xcorr = sg.convolve(upsample(preamble, nsps), rc_tx, mode="same")

    # fc = 15e3
    s = np.real(preamble_rc * np.exp(2 * np.pi * 1j * fc * np.arange(len(preamble_rc))/Fs))
    s /= s.max()

    channels = 24
    sig = np.zeros((len(s),channels))
    sig[:, tx_channel] = 0.1*s
    sig = np.concatenate((s, np.zeros((int(0.25*Fs),))))*0.1/2
    sig = np.tile(np.reshape(sig, (len(sig),1)), (1, channels))
    #r_multichannel = sd.playrec(sig, channels=channels, blocking=True)
    r_multichannel = sig
    np.save('data/r_multichannel.npy', r_multichannel)
    np.save('data/fc.npy', fc)

def get_S_theta(n_path=1, theta_start=-45, theta_end=45, plot_S_theta=True, plot3D_S_theta=False):
    r_multichannel = np.load('data/r_multichannel.npy')
    fc = np.load('data/fc.npy')
    print(f'fc={fc/1000:.1f}kHz')

    Fs = 96000
    R = 3000
    M = 12
    # M_list = [0, 3, 6, 9]
    d = 0.05
    c = 343

    r = r_multichannel[:,:M]
    r_fft = np.fft.fft(r, axis=0)
    freqs = np.fft.fftfreq(len(r[:, 0]), 1/Fs)

    index = np.where((freqs >= fc-R/2) & (freqs < fc+R/2))[0]
    N = len(index)
    fk = freqs[index]       
    yk = r_fft[index, :]    # N*M

    N_theta = 200
    S_theta = np.zeros((N_theta,))
    S_theta3D = np.zeros((N_theta, N))
    theta_start = np.deg2rad(theta_start)
    theta_end = np.deg2rad(theta_end)
    theta_list = np.linspace(theta_start, theta_end, N_theta)
    for n_theta, theta in enumerate(theta_list):
        d_tau = np.sin(theta) * d/c
        # for k in range(N):
        #     S_M = np.exp(-2j * np.pi * fk[k] * d_tau * np.arange(M).reshape(M,1))
        #     S_theta[n_theta] += np.abs(np.vdot(S_M.T, yk[k, :].T))**2
        S_M = np.exp(-2j * np.pi * d_tau * np.dot(fk.reshape(N, 1), np.arange(M).reshape(1, M)))    # N*M
        SMxYk = np.einsum('ij,ji->i', S_M.conj(), yk.T)
        S_theta[n_theta] = np.real(np.vdot(SMxYk, SMxYk))
        S_theta3D[n_theta, :] = np.abs(SMxYk)**2

    # n_path = 1 # number of path
    S_theta_peaks_idx, _ = sg.find_peaks(S_theta, height=0)
    S_theta_peaks = S_theta[S_theta_peaks_idx]
    theta_m_idx = np.argsort(S_theta_peaks)
    theta_m = theta_list[S_theta_peaks_idx[theta_m_idx[-n_path:]]]
    print(theta_m/np.pi*180)

    y_tilde = np.zeros((N,n_path), dtype=complex)
    for k in range(N):
        d_tau_m = np.sin(theta_m) * d/c
        Sk = np.exp(-2j * np.pi * fk[k] * np.arange(M).reshape(M, 1) @ d_tau_m.reshape(1, n_path))
        for i in range(n_path):
            e_pu = np.zeros((n_path,1))
            e_pu[i, 0] = 1
            wk = Sk @ np.linalg.inv(Sk.conj().T @ Sk) @ e_pu
            y_tilde[k, i] = wk.conj().T @ yk[k, :].T

    y_fft = np.zeros((len(r[:, 0]), n_path), complex)
    y_fft[index, :] = y_tilde
    y = np.fft.ifft(y_fft, axis=0)
    np.save('data/signal.npy', np.real(y))

    if plot_S_theta:
        plt.figure()
        plt.plot(theta_list/np.pi*180, S_theta / S_theta.max())
        plt.savefig(f'S_theta-{int(fc/1000)}.png')
    
    if plot3D_S_theta:
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        X = theta_list * 180 / np.pi
        Y = fk
        X, Y = np.meshgrid(X, Y)
        Z = S_theta3D.T / S_theta3D.max()

        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                            linewidth=0, antialiased=False)
        ax.zaxis.set_major_locator(ticker.LinearLocator(10))
        ax.zaxis.set_major_formatter('{x:.02f}')
        ax.view_init(elev=30, azim=60)

        plt.savefig(f'S_theta_3D-{int(fc/1000)}.png')

def receiver_processing(filename='data/signal.npy', K_list=[1], feedforward_taps=20, feedbackward_taps=3, alpha_rls=0.999):
    fc = np.load('data/fc.npy')
    Fs = 96000

    R = 3000
    Ns = 4
    fs = Ns * R
    nsps = int(Fs/R)
    uf = int(Fs / R)
    df = int(uf / Ns)
    time_idx, rc_tx = rrcosfilter(16 * int(1 / R * Fs), 0.5, 1 / R, Fs)
    M = 16
    modem = QAMModem(M)

    r_multichannel = np.load(filename)
    K = 1
    if np.ndim(r_multichannel)==2:
        K = r_multichannel.shape[1]
    elif np.ndim(r_multichannel)==1:
        r_multichannel = r_multichannel.reshape(len(r_multichannel),1)
    v_multichannel = []
    peaks_rx = 0
    preamble = np.load('data/preamble.npy')
    for i in range(K):
        r = np.squeeze(r_multichannel[:, i])
        v = r * np.exp(-2 * np.pi * 1j * fc * np.arange(len(r))/Fs)
        v = sg.decimate(v, df)
        _, rc_rx = rrcosfilter(16 * int(fs / R), 0.5, 1 / R, fs)
        v = np.convolve(v, rc_rx, "full")
        # v /= np.sqrt(np.var(v))
        if np.var(v) > 1e-6:
            v /= np.sqrt(np.var(v))
        else:
            v = np.zeros_like(v)
        if i == 0:
            xcorr_for_peaks = np.abs(sg.fftconvolve(sg.decimate(v, Ns), preamble[::-1].conj()))
            xcorr_for_peaks /= xcorr_for_peaks.max()
            time_axis_xcorr = np.arange(0, len(xcorr_for_peaks)) / R * 1e3  # ms
            peaks_rx, _ = sg.find_peaks(
                xcorr_for_peaks, height=0.2, distance=len(preamble) - 100
            )
        #v = v[peaks_rx[1] * Ns :]
        v = v
        v_multichannel.append(v)
    d = np.tile(preamble, 3)

    v_rls = np.zeros((K, len(v_multichannel[0])), dtype=complex)
    for channel in range(K):
        v = v_multichannel[channel]
        v_rls[channel, 0:len(v)] = v

    def processing(v_rls, d, K=1, filename='data/r_multichannel.npy'):
        channel_list = {1:[0], 2:[0,11], 3:[0,6,11], 4:[0,4,7,11], 8:[0,1,3,4,6,7,9,11], 6:[0,2,4,6,8,10]}
        if filename == 'data/r_multichannel.npy':
            v_rls = v_rls[channel_list[K], :]
        # v_rls = v_rls[random.sample(range(0, 12), K), :]
        delta = 0.001
        Nplus = 2
        n_training = int(4 * (feedforward_taps + feedbackward_taps) / Ns)
        fractional_spacing = 4
        K_1 = 0.0001
        K_2 = K_1 / 10

        # v_rls = np.tile(v, (K, 1))
        d_rls = d
        x = np.zeros((K,feedforward_taps), dtype=complex)
        a = np.zeros((K*feedforward_taps,), dtype=complex)
        b = np.zeros((feedbackward_taps,), dtype=complex)
        c = np.concatenate((a, -b))
        theta = np.zeros((len(d_rls),K), dtype=float)
        d_hat = np.zeros((len(d_rls),), dtype=complex)
        d_tilde = np.zeros((len(d_rls),), dtype=complex)
        d_backward = np.zeros((feedbackward_taps,), dtype=complex)
        e = np.zeros((len(d_rls),), dtype=complex)
        Q = np.eye(K*feedforward_taps + feedbackward_taps) / delta
        pk = np.zeros((K,), dtype=complex)

        cumsum_phi = 0.0
        sse = 0.0
        for i in np.arange(len(d_rls) - 1, dtype=int):
            nb = (i) * Ns + (Nplus - 1) * Ns - 1
            xn = v_rls[
                :, int(nb + np.ceil(Ns / fractional_spacing / 2)) : int(nb + Ns)
                + 1 : int(Ns / fractional_spacing)
            ]
            for j in range(K):
                xn[j,:] *= np.exp(-1j * theta[i,j])
            x_buf = np.concatenate((xn, x), axis=1)
            x = x_buf[:,:feedforward_taps]

            # p = np.inner(x, a.conj()) * np.exp(-1j * theta[i])
            for j in range(K):
                pk[j] = np.inner(x[j,:], a[j*feedforward_taps:(j+1)*feedforward_taps].conj())
            p = pk.sum()

            q = np.inner(d_backward, b.conj())
            d_hat[i] = p - q

            if i > n_training:
                # d_tilde[i] = slicing(d_hat[i], M)
                d_tilde[i] = modem.modulate(np.array(modem.demodulate(np.array(d_hat[i], ndmin=1), 'hard'), dtype=int))
            else:
                d_tilde[i] = d_rls[i]
            e[i] = d_tilde[i] - d_hat[i]
            sse += np.abs(e[i] ** 2)

            # y = np.concatenate((x * np.exp(-1j * theta[i]), d_backward))
            y = np.concatenate((x.reshape(K*feedforward_taps), d_backward))

            # PLL
            phi = np.imag(p * np.conj(d_tilde[i] + q))
            cumsum_phi += phi
            theta[i + 1] = theta[i] + K_1 * phi + K_2 * cumsum_phi

            # RLS
            k = (
                np.matmul(Q, y.T)
                / alpha_rls
                / (1 + np.matmul(np.matmul(y.conj(), Q), y.T) / alpha_rls)
            )
            c += k.T * e[i].conj()
            Q = Q / alpha_rls - np.matmul(np.outer(k, y.conj()), Q) / alpha_rls

            a = c[:K*feedforward_taps]
            b = -c[-feedbackward_taps:]

            d_backward_buf = np.insert(d_backward, 0, d_tilde[i])
            d_backward = d_backward_buf[:feedbackward_taps]

        err = d_rls[n_training + 1 : -1] - d_tilde[n_training + 1 : -1]
        mse = 10 * np.log10(
            np.mean(np.abs(d_rls[n_training + 1 : -1] - d_hat[n_training + 1 : -1]) ** 2)
        )
        n_err = np.sum(np.abs(err) > 0.01)
        return d_hat, mse, n_err, n_training
    
    for K_channels in K_list:
        d_hat, mse, n_err, n_training = processing(v_rls, d, K_channels, filename=filename)
        plt.figure()
        plt.plot(d_hat[n_training:-1].real, d_hat[n_training:-1].imag, ".")
        # modem.plot_const()
        limit = np.log2(M)
        plt.axis('square')
        plt.axis([-limit, limit, -limit, limit])
        plt.xlabel("Re")
        plt.ylabel("Im")
        plt.title(f"channels={K_channels}, MSE={mse:.2f}dB, ERR={n_err}")
        if filename == 'data/r_multichannel.npy':
            plt.savefig(f'{int(fc/1000)}kHz-{K_channels}channel.png')
        elif filename == 'data/signal.npy':
            plt.savefig(f'bf-{int(fc/1000)}kHz-{K_channels}channel.png')