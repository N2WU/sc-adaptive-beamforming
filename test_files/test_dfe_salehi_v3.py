import numpy as np
import scipy.signal as sg
import matplotlib.pyplot as plt

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

def transmit(s,snr,n_rx,el_spacing,R,fc,fs):
    reflection_list = np.asarray([1,0.5]) # reflection gains
    x_tx_list = np.array([50,-10]) #([5,-5]) 
    y_tx_list = np.array([50,50]) #([20,20])
    c = 343
    duration = 10 # amount of padding, basically
    x_rx = np.arange(0, el_spacing*n_rx, el_spacing)
    y_rx = np.zeros_like(x_rx)
    rng = np.random.RandomState(2021)
    r_multichannel = rng.randn(duration * fs, n_rx) / snr
    for i in range(len(reflection_list)):
        x_tx, y_tx = x_tx_list[i], y_tx_list[i]
        reflection = reflection_list[i] #delay and sum not scale
        dx, dy = x_rx - x_tx, y_rx - y_tx
        d_rx_tx = np.sqrt(dx**2 + dy**2)
        delta_tau = d_rx_tx / c
        delay = np.round(delta_tau * fs).astype(int) # sample delay
        for j, delay_j in enumerate(delay):
            r_multichannel[delay_j:delay_j+len(s), j] += reflection * s
    
    r = r_multichannel
    r_fft = np.fft.fft(r, axis=0)
    freqs = np.fft.fftfreq(len(r[:, 0]), 1/fs)
    index = np.where((freqs >= fc-R/2) & (freqs < fc+R/2))[0]
    N = len(index)
    fk = freqs[index]       
    yk = r_fft[index, :]    # N*M

    theta_start = -90
    theta_end = 90
    S_theta = np.arange(theta_start, theta_end, 1,dtype="complex128")
    N_theta = len(S_theta)
    theta_start = np.deg2rad(theta_start)
    theta_end = np.deg2rad(theta_end)
    theta_list = np.linspace(theta_start, theta_end, N_theta)

    for n_theta, theta in enumerate(theta_list):
        d_tau = np.sin(theta) * el_spacing/c
        S_M = np.exp(-2j * np.pi * d_tau * np.dot(fk.reshape(N, 1), np.arange(n_rx).reshape(1, n_rx)))    # N*M
        SMxYk = np.einsum('ij,ji->i', S_M.conj(), yk.T,dtype="complex128")
        S_theta[n_theta] = np.real(np.vdot(SMxYk, SMxYk))

    S_theta_peaks_idx, _ = sg.find_peaks(S_theta, height=0)
    S_theta_peaks = S_theta[S_theta_peaks_idx] # plot and see
    theta_m_idx = np.argsort(S_theta_peaks)
    theta_m = theta_list[S_theta_peaks_idx[theta_m_idx[-len(reflection_list):]]]

    # do beamforming
    y_tilde = np.zeros((N,len(reflection_list)), dtype=complex)
    for k in range(N):
        d_tau_m = np.sin(theta_m) * el_spacing/c
        Sk = np.exp(-2j * np.pi * fk[k] * np.arange(n_rx).reshape(n_rx, 1) @ d_tau_m.reshape(1, len(reflection_list)))
        for i in range(len(reflection_list)):
            e_pu = np.zeros((len(reflection_list), 1))
            e_pu[i, 0] = 1
            wk = Sk @ np.linalg.inv(Sk.conj().T @ Sk) @ e_pu
            y_tilde[k, i] = wk.conj().T @ yk[k, :].T

    y_fft = np.zeros((len(r[:, 0]), len(reflection_list)), complex)
    y_fft[index, :] = y_tilde
    y = np.fft.ifft(y_fft, axis=0)

    # processing the signal we get from bf
    r_multichannel_1 = y
    return r_multichannel_1

def dfe_old(v_rls, d, Ns, feedforward_taps=20, feedbackward_taps=8, alpha_rls=0.5): #, filename='data/r_multichannel.npy'
        K = len(v_rls[:,0])
        delta = 0.001
        Nplus = 2
        n_training = int(4 * (feedforward_taps + feedbackward_taps) / Ns)
        fractional_spacing = ns_fs
        K_1 = 0.0000
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
                d_tilde[i] = (d_hat[i] > 0)*2 - 1
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
        return d_hat, mse, #n_err, n_training

if __name__ == "__main__":
    # initialize
    bits = 7
    rep = 16
    training_rep = 4
    snr_db = np.array([8, 12, 16, 20])#np.arange(-10,20,2)
    n_ff = 20
    n_fb = 4
    R = 3000
    fs = 48000
    ns = int(fs/R)
    ns_fs = 4 # fractionally spaced
    fc = 16e3
    uf = int(fs / R)
    df = int(uf / ns)
    n_rx = 1
    d_lambda = 0.5 #np.array([0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]) #0.5
    el_spacing = d_lambda*343/fc
    mse = np.zeros(len(snr_db),dtype='float')
    # init rrc filters with b=0.5
    _, rc_tx = rrcosfilter(16 * int(1 / R * fs), 0.5, 1 / R, fs)
    _, rc_rx = rrcosfilter(16 * int(fs / R), 0.5, 1 / R, fs)
    # generate the tx BPSK signal
    # init bits (training bits are a select repition of bits)
    d = np.array(sg.max_len_seq(bits)[0]) * 2 - 1.0
    u = np.tile(d,rep)
    # upsample
    u = sg.resample_poly(u,uf,1)
    #u = upsample(u, uf)
    # filter
    u_rrc = u #np.convolve(u, rc_tx, "full")
    # upshift
    s = np.real(u_rrc * np.exp(2j * np.pi * fc * np.arange(len(u_rrc)) / fs))
    s /= np.max(np.abs(s))
    # generate rx signal with ISI
    for ind in range(len(snr_db)):
        d0 = el_spacing
        snr = 10**(0.1 * snr_db[ind])
        # r_multi = transmit(s,snr,n_rx,d0,R,fc,fs) # should come out as an n-by-zero
        r_multi = np.tile(s,(n_rx,1)).T
        peaks_rx = 0
        v_multichannel = []
        for i in range(len(r_multi[0,:])):
            r = np.squeeze(r_multi[:, i])
            v = r * np.exp(-2 * np.pi * 1j * fc * np.arange(len(r))/fs)
            # v
            # v_xcorr = sg.decimate(v, df)
            # v = np.convolve(v, rc_rx, "full")
            if i == 0:
                xcorr_for_peaks = np.abs(sg.fftconvolve(v, sg.resample_poly(d[::-1].conj(),uf,1))) # correalte and sync at sample rate sg.decimate(v, int(ns)),
                xcorr_for_peaks /= xcorr_for_peaks.max()
                time_axis_xcorr = np.arange(0, len(xcorr_for_peaks)) / R * 1e3  # ms
                peaks_rx, _ = sg.find_peaks(
                    xcorr_for_peaks, height=0.2, distance=len(d) - 100
                )
                #plt.figure()
                #plt.plot(np.abs(xcorr_for_peaks))
                #plt.show()
            v = v[int(peaks_rx[1]) :] # * ns
            if i == 0:
                v_multichannel = v
            else:
                v_multichannel = np.vstack((v_multichannel,v))
        # dfe
        d_adj = np.tile(d,rep)
        if n_rx == 1:
            v_multichannel = v_multichannel[None,:]
        # resample v_multichannel for frac spac
        v_multichannel = sg.resample_poly(v_multichannel,ns_fs,ns,axis=1)
        d_hat, mse_out = dfe_old(v_multichannel, d_adj, ns_fs, n_ff, n_fb)
        d_hat_adj = (d_hat > 0) * 2 - 1
        #d_hat = lms(v,d,ns)
        mse[ind] = mse_out #10 * np.log10(np.mean(np.abs(d_adj-d_hat) ** 2))
        # plot const
        plt.subplot(2, 2, int(ind+1))
        plt.scatter(np.real(d_hat), np.imag(d_hat), marker='*')
        plt.axis('square')
        plt.axis([-2, 2, -2, 2])
        plt.title(f'SNR={snr_db[ind]}dB')



    fig, ax = plt.subplots()
    ax.plot(snr_db,mse,'o')
    ax.set_xlabel(r'SNR (dB)')
    ax.set_ylabel('MSE (dB)')
    #ax.set_xticks(np.arange(el_spacing[0],el_spacing[-1],5))
    ax.set_title('MSE vs SNR for BPSK Signal')
    plt.show()
