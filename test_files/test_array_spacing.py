import numpy as np
import scipy.signal as sg
import matplotlib.pyplot as plt

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

def transmit(s,snr,el_spacing,fs,fc,R):
    # initialize
    reflection = np.asarray([1, 0.5]) # reflection channel
    n_rx = 12 # arbitrary
    c = 343 # speed of sound in air
    duration = 1 # 1-second signal
    x_tx = np.array([5,-5])
    y_tx = np.array([20,20])
    x_rx = np.arange(0, el_spacing*n_rx, el_spacing)
    y_rx = np.zeros_like(x_rx)
    rng = np.random.RandomState(2021)
    r_multichannel = rng.randn(duration * fs, n_rx) / snr
    # simulate reflected multi-array channel
    for i in range(len(reflection)):
        dx, dy = x_rx- x_tx[i], y_rx - y_tx[i]
        d_rx_tx = np.sqrt(dx**2 + dy**2)
        delta_tau = d_rx_tx / c
        delay = np.round(delta_tau * fs).astype(int)
        reflect = reflection[i]
        for j, delay_j in enumerate(delay):
            r_multichannel[delay_j:delay_j+len(s), j] += reflect * s

    r = r_multichannel

    r_fft = np.fft.fft(r, axis=0)
    freqs = np.fft.fftfreq(len(r[:, 0]), 1/fs)
    index = np.where((freqs >= fc-R/2) & (freqs < fc+R/2))[0]
    N = len(index)
    fk = freqs[index]       
    yk = r_fft[index, :]    # N*M
    theta_start, theta_end = [-90, 90]
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
    theta_m = theta_list[S_theta_peaks_idx[theta_m_idx[-len(reflection):]]]
    # do beamforming
    y_tilde = np.zeros((N,len(reflection)), dtype=complex)
    for k in range(N):
        d_tau_m = np.sin(theta_m) * el_spacing/c
        Sk = np.exp(-2j * np.pi * fk[k] * np.arange(n_rx).reshape(n_rx, 1) @ d_tau_m.reshape(1, len(reflection)))
        for i in range(len(reflection)):
            e_pu = np.zeros((len(reflection), 1))
            e_pu[i, 0] = 1
            wk = Sk @ np.linalg.inv(Sk.conj().T @ Sk) @ e_pu
            y_tilde[k, i] = wk.conj().T @ yk[k, :].T

    y_fft = np.zeros(len(r[:, 0]), complex)# np.zeros((len(r[:, 0]), len(reflection)), complex)
    y_fft[index] = y_tilde
    y = np.fft.ifft(y_fft, axis=0)
    # processing the signal we get from bf
    r_single = y
    return r_single

def lms(v,d,ns):
    Nplus = 5
    fractional_spacing = 4
    n_training = 24
    d_hat = np.zeros(len(d))
    d_tilde = np.zeros(len(d))
    mu = 0.001
    for i in np.arange(len(d) - 1, dtype=int):
        # resample v with fractional spacing
        nb = (i) * ns + (Nplus - 1) * ns - 1 # basically a sample index
        v_nt = v[ int(nb + np.ceil(ns / fractional_spacing / 2)) : int(nb + ns) 
            + 1 : int(ns / fractional_spacing)]
    # select a' (?)
        if i==0:
            a = np.ones_like(v_nt)
    # calculate d_hat
        d_hat[i] = np.dot(a,v_nt)
    # calculate e (use d=dec(d_hat))
        if i > n_training:
            d_tilde[i] = (d_tilde[i] > 0)*2 - 1
        else:
            d_tilde[i] = d[i]
        err = d_tilde[i] - d_hat[i]
    # select a(n+1) with stochastic gradient descent 
        a += mu*v_nt*err.conj()
    return d_hat

if __name__ == "__main__":
    # initialize
    bits = 7
    rep = 16
    training_rep = 4
    snr_db = 10 # arbitrary
    n_ff = 20
    n_fb = 8
    R = 3000
    fs = 48000
    ns = fs / R
    fc = 16e3
    uf = int(fs / R)
    df = int(uf / ns)
    d_lambda = np.array([0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2])
    el_spacing = d_lambda*343/fc
    mse = np.zeros(len(el_spacing),dtype='float')
    # init rrc filters with b=0.5
    _, rc_tx = rrcosfilter(16 * int(1 / R * fs), 0.5, 1 / R, fs)
    _, rc_rx = rrcosfilter(16 * int(fs / R), 0.5, 1 / R, fs)
    # generate the tx BPSK signal
    # init bits (training bits are a select repition of bits)
    d = np.array(sg.max_len_seq(bits)[0]) * 2 - 1.0
    u = np.tile(d,10)
    # upsample
    u = sg.resample_poly(u,ns,1)
    # filter
    u_rrc = np.convolve(u, rc_tx, "full")
    # upshift
    s = np.real(u_rrc * np.exp(2j * np.pi * fc * np.arange(len(u_rrc)) / fs))
    s /= np.max(np.abs(s))
    # generate rx signal with ISI
    for i in range(len(el_spacing)):
        snr = 10**(0.1 * snr_db)
        d0 = el_spacing[i]
        r = transmit(s,snr,d0,fs,fc,R)
        # downshift
        v = r * np.exp(-2j * np.pi * fc * np.arange(len(r)) / fs)
        # decimate
        # v = sg.decimate(v, df)
        # filter
        v_rrc = np.convolve(v, rc_rx, "full")
        # sync (skip for now)

        # dfe
        d_adj = np.tile(d,1)
        # d_hat = dfe_v2(v,d_adj,n_ff,n_fb,ns)
        d_hat = lms(v,d,ns)
        mse[i] = 10 * np.log10(np.mean(np.abs(d_adj-d_hat) ** 2))

    fig, ax = plt.subplots()
    ax.plot(d_lambda,mse,'o')
    ax.set_xlabel('Element Spacing (factor of Lambda)')
    ax.set_ylabel('MSE (dB)')
    ax.set_xticks(np.arange(d_lambda[0],d_lambda[-1],2))
    ax.set_title('EL Spacing vs MSE for BPSK Signal')
    plt.show()


