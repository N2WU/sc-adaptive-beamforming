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

def transmit(s,snr,n_rx,el_spacing,R,fc,fs):
    reflection_list = np.asarray([1,0.5]) # reflection gains
    x_tx_list = np.array([5,-5])
    y_tx_list = np.array([20,20])
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

def dfe(vk, d, Nd, Tmp, T, Nplus, snr):
    plt.figure()
    K_list = [1, 2, 5, 8]

    for K in K_list:
        Ns = 2
        N = 6 * Ns
        M = np.ceil(Tmp / T)
        delta = 1e-3
        Nt = 4 * (N + M)
        FS = 2
        Kf1 = 0.001
        Kf2 = Kf1 / 10
        Lf = 1
        L = 0.98
        P = np.eye(K * N + M) / delta
        Lbf = 0.99

        v = vk[:K, :]

        f = np.zeros((Nd, K))
        a = np.zeros(K * N)
        b = np.zeros(M)
        c = np.concatenate([a, -b])
        p = np.zeros(K)
        d_tilde = np.zeros(M)
        Sf = np.zeros(K)  # sum of phi_0 ~ phi_n
        x = np.zeros((K, N))
        et = np.zeros(Nd)

        for n in range(Nd):
            nb = (n - 1) * Ns + (Nplus - 1) * Ns
            xn = v[:, nb + np.ceil(Ns / FS / 2).astype(int): nb + Ns: int(Ns / FS)]
            
            for k in range(K):
                xn[k, :] = xn[k, :] * np.exp(-1j * f[n, k])
            
            xn = np.fliplr(xn)
            x = np.concatenate([xn, x[:, :N-1]], axis=1)

            for k in range(K):
                p[k] = np.dot(x[k, :], a[(k - 1) * N: k * N])
            
            psum = np.sum(p)
            q = np.dot(d_tilde, b)
            d_hat = psum - q

            if n > Nt:
                d[n] = (d_hat > 0)*2 - 1  # make decision

            e = d[n] - d_hat
            et[n] = abs(e) ** 2

            # parameter update
            phi = np.imag(p * np.conj(p + e))
            Sf = Lf * Sf + phi
            f[n + 1, :] = f[n, :] + Kf1 * phi + Kf2 * Sf

            y = np.reshape(x.T, (1, K * N))
            y = np.concatenate([y, d_tilde])
            k = np.dot(P / L, y.T) / (1 + np.conj(y).dot(P / L).dot(y))
            c = c + k.T.conj() * np.conj(e)
            P = P / L - k.dot(np.conj(y)).dot(P / L)

            a = c[:K * N]
            b = -c[K * N: K * N + M]
            d_tilde = np.concatenate([[d[n]], d_tilde])[:-1]

        # plot
        plt.subplot(2, 2, K_list.index(K) + 1)
        plt.plot(d_hat[Nt:], '*')
        plt.axis('square')
        plt.axis([-2, 2, -2, 2])
        plt.title(f'K={K} SNR={snr}dB')

    plt.show()
    return d_hat

if __name__ == "__main__":
    # initialize
    bits = 7
    rep = 16
    training_rep = 4
    snr_db = np.arange(-10,20,2)
    n_ff = 20
    n_fb = 8
    R = 3000
    fs = 48000
    ns = fs / R
    fc = 16e3
    uf = int(fs / R)
    df = int(uf / ns)
    n_rx = 12
    d_lambda = 0.5 #np.array([0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]) #0.5
    el_spacing = d_lambda*343/fc
    mse = np.zeros(len(snr_db),dtype='float')
    # init rrc filters with b=0.5
    _, rc_tx = rrcosfilter(16 * int(1 / R * fs), 0.5, 1 / R, fs)
    _, rc_rx = rrcosfilter(16 * int(fs / R), 0.5, 1 / R, fs)
    # generate the tx BPSK signal
    # init bits (training bits are a select repition of bits)
    d = np.array(sg.max_len_seq(bits)[0]) * 2 - 1.0
    u = np.tile(d,100)
    # upsample
    u = sg.resample_poly(u,ns,1)
    # filter
    u_rrc = np.convolve(u, rc_tx, "full")
    # upshift
    s = np.real(u_rrc * np.exp(2j * np.pi * fc * np.arange(len(u_rrc)) / fs))
    s /= np.max(np.abs(s))
    # generate rx signal with ISI
    for i in range(len(snr_db)):
        d0 = el_spacing
        snr = 10**(0.1 * snr_db[i])
        r = transmit(s,snr,n_rx,d0,R,fc,fs) # should come out as an n-by-zero
        # downshift
        v = r[:,0] * np.exp(-2j * np.pi * fc * np.arange(len(r)) / fs)
        # decimate
        # v = sg.decimate(v, df)
        # filter
        v_rrc = np.convolve(v, rc_rx, "full")
        # sync (skip for now)

        # dfe
        d_adj = np.tile(d,1)
        d_hat = dfe_v3(v,d_adj,n_ff,n_fb,ns)
        #d_hat = lms(v,d,ns)
        mse[i] = 10 * np.log10(np.mean(np.abs(d_adj-d_hat) ** 2))


    fig, ax = plt.subplots()
    ax.plot(snr_db,mse,'o')
    ax.set_xlabel(r'SNR (dB)')
    ax.set_ylabel('MSE (dB)')
    #ax.set_xticks(np.arange(el_spacing[0],el_spacing[-1],5))
    ax.set_title('MSE vs SNR for BPSK Signal')
    plt.show()
