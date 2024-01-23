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

def dfe(v,d,n_ff,n_fb,ns):
    delta = 0.001
    Nplus = 3
    n_training = int(4 * (n_ff + n_fb) / ns)
    fractional_spacing = 4
    d_rls = d
    x = np.zeros(n_ff, dtype=complex)
    a = np.zeros(n_ff, dtype=complex)
    b = np.zeros(n_fb, dtype=complex)
    c = np.concatenate((a, -b))
    e = np.zeros((len(d_rls),), dtype=complex)
    theta = np.zeros(len(d_rls), dtype=float)
    d_hat = np.zeros(len(d_rls), dtype=complex)
    d_backward = np.zeros(n_fb, dtype=complex)
    d_tilde = np.zeros((len(d_rls),), dtype=complex)
    Q = np.eye(n_ff + n_fb) / delta
    pk = np.zeros(1, dtype=complex)
    cumsum_phi = 0.0
    sse = 0.0

    for i in np.arange(len(d_rls) - 1, dtype=int):
        nb = (i) * ns + (Nplus - 1) * ns - 1
        xn = v[ int(nb + np.ceil(ns / fractional_spacing / 2)) : int(nb + ns)
            + 1 : int(ns / fractional_spacing)
        ]
        xn *= np.exp(-1j * theta[i])
        x_buf = np.concatenate((xn, x), axis=0)
        x = x_buf[:n_ff]
        pk = np.inner(x, a[:n_ff].conj())
        p = pk.sum()
        q = np.inner(d_backward, b.conj())
        d_hat[i] = p - q
        if i > n_training:
            d_tilde[i] = (d_tilde[i] > 0)*2 - 1
        else:
            d_tilde[i] = d_rls[i]
        e[i] = d_tilde[i] - d_hat[i]
        y = np.concatenate((x.reshape(n_ff), d_backward))
        # PLL
        phi = np.imag(p * np.conj(d_tilde[i] + q))
        cumsum_phi += phi
        theta[i + 1] = theta[i] + 0.0 * phi + 0.0 * cumsum_phi
        # RLS
        k = (
            np.matmul(Q, y.T)
            / 0.5
            / (1 + np.matmul(np.matmul(y.conj(), Q), y.T) / 0.5)
        )
        c += k.T * e[i].conj()
        Q = Q / 0.5 - np.matmul(np.outer(k, y.conj()), Q) / 0.5

        a = c[:n_ff]
        b = -c[-n_fb:]

        d_backward_buf = np.insert(d_backward, 0, d_tilde[i])
        d_backward = d_backward_buf[:n_fb]
    return d_hat

def dfe_v2(v,d,n_ff,n_fb,ns):
    Nplus = 5
    n_training = int(4 * (n_ff + n_fb) / ns)
    fractional_spacing = 4
    a = np.zeros(n_ff, dtype=complex)
    b = np.zeros(n_fb, dtype=complex)
    d_tilde = np.zeros(len(d), dtype=float)
    d_hat = np.zeros(len(d), dtype=float)
    d_backward = np.zeros(n_fb, dtype=complex)
    theta = np.zeros(len(d), dtype=float)
    cumsum_phi = 0.0

    for i in np.arange(len(d) - 1, dtype=int):
        # 1. delay v by N samples
        nb = (i) * ns + (Nplus - 1) * ns - 1
        v_nt = v[ int(nb + np.ceil(ns / fractional_spacing / 2)) : int(nb + ns) 
            + 1 : int(ns / fractional_spacing)]
        v_nt *= np.exp(-1j * theta[i])
        print(len(v_nt))
        print(len(d))
        # 2a. form a' taps of FF filter
        # 2b. store N samples of v
        rvv = np.correlate(v_nt,v_nt[::-1],mode='full')
        Rvv = np.eye(len(rvv))*rvv
        Rvd = np.correlate(v_nt,d,mode='full')
        a = np.dot(np.linalg.inv(Rvv),Rvd)
        v_buf = np.concatenate((v_nt, v_n), axis=0)
        v_n = v_buf[:n_ff]
        # 3. calculate pn
        p = np.sum(np.inner(v_n, a[:n_ff].conj()))
        # 4a. form b' taps of FB filter
        # 4b. store N samples of d_tilde (prev. detected symbols)
        # 5. calcualte qn
        q = np.inner(d_backward, b.conj())
        # 6a. calculate d_hat
        # 6b. calculate d_tilde
        d_hat[i] = p - q
        if i > n_training:
            d_tilde[i] = (d_tilde[i] > 0)*2 - 1
        else:
            d_tilde[i] = d[i]
        e = d_tilde[i] - d_hat[i]
        phi = np.imag(p * np.conj(d_tilde[i] + q))
        cumsum_phi += phi
        theta[i + 1] = theta[i] + 0.0 * phi + 0.0 * cumsum_phi
        c += e.conj()
        a = c[:n_ff]
        b = -c[-n_fb:]
        d_backward_buf = np.insert(d_backward, 0, d_tilde[i])
        d_backward = d_backward_buf[:n_fb]
    return d_hat

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

def rls(un,err,n):
    forget_fac = 0.99
    # the p-order is just the length of un
    rls = np.zeros((n,len(un)),dtype=complex)
    for i in range(n):
        rls[i,:] = 2 * (forget_fac**(n-i))*err[i]*un
    rls_return = np.sum(rls,axis=0)
    return rls_return

def dfe_v3(v,d,n_ff,n_fb,ns):
    # init
    n_training = int(4 * (n_ff + n_fb) / ns)
    # fractional_spacing = 4
    d_tilde = np.zeros(len(d), dtype=float)
    d_hat = np.zeros(len(d), dtype=float)
    err = np.zeros_like(d_hat,dtype=float)
    c = np.ones(n_ff+n_fb,dtype=complex)
    # already assuming RRC does the dirty work
    # step through:
    for i in np.arange(int(n_ff/2),len(d) - 1, dtype=int):
        v_n = v[int(i-n_ff/2):int(i+n_ff/2)] # goes forward and back
        # find d_hat
        d_tilde_n = d_tilde[-n_fb:]
        u_n = np.append(v_n,d_tilde_n)
        d_hat[i] = np.inner(c.conj(),u_n)
        if i > n_training:
            d_tilde[i] = (d_tilde[i] > 0)*2 - 1
        else:
            d_tilde[i] = d[i]
        # find e
        err[i] = d_tilde[i] - d_hat[i]
        # update c
        c = c + rls(u_n,err[:i],i)
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
