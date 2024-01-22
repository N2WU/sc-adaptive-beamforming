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

def transmit(s,snr):
    h = np.asarray([0.5, 0.71, 0.5]) # from salehi
    sigma = 1 / (2 * snr)
    n = np.sqrt(sigma)*np.random.randn(len(s)+len(h)-1)
    r = np.convolve(h,s)+n
    return r

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
        snr = 10**(0.1 * snr_db[i])
        r = transmit(s,snr)
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
    ax.plot(snr_db,mse,'o')
    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('MSE (dB)')
    ax.set_xticks(np.arange(snr_db[0],snr_db[-1],5))
    ax.set_title('SNR vs MSE for BPSK Signal')
    plt.show()
