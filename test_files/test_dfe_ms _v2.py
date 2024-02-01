import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sg

def dec4psk(x):
    xr = np.real(x)
    xi = np.imag(x)
    dr = np.sign(xr)
    di = np.sign(xi)
    d = dr + 1j * di
    d = d / np.sqrt(2)
    return d

def rcos(alpha, Ns, trunc):
    tn = np.arange(-trunc * Ns, trunc * Ns) / Ns
    p = np.sinc(tn) * np.cos(np.pi * alpha * tn) / (1 - 4 * alpha**2 * tn**2)
    p[np.isnan(p)] = 0  # Replace NaN with 0
    p[np.isinf(p)] = 0
    p[Ns * trunc] = 1
    return p

def fdel(v, u):
    N = np.ceil(np.log2(len(v)))  # First 2^N that will do
    times = (np.fft.fft(v, int(2**N)) * np.conj(np.fft.fft(u, int(2**N))))
    x = np.fft.ifft(times)
    del_val = np.argmax(x)
    return del_val, x

def fdop(v, u, fs, Ndes):
    x = v * np.conj(u)
    N = int(np.ceil(np.log2(len(x))))  # First 2^N that will do
    if Ndes > N:
        N = Ndes

    X = np.fft.fft(x, 2**N) / len(x)
    X = np.fft.fftshift(X)

    f = ((np.arange(1, 2**N + 1) - 2**(N - 1) - 1) / (2**N)) * fs

    m, i = np.max(X), np.argmax(X)
    fd = f[i]

    return fd, X, N

def fil(d, p, Ns):
    d = d.astype(complex)
    N = len(d)
    Lp = len(p)
    Ld = Ns * N
    u = np.zeros(Lp + Ld - Ns,dtype=complex)
    for n in range(len(d)):
        window = np.arange(int(n*Ns), int(n*Ns + Lp))
        u[window] =  u[window] + d[n] * p
    return u

def pwr(x):
    return np.sum(np.abs(x)**2) / len(x)

def gen_s(fc,Fs):
    dp = np.array([1, -1, 1, -1, 1, 1, -1, -1, 1, 1, 1, 1, 1]) #bpsk preamble
    Nd = 3000
    Nz = 100
    fs = Fs / 4
    Ts = 1 / fs
    alpha = 0.25
    trunc = 4
    Ns = 7
    T = Ns * Ts
    R = 1 / T
    B = R * (1 + alpha)
    Nso = Ns

    g = rcos(alpha, Ns, trunc)
    up = fil(dp, g, Ns)
    lenu = len(up)

    d = np.sign(np.random.randn(Nd))# bpsk data
    ud = fil(d, g, Ns)
    u = np.concatenate([up, np.zeros(Nz * Ns), ud]) # preamble _____ data
    us = sg.resample_poly(u, Fs, fs)
    s = np.real(us * np.exp(1j * 2 * np.pi * fc * np.arange(len(us)) / Fs))
    return s

def transmit(s,el_spacing,n_rx,snr):
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
    return r_multichannel  

def dfe_func(vk, d, Nd, Tmp, T, Nplus, snr):
    plt.figure()
    K_list = np.array([1, 2, 5, 8])

    for K_index ,K in enumerate(K_list):
        Ns = 2
        N = 6 * Ns
        M = int(np.ceil(Tmp / T))
        delta = 1e-3
        Nt = 4 * (N + M)
        FS = 2
        Kf1 = 0.001
        Kf2 = Kf1 / 10
        Lf = 1
        L = 0.98
        P = np.eye(int(K * N + M)) / delta
        Lbf = 0.99

        v = vk[:K,:]

        f = np.zeros((Nd+1, K), dtype=complex)
        a = np.zeros(int(K * N), dtype=complex)
        b = np.zeros(M, dtype=complex)
        c = np.concatenate([a, -b])
        p = np.zeros(K, dtype=complex)
        d_tilde = np.zeros(M, dtype=complex)
        Sf = np.zeros(K)  # sum of phi_0 ~ phi_n
        x = np.zeros((K, N), dtype=complex)
        et = np.zeros(Nd, dtype=complex)
        d_hat = np.zeros(Nd, dtype=complex)

        for n in range(Nd):
            nb = n* Ns + (Nplus - 1) * Ns # check these
            xn = v[:, nb + np.ceil(Ns / FS / 2).astype(int): nb + Ns+1: int(Ns / FS)]
            
            for k in range(K):
                xn[k, :] = xn[k, :] * np.exp(-1j * f[n, k])
            
            xn = np.fliplr(xn)
            if K==0:
                x = np.append(xn, x)
                x = x[:N]
                for k in range(1,K):
                    p[k] = np.dot(x, a[(k - 1) * N: k * N])
            else:    
                x = np.concatenate((xn, x),axis=1)
                x = x[:,:N]
                for k in range(1,K):
                    p[k] = np.dot(x[k, :], a[(k - 1) * N: k * N])
        
            psum = np.sum(p)
            q = np.dot(d_tilde, b)
            d_hat[n] = psum - q

            if n > Nt:
                d[n] = (d_hat[n] > 0)*2 - 1  # make decision

            e = d[n] - d_hat[n]
            et[n] = abs(e) ** 2

            # parameter update
            phi = np.imag(p * np.conj(p + e))
            Sf = Lf * Sf + phi
            f[n + 1, :] = f[n, :] + Kf1 * phi + Kf2 * Sf

            y = np.reshape(x.T, (K * N))
            y = np.concatenate([y, d_tilde])
            k = np.dot(P / L, y.T) / (1 + np.conj(y).dot(P / L).dot(y))
            c = c + k.T.conj() * np.conj(e)
            P = P / L - k * np.conj(y) * (P / L)

            a = c[:K * N]
            b = -c[K * N: K * N + M]
            d_tilde = np.concatenate([[d[n]], d_tilde])[:-1]

        # plot
        plt.subplot(2, 2, int(K_index+1))
        plt.scatter(np.real(d_hat), np.imag(d_hat), marker='*')
        plt.axis('square')
        plt.axis([-2, 2, -2, 2])
        plt.title(f'K={K} SNR={snr}dB')

    plt.show()
    return d_hat

if __name__ == "__main__":
    # Example usage:
    fc = 17000
    Fs = 44100
    K0 = 2
    s = gen_s(fc,Fs,K0)
    r_multi = transmit(s,el_spacing=0.05,n_rx=8)
    Nd = 3000
    d = np.sign(np.random.randn(Nd))
    Tmp = 40 / 1000
    Ns = 7
    Fs = 44100
    fs = Fs / 4
    Ts = 1 / fs
    T = Ns * Ts
    Nplus = 4
    snr = 15
    d_hat = dfe_func(vk, d, Nd, Tmp, T, Nplus, snr)
    print(d_hat)