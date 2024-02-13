import numpy as np
import scipy.signal as sg
import matplotlib.pyplot as plt

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
    p = np.sum(np.abs(x)**2)/len(x)
    return p

def dec4psk(x):
    xr = np.real(x)
    xi = np.imag(x)
    dr = np.sign(xr)
    di = np.sign(xi)
    d = dr+1j*di
    d = d/np.sqrt(2)
    return d

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
    n_path = len(reflection_list)  
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
    

    M = 12
    r = r_multichannel[:,:M]
    r_fft = np.fft.fft(r, axis=0)
    freqs = np.fft.fftfreq(len(r[:, 0]), 1/fs)

    index = np.where((freqs >= fc-R/2) & (freqs < fc+R/2))[0]
    N = len(index)
    fk = freqs[index]       
    yk = r_fft[index, :]    # N*M
    theta_start = -45
    theta_end = 45
    N_theta = 200
    S_theta = np.zeros((N_theta,))
    S_theta3D = np.zeros((N_theta, N))
    theta_start = np.deg2rad(theta_start)
    theta_end = np.deg2rad(theta_end)
    theta_list = np.linspace(theta_start, theta_end, N_theta)
    for n_theta, theta in enumerate(theta_list):
        d_tau = np.sin(theta) * el_spacing/c
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
    # print(theta_m/np.pi*180)

    y_tilde = np.zeros((N,n_path), dtype=complex)
    for k in range(N):
        d_tau_m = np.sin(theta_m) * el_spacing/c
        try:
            d_tau_new = d_tau_m.reshape(1, n_path)
        except:
            d_tau_new = np.append(d_tau_m, [0])
            d_tau_new = d_tau_new.reshape(1, n_path)
        Sk = np.exp(-2j * np.pi * fk[k] * np.arange(M).reshape(M, 1) @ d_tau_new)
        for i in range(n_path):
            e_pu = np.zeros((n_path,1))
            e_pu[i, 0] = 1
            wk = Sk @ np.linalg.inv(Sk.conj().T @ Sk) @ e_pu
            y_tilde[k, i] = wk.conj().T @ yk[k, :].T

    y_fft = np.zeros((len(r[:, 0]), n_path), complex)
    y_fft[index, :] = y_tilde
    y = np.fft.ifft(y_fft, axis=0)

    # processing the signal we get from bf
    r_multichannel_1 = y
    return r_multichannel, wk

def transmit_dl(s_dl,snr,n_rx,el_spacing,R,fc,fs):
    reflection_list = np.asarray([1,0.5]) # reflection gains
    n_path = len(reflection_list)  
    x_rx_list = np.array([5,-5]) 
    y_rx_list = np.array([20,20])
    c = 343
    duration = 10 # amount of padding, basically
    x_tx = np.arange(0, el_spacing*n_rx, el_spacing)
    y_tx = np.zeros_like(x_tx)
    rng = np.random.RandomState(2021)
    r_single = rng.randn(duration * fs) / snr
    r_single = r_single.astype('complex')
    for i in range(len(reflection_list)):
        x_rx, y_rx = x_rx_list[i], y_rx_list[i]
        reflection = reflection_list[i] #delay and sum not scale
        dx, dy = x_rx - x_tx, y_rx - y_tx
        d_rx_tx = np.sqrt(dx**2 + dy**2)
        delta_tau = d_rx_tx / c
        delay = np.round(delta_tau * fs).astype(int) # sample delay
        for j, delay_j in enumerate(delay):
            r_single[delay_j:delay_j+len(s)] += reflection * s_dl[j,:]
    return r_single

def dfe_old(v_rls, d, Ns, feedforward_taps=20, feedbackward_taps=8, alpha_rls=0.98):
        K = len(v_rls[:,0])
        delta = 0.001
        Nplus = 2
        n_training = int(4 * (feedforward_taps + feedbackward_taps) / Ns)
        fractional_spacing = ns_fs
        K_1 = 0.0000 # PLL shits everything up
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
            #k_buf = Q/alpha_rls
            #k_denom = 1+np.matmul(np.matmul(y.conj(),k_buf),y.T)
            #k = np.matmul(k_buf,y.T) / k_denom
            c += k.T * e[i].conj()
            Q = Q / alpha_rls - np.matmul(np.outer(k, y.conj()), Q) / alpha_rls
            #Q = Q / alpha_rls - np.matmul(np.matmul(k.reshape(1,-1), y.conj()), Q) / alpha_rls 
            # Q = Q / alpha_rls - np.matmul(k.reshape(1,-1), y.conj()) * Q / alpha_rls 

            a = c[:K*feedforward_taps]
            b = -c[-feedbackward_taps:]
            # b = -c[K*N:K*N+M]

            d_backward_buf = np.insert(d_backward, 0, d_tilde[i])
            d_backward = d_backward_buf[:feedbackward_taps]

        err = d_rls[n_training + 1 : -1] - d_tilde[n_training + 1 : -1]
        mse = 10 * np.log10(
            np.mean(np.abs(d_rls[n_training + 1 : -1] - d_hat[n_training + 1 : -1]) ** 2)
        )
        n_err = np.sum(np.abs(err) > 0.01)
        return d_hat, mse, #n_err, n_training

def dfe_matlab(v_rls, d, Ns, Nd, feedforward_taps=20, feedbackward_taps=8, alpha_rls=0.5): #, filename='data/r_multichannel.npy'
        K = len(v_rls[:,0]) # maximum
        Ns = 2
        N = int(6 * Ns)
        M = int(0)
        delta = 0.001
        Nt = 4*(N+M)
        FS = 2
        Kf1 = 0.001
        Kf2 = Kf1/10
        Lf1 = 1
        L = 0.98
        P = np.eye(int(K*N+M))/delta
        Lbf = 0.99
        Nplus = 4

        v = v_rls[:K,]

        f = np.zeros((Nd,K))
        
        a = np.zeros(int(K*N), dtype=complex)
        b = np.zeros(M, dtype=complex)
        c = np.append(a, -b)
        p = np.zeros(int(K),dtype=complex)
        d_tilde = np.zeros(M, dtype=complex)
        Sf = np.zeros(K)
        x = np.zeros((K,N), dtype=complex)
        et = np.zeros(Nd, dtype=complex)
        d_hat = np.zeros_like(d, dtype=complex)

        for n in range(Nd-1):
            nb = (n) * Ns + (Nplus - 1) * Ns - 1
            xn = v[
                :, int(nb + np.ceil(Ns / FS / 2)) : int(nb + Ns)
                + 1 : int(Ns / FS)
            ]
            for k in range(K):
               xn[k,:] *= np.exp(-1j * f[n,k])
            xn = np.fliplr(xn)
            x = np.concatenate((xn, x), axis=1)
            x = x[:,:N] # in matlab, this appends zeros

            for k in range(K):
                p[k] = np.inner(x[k,:], a[k*N:(k+1)*N].conj())
            psum = p.sum()

            q = np.inner(d_tilde, b.conj())
            d_hat[n] = psum - q

            if n > Nt:
                #d[n] = (d_hat[n] > 0)*2 - 1
                d[n] = dec4psk(d_hat[n])

            e = d[n]- d_hat[n]
            et[n] = np.abs(e ** 2)

            # y = np.concatenate((x * np.exp(-1j * theta[i]), d_backward))
            # y = np.concatenate((x.reshape(K*feedforward_taps), d_backward))

            # PLL
            phi = np.imag(p * np.conj(p+e))
            Sf = Sf + phi
            f[n,:] = f[n,:] + Kf1*phi + Kf2*Sf

            y = np.reshape(x.T,(1,int(K*N)))
            y = np.append(y,d_tilde)

            # RLS
            k = (
                np.matmul(P, y.T)
                / L
                / (1 + np.matmul(np.matmul(y.conj(), P), y.T) / L)
            )
            c += k.T * e.conj()
            P = P / L - np.matmul(np.outer(k, y.conj()), P) / L

            a = c[:int(K*N)]
            b = -c[int(K*N):int(K*N+M)]
            # b = -c[K*N:K*N+M]
            d_tilde_buf = np.append(d[n], d_tilde)
            d_tilde = d_tilde_buf[:M]

        mse = 10 * np.log10(
            np.mean(np.abs(d[Nt : -1] - d_hat[Nt : -1]) ** 2)
        )
        return d_hat, mse, #n_err, n_training

if __name__ == "__main__":
    # initialize
    bits = 7
    rep = 16
    training_rep = 4
    snr_db = np.array([300, 300, 300, 300]) #np.arange(-10,20,2)
    n_ff = 20
    n_fb = 4
    # R = 3000
    fs = 44100
    Fs = fs/4
    ts = 1/Fs
    ns = 7 #int(fs/R)
    T = ns*ts
    R = 1/T

    ns_fs = 2 # fractionally spaced
    fc = 17e3
    uf = int(fs / R)
    df = int(uf / ns)
    n_rx = 12
    d_lambda = 0.5 #np.array([0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2]) #
    el_spacing = d_lambda*343/fc
    mse = np.zeros(len(snr_db),dtype='float')
    mse_dl = np.zeros(len(snr_db),dtype='float')
    # init rrc filters with b=0.5
    _, rc_tx = rrcosfilter(16 * int(1 / R * fs), 0.5, 1 / R, fs)
    _, rc_rx = rrcosfilter(16 * int(fs / R), 0.5, 1 / R, fs)
    # generate the tx BPSK signal
    # init bits (training bits are a select repition of bits)
    dp = np.array([1, -1, 1, -1, 1, 1, -1, -1, 1, 1, 1, 1, 1])*(1+1j)/np.sqrt(2)
    Nz=100
    g=rcos(0.25,ns,4)
    d = np.tile(dp,rep)
    Nd=len(d)
    up=fil(dp,g,ns)
    ud=fil(d,g,ns)
    u=np.concatenate((up, np.zeros(Nz*ns), ud))
    us = sg.resample_poly(u,fs,Fs)
    # upshift
    s = np.real(us * np.exp(2j * np.pi * fc * np.arange(len(us)) / fs))
    s /= np.max(np.abs(s))
    # generate rx signal with ISI
    for ind in range(len(snr_db)):
        d0 = el_spacing
        snr = 10**(0.1 * snr_db[ind])
        r_multi, wk = transmit(s,snr,n_rx,d0,R,fc,fs) # should come out as an n-by-zero
        # r_multi = np.tile(s,(n_rx,1)).T
        peaks_rx = 0
        v_multichannel = []
        for i in range(len(r_multi[0,:])):
            r = np.squeeze(r_multi[:, i])
            v = r * np.exp(-2 * np.pi * 1j * fc * np.arange(len(r))/fs)
            v_xcorr = np.copy(v)
            #v = np.convolve(v, rc_rx, "full")
            #v = sg.decimate(v, df)
            """
            if i == 0:
                xcorr_for_peaks = np.abs(sg.fftconvolve(v_xcorr, sg.resample_poly(dp[::-1].conj(),uf,1))) # correalte and sync at sample rate sg.decimate(v, int(ns)),
                xcorr_for_peaks /= xcorr_for_peaks.max()
                time_axis_xcorr = np.arange(0, len(xcorr_for_peaks)) / R * 1e3  # ms
                peaks_rx, _ = sg.find_peaks(
                    xcorr_for_peaks, height=0.2, distance=len(dp) - 100
                )
                #plt.figure()
                #plt.plot(np.abs(xcorr_for_peaks))
                #plt.show()
            v = v[int(peaks_rx[1]) :] # * ns
            """
            v = u # right now v is huge
            v[-1] = 0
            v = v/np.sqrt(pwr(v))
            z = np.sqrt(1/(2*snr))*np.random.randn(np.size(v)) + 1j*np.sqrt(1/(2*snr))*np.random.randn(np.size(v))
            v = v + z
            v = v[len(up)+Nz*ns+4*ns+1:]
            if i == 0:
                v_multichannel = v
            else:
                v_multichannel = np.vstack((v_multichannel,v))
        # dfe

        if n_rx == 1:
            v_multichannel = v_multichannel[None,:]
        # resample v_multichannel for frac spac
        v_multichannel = sg.resample_poly(v_multichannel,ns_fs,ns,axis=1)

        d_hat, mse_out = dfe_matlab(v_multichannel, d, ns, len(d), n_ff, n_fb)
        mse[ind] = 10 * np.log10(
            np.mean(np.abs(d[300 :] - d_hat[300 :]) ** 2)
        )
        
        # plot const
        plt.subplot(2, 2, int(ind+1))
        plt.scatter(np.real(d_hat), np.imag(d_hat), marker='*')
        plt.axis('square')
        plt.axis([-2, 2, -2, 2])
        plt.title(f'SNR={snr_db[ind]} dB') #(f'd0={d_lambda[ind]}'r'$\lambda$')

        # Apply wk and use for downlink
        """
        s_dl = np.dot(np.reshape(wk,[-1,1]),np.reshape(s,[1,-1]))
        r_single = transmit_dl(s_dl,snr,n_rx,d0,R,fc,fs)
        peaks_dl = 0
        v_dl = r_single * np.exp(-2 * np.pi * 1j * fc * np.arange(len(r_single))/fs)
        xcorr_dl = np.abs(sg.fftconvolve(v_dl, sg.resample_poly(d[::-1].conj(),uf,1))) # correalte and sync at sample rate sg.decimate(v, int(ns)),
        xcorr_dl /= xcorr_dl.max()
        time_axis_xcorr = np.arange(0, len(xcorr_dl)) / R * 1e3  # ms
        peaks_dl, _ = sg.find_peaks(
            xcorr_dl, height=0.2, distance=len(d) - 100
        )
        v_dl = v_dl[int(peaks_dl[1]) :] # * ns
        d_adj = np.tile(d,rep)
        v_dl = v_dl[None,:]
        # resample v_multichannel for frac spac
        v_dl = sg.resample_poly(v_dl,ns_fs,ns,axis=1)
        d_hat_dl, mse_out_dl = dfe_old(v_dl, d_adj, ns_fs, n_ff, n_fb)
        mse_dl[ind] = mse_out_dl
        plt.subplot(2, 2, int(ind+1))
        plt.scatter(np.real(d_hat_dl), np.imag(d_hat_dl), marker='*')
        plt.axis('square')
        plt.axis([-2, 2, -2, 2])
        plt.title(f'SNR={snr_db[ind]} dB')
        plt.suptitle('Uplink and Downlink BPSK Constellation Diagrams')
        plt.legend(['Uplink','Downlink'])
        """

    fig, ax = plt.subplots()
    ax.plot(snr_db,mse,'o')
    ax.set_xlabel(r'SNR (dB)')
    ax.set_ylabel('MSE (dB)')
    # ax.set_xticks(np.arange(el_spacing[0],el_spacing[-1]))
    ax.set_title('MSE vs SNR for BPSK Signal')
    plt.show()
    print(mse)
    """
    fig, ax = plt.subplots()
    ax.plot(snr_db,mse_dl,'o')
    ax.set_xlabel(r'SNR (dB)')
    ax.set_ylabel('MSE (dB)')
    # ax.set_xticks(np.arange(el_spacing[0],el_spacing[-1]))
    ax.set_title('MSE vs SNR for BPSK Signal Downlink')
    plt.show()

    """
