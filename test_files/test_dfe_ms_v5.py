import numpy as np
import scipy.signal as sg
import matplotlib.pyplot as plt

# 2024-02-22: init commit
# this code simulates two-path refelct environment from uplink to downlink

def rcos(alpha, Ns, trunc):
    tn = np.arange(-trunc * Ns, trunc * Ns+1) / Ns
    p = np.sin(np.pi* tn)/(np.pi*tn) * np.cos(np.pi * alpha * tn) / (1 - 4 * (alpha**2) * (tn**2))
    p[np.isnan(p)] = 0  # Replace NaN with 0
    p[np.isinf(p)] = 0
    p[-1] = 0
    p[int(Ns*trunc)] = 1
    return p

def fdel(v, u):
    N = np.ceil(np.log2(len(v)))  # First 2^N that will do
    x = np.fft.ifft(np.fft.fft(v, int(2**N), ) *
    np.conj(np.fft.fft(u, int(2**N)))
    )
    del_val = np.argmax(np.abs(x))
    return del_val, x

def fdop(v, u, fs, Ndes):
    x = v * np.conj(u)
    N = int(np.ceil(np.log2(len(x))))  # First 2^N that will do
    if Ndes > N:
        N = Ndes

    X = np.fft.fft(x, 2**N) / len(x)
    X = np.fft.fftshift(X)

    f = (np.arange(2**N)-(2**N)/2 -1)/(2**N)*fs
    #f = ((np.arange(1, 2**N + 1) - 2**(N - 1) - 1) / (2**N)) * fs

    i = np.argmax(np.abs(X))
    fd = f[i+1]

    return fd, X, N

def fil(d, p, Ns):
    d = d.astype(complex)
    N = len(d)
    Lp = len(p)
    Ld = Ns * N
    u = np.zeros(Lp + Ld - Ns,dtype=complex)
    for n in range(N-1):
        window = np.arange(int(n*Ns), int(n*Ns + Lp))
        u[window] =  u[window] + d[n] * p
    return u

def pwr(x):
    p = np.sum(np.abs(x)**2)/len(x)
    return p

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

def transmit_simple(s,snr,n_rx):
    a = 1/350
    r = sg.resample_poly(s,np.rint(10**4),np.rint((1+a)*10**4))
    rng = np.random.RandomState(2021)
    r_multi = rng.randn(n_rx,len(r)) / snr
    r_multi += np.tile(r,(n_rx,1))
    return r_multi.T

def transmit_passband(v,snr,Fs,fs,fc):
    vs = sg.resample_poly(v,Fs,fs)
    s = np.real(vs*np.exp(1j*2*np.pi*fc*np.arange(len(vs))/Fs))
    a = 1/350
    r = sg.resample_poly(s,np.rint(10**4),np.rint((1+a)*(10**4)))
    zr = np.sqrt(1/(2*snr))*np.random.randn(len(r))
    zi = np.sqrt(1/(2*snr))*np.random.randn(len(r))
    z = zr + 1j*zi
    r = r + z
    vr = 2*r*np.exp(-1j*2*np.pi*fc*np.arange(len(r))/Fs)
    v = sg.resample_poly(vr,1,Fs/fs)
    return v

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

def dec4psk(x):
    xr = np.real(x)
    xi = np.imag(x)
    dr = np.sign(xr)
    di = np.sign(xi)
    d = dr+1j*di
    d = d/np.sqrt(2)
    return d

def dfe_matlab(vk, d, Ns, Nd, M): 
        K = len(vk[:,0]) # maximum
        Ns = 2
        N = int(6 * Ns)
        # M = np.rint(0)
        delta = 10**(-3)
        Nt = 4*(N+M)
        FS = 2
        Kf1 = 0.001
        Kf2 = Kf1/10
        Lf1 = 1
        L = 0.98
        P = np.eye(int(K*N+M))/delta
        Lbf = 0.99
        Nplus = 4

        v = vk[:K,]

        f = np.zeros((Nd,K),dtype=complex)
        
        a = np.zeros(int(K*N), dtype=complex)
        b = np.zeros(M, dtype=complex)
        c = np.append(a, -b)
        p = np.zeros(int(K),dtype=complex)
        d_tilde = np.zeros(M, dtype=complex)
        Sf = np.zeros(K)
        x = np.zeros((K,N), dtype=complex)
        a_log = np.zeros_like(x, dtype=complex)
        et = np.zeros(Nd, dtype=complex)
        d_hat = np.zeros_like(d, dtype=complex)

        for n in range(Nd-1):
            nb = (n) * Ns + (Nplus - 1) * Ns
            xn = v[
                :, int(nb + np.ceil(Ns / FS / 2)-1) : int(nb + Ns)
                : int(Ns / FS)
            ]
            for k in range(K):
               xn[k,:] *= np.exp(-1j * f[n,k])
            xn = np.fliplr(xn)
            x = np.concatenate((xn, x), axis=1)
            x = x[:,:N] # in matlab, this appends zeros

            for k in range(K):
                p[k] = np.inner(x[k,:], np.conj(a[(k*N):(k+1)*N]))
            psum = p.sum()
            #print(psum)

            q = np.inner(d_tilde, b.conj())
            d_hat[n] = psum - q

            if n > Nt:
                d[n] = dec4psk(d_hat[n])

            e = d[n]- d_hat[n]
            et[n] = np.abs(e ** 2)

            # PLL
            phi = np.imag(p * np.conj(p+e))
            Sf = Sf + phi
            f[n+1,:] = f[n,:] + Kf1*phi + Kf2*Sf

            y = np.reshape(x,int(K*N))
            y = np.append(y,d_tilde)

            # RLS
            k = (
                np.matmul(P, y) / L
                / (1 + np.matmul(np.matmul(y.conj(), P), y) / L)
            )
            c += k * np.conj(e)
            P = P / L - np.matmul(np.outer(k, y.conj()), P) / L

            a = c[:int(K*N)]
            b = -c[int(K*N):int(K*N+M)]
            d_tilde = np.append(d[n], d_tilde)
            d_tilde = d_tilde[:M]

        mse = 10 * np.log10(
            np.mean(np.abs(d[Nt : -1 ] - d_hat[Nt : -1]) ** 2)
        )
        if np.isnan(mse):
            mse = 100
        return d_hat, mse, #n_err, n_training

if __name__ == "__main__":
    Nd = 3000
    Nz = 100
    # init bits (training bits are a select repition of bits)
    
    dp = np.array([1, -1, 1, -1, 1, 1, -1, -1, 1, 1, 1, 1, 1])*(1+1j)/np.sqrt(2)
    fc = 17e3
    Fs = 44100
    fs = Fs/4
    Ts = 1/fs
    alpha = 0.25
    trunc = 4
    Ns = 7
    T = Ns*Ts
    R = 1/T
    B = R*(1+alpha)
    Nso = Ns
    uf = np.rint(fs / R)

    n_rx = 12
    d_lambda = 0.5 #np.array([0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2]) #
    el_spacing = d_lambda*343/fc

    g = rcos(alpha,Ns,trunc)
    up = fil(dp,g,Ns)
    lenu = len(up)

    d = np.sign(np.random.randn(Nd))+1j*np.sign(np.random.randn(Nd))
    d /= np.sqrt(2)
    ud = fil(d,g,Ns)
    u = np.concatenate((up, np.zeros(Nz*Ns), ud))
    us = sg.resample_poly(u,Fs,fs)
    # upshift
    s = np.real(us * np.exp(2j * np.pi * fc * np.arange(len(us)) / Fs))
    # s /= np.max(np.abs(s))

    snr_db = np.array([30, 31, 32, 33])
    mse = np.zeros_like(snr_db)
    mse_dl = np.zeros_like(snr_db)

    load = False

    K0 = 10
    Ns = 7
    Nplus = 4
    # generate rx signal with ISI
    for ind in range(len(snr_db)):
        #for repeat in range(4):
        snr = 10**(0.1 * snr_db[ind])
        d0 = el_spacing
        for i in range(K0):
            Tmp = 40/1000
            tau = (3 + np.random.randint(1,33,6))/1000
            h = np.exp(-tau/Tmp)
            h /= np.sqrt(np.sum(np.abs(h)))
            taus = np.rint(tau/Ts)
            v = np.zeros(len(u) + int(np.max(taus)),dtype=complex)
            for p in range(len(tau)):
                taup = int(taus[p])
                vp = np.append(np.zeros(taup),h[p]*u)
                lendiff = len(v) - len(vp)
                if lendiff > 0:
                    vp = np.append(vp,np.zeros(lendiff))
                elif lendiff < 0:
                    v = np.append(v,np.zeros(-lendiff))
                v = v+vp
            v /= np.sqrt(pwr(v))
            v = transmit_passband(v,snr,Fs,fs,fc)
            vp = v[:len(up)+Nz*Ns]
            delval,_ = fdel(vp,up)
            vp1 = vp[delval:delval+len(up)]
            fde,_,_ = fdop(vp1,up,fs,12)
            v = v*np.exp(-1j*2*np.pi*np.arange(len(v))*fde*Ts)
            v = sg.resample_poly(v,np.rint(10**4),np.rint((1/(1+fde/fc))*(10**4)))

            v = v[delval:delval+len(u)]
            v = v[lenu+Nz*Ns+trunc*Ns+1:] #assuming above just chops off preamble
            v = sg.resample_poly(v,2,Ns)
            v = np.concatenate((v,np.zeros(Nplus*2))) # should occur after 
            if i == 0:
                v_multichannel = v
                lenvm = len(v)
            else:
                lendiff = lenvm - len(v)
                if lendiff > 0:
                    v = np.append(v,np.zeros(lendiff))
                elif lendiff < 0:
                    if i == 1:
                        v_multichannel = v_multichannel.reshape(1,-1)
                    v_multichannel = np.concatenate((v_multichannel,np.zeros((len(v_multichannel[:,0]),-lendiff))),axis=1)
                v_multichannel = np.vstack((v_multichannel,v))
                lenvm = len(v_multichannel[0,:])
        if n_rx == 1:
            v_multichannel = v_multichannel[None,:]

        vk = np.copy(v_multichannel)

        if load:
            vk_real = np.load('data/vk_real.npy')
            vk_imag = np.load('data/vk_imag.npy')
            vk = vk_real + 1j*vk_imag
            d_real = np.load('data/d_real.npy')
            d_imag = np.load('data/d_imag.npy')
            d = d_real + 1j*d_imag
            d = d.flatten()
        else: 
            np.save('data/vk_real.npy', np.real(vk))
            np.save('data/vk_imag.npy', np.imag(vk))
            np.save('data/d_real.npy', np.real(d))
            np.save('data/d_imag.npy', np.imag(d))

        # Tmp = 40/1000 # maybe you could move this to peaks_rx[]
        M = np.rint(Tmp/T) # just creates the n_fb value
        M = int(M)
        d_hat, mse_out = dfe_matlab(vk, d, Ns, Nd, M)

        mse[ind] = mse_out

        # plot const
        plt.subplot(2, 2, int(ind+1))
        plt.scatter(np.real(d_hat), np.imag(d_hat), marker='x')
        plt.axis('square')
        plt.axis([-2, 2, -2, 2])
        plt.title(f'SNR={snr_db[ind]} dB') #(f'd0={d_lambda[ind]}'r'$\lambda$')

    fig, ax = plt.subplots()
    ax.plot(snr_db,mse,'o')
    ax.set_xlabel(r'SNR (dB)')
    ax.set_ylabel('MSE (dB)')
    ax.set_title('MSE vs SNR for QPSK Signal')
    plt.show()
    print(mse)
