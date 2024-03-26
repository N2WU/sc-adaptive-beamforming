import numpy as np
import scipy.signal as sg
import sounddevice as sd

# 2024-02-27: initial commit
# this code simulates noisy two-path reflected environment
# then transmits the adjusted signal back

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

def testbed(s_tx,n_tx,n_rx,Fs):
    # testbed constants
    sd.default.device = "ASIO MADIface USB"
    sd.default.channels = 20 
    sd.default.samplerate = Fs

    # devices [0-15] are for array
    # devices [16-20] are for users
    tx = np.zeros((len(s_tx[:,0]),20))
    if n_tx < n_rx:
        # implies uplink
        s_tx = s_tx.flatten()
        for ch in range(16,16+n_tx):
            tx[:,ch] = s_tx
        print("Transmitting...")
        rx_raw = sd.playrec(tx * 0.5, samplerate=Fs, blocking=True)
        rx = rx_raw[:,:n_rx]
        print("Received")
    else:
        # downlink, meaning s_tx is already weighted array
        for ch in range(n_tx):
            tx[:,ch] = s_tx[:,ch]
        print("Transmitting...")
        rx_raw = sd.playrec(tx * 0.5, samplerate=Fs, blocking=True)
        rx = rx_raw[:,16:16+n_rx] #testbed specific
        print("Received")
    return rx

def downlink(v_dl,wk,Fs,fs,fc,n_rx,n_tx):
    vs = sg.resample_poly(v_dl,Fs,fs)
    s = np.real(vs*np.exp(1j*2*np.pi*fc*np.arange(len(vs))/Fs))
    r = np.zeros(len(s))
    # slightly confused
    # generate angle steering vec and apply
    s_tx = np.dot(np.reshape(wk,[-1,1]),np.reshape(np.fft.fft(s),[1,-1]))
    s_tx = np.fft.ifft(s_tx)

    r = testbed(s_tx.T,n_tx,n_rx,Fs) # s-by-nrx

    r = np.squeeze(r)
    vr = r * np.exp(-1j*2*np.pi*fc*np.arange(len(r))/Fs)
    v_single = sg.resample_poly(vr,1,Fs/fs)
    vps = v_single[:len(up)+Nz*Ns]
    delvals,_ = fdel(vps,up)
    adj_len = delvals+len(up)
    if adj_len > len(vps): 
        delvals = 0 # don't adjust with xcorr if it would otherwise ruin signal
    vp1s = vps[delvals:delvals+len(up)]
    fdes,_,_ = fdop(vp1s,up,fs,12)
    if fdes == -fs/2:
            fdes = 0 # forced
    v_single = v_single*np.exp(-1j*2*np.pi*np.arange(len(v_single))*fdes*Ts)
    v_single = sg.resample_poly(v_single,np.rint(10**4),np.rint((1/(1+fdes/fc))*(10**4)))
    
    v_single = v_single[lenu+Nz*Ns+trunc*Ns+1:] #assuming above just chops off preamble
    v_single = sg.resample_poly(v_single,2,Ns)
    v_single = np.concatenate((v_single,np.zeros(Nplus*2))) # should occur after
    vk = v_single.reshape(1,-1) 

    return vk

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
        return d_hat[Nt : -1], mse, #n_err, n_training

if __name__ == "__main__":
    n_tx = 8
    n_rx = 1
    # generate an s-by-n_tx signal
    Nd = 3000
    Nz = 100
    # init bits (training bits are a select repition of bits)
    
    dp = np.array([1, -1, 1, -1, 1, 1, -1, -1, 1, 1, 1, 1, 1])*(1+1j)/np.sqrt(2)
    fc = 6.5e3
    Fs = 96000
    fs = Fs/4
    Ts = 1/fs
    alpha = 0.25
    trunc = 4
    Ns = 7
    T = Ns*Ts
    R = 1/T
    B = R*(1+alpha)
    Nso = Ns
    uf = int(fs / R)

    g = rcos(alpha,Ns,trunc)
    up = fil(dp,g,Ns)
    lenu = len(up)

    d = np.sign(np.random.randn(Nd))+1j*np.sign(np.random.randn(Nd))
    d /= np.sqrt(2)
    ud = fil(d,g,Ns)
    u = np.concatenate((up, np.zeros(Nz*Ns), ud))
    us = sg.resample_poly(u,Fs,fs)
    
    Ns = 7
    Nplus = 4
    
    v = np.copy(u)
    v_dl = np.copy(v)

    angle_deg = 0

    wk_real = np.load('data/wk_real.npy')
    wk_imag = np.load('data/wk_imag.npy')
    wk = wk_real + 1j*wk_imag

    vk = downlink(v_dl,wk,Fs,fs,fc,n_rx,n_tx)

    np.save('data/vk_dl_real.npy', np.real(vk))
    np.save('data/vk_dl_imag.npy', np.imag(vk))
    np.save('data/d_dl_real.npy', np.real(d))
    np.save('data/d_dl_imag.npy', np.imag(d))

    d_hat_dl, mse_out_dl = dfe_matlab(vk, d, Ns, Nd, int(0))

    print(mse_out_dl)
