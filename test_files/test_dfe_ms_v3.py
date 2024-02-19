import numpy as np
import scipy.signal as sg
import matplotlib.pyplot as plt

def rcos(alpha, Ns, trunc):
    tn = np.arange(-trunc * Ns, trunc * Ns+1) / Ns
    p = np.sin(np.pi* tn)/(np.pi*tn) * np.cos(np.pi * alpha * tn) / (1 - 4 * (alpha**2) * (tn**2))
    p[np.isnan(p)] = 0  # Replace NaN with 0
    p[np.isinf(p)] = 0
    p[-1] = 1
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

    f = (np.arange(2**N)-(2**N)/2 -1)/(2**N)*fs
    #f = ((np.arange(1, 2**N + 1) - 2**(N - 1) - 1) / (2**N)) * fs

    m, i = np.max(X), np.argmax(X)
    fd = f[i]

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

def dec4psk(x):
    xr = np.real(x)
    xi = np.imag(x)
    dr = np.sign(xr)
    di = np.sign(xi)
    d = dr+1j*di
    d = d/np.sqrt(2)
    return d

def dfe_matlab(vk, d, Ns, Nd): #, filename='data/r_multichannel.npy'
        K = len(vk[:,0]) # maximum
        Ns = 2
        N = int(6 * Ns)
        M = int(0)
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

        for n in range(1,Nd-1):
            nb = (n-1) * Ns + (Nplus - 1) * Ns
            xn = v[
                :, int(nb + np.ceil(Ns / FS / 2)) : int(nb + Ns + 1)
                : int(Ns / FS)
            ]
            for k in range(K):
               xn[k,:] *= np.exp(-1j * f[n,k])
            xn = np.fliplr(xn)
            x = np.concatenate((xn, x), axis=1)
            x = x[:,:N] # in matlab, this appends zeros

            for k in range(1,K):
                p[k] = np.inner(x[k,:], a[(k-1)*N:k*N].conj())
            psum = p.sum()

            q = np.inner(d_tilde, b.conj())
            d_hat[n] = psum - q

            if n > Nt:
                #d[n] = (d_hat[n] > 0)*2 - 1
                d[n] = dec4psk(d_hat[n])

            e = d[n]- d_hat[n]
            et[n] = np.abs(e ** 2)

            # PLL
            phi = np.imag(p * np.conj(p+e))
            Sf = Sf + phi
            f[n+1,:] = f[n,:] + Kf1*phi + Kf2*Sf

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

    g=rcos(alpha,Ns,trunc)
    up = fil(dp,g,Ns)
    lenu = len(up)

    d=np.sign(np.random.randn(Nd))+1j*np.sign(np.random.randn(Nd))
    d /= np.sqrt(2)
    ud = fil(d,g,Ns)
    u=np.concatenate((up, np.zeros(Nz*Ns), ud))
    us = sg.resample_poly(u,Fs,fs)
    # upshift
    s = np.real(us * np.exp(2j * np.pi * fc * np.arange(len(us)) / Fs))
    #s /= np.max(np.abs(s))

    snr_db = np.array([300,300,300,300])
    mse = np.zeros_like(snr_db)

    K0 = 10
    Ns = 7
    Nplus = 4
    # generate rx signal with ISI
    for ind in range(len(snr_db)):
        snr = 10**(0.1 * snr_db[ind])
        for k in range(K0):
            Tmp = 0
            v = np.zeros(len(u),dtype=complex)
            vp = u
            vp[-1] = 0
            v = v+vp # right now v is huge
            
            v /= np.sqrt(pwr(v))
            v0 = v
            v = v0
            z = np.sqrt(1/(2*snr))*np.random.randn(np.size(v)) + 1j*np.sqrt(1/(2*snr))*np.random.randn(np.size(v))
            v = v + z - z
            v = v[lenu+Nz*Ns+trunc*Ns+1:]

            v = sg.resample_poly(v,2,Ns)
            v = np.concatenate((v,np.zeros(Nplus*2)))
            if k==0:
                vk = np.zeros((K0,len(v)),dtype=complex)
            vk[k,:] = v
        #np.save('data/vk_real.npy',np.real(vk))
        #np.save('data/vk_imag.npy',np.imag(vk))
        #np.save('data/d_real.npy',np.real(d))
        #np.save('data/d_imag.npy',np.imag(d))
        vk_real = np.load('data/vk_real.npy')
        vk_imag = np.load('data/vk_imag.npy')
        vk = vk_real + 1j*vk_imag
        d_real = np.load('data/d_real.npy')
        d_imag = np.load('data/d_imag.npy')
        d = d_real + 1j*d_imag

        d_hat, mse_out = dfe_matlab(vk, d, Ns, Nd)
        #mse[ind] = 10 * np.log10(
        #    np.mean(np.abs(d[300 :] - d_hat[300 :]) ** 2)
        #)
        mse[ind] = mse_out

        # plot const
        plt.subplot(2, 2, int(ind+1))
        plt.scatter(np.real(d_hat), np.imag(d_hat), marker='*')
        plt.axis('square')
        plt.axis([-2, 2, -2, 2])
        plt.title(f'SNR={snr_db[ind]} dB') #(f'd0={d_lambda[ind]}'r'$\lambda$')

    fig, ax = plt.subplots()
    ax.plot(snr_db,mse,'o')
    ax.set_xlabel(r'SNR (dB)')
    ax.set_ylabel('MSE (dB)')
    ax.set_title('MSE vs SNR for BPSK Signal')
    plt.show()
    print(mse)
