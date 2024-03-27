import numpy as np
import scipy.signal as sg
import matplotlib.pyplot as plt

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
    delta = 10**(-3)
    Nt = 4*(N+M)
    FS = 2
    Kf1 = 0.001
    Kf2 = Kf1/10
    Lf1 = 1
    L = 0.99
    P = np.eye(int(K*N+M))/delta
    Lbf = 0.99
    Nplus = 6

    v = vk

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
    Ns = 7
    Nd = 3000
    M = int(63)
    
    # load
    vk_dl_real = np.load('data/vk_dl_real.npy')
    vk_dl_imag = np.load('data/vk_dl_imag.npy')
    vk_dl = vk_dl_real + 1j*vk_dl_imag
    d_dl_real = np.load('data/d_dl_real.npy')
    d_dl_imag = np.load('data/d_dl_imag.npy')
    d_dl = d_dl_real + 1j*d_dl_imag
    d_dl = d_dl.flatten()

    vk_ul_real = np.load('data/vk_ul_real.npy')
    vk_ul_imag = np.load('data/vk_ul_imag.npy')
    vk_ul = vk_ul_real + 1j*vk_ul_imag
    d_ul_real = np.load('data/d_ul_real.npy')
    d_ul_imag = np.load('data/d_ul_imag.npy')
    d_ul = d_ul_real + 1j*d_ul_imag
    d_ul = d_ul.flatten()

    # DFE
    d_hat_ul, mse_ul = dfe_matlab(vk_ul,d_ul,Ns,Nd,M) 
    d_hat_dl, mse_dl = dfe_matlab(vk_dl,d_dl,Ns,Nd,M) 

    # uplink constellation diagram
    plt.subplot(1, 2, 1)
    plt.scatter(np.real(d_hat_ul), np.imag(d_hat_ul), marker='x')
    plt.axis('square')
    plt.axis([-2, 2, -2, 2])
    plt.title(f'Uplink QPSK Constellation')

    # downlink constellation diagram
    plt.subplot(1, 2, 2)
    plt.scatter(np.real(d_hat_dl), np.imag(d_hat_dl), marker='x')
    plt.axis('square')
    plt.axis([-2, 2, -2, 2])
    plt.title(f'Downlink QPSK Constellation') 
    plt.show()

    # s(theta)
    theta_start = -45
    theta_end = 45
    N_theta = 200
    deg_theta = np.linspace(theta_start,theta_end,N_theta)
    S_theta = np.load('data/S_theta.npy')
    est_deg = np.argmax(S_theta)
    deg_ax = np.flip(deg_theta)
    plt.plot(deg_ax,S_theta)
    #plt.axvline(x=true_angle, linestyle="--", color="red")
    plt.axvline(x=deg_ax[est_deg], linestyle="--", color="blue")
    plt.text(deg_ax[est_deg], np.max(S_theta), f'Est Angle={deg_ax[est_deg]}')
    #plt.text(true_angle, np.max(S_theta)*1e-5, f'True Angle={true_angle}')
    plt.title(f'S(Theta) Open Air, M={12}, B = 3.9 kHz, d0 =5cm')
    plt.show() 

    # uplink/downlink MSE (could probably re-run with iterations)
    print("Uplink MSE: ", mse_ul)
    print("Downlink MSE: ", mse_dl) 


