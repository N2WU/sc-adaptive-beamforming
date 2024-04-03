import numpy as np
import scipy.signal as sg
import matplotlib.pyplot as plt
import gc

def dec4psk(x):
    xr = np.real(x)
    xi = np.imag(x)
    dr = np.sign(xr)
    di = np.sign(xi)
    d = dr+1j*di
    d = d/np.sqrt(2)
    return d

def dfe_matlab(vk, d, N, Nd, M): 
    K = len(vk[:,0]) # maximum
    Ns = 2
    #N = int(6 * Ns)
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

    v = np.copy(vk)

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
    Nd = 3000
    
    # load
    vk_dl_nobf_real = np.load('data/dl/vk_dl_bf_real.npy')
    vk_dl_nobf_imag = np.load('data/dl/vk_dl_bf_imag.npy')

    vk_dl_bf_real = np.load('data/dl/vk_dl_nobf_real.npy')
    vk_dl_bf_imag = np.load('data/dl/vk_dl_nobf_imag.npy')
    vk_dl_nobf = vk_dl_nobf_real + 1j*vk_dl_nobf_imag
    vk_dl_bf = vk_dl_bf_real + 1j*vk_dl_bf_imag

    d_dl_real = np.load('data/dl/d_dl_real.npy')
    d_dl_imag = np.load('data/dl/d_dl_imag.npy')
    d_dl = d_dl_real + 1j*d_dl_imag
    d_dl = d_dl.flatten()

    vk_ul_bf_real = np.load('data/ul/vk_ul_bf_real.npy')
    vk_ul_bf_imag = np.load('data/ul/vk_ul_bf_imag.npy')
    vk_ul_bf = vk_ul_bf_real + 1j*vk_ul_bf_imag
    vk_ul_nobf_real = np.load('data/ul/vk_ul_nobf_real.npy')
    vk_ul_nobf_imag = np.load('data/ul/vk_ul_nobf_imag.npy')
    vk_ul_nobf = vk_ul_nobf_real + 1j*vk_ul_nobf_imag

    d_ul_real = np.load('data/ul/d_ul_real.npy')
    d_ul_imag = np.load('data/ul/d_ul_imag.npy')
    d_ul = d_ul_real + 1j*d_ul_imag
    d_ul = d_ul.flatten()

    S_theta = np.load('data/ul/S_theta.npy') #archive_03_27_-10deg/

    M_bf = int(5)
    M_nobf = int(10)
    N_bf = int(10)
    N_nobf = int(12)

    # DFE
    d_hat_ul_bf, mse_ul_bf = dfe_matlab(vk_ul_bf,d_ul,N_bf,Nd,M_bf)
    _, mse_ul_bf_sametaps = dfe_matlab(vk_ul_bf,d_ul,N_nobf,Nd,M_nobf)
    d_hat_ul_nobf, mse_ul_nobf = dfe_matlab(vk_ul_nobf,d_ul,N_nobf,Nd,M_nobf) 

    d_hat_dl_bf, mse_dl_bf = dfe_matlab(vk_dl_bf,d_dl,N_bf,Nd,M_bf)
    _, mse_dl_bf_sametaps = dfe_matlab(vk_dl_bf,d_dl,N_nobf,Nd,M_nobf)
    d_hat_dl_nobf, mse_dl_nobf = dfe_matlab(vk_dl_nobf,d_dl,N_bf,Nd,M_bf) 


    # uplink constellation diagram
    #plt.legend([r'BF, $N_{{FF}}=${}, $N_{{FB}}=${}'.format(N_bf, M_bf), 
        #r'BF, $N_{{FF}}=${}, $N_{{FB}}=${} '.format(N_nobf, M_nobf)])
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.scatter(np.real(d_hat_ul_nobf), np.imag(d_hat_ul_nobf), marker='x', alpha=0.5)
    plt.axis('square')
    plt.axis([-2, 2, -2, 2])
    plt.title(r'UL No BF, $N_{{FF}}=${}, $N_{{FB}}=${}, MSE={}'.format(N_nobf, M_nobf, round(mse_ul_nobf,2)))
    # uplink constellation diagram
    plt.subplot(1, 2, 2)
    plt.scatter(np.real(d_hat_ul_bf), np.imag(d_hat_ul_bf), marker='x', alpha=0.5)
    plt.axis('square')
    plt.axis([-2, 2, -2, 2])
    plt.title(r'UL BF, $N_{{FF}}=${}, $N_{{FB}}=${}, MSE={}'.format(N_bf, M_bf,round(mse_ul_bf,2))) 
    
    # downlink constellation diagram
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.scatter(np.real(d_hat_dl_nobf), np.imag(d_hat_dl_nobf), marker='x', alpha=0.5)
    plt.axis('square')
    plt.axis([-2, 2, -2, 2])
    plt.title(r'DL No BF, $N_{{FF}}=${}, $N_{{FB}}=${}, MSE={}'.format(N_nobf, M_nobf, round(mse_dl_nobf,2)))
    # downlink constellation diagram
    plt.subplot(1, 2, 2)
    plt.scatter(np.real(d_hat_dl_bf), np.imag(d_hat_dl_bf), marker='x', alpha=0.5)
    plt.axis('square')
    plt.axis([-2, 2, -2, 2])
    plt.title(r'DL BF, $N_{{FF}}=${}, $N_{{FB}}=${}, MSE={}'.format(N_nobf, M_nobf, round(mse_dl_bf,2))) 

    # s(theta)
    est_deg = np.load('data/ul/ang_est.npy')
    est_deg = np.reshape(est_deg, (-1,1))
    theta_start = -45
    theta_end = 45
    N_theta = 200
    deg_theta = np.linspace(theta_start,theta_end,N_theta)
    deg_ax = deg_theta
    true_angle = [15]
    plt.figure()
    plt.plot(deg_ax,S_theta)
    for i in range(est_deg.size):
            angle = est_deg[i][0]
            plt.axvline(x=true_angle[i], linestyle="--", color="red")
            plt.axvline(x=est_deg[i], linestyle="--", color="blue")
            plt.text(est_deg[i]+2, np.max(S_theta), f'Est Angle={"{:.2f}".format(angle)}') #shorten to 2 {"{:.2f}".format(mse_ul_nobf)}
            plt.text(true_angle[i]+5, np.max(S_theta)/10, f'True Angle={"{:.2f}".format(true_angle[i])}')
    plt.title(r'$S(\theta)$ for Open-Air, M=12, $f_c=6.5$kHz, $d_0$=5cm')
    plt.xlabel(r'Angle ($^\circ$)')
    plt.ylabel(r"$S(\theta)$, Magnitude$^2$")
    
    # uplink/downlink MSE (could probably re-run with iterations)
    print("MSE UL No BF:", mse_ul_nobf)
    print("MSE UL BF:", mse_ul_bf)
    print("MSE UL BF Less Taps:", mse_ul_bf_sametaps)

    print("MSE DL No BF:", mse_dl_nobf)
    print("MSE DL BF:", mse_dl_bf)
    print("MSE DL BF Less Taps:", mse_dl_bf_sametaps)

    plt.show() 

