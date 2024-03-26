import numpy as np
import scipy.signal as sg
import matplotlib.pyplot as plt

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

def array_conditions(fc,B,n_rx,el_spacing):
    c = 343
    theta_p = 0
    lambda_max = c/(fc - B/2)
    lambda_min = c/(fc + B/2)

    delmin = lambda_max/(el_spacing*n_rx)
    delmax = lambda_min/el_spacing
    theta_q_deg = np.linspace(-90,90)
    theta_q = np.deg2rad(theta_q_deg)
    theta_p = np.deg2rad(theta_p)

    dataq = np.abs(np.sin(theta_p) - np.sin(theta_q))
    delmindat = np.squeeze(np.where(dataq < delmin))
    delmaxdat = np.squeeze(np.where(dataq > delmax))
    try:
        ang_res = (theta_q_deg[np.amax(delmindat)]-theta_q_deg[np.amin(delmindat)])/2
    except:
        ang_res = 0
    try:
        amb_res = (theta_q_deg[delmaxdat[int(len(delmaxdat)/2)]] - theta_q_deg[delmaxdat[int(len(delmaxdat)/2-1)]])/2
    except:
        amb_res = (theta_q_deg[-1]-theta_q_deg[0])/2
    return ang_res, amb_res

def transmit(v,snr,Fs,fs,fc,n_rx,d0,bf):
    reflection_list = np.asarray([1,0.5]) # reflection gains
    n_path = len(reflection_list)  
    x_tx_list = np.array([5,-5]) 
    y_tx_list = np.array([20,20])
    c = 343
    x_rx = d0 * np.arange(n_rx)
    x_rx = x_rx - d0*n_rx/2 #center on origin
    y_rx = np.zeros_like(x_rx)
    vs = sg.resample_poly(v,Fs,fs)
    #vs = sg.resample_poly(v,4,1)
    #vs = np.copy(v)
    s = np.real(vs*np.exp(1j*2*np.pi*fc*np.arange(len(vs))/Fs)) #Fs
    a = 1/c
    r_multi = np.random.randn(int(2*len(s)), n_rx) / snr
    for i in range(len(reflection_list)):
        x_tx, y_tx = x_tx_list[i], y_tx_list[i]
        reflection = reflection_list[i] #delay and sum not scale
        true_angle = np.rad2deg(np.arctan(np.abs(x_tx/y_tx)))
        dx, dy = x_rx - x_tx, y_rx - y_tx
        d_rx_tx = np.sqrt(dx**2 + dy**2)
        delta_tau = d_rx_tx / c
        #delta_tau = el_spacing/c * np.sin(np.deg2rad(true_angle))
        delay = np.round(delta_tau * Fs).astype(int) # sample delay, grrrr
        for j, delay_j in enumerate(delay):
            r_multi[delay_j:delay_j+len(s), j] += reflection * s
    peaks_rx = 0
    wk = np.ones(n_rx)
    deg_diff = 0
    if bf == 1:
        # now beamforming and weights
        r = r_multi #[:,:M]
        M = int(n_rx)
        r_fft = np.fft.fft(r, axis=0)
        freqs = np.fft.fftfreq(len(r[:, 0]), 1/fs) # this is the same as 1/Fs with no adjustment
        freqs = freqs*Fs/fs

        index = np.where((freqs >= fc-R/2) & (freqs < fc+R/2))[0]
        N = len(index) #uhh???
        if N > 0:
            fk = freqs[index]       
            yk = r_fft[index, :]    # N*M
            theta_start = -45
            theta_end = 45
            N_theta = 200
            deg_theta = np.linspace(theta_start,theta_end,N_theta)
            S_theta = np.zeros((N_theta,))
            S_theta3D = np.zeros((N_theta, N))
            theta_start = np.deg2rad(theta_start)
            theta_end = np.deg2rad(theta_end)
            theta_list = np.linspace(theta_start, theta_end, N_theta)
            for n_theta, theta in enumerate(theta_list):
                d_tau = np.sin(theta) * d0/c
                # for k in range(N):
                #     S_M = np.exp(-2j * np.pi * fk[k] * d_tau * np.arange(M).reshape(M,1))
                #     S_theta[n_theta] += np.abs(np.vdot(S_M.T, yk[k, :].T))**2
                S_M = np.exp(-2j * np.pi * d_tau * np.dot(fk.reshape(N, 1), np.arange(M).reshape(1, M)))    # N*M
                SMxYk = np.einsum('ij,ji->i', S_M.conj(), yk.T)
                S_theta[n_theta] = np.real(np.vdot(SMxYk, SMxYk)) #real part
                S_theta[n_theta] = np.abs(S_theta[n_theta])**2 # doesn't necessarily make a difference
                S_theta3D[n_theta, :] = np.abs(SMxYk)**2

            # n_path = 1 # number of path
            S_theta_peaks_idx, _ = sg.find_peaks(S_theta, height=0) #S_theta
            S_theta_peaks = S_theta[S_theta_peaks_idx] #S_theta
            theta_m_idx = np.argsort(S_theta_peaks)
            theta_m = theta_list[S_theta_peaks_idx[theta_m_idx[-n_path:]]]
            print(theta_m/np.pi*180)

            y_tilde = np.zeros((N,n_path), dtype=complex)
            for k in range(N):
                d_tau_m = np.sin(theta_m) * d0/c
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
            bf_flag = True
        else:
            y = np.copy(vk).T
            bf_flag = False
        
        est_deg = np.argmax(S_theta)
        deg_ax = np.flip(deg_theta)
        deg_diff = np.abs(true_angle - deg_ax[est_deg])
        """
        print(true_angle)
        plt.plot(deg_ax,S_theta)
        plt.axvline(x=true_angle, linestyle="--", color="red")
        plt.axvline(x=deg_ax[est_deg], linestyle="--", color="blue")
        plt.text(deg_ax[est_deg], np.max(S_theta), f'Est Angle={deg_ax[est_deg]}')
        plt.text(true_angle, np.max(S_theta)*1e-5, f'True Angle={true_angle}')
        plt.title(f'S(Theta) for 10 dB, M={n_rx}, B = 3.9 kHz, d0 ={d0}')
        plt.show()
        """
        
        r_multi = np.copy(y)
    for i in range(len(r_multi[0,:])):
        r = np.squeeze(r_multi[:, i])
        vr = r * np.exp(-1j*2*np.pi*fc*np.arange(len(r))/Fs) #Fs
        v = sg.resample_poly(vr,1,Fs/fs)
        #v = sg.resample_poly(vr,1,4)
        vp = v[:len(up)+Nz*Ns]
        delval,_ = fdel(vp,up)
        vp1 = vp[delval:delval+len(up)]
        lendiff = len(up)-len(vp1)
        if lendiff > 0:
            vp1 = np.append(vp1, np.zeros(lendiff))
        fde,_,_ = fdop(vp1,up,fs,12)
        v = v*np.exp(-1j*2*np.pi*np.arange(len(v))*fde*Ts)
        v = sg.resample_poly(v,np.rint(10**4),np.rint((1/(1+fde/fc))*(10**4)))
        
        v = v[delval:delval+len(u)]
        v = v[lenu+Nz*Ns+trunc*Ns+1:] #assuming above just chops off preamble
        v = sg.resample_poly(v,2,Ns)
        v = np.concatenate((v,np.zeros(Nplus*2)))
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
    vk = np.copy(v_multichannel)

    return vk, wk, deg_diff

def transmit_dl(v_dl,wk,snr,n_rx,el_spacing,R,fc,fs):
    vs = sg.resample_poly(v_dl,Fs,fs)
    s_dls = np.real(vs*np.exp(1j*2*np.pi*fc*np.arange(len(vs))/Fs))
    # apply wk here
    #s_dl = wk * s_dl
    #s_dl = np.tile(s_dl,(n_rx,1))
    s_dl = np.dot(np.reshape(wk,[-1,1]),np.reshape(s_dls,[1,-1]))
    reflection_list = np.asarray([1,0.5]) # reflection gains 
    x_rx_list = np.array([5,-5]) 
    y_rx_list = np.array([20,20])
    c = 343
    duration = 10 # amount of padding, basically
    x_tx = np.arange(0, el_spacing*n_rx, el_spacing)
    y_tx = np.zeros_like(x_tx)
    rng = np.random.RandomState(2021)
    r_single = rng.randn(int(duration * fs)) / snr
    r_single = r_single.astype('complex')
    for i in range(len(reflection_list)):
        x_rx, y_rx = x_rx_list[i], y_rx_list[i]
        reflection = reflection_list[i] #delay and sum not scale
        dx, dy = x_rx - x_tx, y_rx - y_tx
        d_rx_tx = np.sqrt(dx**2 + dy**2)
        delta_tau = d_rx_tx / c
        delay = np.round(delta_tau * fs).astype(int) # sample delay
        for j, delay_j in enumerate(delay):
            r_single[delay_j:delay_j+len(s_dl[0,:])] += reflection * s_dl[j,:]
    r = np.squeeze(r_single)
    vr = r * np.exp(-1j*2*np.pi*fc*np.arange(len(r))/Fs)
    v_single = sg.resample_poly(vr,1,Fs/fs)
    return v_single

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
    Nd = 3000
    Nz = 100
    # init bits (training bits are a select repition of bits)
    
    dp = np.array([1, -1, 1, -1, 1, 1, -1, -1, 1, 1, 1, 1, 1])*(1+1j)/np.sqrt(2)
    fc = 6.5e3
    Fs = 44100
    fs = Fs/4
    Ts = 1/fs
    alpha = 0.25
    trunc = 4
    Ns = 7
    T = Ns*Ts
    R = 1/T
    B = R*(1+alpha)/2
    Nso = Ns
    uf = int(fs / R)

    el_num = np.array([6, 10, 12, 16])
    el_num = np.arange(6,18,2)
    el_spacing = 0.05 #np.array([0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3])
    d_lambda = el_spacing*fc/343

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

    snr_db = 10 #np.array([5, 8, 12, 15])
    mse = np.zeros_like(el_num)
    mse_dl = np.zeros_like(mse)
    d_hat_cum = np.zeros((len(mse),Nd-88-1), dtype=complex) # has to change if Nt changes :(
    d_hat_dl_cum = np.zeros_like(mse,dtype=complex)

    mse_wk = np.zeros_like(mse)
    d_hat_cum_wk = np.zeros((len(mse),Nd-88-1), dtype=complex) # has to change if Nt changes :(
    deg_diff_cum = np.zeros_like(mse,dtype=float)

    load = False
    downlink = False
    beamform = range(2)
    check_conditions = False
    Ns = 7
    Nplus = 4
    # generate rx signal with ISI
    for ind in range(len(mse)):
        for bf in beamform:
            snr = 10**(0.1 * snr_db)
            d0 = el_spacing
            n_rx = el_num[ind]
            v = np.copy(u)
            v /= np.sqrt(pwr(v))
            if downlink:
                v_dl = np.copy(v)
            vk, wk, deg_diff = transmit(v,snr,Fs,fs,fc,n_rx,d0,bf) # this already does rough phase alignment
            deg_diff_cum[ind] = deg_diff
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

            M = int(10)

            if bf == 1:
                d_hat_wk, mse_out_wk = dfe_matlab(vk, d, Ns, Nd, M)
                d_hat_cum_wk[ind,:] = d_hat_wk
                mse_wk[ind] = mse_out_wk
            elif bf == 0:
                #vk = 1/n_rx * np.sum(vk[::2,:],axis=0)
                #vk = np.reshape(vk,(1,-1))
                d_hat, mse_out = dfe_matlab(vk, d, Ns, Nd, M)
                d_hat_cum[ind,:] = d_hat
                mse[ind] = mse_out

            if downlink: # ouch
                v_single = transmit_dl(v_dl,wk,snr+5,n_rx,el_spacing,R,fc,fs)
                vps = v_single[:len(up)+Nz*Ns]
                delvals,_ = fdel(vps,up)
                vp1s = vps[delvals:delvals+len(up)]
                fdes,_,_ = fdop(vp1s,up,fs,12)
                v_single = v_single*np.exp(-1j*2*np.pi*np.arange(len(v_single))*fdes*Ts)
                v_single = sg.resample_poly(v_single,np.rint(10**4),np.rint((1/(1+fdes/fc))*(10**4)))
                
                v_single = v_single[delvals:delvals+len(u)]
                v_single = v_single[lenu+Nz*Ns+trunc*Ns+1:] #assuming above just chops off preamble
                v_single = sg.resample_poly(v_single,2,Ns)
                v_single = np.concatenate((v_single,np.zeros(Nplus*2))) # should occur after
                v_single = v_single.reshape(1,-1) 
                d_hat_dl, mse_out_dl = dfe_matlab(v_single, d, Ns, Nd, M)
                mse_dl[ind] = mse_out_dl
                d_hat_dl_cum[ind,:] = d_hat_dl
    """
    for ind in range(len(mse)):
        # plot const
        plt.subplot(2, 2, int(ind+1))
        plt.scatter(np.real(d_hat_cum[ind,:]), np.imag(d_hat_cum[ind,:]), marker='x')
        plt.scatter(np.real(d_hat_cum_wk[ind,:]), np.imag(d_hat_cum_wk[ind,:]), marker='x', color="orange")
        plt.legend(['MRC Only','Beamforming'])
        plt.axis('square')
        plt.axis([-2, 2, -2, 2])
        plt.title(f'SNR={snr_db} dB, M={el_num[ind]}, fc={fc}, d0={d0}') #(f'd0={d_lambda[ind]}'r'$\lambda$') 
    """
    fig, ax = plt.subplots()
    #ax.plot(el_num,mse,'-o')
    ax.plot(el_num,mse_wk,'-o',color="orange")
    ax.set_xlabel(r'Number of Elements')
    ax.set_ylabel('MSE (dB)')
    ax.set_title(f'UL MSE for 2-path channel, SNR={snr_db}dB, d0={d0}, fc={fc}')
    #ax.legend(['MRC Only','Beamforming'])

    if downlink:
        fig1, ax1 = plt.subplots()
        ax1.plot(snr_db,mse_dl,'o')
        ax1.set_xlabel(r'SNR (dB)')
        ax1.set_ylabel('MSE (dB)')
        ax1.set_title('MSE vs SNR for QPSK Signal DL')
        
        for ind in range(len(snr_db)):
            plt.subplot(2, 2, int(ind+1))
            plt.scatter(np.real(d_hat_dl_cum[ind,:]), np.imag(d_hat_dl_cum[ind,:]), marker='x')
            plt.axis('square')
            plt.axis([-2, 2, -2, 2])
            plt.title(f'SNR={snr_db[ind]} dB') #(f'd0={d_lambda[ind]}'r'$\lambda$')
    print(deg_diff_cum)
    plt.show()
    if check_conditions == True:
        for i in range(len(mse)):
            print(el_num[i], array_conditions(fc,B,el_num[i],el_spacing), deg_diff_cum[i])
