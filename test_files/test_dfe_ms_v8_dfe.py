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
    x_rx = x_rx # - d0*n_rx/2 #center on origin
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
        plt.axvline(x=theta_m[1]/np.pi*180, linestyle="--", color="blue")
        plt.axvline(x=-true_angle, linestyle="--", color="red")
        plt.axvline(x=theta_m[0]/np.pi*180, linestyle="--", color="blue")
        plt.text(theta_m[1]/np.pi*180, np.max(S_theta), f'Est Angle={"{:.2f}".format(deg_ax[est_deg])}')
        plt.text(true_angle,  0.5e23, f'True Angle={"{:.2f}".format(true_angle)}')
        plt.text(theta_m[0]/np.pi*180, np.max(S_theta), f'Est Angle={"{:.2f}".format(deg_ax[est_deg])}')
        plt.text(-true_angle, 0.5e23, f'True Angle={"{:.2f}".format(-true_angle)}')
        plt.title(f'S(Theta) for {10*np.log10(snr)} dB, M={n_rx}, B = 3.9 kHz, d0 ={d0}')
        plt.xlabel("Angle of Arrival (Degrees)")
        plt.xlim(-30,30)
        plt.ylabel("Power Magnitude Squared")
        plt.show()
        """
        
        r_multi = np.copy(y)  
    delvals = np.zeros((n_rx,1024))
    for i in range(len(r_multi[0,:])):
        r = np.squeeze(r_multi[:, i])
        vr = r * np.exp(-1j*2*np.pi*fc*np.arange(len(r))/Fs) #Fs
        v = sg.resample_poly(vr,1,Fs/fs)
        #v = sg.resample_poly(vr,1,4)
        vp = v[:len(up)+Nz*Ns]
        delval,xvals = fdel(vp,up)
        #delvals[i,:] = xvals
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
    """
    for i in range(2,n_rx,2):
        plt.plot(np.abs(delvals[i,:]))
    plt.xlabel("Samples")
    plt.ylabel("Cross-Correlation Magnitude")
    plt.legend(["m=2","m=4","m=6"])
    plt.xlim(600,700)
    plt.show()
    """
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
    Kf1 = 0.000
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

def lineq(vk, d, Ns, Nd): 
    M = 0
    K = len(vk[:,0]) # maximum
    Ns = 2
    N = int(6 * Ns)
    delta = 0 #10**(-3)
    Nt = 4*(N+M)
    FS = 2
    Kf1 = 0.000
    L = 0.99
    P = np.eye(int(K*N+M))/delta
    Lbf = 0.99
    Nplus = 6

    v = np.copy(vk)

    f = np.zeros((Nd,K),dtype=complex)
    
    a = np.zeros(int(K*N), dtype=complex)
    #b = np.zeros(M, dtype=complex)
    #c = np.append(a, -b)
    p = np.zeros(int(K),dtype=complex)
    #d_tilde = np.zeros(M, dtype=complex)
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

        x = x[:,:N] # in matlab, this appends zeros

        for k in range(K):
            p[k] = np.inner(x[k,:], np.conj(a[(k*N):(k+1)*N]))
        psum = p.sum()
        #print(psum)

        d_hat[n] = psum

        if n > Nt:
            d[n] = dec4psk(d_hat[n])

        e = d[n]- d_hat[n]
        et[n] = np.abs(e ** 2)


        y = np.reshape(x,int(K*N))

        # RLS
        k = (
            np.matmul(P, y) / L
            / (1 + np.matmul(np.matmul(y.conj(), P), y) / L)
        )
        a += k * np.conj(e)
        P = P / L - np.matmul(np.outer(k, y.conj()), P) / L

        a = a[:int(K*N)]

    mse = 10 * np.log10(
        np.mean(np.abs(d[Nt : -1 ] - d_hat[Nt : -1]) ** 2)
    )
    if np.isnan(mse):
        mse = 100
    return d_hat[Nt : -1], mse, #n_err, n_training

def lms(v,d):
    Nd = 3000
    ns = 2
    Nplus = 6
    FS = 2
    n_training = 88
    d_hat = np.zeros(len(d), dtype=complex)
    mu = 0.1
    #v = v.flatten()
    vn = np.copy(v)
    for i in np.arange(len(d) - 1, dtype=int):
        # resample v with fractional spacing
        nb = (i) * ns + (Nplus - 1) * ns - 1 # basically a sample index
        v_nt = vn[ int(nb + np.ceil(ns / FS / 2)) : int(nb + ns) 
            + 1 : int(ns / FS)]
        if len(v_nt) == 0:
            v_nt = np.zeros_like(a)
    # select a' (?)
        if i==0:
            a = np.ones_like(v_nt)
    # calculate d_hat
        d_hat[i] = np.inner(a,v_nt) #- d[i]
    # calculate e (use d=dec(d_hat))
        if i > n_training:
            d[i] = dec4psk(d_hat[i])

        err = d[i] - d_hat[i]
    # select a(n+1) with stochastic gradient descent 
        a += mu*v_nt*err.conj()
    return d_hat[n_training:-50]

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

    n_rx = 8
    d_lambda = 0.5 #np.array([0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2]) #
    el_spacing = d_lambda*343/fc
    el_spacing = 0.05
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

    snr_db = np.array([15, 15]) #, 15])
    mse = np.zeros_like(snr_db)
    mse_dl = np.zeros_like(snr_db)
    d_hat_cum = np.zeros((len(snr_db),Nd-88-1), dtype=complex) # has to change if Nt changes :(
    d_hat_dl_cum = np.zeros_like(d_hat_cum,dtype=complex)

    mse_wk = np.zeros_like(snr_db)
    d_hat_cum_wk = np.zeros((len(snr_db),Nd-88-1), dtype=complex) # has to change if Nt changes :(
    deg_diff_cum = np.zeros_like(mse,dtype=float)

    load = False
    downlink = False
    #beamform = range(2)
    check_conditions = True
    if check_conditions == True:
        print(array_conditions(fc,B,n_rx,el_spacing))
    K0 = n_rx
    Ns = 7
    Nplus = 4
    # generate rx signal with ISI
    for ind in range(len(snr_db)):
    #for bf in beamform:    
        #for repeat in range(4):
        snr = 10**(0.1 * snr_db[ind])
        d0 = el_spacing
        #for i in range(K0):
        v = np.copy(u) #np.zeros(len(u), dtype=complex)
        v /= np.sqrt(pwr(v))
        vk, wk, deg_diff = transmit(v,snr,Fs,fs,fc,n_rx,d0,True) # this already does rough phase alignment
        deg_diff_cum[ind] = deg_diff
        if ind==1:
            d_hat1, mse_out = dfe_matlab(vk, d, Ns, Nd, int(10)) #lineq(vk, d, Ns, Nd)
            d_hat1 = lms(vk[0,:],d)
            d_hat1 = d_hat1 * 4 * np.exp(1j*np.pi/4)
        if ind==0:
            M = int(10)
            d_hat2, mse_out = dfe_matlab(vk, d, Ns, Nd, M)

    text_blk = ["Lin. Eq FF=12","DFE, FF=12 FB=10"]
    # plot const
    plt.subplot(1, 2, 1)
    plt.scatter(np.real(d_hat1[-500:]), np.imag(d_hat1[-500:]), marker='x')
    plt.axis('square')
    plt.axis([-2, 2, -2, 2])
    plt.title("Lin. Eq FF=12") #(f'd0={d_lambda[ind]}'r'$\lambda$') 
    plt.subplot(1, 2, 2)
    plt.scatter(np.real(d_hat2[-50:]), np.imag(d_hat2[-50:]), marker='x')
    plt.axis('square')
    plt.axis([-2, 2, -2, 2])
    plt.title("DFE, FF=12 FB=10") #(f'd0={d_lambda[ind]}'r'$\lambda$') 

    plt.show()
