import numpy as np
import scipy.signal as sg
import matplotlib.pyplot as plt
#plt.rcParams['text.usetex'] = True


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

def transmit(v,snr,Fs,fs,fc,n_rx,d0,bf,nb):
    reflection_list = np.asarray([1,0.5]) # reflection gains
    n_path = len(reflection_list)  
    x_tx_list = np.array([5,-5]) 
    y_tx_list = np.array([15,15])
    c = 343
    x_rx = d0 + d0 * np.arange(n_rx) #starting on origin
    # x_rx = x_rx # - d0*n_rx/2 #center on origin
    y_rx = np.zeros_like(x_rx)
    vs = sg.resample_poly(v,Fs,fs)
    #vs = sg.resample_poly(v,4,1)
    #vs = np.copy(v)
    s = np.real(vs*np.exp(1j*2*np.pi*fc*np.arange(len(vs))/Fs)) #Fs
    a = 1/c
    r_multi = np.random.randn(int(2*len(s)), n_rx) / snr
    delays = np.zeros((len(reflection_list),n_rx))
    for i in range(len(reflection_list)):
        x_tx, y_tx = x_tx_list[i], y_tx_list[i]
        reflection = reflection_list[i] #delay and sum not scale
        true_angle = np.rad2deg(np.arctan(np.abs(x_tx/y_tx)))
        dx, dy = x_rx - x_tx, y_rx - y_tx
        d_rx_tx = np.sqrt(dx**2 + dy**2)
        delta_tau = d_rx_tx / c
        #delta_tau = el_spacing/c * np.sin(np.deg2rad(true_angle))
        delay = np.round(delta_tau * Fs).astype(int) # sample delay, grrrr
        delay =(i*100 + delay).astype(int)
        delays[i,:] = delay
        for j, delay_j in enumerate(delay):
            r_multi[delay_j:delay_j+len(s), j] += reflection * s
    peaks_rx = 0
    wk = np.ones(n_rx)
    deg_diff = 0
    if bf == 1:
        if nb == 0:
            # now beamforming and weights
            r = r_multi #[:,:M]
            M = int(n_rx)
            r_fft = np.fft.fft(r, axis=0)
            freqs = np.fft.fftfreq(len(r[:, 0]), 1/Fs) # this is the same as 1/Fs with no adjustment
            #freqs = freqs*Fs/fs

            index = np.where((freqs >= fc-R/2) & (freqs < fc+R/2))[0]
            N = len(index) #uhh???
            #W_out = np.ones((n_rx,N),dtype=complex)
            fk = freqs[index]       
            yk = r_fft[index, :]    # N*M
            theta_start = -30
            theta_end = 30
            N_theta = 200
            deg_theta = np.linspace(theta_start,theta_end,N_theta)
            S_theta = np.zeros((N_theta,))
            S_theta2 = np.zeros((N_theta,))
            S_theta3 = np.zeros((N_theta,))
            S_theta3D = np.zeros((N_theta, N))
            theta_start = np.deg2rad(theta_start)
            theta_end = np.deg2rad(theta_end)
            theta_list = np.linspace(theta_start, theta_end, N_theta)
            for n_theta, theta in enumerate(theta_list):
                d_tau = np.sin(theta) * d0/c

                S_M = np.exp(-2j * np.pi * d_tau * np.dot(fk.reshape(N, 1), np.arange(M).reshape(1, M)))    # N*M
                SMxYk = np.einsum('ij,ji->i', S_M.conj(), yk.T, dtype='complex128')
                #SMxYk = np.sum(S_M.conj)
                klen = len(SMxYk)
                for i in range(n_rx):
                    S_M = np.exp(-2j * np.pi * fc * d_tau * np.arange(M).reshape(M,1))
                    yfk = yk[int(klen/2),:].T
                    S_theta2[n_theta] += np.abs(S_M.conj().T @ yfk)**2
                S_theta3[n_theta] = np.abs(S_M.conj().T @ yfk)**2
                S_theta[n_theta] = np.real(np.vdot(SMxYk, SMxYk)) #real part
                S_theta[n_theta] = S_theta[n_theta] # doesn't necessarily make a difference
                S_theta3D[n_theta, :] = np.abs(SMxYk)**2

            # n_path = 1 # number of path
            S_theta_peaks_idx, _ = sg.find_peaks(S_theta, height=0) #S_theta
            S_theta_peaks = S_theta[S_theta_peaks_idx] #S_theta
            theta_m_idx = np.argsort(S_theta_peaks)
            theta_m = theta_list[S_theta_peaks_idx[theta_m_idx[-n_path:]]]
            print(theta_m/np.pi*180)
            ang_est = np.rad2deg(theta_list[np.argmax(S_theta)])
            y_tilde = np.zeros((N,n_path), dtype=complex)
            Sk = np.zeros((n_rx,n_path), dtype=complex)
            for k in range(N):
                for p in range(n_path):
                    d_tau_p = np.sin(theta_m[p])*d0/c
                    Sk[:,p] = np.exp(-2j * np.pi * fk[k] * np.arange(M)*d_tau_p)
                for i in range(n_path):
                    e_pu = np.zeros(n_path)
                    e_pu[i] = 1
                    wk = Sk @ np.linalg.inv(Sk.conj().T @ Sk) @ e_pu
                    #if i==1:
                        #W_out[:,k] = wk.flatten()
                    y_tilde[k, i] = wk.conj().T @ yk[k, :].T

            y_fft = np.zeros((len(r[:, 0]), n_path), complex)
            y_fft[index, :] = y_tilde
            y = np.fft.ifft(y_fft, axis=0)
            r_multi = np.copy(y) 
        if nb==1:
            # now beamforming and weights
            r = r_multi #[:,:M]
            M = int(n_rx)
            r_fft = np.fft.fft(r, axis=0)
            freqs = np.fft.fftfreq(len(r[:, 0]), 1/Fs) # this is the same as 1/Fs with no adjustment
            #freqs = freqs*Fs/fs

            index = np.where((freqs >= fc-R/2) & (freqs < fc+R/2))[0]
            #index = np.ones_like(index)*fc
            index = index.astype(int)
            N = len(index) #uhh???
            #W_out = np.ones((n_rx,N),dtype=complex)
            fk = freqs[index]       
            yk = r_fft[index, :]    # N*M
            theta_start = -30
            theta_end = 30
            N_theta = 200
            deg_theta = np.linspace(theta_start,theta_end,N_theta)
            S_theta = np.zeros((N_theta,))
            S_theta2 = np.zeros((N_theta,))
            S_theta3 = np.zeros((N_theta,))
            S_theta3D = np.zeros((N_theta, N))
            theta_start = np.deg2rad(theta_start)
            theta_end = np.deg2rad(theta_end)
            theta_list = np.linspace(theta_start, theta_end, N_theta)
            for n_theta, theta in enumerate(theta_list):
                d_tau = np.sin(theta) * d0/c
                #SMxYk = np.sum(S_M.conj)
                S_M = np.exp(-2j * np.pi * fc * d_tau * np.arange(M).reshape(M,1))
                S_theta[n_theta] = np.abs(S_M.conj().T @ yk[int(len(yk)/2),:])**2

            # n_path = 1 # number of path
            S_theta_peaks_idx, _ = sg.find_peaks(S_theta, height=0) #S_theta
            S_theta_peaks = S_theta[S_theta_peaks_idx] #S_theta
            theta_m_idx = np.argsort(S_theta_peaks)
            theta_m = theta_list[S_theta_peaks_idx[theta_m_idx[-n_path:]]]
            print(theta_m/np.pi*180)
            ang_est = np.rad2deg(theta_list[np.argmax(S_theta)])
            y_tilde = np.zeros((N,n_path), dtype=complex)
            Sk = np.zeros((n_rx,n_path), dtype=complex)
            for p in range(n_path):
                d_tau_p = np.sin(theta_m[p])*d0/c
                Sk[:,p] = np.exp(-2j * np.pi * fc * np.arange(M)*d_tau_p)
            for i in range(n_path):
                e_pu = np.zeros(n_path)
                e_pu[i] = 1
                wk = Sk @ np.linalg.inv(Sk.conj().T @ Sk) @ e_pu
                y_tilde[:, i] = wk.conj().T @ yk.T

            y_fft = np.zeros((len(r[:, 0]), n_path), complex)
            y_fft[index, :] = y_tilde #work on this later
            y = np.fft.ifft(y_fft, axis=0)
            r_multi = np.copy(y) 
    elif bf ==0:
        ang_est = 0
    xvals = np.zeros((len(r_multi[0,:]),1024), dtype=complex)
    #for i in range(len(r_multi[0,:])):
    #    plt.plot(np.real(r_multi[:,i]))
    #plt.show()
    xval = 0
    for i in range(len(r_multi[0,:])):
        r = np.squeeze(r_multi[:, i])
        vr = r * np.exp(-1j*2*np.pi*fc*np.arange(len(r))/Fs) #Fs
        v = sg.resample_poly(vr,1,Fs/fs)
        #v = sg.resample_poly(vr,1,4)
        vp = v[:len(up)+Nz*Ns]
        if i==0:
            delval,xval = fdel(vp,up)
        #if bf==0:
        delval, xval = fdel(vp,up)
        xvals[i] = np.abs(xval) #np.abs(xval)/np.amax(np.abs(xval))
        v = v[delval:delval+len(u)]
        v = v[lenu+Nz*Ns+trunc*Ns+1:] #assuming above just chops off preamble
        v = sg.resample_poly(v,2,Ns)
        v = np.concatenate((v,np.zeros(Nplus*2)))
        if i == 0:
            v_multichannel = np.copy(v)
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
    for i in range(len(r_multi[0,:])):
        plt.plot(xvals[i,:])
    plt.title("Beamformed Cross-Correlation, M=4")
    plt.legend(["p=0","p=1"])
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.show()
    """
    return vk, ang_est, deg_diff

def transmit_dl(v_dl,ang_est,snr,n_rx,el_spacing,R,fc,fs,bfdl,nb):
    vs = sg.resample_poly(v_dl,Fs,fs)
    s_d = np.real(vs*np.exp(1j*2*np.pi*fc*np.arange(len(vs))/Fs))
    delays = el_spacing/343 * np.sin(np.deg2rad(ang_est))*np.arange(n_rx)
    s_tx = np.zeros((n_rx,int(np.amax(delays) + len(s_d))))
    s_fft = np.fft.fft(s_d)
    S_tilde = np.zeros((len(s_fft),n_rx),dtype=complex)
    # apply wk here
    if bfdl == 1:
        ntheta = 2
        theta_span = np.deg2rad(np.linspace(-45,45,ntheta))
        theta_span = np.asarray([0, -17]) #20
        delay_span = el_spacing/343 * np.sin(np.deg2rad(theta_span))
        # form Pk
        if nb==0:
            for k in range(len(s_fft)):
                #wk = np.exp(-2j*np.pi*k*np.reshape(delay_span,(-1,1))@np.reshape(np.arange(n_rx),(1,-1)))
                #wk_tilde
                Wk = np.zeros((n_rx,ntheta), dtype=complex)
                fk = s_fft[k]
                for theta in range(ntheta):
                    #Sk[:,p] = np.exp(-2j * np.pi * fk[k] * np.arange(M)*d_tau_p)
                    #wk = Sk @ np.linalg.inv(Sk.conj().T @ Sk) @ e_pu
                    Wk[:,theta] = 1/np.sqrt(n_rx) * np.exp(-2j*np.pi*k*np.arange(n_rx)*delay_span[theta])

                #Wk = 1/np.sqrt(n_rx) * np.exp(-2j*np.pi*k*np.reshape(delay_span,(-1,1)) * np.reshape(np.arange(n_rx),(1,-1)))

                wk = 1/np.sqrt(n_rx) * np.exp(-2j*np.pi*k*delays)
                try:
                    Pk = np.eye(n_rx) - Wk @ np.linalg.inv(Wk.conj().T @ Wk) @ Wk.conj().T
                #Pk = np.eye(n_rx) - Wk / (Wk.conj().T @ Wk) @ Wk.conj().T
                except:
                    Pk = np.eye(n_rx) - Wk @ Wk.conj().T #/(Wk.conj().T @ Wk)
            #        #print("failed")
                #Pk = np.eye(n_rx) - wk / (wk.conj().T @ wk) @ wk.conj().T
                wk_tilde = Pk@wk
                wk_tilde /= np.linalg.norm(wk_tilde)
                S_tilde[k,:] = wk_tilde * s_fft[k]
            for i in range(n_rx):
                s_tx[i,:] = np.real(np.fft.ifft(S_tilde[:,i]))   
                s_tx[i,:] /= np.sqrt(pwr(s_tx[i,:]))
        if nb==1:
            Wk = np.zeros((n_rx,ntheta), dtype=complex)
            for theta in range(ntheta):
                Wk[:,theta] = 1/np.sqrt(n_rx) * np.exp(-2j*np.pi*fc*np.arange(n_rx)*delay_span[theta])
            wk = 1/np.sqrt(n_rx) * np.exp(-2j*np.pi*fc*delays)
            try:
                Pk = np.eye(n_rx) - Wk @ np.linalg.inv(Wk.conj().T @ Wk) @ Wk.conj().T
            except:
                Pk = np.eye(n_rx) - Wk @ Wk.conj().T #/(Wk.conj().T @ Wk)
            wk_tilde = Pk@wk
            wk_tilde /= np.linalg.norm(wk_tilde)
            S_tilde = np.reshape(wk_tilde,(-1,1)) @ np.reshape(s_fft,(1,-1))
            for i in range(n_rx):
                s_tx[i,:] = np.real(np.fft.ifft(S_tilde[i,:]))   
                s_tx[i,:] /= np.sqrt(pwr(s_tx[i,:]))      
    elif bfdl == 0:
        for i in range(n_rx):
            s_tx[i,:len(s_d)] = s_d/np.linalg.norm(s_d)
            s_tx[i,:len(s_d)] = s_d # equal power but in one element
            s_tx[i,:] /= np.sqrt(pwr(s_tx[0,:]))     
        #print(pwr(s_tx[0,:]))
    #print(np.linalg.norm(s_tx))
    x_rx = np.array([5])
    y_rx = np.array([15])
    c = 343
    x_d = d0 + d0 * np.arange(n_rx)
    x_refl = d0*np.arange(-n_rx,0)
    x_tx_list = np.append(x_refl,x_d)
    #x_tx = x_tx - d0*n_rx/2 #center on origin
    y_tx_list = np.zeros_like(x_tx_list)
    r_single = np.random.randn(int(2*len(s_tx[0,:]))).astype('complex') / snr
    array_h = np.ones(n_rx)
    reflection_list = np.append(0.1*array_h, array_h)
    delay_list = np.append(np.zeros(int(n_rx)),np.ones(int(n_rx)))
    # make s_tx a reflection
    s_tx_ref = np.zeros((2*n_rx,len(s_tx[0,:])), dtype=complex)
    s_tx_ref[n_rx:,:] = s_tx
    s_tx_ref[:n_rx,:] = np.flipud(s_tx)
    for i in range(len(reflection_list)):
        x_tx, y_tx = x_tx_list[i], y_tx_list[i]
        reflection = reflection_list[i] #delay and sum not scale
        dx, dy = x_rx - x_tx, y_rx - y_tx
        d_rx_tx = np.sqrt(dx**2 + dy**2)
        delta_tau = d_rx_tx / c
        delay = np.round(delta_tau * Fs).astype(int) # sample delay
        delay = delay[0]
        #delay =(delay_list[i]*100 + delay).astype(int)
        r_single[delay:delay+len(s_tx[0,:])] += reflection * s_tx_ref[i,:]
    r = np.squeeze(r_single)
    vr = r * np.exp(-1j*2*np.pi*fc*np.arange(len(r))/Fs)
    v_single = sg.resample_poly(vr,1,Fs/fs)
    vps = v_single[:len(up)+Nz*Ns]
    delvals,xvals = fdel(vps,up)
    
    v_single = v_single[delvals:delvals+len(u)]
    v_single = v_single[lenu+Nz*Ns+trunc*Ns+1:] #assuming above just chops off preamble
    v_single = sg.resample_poly(v_single,2,Ns)
    v_single = np.concatenate((v_single,np.zeros(Nplus*2))) # should occur after
    v_single = v_single.reshape(1,-1) 
    return v_single

def dec4psk(x):
    xr = np.real(x)
    xi = np.imag(x)
    dr = np.sign(xr)
    di = np.sign(xi)
    d = dr+1j*di
    d = d/np.sqrt(2)
    return d

def dfe_matlab(vk, d_rls, N, Nd, M): 
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
    d = np.copy(d_rls)

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
        # d_backward
        d_hat[n] = psum - q

        if n > Nt:
            d[n] = dec4psk(d_hat[n])

        #if n>Nt:
            #d_tilde[n] = dec4psk(d_hat[n])
        #else:
            #d_tilde[n] = d[n]
        
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
        #b = -c[int(K*N):int(K*N+M)]
        b = -c[-M:]
        #d_tilde = np.append(d[n], d_tilde)
        #d_tilde = d_tilde[:M]

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
    Fs = 48000
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

    n_rx = 4
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

    snr_db = np.array([5, 10, 15, 20]) #, 15])
    mse = np.zeros_like(snr_db)
    mse_dl_bf = np.zeros_like(snr_db)
    M_nobf = int(5)
    M_bf = int(3)
    N_nobf = int(20)
    N_bf = int(12)

    d_hat_cum_nobf = np.zeros((len(snr_db),Nd-(4*(N_nobf+M_nobf))-1), dtype=complex)
    d_hat_ul_cum_bf = np.zeros((len(snr_db),Nd-(4*(N_nobf+M_nobf))-1), dtype=complex)
    d_hat_ul_cum_nobf = np.zeros((len(snr_db),Nd-(4*(N_nobf+M_nobf))-1), dtype=complex)
    mse_ul_nobf = np.zeros_like(snr_db)
    d_hat_ul_cum_bf_nb = np.zeros((len(snr_db),Nd-(4*(N_nobf+M_nobf))-1), dtype=complex)
    d_hat_ul_cum_bf_taps = np.zeros((len(snr_db),Nd-(4*(N_bf+M_bf))-1), dtype=complex)
    d_hat_ul_cum_bf_taps_nb = np.zeros((len(snr_db),Nd-(4*(N_bf+M_bf))-1), dtype=complex)
    mse_ul_bf_taps = np.zeros_like(snr_db)
    mse_ul_bf = np.zeros_like(snr_db)
    mse_ul_bf_taps_nb = np.zeros_like(snr_db)
    mse_ul_bf_nb = np.zeros_like(snr_db)

    mse_wk = np.zeros_like(snr_db)
    d_hat_cum_bf = np.zeros((len(snr_db),Nd-(4*(N_nobf+M_nobf))-1), dtype=complex)
    mse_wk_taps = np.zeros_like(snr_db)
    d_hat_cum_bf_taps = np.zeros((len(snr_db),Nd-(4*(N_bf+M_bf))-1), dtype=complex)
    deg_diff_cum = np.zeros_like(mse,dtype=float)

    d_hat_dl_cum_bf = np.zeros((len(snr_db),Nd-(4*(N_nobf+M_nobf))-1), dtype=complex)
    d_hat_dl_cum_nobf = np.zeros((len(snr_db),Nd-(4*(N_nobf+M_nobf))-1), dtype=complex)
    mse_dl_nobf = np.zeros_like(snr_db)
    d_hat_dl_cum_bf_taps = np.zeros((len(snr_db),Nd-(4*(N_bf+M_bf))-1), dtype=complex)
    d_hat_dl_cum_bf_nb = np.zeros((len(snr_db),Nd-(4*(N_nobf+M_nobf))-1), dtype=complex)
    d_hat_dl_cum_bf_taps_nb = np.zeros((len(snr_db),Nd-(4*(N_bf+M_bf))-1), dtype=complex)
    mse_dl_bf_taps = np.zeros_like(snr_db)
    mse_dl_bf = np.zeros_like(snr_db)
    mse_dl_bf_taps_nb = np.zeros_like(snr_db)
    mse_dl_bf_nb = np.zeros_like(snr_db)

    load = False
    downlink = False
    beamform = range(2)
    K0 = n_rx
    Ns = 7
    Nplus = 4
    # generate rx signal with ISI
    #for ind in range(len(snr_db)):
    snr = 10**(0.1 * snr_db[0])
    d0 = el_spacing
    v = np.copy(u) #np.zeros(len(u), dtype=complex)
    v /= np.sqrt(pwr(v))
    v_dl = np.copy(v)
    #for bf in beamform:    
    #    if bf == 1:
    #vk_bf1, ang_est, deg_diff = transmit(v,snr,Fs,fs,fc,n_rx,d0,1) # this already does rough phase alignment
                #vk_bf = np.copy(vk_bf1)
                #d_hat_wk, mse_out_wk = dfe_matlab(vk_bf, d, N_nobf, Nd, M_nobf)
                #d_hat_cum_bf[ind,:] = d_hat_wk
                #mse_wk[ind] = mse_out_wk
                #vk_bf_taps = np.copy(vk_bf)
                #d_hat_wk_taps, mse_out_taps = dfe_matlab(vk_bf_taps, d, N_bf, Nd, M_bf)
                #d_hat_cum_bf_taps[ind,:] = d_hat_wk_taps
                #mse_wk_taps[ind] = mse_out_taps
    """
            elif bf == 0:
                #vk = 1/n_rx * np.sum(vk[::2,:],axis=0)
                #vk = np.reshape(vk,(1,-1))
                vk_nobf1, ang_est, deg_diff = transmit(v,snr,Fs,fs,fc,n_rx,d0,bf) # this already does rough phase alignment
                vk_nobf = np.copy(vk_nobf1)
                d_hat, mse_out = dfe_matlab(vk_nobf, d, N_nobf, Nd, M_nobf)
                d_hat_cum_nobf[ind,:] = d_hat
                mse[ind] = mse_out
    """
        #if downlink: # ouch
    
    for ind in range(len(snr_db)):
        snr = 10**(0.1 * snr_db[ind])      
        vk_nobf, ang_est, deg_diff = transmit(v,snr,Fs,fs,fc,n_rx,d0,0,0)
        d_hat_ul_nobf, mse = dfe_matlab(vk_nobf, d, N_nobf, Nd, M_nobf)
        d_hat_ul_cum_nobf[ind,:] = d_hat_ul_nobf
        mse_ul_nobf[ind] = mse
    for ind in range(len(snr_db)):
        snr = 10**(0.1 * snr_db[ind])    
        vk_bf, ang_est, deg_diff = transmit(v,snr,Fs,fs,fc,n_rx,d0,1,0)
        d_hat_ul_bf, mse = dfe_matlab(vk_bf, d, N_nobf, Nd, M_nobf)
        d_hat_ul_cum_bf[ind,:] = d_hat_ul_bf
        mse_ul_bf[ind] = mse
    for ind in range(len(snr_db)):
        snr = 10**(0.1 * snr_db[ind])    
        vk_bf_taps, ang_est, deg_diff = transmit(v,snr,Fs,fs,fc,n_rx,d0,1,0)
        d_hat_ul_bf_taps, mse = dfe_matlab(vk_bf_taps, d, N_bf, Nd, M_bf)
        d_hat_ul_cum_bf_taps[ind,:] = d_hat_ul_bf_taps
        mse_ul_bf_taps[ind] = mse
    for ind in range(len(snr_db)): #narrowband
        snr = 10**(0.1 * snr_db[ind])    
        vk_bf, ang_est, deg_diff = transmit(v,snr,Fs,fs,fc,n_rx,d0,1,1)
        d_hat_ul_bf, mse = dfe_matlab(vk_bf, d, N_nobf, Nd, M_nobf)
        d_hat_ul_cum_bf_nb[ind,:] = d_hat_ul_bf
        mse_ul_bf_nb[ind] = mse
    for ind in range(len(snr_db)):
        snr = 10**(0.1 * snr_db[ind])    
        vk_bf_taps, ang_est, deg_diff = transmit(v,snr,Fs,fs,fc,n_rx,d0,1,1)
        d_hat_ul_bf_taps, mse = dfe_matlab(vk_bf_taps, d, N_bf, Nd, M_bf)
        d_hat_ul_cum_bf_taps_nb[ind,:] = d_hat_ul_bf_taps
        mse_ul_bf_taps_nb[ind] = mse
    
    if downlink:
        for ind in range(len(snr_db)):
            snr = 10**(0.1 * snr_db[ind])      
            vk_nobf = transmit_dl(v_dl,ang_est,snr,n_rx,el_spacing,R,fc,fs,0,0)
            d_hat_dl_nobf, mse = dfe_matlab(vk_nobf, d, N_nobf, Nd, M_nobf)
            d_hat_dl_cum_nobf[ind,:] = d_hat_dl_nobf
            mse_dl_nobf[ind] = mse
        for ind in range(len(snr_db)):
            snr = 10**(0.1 * snr_db[ind])    
            vk_bf = transmit_dl(v_dl,ang_est,snr,n_rx,el_spacing,R,fc,fs,1,0)
            d_hat_dl_bf, mse = dfe_matlab(vk_bf, d, N_nobf, Nd, M_nobf)
            d_hat_dl_cum_bf[ind,:] = d_hat_dl_bf
            mse_dl_bf[ind] = mse
        for ind in range(len(snr_db)):
            snr = 10**(0.1 * snr_db[ind])    
            vk_bf_taps = transmit_dl(v_dl,ang_est,snr,n_rx,el_spacing,R,fc,fs,1,0)
            d_hat_dl_bf_taps, mse = dfe_matlab(vk_bf_taps, d, N_bf, Nd, M_bf)
            d_hat_dl_cum_bf_taps[ind,:] = d_hat_dl_bf_taps
            mse_dl_bf_taps[ind] = mse
        for ind in range(len(snr_db)): #start narrowband
            snr = 10**(0.1 * snr_db[ind])    
            vk_bf = transmit_dl(v_dl,ang_est,snr,n_rx,el_spacing,R,fc,fs,1,1)
            d_hat_dl_bf, mse = dfe_matlab(vk_bf, d, N_nobf, Nd, M_nobf)
            d_hat_dl_cum_bf_nb[ind,:] = d_hat_dl_bf
            mse_dl_bf_nb[ind] = mse
        for ind in range(len(snr_db)):
            snr = 10**(0.1 * snr_db[ind])    
            vk_bf_taps = transmit_dl(v_dl,ang_est,snr,n_rx,el_spacing,R,fc,fs,1,1)
            d_hat_dl_bf_taps, mse = dfe_matlab(vk_bf_taps, d, N_bf, Nd, M_bf)
            d_hat_dl_cum_bf_taps_nb[ind,:] = d_hat_dl_bf_taps
            mse_dl_bf_taps_nb[ind] = mse
    """
    for ind in range(len(mse)):
        # plot const
        plt.subplot(2, 2, int(ind+1))
        plt.scatter(np.real(d_hat_cum[ind,:]), np.imag(d_hat_cum[ind,:]), marker='x')
        plt.scatter(np.real(d_hat_cum_wk[ind,:]), np.imag(d_hat_cum_wk[ind,:]), marker='x', color="orange")
        plt.legend(['MRC Only','Beamforming'])
        plt.axis('square')
        plt.axis([-2, 2, -2, 2])
        plt.title(f'SNR={snr_db[ind]} dB, M={n_rx}, fc={fc}, d0={d0}') #(f'd0={d_lambda[ind]}'r'$\lambda$') 
    """
        #fig1, ax1 = plt.subplots()
    
    plt.figure()
    plt.plot(snr_db,mse_ul_nobf,'-o',color='blue')
    plt.plot(snr_db,mse_ul_bf_taps,'-o',color='green' )
    plt.plot(snr_db,mse_ul_bf,'-o', color='orange')
    plt.plot(snr_db,mse_ul_bf_taps_nb,'--o',color='green' )
    plt.plot(snr_db,mse_ul_bf_nb,'--o', color='orange')
    plt.xlabel(r'SNR (dB)')
    plt.ylabel('MSE (dB)')
    plt.title(r'MSE vs SNR Uplink, Varied $N_{FF}$ and $N_{FB}$ Taps')
    plt.legend([r'No BF, $N_{{FF}}=${}, $N_{{FB}}=${}'.format(N_nobf, M_nobf),
                r'BF, $N_{{FF}}=${}, $N_{{FB}}=${}'.format(N_bf, M_bf),
                r'BF, $N_{{FF}}=${}, $N_{{FB}}=${}'.format(N_nobf, M_nobf),
                r'Narrowband BF, $N_{{FF}}=${}, $N_{{FB}}=${}'.format(N_bf, M_bf),
                r'Narrowband BF, $N_{{FF}}=${}, $N_{{FB}}=${}'.format(N_nobf, M_nobf)]) 
    #plt.figure()
    """
    for ind in range(len(snr_db)):
        plt.subplot(2, 2, int(ind+1))
        plt.scatter(np.real(d_hat_ul_cum_nobf[ind,:]), np.imag(d_hat_ul_cum_nobf[ind,:]), marker='x', alpha=0.5, color='blue')
        plt.scatter(np.real(d_hat_ul_cum_bf_taps[ind,:]), np.imag(d_hat_ul_cum_bf_taps[ind,:]), marker='x', alpha=0.5, color='green')
        plt.axis('square')
        plt.axis([-2, 2, -2, 2])
        plt.xticks([],[])
        plt.yticks([],[])
        if ind == 2 or 3:
            plt.xlabel("In-Phase")
        elif ind== 0 or 1:
            plt.xlabel("")
        plt.ylabel("Quadrature")
        plt.legend([r'No BF, $N_{{FF}}=${}, $N_{{FB}}=${}'.format(N_nobf, M_nobf)
                    ,r'BF, $N_{{FF}}=${}, $N_{{FB}}=${}'.format(N_bf, M_bf)])
        plt.title(r'Uplink, SNR={} dB'.format(snr_db[ind]))
    """
    #plt.show()
    
    
    if downlink:
        #fig1, ax1 = plt.subplots()
        plt.figure()
        plt.plot(snr_db,mse_dl_nobf,'-o',color='blue')
        plt.plot(snr_db,mse_dl_bf_taps,'-o',color='green' )
        plt.plot(snr_db,mse_dl_bf,'-o', color='orange')
        plt.xlabel(r'SNR (dB)')
        plt.ylabel('MSE (dB)')
        plt.title(r'MSE vs SNR Downlink, Varied $N_{FF}$ and $N_{FB}$ Taps')
        plt.legend([r'No BF, $N_{{FF}}=${}, $N_{{FB}}=${}'.format(N_nobf, M_nobf),
                    r'BF, $N_{{FF}}=${}, $N_{{FB}}=${}'.format(N_bf, M_bf),
                    r'BF, $N_{{FF}}=${}, $N_{{FB}}=${}'.format(N_nobf, M_nobf)]) 
        #plt.figure()
        """
        for ind in range(len(snr_db)):
            plt.subplot(2, 2, int(ind+1))
            plt.scatter(np.real(d_hat_dl_cum_nobf[ind,:]), np.imag(d_hat_dl_cum_nobf[ind,:]), marker='x', alpha=0.5, color='blue')
            plt.scatter(np.real(d_hat_dl_cum_bf_taps[ind,:]), np.imag(d_hat_dl_cum_bf_taps[ind,:]), marker='x', alpha=0.5, color='green')
            plt.axis('square')
            plt.axis([-2, 2, -2, 2])
            plt.xticks([],[])
            plt.yticks([],[])
            if ind == 2 or 3:
                plt.xlabel("In-Phase")
            elif ind== 0 or 1:
                plt.xlabel("")
            plt.ylabel("Quadrature")
            plt.legend([r'No BF, $N_{{FF}}=${}, $N_{{FB}}=${}'.format(N_nobf, M_nobf)
                        ,r'BF, $N_{{FF}}=${}, $N_{{FB}}=${}'.format(N_bf, M_bf)])
            plt.title(r'Downlink, SNR={} dB'.format(snr_db[ind]))
        """
    plt.show()
    

