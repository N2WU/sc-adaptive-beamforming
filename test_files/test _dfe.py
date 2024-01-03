import numpy as np
import scipy.signal as sg
import matplotlib.pyplot as plt


def process(v,x):
    #k = np.arange(len(v))
    L = 2 # training length
    i = np.arange(L)
    #x = np.tile(d_upsampled,round(len(v)/len(d_upsampled)))
    x_est = np.zeros_like(v)
    h = np.zeros(L)
    h = v[0:L]/x[0:L]
    x_est[0] = x[0] # training length of 1, basically
    for k in range(1,len(v)-1):
        x_est[k] = (v[k]-h[1]*x_est[k-1])/h[0]
        if x_est[k] < 0:
            x_est[k] = 1
        else:
            x_est[k] = -1
    return x_est

if __name__ == "__main__":
    bits = 7
    rep = 16
    R = 3000
    fs = 48000
    ns = fs / R
    fc = 15e3
    n_tx = 12
    reflection = np.linspace(1,0,n_tx+1)
    reflection = reflection[0:n_tx]
    SNR_db = 10
    SNR = np.power(10, SNR_db/10)

    # generate the signal to be transmitted
    d = np.array(sg.max_len_seq(bits)[0]) * 2 - 1.0
    tbits = np.array(sg.max_len_seq(bits)[0])
    u = np.tile(d, rep)
    u = sg.resample_poly(u, ns, 1)
    s = np.real(u * np.exp(2j * np.pi * fc * np.arange(len(u)) / fs))
    s /= np.max(np.abs(s))

    s_pad = np.concatenate((np.zeros((int(fs*0.1),)), s, np.zeros((int(fs*0.1),))))
    s_multi = np.tile(s,(n_tx,1))/n_tx
    # r_singlechannel[delay_j:delay_j+len(self.s_tx[:,1])] += self.s_tx[:,j] * reflection #(eliminated for now)
    # pass through the channel
    delay = 100*(1+np.arange(n_tx))
    r = np.random.randn(s_pad.shape[0]) / SNR
    for i in range(n_tx):
        r[delay[i]:delay[i]+len(s)] += s_multi[i,:] * reflection[i]

    # xcorr
    v = r * np.exp(-2j * np.pi * fc * np.arange(len(r)) / fs)
    xcor = sg.fftconvolve(v,sg.resample_poly(d[::-1].conj(),ns,1),'full')
    d_upsampled = sg.resample_poly(d,ns,1)
    x = np.tile(d_upsampled,round(len(v)/len(d_upsampled)))
    d_hat = process(v,x)
    mse = 10 * np.log10(np.mean(np.abs(x[0:len(d_hat)] - d_hat) ** 2))
    print("MSE is: ", mse)

    # plot
    plt.plot(np.abs(xcor))
    plt.xlim([np.argmax(xcor) - 100, np.argmax(xcor) + 500])
    plt.xlabel('Delay [samples]')
    plt.ylabel('Cross-correlation')
    plt.show()


""" 
        #r = np.copy(s)
        r = s_multi[i,:] + 0.3 * np.roll(s_multi[i,:], 300)
        r += 0.01 * np.random.randn(s.shape[0])
        # add reflection metric
        r_reflect = 0.5*np.copy(s_multi[i,:])
        r_reflect = r_reflect + 0.3 * np.roll(r, 600)
        r_reflect += 0.01 * np.random.randn(s.shape[0])

        r += r_reflect
"""