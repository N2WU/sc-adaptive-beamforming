import numpy as np
import scipy.signal as sg
import matplotlib.pyplot as plt

def dfe(v):
    d_hat = np.zeros(len(v) - ff_filter_len + 1)
    return d_hat

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
    ff_filter_len = 20
    fb_filter_len = 8
    training_len = 

    # generate the signal to be transmitted
    d = np.array(sg.max_len_seq(bits)[0]) * 2 - 1.0
    tbits = np.array(sg.max_len_seq(bits)[0])
    u = np.tile(d, rep)
    u = sg.resample_poly(u, ns, 1)
    u_train = u[0:training_len]
    s_train = np.real(u_train * np.exp(2j * np.pi * fc * np.arange(len(u_train)) / fs))
    s_train /= np.max(np.abs(s_train))
    s = np.real(u * np.exp(2j * np.pi * fc * np.arange(len(u)) / fs))
    s /= np.max(np.abs(s))

    s_pad = np.concatenate((np.zeros((int(fs*0.1),)), s, np.zeros((int(fs*0.1),))))
    s_multi = np.tile(s,(n_tx,1))/n_tx
    # r_singlechannel[delay_j:delay_j+len(self.s_tx[:,1])] += self.s_tx[:,j] * reflection #(eliminated for now)
    chan_op = 
    chan_len = 
    # LMS update of taps
    ff_filter = np.zeros(ff_filter_len)
    fb_filter = np.zeros(fb_filter_len)

    ff_filter_ip = np.zeros(ff_filter_len)
    fb_filter_ip = np.zeros(fb_filter_len)

    fb_filter_op = 0

    Rvv0 = np.dot(chan_op, chan_op) / (training_len + chan_len - 1)
    max_step_size = 2 / (ff_filter_len * Rvv0 + fb_filter_len * (1))
    step_size = 0.125 * max_step_size

    for i1 in range(training_len - ff_filter_len + 1):
        ff_filter_ip[1:] = ff_filter_ip[:-1]
        ff_filter_ip[0] = chan_op[i1]
        ff_filter_op = np.dot(ff_filter, ff_filter_ip)

        ff_and_fb = ff_filter_op - fb_filter_op
        error = ff_and_fb - u_train[i1]

        temp = ff_and_fb < 0
        quantizer_op = 1 - 2 * temp

        ff_filter = ff_filter - step_size * error * ff_filter_ip
        fb_filter = fb_filter + step_size * error * fb_filter_ip

        fb_filter_ip[1:] = fb_filter[:-1]
        fb_filter_ip[0] = quantizer_op

        fb_filter_op = np.dot(fb_filter, fb_filter_ip)
    # pass through the channel
    delay = 100*(1+np.arange(n_tx))
    r = np.random.randn(s_pad.shape[0]) / SNR
    r_train = np.random.randn(s_train.shape[0]) / SNR 
    for i in range(n_tx):
        r[delay[i]:delay[i]+len(s)] += s_multi[i,:] * reflection[i]
        r_train[delay[i]:delay[i]+len(s_train)] += s_train[i,:] * reflection[i]
    # xcorr
    v = r * np.exp(-2j * np.pi * fc * np.arange(len(r)) / fs)
    xcor = sg.fftconvolve(v,sg.resample_poly(d[::-1].conj(),ns,1),'full')
 
    peaks_rx, _ = sg.find_peaks(xcor, distance=len(sg.resample_poly(d,ns,1)))
    print(peaks_rx)
    v = v[round(peaks_rx[1] * ns) :] #this one

    d_hat = dfe(v)
    
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