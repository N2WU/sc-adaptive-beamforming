import numpy as np
import scipy.signal as sg
import matplotlib.pyplot as plt

def transmit(x,fade_chan,SNR):
    chan_len = len(fade_chan)
    noise_var = 1 / (2 * SNR)
    noise = np.sqrt(noise_var)*np.random.randn(len(x)+chan_len-1)
    channel_output = np.convolve(fade_chan,x)+noise
    return channel_output

def transmit2(s,SNR,n_tx,reflection):
    delay = 100*(1+np.arange(n_tx))
    r = np.random.randn(s_pad.shape[0]) / SNR
    for i in range(n_tx):
        r[delay[i]:delay[i]+len(s)] += s_multi[i,:] * reflection[i]
    return r

def filt_init(channel_out,x,ff_len,fb_len,chan_len):
    training_len = len(channel_out)
    # initialize filters
    ff_filter = np.zeros(ff_len,dtype=np.complex64)
    fb_filter = np.zeros(fb_len,dtype=np.complex64)
    ff_input = np.zeros(ff_len,dtype=np.complex64)
    fb_input = np.zeros(fb_len,dtype=np.complex64)
    fb_out = np.zeros(1,dtype=np.complex64)
    # estimate autocorrelation
    Rvv0 = np.dot(channel_out, channel_out)/(training_len + chan_len - 1)
    max_step_size = 2 / (ff_len * Rvv0 + fb_len * (1))
    step_size = 0.125*max_step_size
    # steady state
    for i in range(training_len - ff_len + 1):
        ff_input[1:]=ff_input[:-1]
        ff_input[0] = channel_out[i]
        ff_out = np.dot(ff_filter, ff_input)
        ff_fb = ff_out-fb_out
        error = ff_fb-x[i]

        temp = ff_fb < 0
        quantizer_output = 1-2*temp
        ff_filter = ff_filter-step_size*error*ff_input
        fb_filter = fb_filter+step_size*error*fb_input

        fb_input[1:] = fb_filter[:-1]
        fb_input[0] = quantizer_output
        fb_out = np.dot(fb_filter, fb_input)
    return ff_filter, fb_filter

def dfe(v,ff,fb,ff_len,fb_len):
    d_hat = np.zeros(len(v) - ff_len + 1)
    ff_filter_ip = np.zeros(ff_len, dtype=np.complex64)
    fb_filter_ip = np.zeros(fb_len, dtype=np.complex64)
    fb_filter_op = np.zeros(1,dtype=np.complex64)

    for i1 in range(len(v) - ff_len + 1):
        ff_filter_ip[1:] = ff_filter_ip[:-1]
        ff_filter_ip[0] = v[i1]
        ff_filter_op = np.dot(ff, ff_filter_ip)

        ff_and_fb = ff_filter_op - fb_filter_op

        temp = ff_and_fb < 0
        d_hat[i1] = 1 - 2 * temp

        fb_filter_ip[1:] = fb[:-1]
        fb_filter_ip[0] = d_hat[i1]

        fb_filter_op = np.dot(fb, fb_filter_ip)
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
    ff_len = 20
    fb_len = 8
    training_rep = 8

    # generate the signal to be transmitted
    d = np.array(sg.max_len_seq(bits)[0]) * 2 - 1.0
    tbits = np.array(sg.max_len_seq(bits)[0])
    u = np.tile(d, rep)
    u = sg.resample_poly(u, ns, 1)
    u_train = np.tile(d, training_rep)
    u_train = sg.resample_poly(u_train, ns, 1)
    training_len = len(u_train)
    s_train = np.real(u_train * np.exp(2j * np.pi * fc * np.arange(len(u_train)) / fs))
    s_train /= np.max(np.abs(s_train))
    s = np.real(u * np.exp(2j * np.pi * fc * np.arange(len(u)) / fs))
    s /= np.max(np.abs(s))

    s_pad = np.concatenate((np.zeros((int(fs*0.1),)), s, np.zeros((int(fs*0.1),))))
    s_multi = np.tile(s,(n_tx,1))/n_tx
    
    fade_chan = np.array([1, 0.5])
    fade_chan = fade_chan / np.linalg.norm(fade_chan)
    #channel_out = transmit(u_train,fade_chan,SNR)
    r_train = transmit(s_train,fade_chan,SNR)
    v_train = r_train * np.exp(-2j * np.pi * fc * np.arange(len(r_train)) / fs)
    #v_train = v_train[:len(u_train)]
    ff,fb = filt_init(v_train,u_train,ff_len,fb_len,len(fade_chan))
    #ff,fb = filt_init(channel_out,training_len,u_train,ff_len,fb_len,len(fade_chan))
    # pass through the channel
    #r_train = transmit2(s_train,SNR,n_tx,reflection)
    #r = transmit2(s,SNR,n_tx,reflection)
    r = transmit(s,fade_chan,SNR)
    # xcorr
    v = r * np.exp(-2j * np.pi * fc * np.arange(len(r)) / fs)
    xcor = sg.fftconvolve(v,sg.resample_poly(d[::-1].conj(),ns,1),'full')
 
    peaks_rx, _ = sg.find_peaks(xcor, distance=len(sg.resample_poly(d,ns,1)))
    #v = v[round(peaks_rx[1] * ns) :] #this one

    u_hat = dfe(v,ff,fb,ff_len,fb_len)
    #MSE
    print(f"u_train-v_train: {len(u_train)-len(v_train)}")
    mse = 10 * np.log10(np.mean(np.abs(u[:len(u_hat)] - u_hat) ** 2))
    print(f"MSE: {mse}")