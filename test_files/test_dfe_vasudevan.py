import numpy as np
import scipy.signal as sg
import matplotlib.pyplot as plt
# https://github.com/vineel49/dfe/blob/master/run_me_bpsk.m

def transmit(x,L,fade_chan,noise_var):
    chan_len = len(fade_chan)
    noise = np.sqrt(noise_var)*np.random.randn(L+chan_len-1)
    channel_output = sg.convolve(fade_chan,x[0:L])+noise
    return channel_output

def generate(x,SNR):
    y = np.zeros_like(x)
    h_len = 2
    h = np.random.rand(h_len)/SNR
    a = 0.5
    y[0]=x[0]
    for k in range(1,len(x)):
        y[k] = (h[0]*x[k] + np.random.randn(1)/SNR) + a*(h[1]*x[k-1] + np.random.randn(1)/SNR)
    return y

def filt_init(channel_output,L,x,K1,K2):
    # initialize filters
    ff_filter = np.zeros(K1+1)
    fb_filter = np.zeros(K2)
    ff_input = np.zeros(K1+1)
    fb_input = np.zeros(K2)
    fb_output = 0
    # estimate autocorrelation
    Rvv0 = (channel_output*channel_output.T)/(L+3-1)
    max_step_size = 2/((K1+1)*Rvv0 + K2)
    step_size = 0.125*max_step_size
    # steady state
    for i in range(L-K2):
        ff_input[1:]=ff_input[0:-1]
        ff_input[0] = channel_output[i]
        ff_output = ff_filter@ff_input.T
        ff_fb = ff_output-fb_output
        error = ff_fb-x[i]

        temp = ff_fb < 0
        quantizer_output = 1-2*temp
        ff_filter = ff_filter-step_size*error*ff_input
        fb_filter = fb_filter-step_size*error*fb_input

        fb_input[1:] = fb_input[0:-1]
        fb_input[0] = quantizer_output
        fb_output = fb_filter@fb_input.T
    return ff_filter, fb_filter

def dfe(y,K1,K2,ff,fb):
    x_est = np.zeros(len(y)-K2+1)
    ff_input = np.zeros(K1+1)
    fb_input = np.zeros(K2)
    fb_output = 0
    for i in range(len(y)-K2):
        ff_input[1:] = ff_input[0:-1]
        ff_input[0] = y[i]
        ff_output = ff@ff_input.T
        ff_fb = ff_output-fb_output
        temp = ff_fb < 0
        x_est[i] = 1-2*temp
        fb_input[1:] = fb[0:-1]
        fb_input[0] = x_est[i]
        fb_output = fb@fb_input.T
    return x_est

if __name__ == "__main__":
    bits = 7
    K1 = 2 # FB
    K2 = 5 # FF

    SNR_db = 10
    SNR = np.power(10, SNR_db/10)
    noise_var = 1/(2*SNR)
    fade_chan =  np.array([0.407, 0.815, 0.407])
    fade_chan = fade_chan/np.linalg.norm(fade_chan)
    L = 5 # training sequence

    # generate the tx BPSK signal
    d = np.array(sg.max_len_seq(bits)[0]) * 2 - 1.0
    x = np.tile(d,10)

    # train input symbol
    channel_output_train = transmit(x,L,fade_chan,noise_var)

    # initialize filters
    ff,fb = filt_init(channel_output_train,L,x,K1,K2)

    # generate rx signal with ISI
    y = transmit(x,len(x),fade_chan,noise_var)

    # process
    x_est = dfe(y,K1,K2,ff,fb)
    x_hat = x_est
    mse = 10 * np.log10(np.mean(np.abs(x[:len(x_hat)] - x_hat) ** 2))
    print("MSE is: ", mse)
