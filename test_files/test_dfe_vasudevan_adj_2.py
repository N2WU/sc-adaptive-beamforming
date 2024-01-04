import numpy as np
import scipy.signal as sg
# https://github.com/vineel49/dfe/blob/master/run_me_bpsk.m

# functions
def transmit(x,fade_chan,SNR):
    chan_len = len(fade_chan)
    noise_var = 1 / (2 * SNR)
    noise = np.sqrt(noise_var)*np.random.randn(len(x)+chan_len-1)
    channel_output = np.convolve(fade_chan,x)+noise
    return channel_output

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
    dec_seq = np.zeros(len(v) - ff_len + 1)
    ff_filter_ip = np.zeros(ff_len, dtype=np.complex64)
    fb_filter_ip = np.zeros(fb_len, dtype=np.complex64)
    fb_filter_op = np.zeros(1,dtype=np.complex64)

    for i1 in range(len(v) - ff_len + 1):
        ff_filter_ip[1:] = ff_filter_ip[:-1]
        ff_filter_ip[0] = v[i1]
        ff_filter_op = np.dot(ff, ff_filter_ip)

        ff_and_fb = ff_filter_op - fb_filter_op

        temp = ff_and_fb < 0
        dec_seq[i1] = 1 - 2 * temp

        fb_filter_ip[1:] = fb[:-1]
        fb_filter_ip[0] = dec_seq[i1]

        fb_filter_op = np.dot(fb, fb_filter_ip)
    return dec_seq

# Parameters
bits = 7
rep = 16
training_rep = 4
snr_dB = 10
ff_filter_len = 20
fb_filter_len = 8
R = 3000
fs = 48000
ns = fs / R
fc = 16e3

d = np.array(sg.max_len_seq(bits)[0]) * 2 - 1.0
# Training phase
training_sym = np.tile(d, training_rep)
training_seq = np.real(training_sym * np.exp(2j * np.pi * fc * np.arange(len(training_sym)) / fs))
training_seq /= np.max(np.abs(training_seq))
# training_seq = training_sym
# Data transmission phase
data_sym = np.tile(d, rep)
data_seq = np.real(data_sym * np.exp(2j * np.pi * fc * np.arange(len(data_sym)) / fs))
data_seq /= np.max(np.abs(data_seq))
# data_seq = data_sym
training_len = len(training_seq)
data_len = len(data_seq)

# SNR parameters
snr = 10**(0.1 * snr_dB)

fade_chan = [1, 0.5]#[0.407, 0.815, 0.407]
fade_chan = fade_chan / np.linalg.norm(fade_chan)
chan_len = len(fade_chan)

chan_op_pb = transmit(training_seq,fade_chan,snr)
chan_op = chan_op_pb * np.exp(-2j * np.pi * fc * np.arange(len(chan_op_pb)) / fs)
# chan_op = chan_op_pb

# LMS update of taps

ff_filter,fb_filter = filt_init(chan_op,training_seq,ff_filter_len,fb_filter_len,chan_len)

chan_op_pb = transmit(data_seq,fade_chan,snr)
chan_op = chan_op_pb * np.exp(-2j * np.pi * fc * np.arange(len(chan_op_pb)) / fs)
# chan_op = chan_op_pb

dec_sym = dfe(chan_op,ff_filter,fb_filter,ff_filter_len,fb_filter_len)

# MSE
data_sym_up = sg.resample_poly(data_sym,ns,1)
data_sym_up = (data_sym_up > 0)*2 - 1
mse = 10 * np.log10(np.mean(np.abs(data_sym[:len(dec_sym)] - dec_sym) ** 2))
print(f"MSE: {mse}")

#print(f"data_sym: {data_sym}")
#print(f"len: {len(data_sym)-len(dec_sym)}")
#print(f"dec_sym: {dec_sym}")