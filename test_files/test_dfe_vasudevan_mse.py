import numpy as np
import scipy.signal as sg
import matplotlib.pyplot as plt
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

def rrcosfilter(N, alpha, Ts, Fs):
    T_delta = 1 / float(Fs)
    time_idx = ((np.arange(N) - N / 2)) * T_delta
    sample_num = np.arange(N)
    h_rrc = np.zeros(N, dtype=float)

    for x in sample_num:
        t = (x - N / 2) * T_delta
        if t == 0.0:
            h_rrc[x] = 1.0 - alpha + (4 * alpha / np.pi)
        elif alpha != 0 and t == Ts / (4 * alpha):
            h_rrc[x] = (alpha / np.sqrt(2)) * (
                ((1 + 2 / np.pi) * (np.sin(np.pi / (4 * alpha))))
                + ((1 - 2 / np.pi) * (np.cos(np.pi / (4 * alpha))))
            )
        elif alpha != 0 and t == -Ts / (4 * alpha):
            h_rrc[x] = (alpha / np.sqrt(2)) * (
                ((1 + 2 / np.pi) * (np.sin(np.pi / (4 * alpha))))
                + ((1 - 2 / np.pi) * (np.cos(np.pi / (4 * alpha))))
            )
        else:
            h_rrc[x] = (
                np.sin(np.pi * t * (1 - alpha) / Ts)
                + 4 * alpha * (t / Ts) * np.cos(np.pi * t * (1 + alpha) / Ts)
            ) / (np.pi * t * (1 - (4 * alpha * t / Ts) * (4 * alpha * t / Ts)) / Ts)

    return time_idx, h_rrc

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
snr_dB = np.arange(-10,20)
ff_filter_len = 20
fb_filter_len = 8
R = 3000
fs = 48000
ns = fs / R
fc = 16e3
uf = int(fs / R)
df = int(uf / ns)

d = np.array(sg.max_len_seq(bits)[0]) * 2 - 1.0
# Training phase
training_sym = np.tile(d, training_rep)
training_seq = training_sym
training_seq /= np.max(np.abs(training_seq))

# Data transmission phase
data_sym = np.tile(d, rep)
data_seq = data_sym
data_seq /= np.max(np.abs(data_seq))

training_len = len(training_seq)
data_len = len(data_seq)

fade_chan = [1, 0.5]#[0.407, 0.815, 0.407]
fade_chan = fade_chan / np.linalg.norm(fade_chan)
chan_len = len(fade_chan)

mse = np.zeros(len(snr_dB))

for i in range(len(snr_dB)):
    # SNR parameters
    snr = 10**(0.1 * snr_dB[i])
    chan_op_train = transmit(training_seq,fade_chan,snr)
    # LMS update of taps
    ff_filter,fb_filter = filt_init(chan_op_train,training_sym,ff_filter_len,fb_filter_len,chan_len) 
    chan_op_data = transmit(data_seq,fade_chan,snr)
    dec_sym = dfe(chan_op_data,ff_filter,fb_filter,ff_filter_len,fb_filter_len)
    dec_sym = (dec_sym > 0)*2 - 1
    data_sym = (data_sym > 0)*2 - 1
    try:
        mse[i] = 10 * np.log10(np.mean(np.abs(data_sym[:len(dec_sym)]- dec_sym) ** 2))
    except ZeroDivisionError:
        mse[i] = -99

plt.plot(snr_dB,mse,'o')
plt.xlabel('SNR (dB)')
plt.ylabel('MSE (dB)')
plt.title('SNR vs MSE for BPSK Signal')
plt.show()