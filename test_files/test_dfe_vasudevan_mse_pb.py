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
snr_dB = np.arange(-10,20,2)
ff_filter_len = 20
fb_filter_len = 8
R = 3000
fs = 48000
ns = fs / R
fc = 16e3
uf = int(fs / R)
df = int(uf / ns)
use_rrc = np.array([False,True])
mse = np.zeros((len(use_rrc),len(snr_dB)),dtype='float')

for k in range(len(use_rrc)):
    d = np.array(sg.max_len_seq(bits)[0]) * 2 - 1.0
    _, rc_tx = rrcosfilter(16 * int(1 / R * fs), 0.5, 1 / R, fs)
    _, rc_rx = rrcosfilter(16 * int(fs / R), 0.5, 1 / R, fs)
    # Training phase
    training_sym = np.tile(d, training_rep)
    # upsample
    training_sym_up = sg.resample_poly(training_sym,ns,1)
    if use_rrc[k] == True:
        training_sym_up = np.convolve(training_sym_up, rc_tx, "full")
        # training_sym_up = training_sym_up[:int(len(training_sym)*ns)]
    training_seq = np.real(training_sym_up * np.exp(2j * np.pi * fc * np.arange(len(training_sym_up)) / fs))
    training_seq /= np.max(np.abs(training_seq))

    # Data transmission phase
    data_sym = np.tile(d, rep)
    # upsample
    data_sym_up = sg.resample_poly(data_sym,ns,1)
    if use_rrc[k] == True:
        data_sym_up = np.convolve(data_sym_up, rc_tx, "full")
        # data_sym_up = data_sym_up[:int(len(data_sym)*ns)]
    data_seq = np.real(data_sym_up * np.exp(2j * np.pi * fc * np.arange(len(data_sym_up)) / fs))
    data_seq /= np.max(np.abs(data_seq))
    training_len = len(training_seq)
    data_len = len(data_seq)

    fade_chan = [1, 0.5]#[0.407, 0.815, 0.407]
    fade_chan = fade_chan / np.linalg.norm(fade_chan)
    chan_len = len(fade_chan)

    for i in range(len(snr_dB)):
        # SNR parameters
        snr = 10**(0.1 * snr_dB[i])
        chan_op_train_pb = transmit(training_seq,fade_chan,snr)
        chan_op_train = chan_op_train_pb * np.exp(-2j * np.pi * fc * np.arange(len(chan_op_train_pb)) / fs)
        if use_rrc[k] == True:
            chan_op_train = np.convolve(chan_op_train, rc_rx, "full")
            chan_op_train = chan_op_train[:len(chan_op_train_pb)]
        # LMS update of taps
        ff_filter,fb_filter = filt_init(chan_op_train,training_sym_up,ff_filter_len,fb_filter_len,chan_len) 

        chan_op_data_pb = transmit(data_seq,fade_chan,snr)
        chan_op_data = chan_op_data_pb * np.exp(-2j * np.pi * fc * np.arange(len(chan_op_data_pb)) / fs)
        if use_rrc[k] == True:
            chan_op_data = np.convolve(chan_op_data, rc_rx, "full")
            chan_op_data = chan_op_data[:len(chan_op_data_pb)]
        dec_sym = dfe(chan_op_data,ff_filter,fb_filter,ff_filter_len,fb_filter_len)
        dec_sym = (dec_sym > 0)*2 - 1
        data_sym_up = (data_sym_up > 0)*2 - 1 #data sym or data sym up?
        with np.errstate(divide='ignore', invalid='ignore'):
            mse[k,i] = 10 * np.log10(np.mean(np.abs(data_sym_up[:len(dec_sym)]- dec_sym) ** 2))
        if np.isneginf(mse[k,i]) == 1:
            mse[k,i] = -40

fig, ax = plt.subplots()
ax.plot(snr_dB,mse[0,:],'o',snr_dB,mse[1,:],'s')
ax.set_xlabel('SNR (dB)')
ax.set_ylabel('MSE (dB)')
ax.legend(['No RRC','RRC'])
ax.set_xticks(np.arange(snr_dB[0],snr_dB[-1],5))
ax.set_title('SNR vs MSE for BPSK Signal')
plt.show()