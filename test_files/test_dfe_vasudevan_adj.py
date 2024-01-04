import numpy as np
# https://github.com/vineel49/dfe/blob/master/run_me_bpsk.m

# Parameters
training_len = 10**5
snr_dB = 10
ff_filter_len = 20
fb_filter_len = 8
data_len = 10**6

# SNR parameters
snr = 10**(0.1 * snr_dB)
noise_var = 1 / (2 * snr)

# Training phase
training_a = np.random.randint(0, 2, training_len)
training_seq = 1 - 2 * training_a

fade_chan = [1, 0.5]#[0.407, 0.815, 0.407]
fade_chan = fade_chan / np.linalg.norm(fade_chan)
chan_len = len(fade_chan)

noise = np.random.normal(0, np.sqrt(noise_var), training_len + chan_len - 1)
chan_op = np.convolve(fade_chan, training_seq) + noise

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
    error = ff_and_fb - training_seq[i1]

    temp = ff_and_fb < 0
    quantizer_op = 1 - 2 * temp

    ff_filter = ff_filter - step_size * error * ff_filter_ip
    fb_filter = fb_filter + step_size * error * fb_filter_ip

    fb_filter_ip[1:] = fb_filter[:-1]
    fb_filter_ip[0] = quantizer_op

    fb_filter_op = np.dot(fb_filter, fb_filter_ip)

# Data transmission phase
data_a = np.random.randint(0, 2, data_len)
data_seq = 1 - 2 * data_a

noise = np.random.normal(0, np.sqrt(noise_var), data_len + chan_len - 1)
chan_op = np.convolve(fade_chan, data_seq) + noise

dec_seq = np.zeros(data_len - ff_filter_len + 1)
ff_filter_ip = np.zeros(ff_filter_len)
fb_filter_ip = np.zeros(fb_filter_len)
fb_filter_op = 0

for i1 in range(data_len - ff_filter_len + 1):
    ff_filter_ip[1:] = ff_filter_ip[:-1]
    ff_filter_ip[0] = chan_op[i1]
    ff_filter_op = np.dot(ff_filter, ff_filter_ip)

    ff_and_fb = ff_filter_op - fb_filter_op

    temp = ff_and_fb < 0
    dec_seq[i1] = 1 - 2 * temp

    fb_filter_ip[1:] = fb_filter[:-1]
    fb_filter_ip[0] = dec_seq[i1]

    fb_filter_op = np.dot(fb_filter, fb_filter_ip)

# Demapping symbols back to bits
dec_a = dec_seq < 0

# MSE
mse = 10 * np.log10(np.mean(np.abs(data_a[:len(dec_a)] - dec_a) ** 2))
print(f"MSE: {mse}")
