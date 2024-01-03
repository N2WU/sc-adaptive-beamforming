import numpy as np
import scipy.signal as sg
import matplotlib.pyplot as plt

def generate(x,SNR):
    y = np.zeros_like(x)
    L = 2
    h = np.random.rand(L)/SNR
    a = 0.5
    y[0]=x[0]
    for k in range(1,len(x)):
        y[k] = (h[0]*x[k] + np.random.randn(1)/SNR) + a*(h[1]*x[k-1] + np.random.randn(1)/SNR)
    return y

def process(y,x):
    K1 = 2 # FB
    K2 = 5 # FF
    L = 10 # training data
    x_est = np.zeros_like(y)
    x_est[0:L] = x[0:L]
    c_ff = sg.firwin(K1+1,0.5)
    c_fb = sg.firwin(K2,0.5)
    c = np.concatenate((c_ff,c_fb))# tap coefficients of filter
    v = y
    for k in range(L,len(y)-L):
        fb_sum = 0
        for j in range(-K1,0):
            fb_sum = fb_sum + c[j]*v[k-j]
        ff_sum = 0
        for j in range(1,K2):
            ff_sum = ff_sum + c[j]*x_est[k-j]
        x_est[k] = ff_sum + fb_sum
        if x_est[k] >= 0:
            x_est[k] = 1
        else:
            x_est[k] = -1
    return x_est

if __name__ == "__main__":
    bits = 7
    SNR_db = 10
    SNR = np.power(10, SNR_db/10)
    # generate the tx BPSK signal
    d = np.array(sg.max_len_seq(bits)[0]) * 2 - 1.0
    x = np.tile(d,100)
    # generate rx signal with ISI
    y = generate(x,SNR)
    x_hat = process(y,x)
    mse = 10 * np.log10(np.mean(np.abs(x - x_hat) ** 2))
    print("MSE is: ", mse)
