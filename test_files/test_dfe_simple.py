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
    L = 1 # training length
    x_est = np.zeros_like(y)
    h = np.zeros(L+1)
    x_est[0:L] = x[0:L] # training
    a = 0.5 # attenuation factor
    for k in range(1,len(y)-1):
        h = np.flip(((x[k-L:k+1]).T * x[k-L:k+1])**-1 *(x[k-L:k+1]).T * y[k-L:k+1]) # (x'x)^-1 x' y
        x_est[k] = (y[k]-a*h[1]*x_est[k-1])/h[0]
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
    x = np.tile(d,10)
    # generate rx signal with ISI
    y = generate(x,SNR)
    x_hat = process(y,x)
    mse = 10 * np.log10(np.mean(np.abs(x - x_hat) ** 2))
    print("MSE is: ", mse, "dB")
