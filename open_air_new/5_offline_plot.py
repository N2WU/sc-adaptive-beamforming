import numpy as np
import scipy.signal as sg
import matplotlib.pyplot as plt

if __name__ == "__main__":   
    # load
    """
    dhat_dl_real = np.load('data/dhat_dl_real.npy')
    dhat_dl_imag = np.load('data/dhat_dl_imag.npy')
    dhat_dl = dhat_dl_real + 1j*dhat_dl_imag
    dhat_dl = dhat_dl.flatten()

    dhat_ul_real = np.load('data/dhat_ul_real.npy')
    dhat_ul_imag = np.load('data/dhat_ul_imag.npy')
    dhat_ul = dhat_ul_real + 1j*dhat_ul_imag
    dhat_ul = dhat_ul.flatten()

    # uplink constellation diagram
    plt.subplot(1, 2, 1)
    plt.scatter(np.real(dhat_ul), np.imag(dhat_ul), marker='x')
    plt.axis('square')
    plt.axis([-2, 2, -2, 2])
    plt.title(f'Uplink QPSK Constellation')

    # downlink constellation diagram
    plt.subplot(1, 2, 2)
    plt.scatter(np.real(dhat_dl), np.imag(dhat_dl), marker='x')
    plt.axis('square')
    plt.axis([-2, 2, -2, 2])
    plt.title(f'Downlink QPSK Constellation') 
    plt.show() 
    """

    data = np.load('data/d_hat_cum3_16.npy')
    d_hat_6 = data[1,:]
    data = np.load('data/d_hat_cum2_10.npy')
    d_hat_10 = data[3,:]
    data = np.load('data/d_hat_cum1_12.npy')
    d_hat_12 = data[3,:]
    data = np.load('data/d_hat_cum3_16.npy')
    d_hat_16 = data[3,:]

    plt.subplot(2, 2, 1)
    plt.scatter(np.real(d_hat_6), np.imag(d_hat_6), marker='x')
    plt.axis('square')
    plt.axis([-2, 2, -2, 2])
    plt.title(f'M=6')
    plt.subplot(2, 2, 2)
    plt.scatter(np.real(d_hat_10), np.imag(d_hat_10), marker='x')
    plt.axis('square')
    plt.axis([-2, 2, -2, 2])
    plt.title(f'M=10')
    plt.subplot(2, 2, 3)
    plt.scatter(np.real(d_hat_12), np.imag(d_hat_12), marker='x')
    plt.axis('square')
    plt.axis([-2, 2, -2, 2])
    plt.title(f'M=12')
    plt.subplot(2, 2, 4)
    plt.scatter(np.real(d_hat_16), np.imag(d_hat_16), marker='x')
    plt.axis('square')
    plt.axis([-2, 2, -2, 2])
    plt.title(f'M=16')
    plt.show()


