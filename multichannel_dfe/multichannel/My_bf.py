from My_functions import *
import os

generate_r = 1          # generate new signal, otherwise read the saved signal
processing_no_bf = 1    # process the multichannel signal before bf
get_S = 1               # get S_theta and do bf
processing_bf = 1       # process the signal after bf

n_path = 1
fc = 7.5e3
tx_channel = 16
K_list = [1,2,4,8]            # the number of channels you want to use

if not os.path.exists('data'):
    os.makedirs('data')

if generate_r:
    receive_signal(fc=fc, tx_channel=tx_channel)

if processing_no_bf:
    receiver_processing(filename='data/r_multichannel.npy', 
                        K_list=K_list, feedforward_taps=200, feedbackward_taps=20, alpha_rls=0.9999)

if get_S:
    get_S_theta(n_path=n_path, theta_start=-45, theta_end=45, 
                plot_S_theta=True, plot3D_S_theta=False)

if processing_bf:
    receiver_processing(filename='data/signal.npy', 
                        K_list=[n_path], feedforward_taps=20, feedbackward_taps=3, alpha_rls=0.9999)