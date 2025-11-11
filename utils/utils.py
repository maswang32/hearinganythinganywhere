import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import pandas as pd
import torch
import os

def plot_spectrogram(S):
    """
    Plots the spectrogram of a given matrix S.

    Parameters:
    S (numpy.ndarray): The input matrix representing the spectrogram.

    Returns:
    matplotlib.figure.Figure: The figure object.
    """
    fig = plt.figure()
    plt.imshow(np.abs(S), aspect='auto', origin='lower', cmap='hot')
    plt.colorbar()
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.show()
    return fig


def arni_sample_from_csv(csv_path_in, num_samples=10):
    """
    Sample RIRs from a CSV file based on the number of closed panels.

    Parameters:
    csv_path_in (str): The path to the input CSV file.
    num_samples (int): The number of RIRs to sample per subset. Default is 10.

    Returns:
    tuple: A tuple containing two elements:
        - dfs (dict): A dictionary where the keys are the starting numbers of the subsets and the values are the sampled RIRs.
        - div (list): A list of lists representing the divisions of the subsets.
    """
    df = pd.read_csv(csv_path_in)

    # sample num_samples RIR per subset 
    dfs = {} 
    div = [[i, i+4] for i in range(0, 51, 5)]
    div[-1][1] = 55 # fix the last division
    for i_div in div:
        dfs[i_div[0]] = df[df['num-closed'].between(i_div[0], i_div[1])].sample(n=num_samples)

    return dfs, div

def compute_echo(ir, fs, N=1024, preDelay=0):
    """
    Computes the mixing time and echo density of an impulse response (IR).

    Args:
        ir (numpy.ndarray): The impulse response signal.
        fs (int): The sampling frequency of the IR.
        N (int, optional): The analysis window length. Defaults to 1024.
        preDelay (int, optional): The pre-delay of the IR. Defaults to 0.

    Returns:
        tuple: A tuple containing the mixing time (in milliseconds) and the echo density.

    Raises:
        ValueError: If the length of the IR is shorter than the analysis window length.

    References:
    Threshold of normalized echo density at which to determine "mixing time"
    Abel & Huang (2006) uses a value of 1.
    Pytorch translation of echoDensity.m from https://github.com/SebastianJiroSchlecht/fdnToolbox
    """
    mixingThresh = 0.9

    # preallocate
    s = np.zeros(len(ir))
    echo_dens = np.zeros(len(ir))

    wTau = np.hanning(N)
    wTau = wTau / np.sum(wTau)

    halfWin = N // 2

    if len(ir) < N:
        raise ValueError('IR shorter than analysis window length (1024 samples). Provide at least an IR of some 100 msec.')

    sparseInd = np.arange(0, len(ir), 500)
    for n in sparseInd:
        # window at the beginning (increasing window length)
        # n = 1 to 513
        if n <= halfWin + 1:
            hTau = ir[0:n + halfWin]
            wT = wTau[-halfWin - n:]

        # window in the middle (constant window length)
        # n = 514 to end-511
        elif n > halfWin + 1 and n <= len(ir) - halfWin + 1:
            hTau = ir[n - halfWin:n + halfWin]
            wT = wTau

        # window at the end (decreasing window length)
        # n = (end-511) to end
        elif n > len(ir) - halfWin + 1:
            hTau = ir[n - halfWin:]
            wT = wTau[:len(hTau)]

        else:
            raise ValueError('Invalid n Condition')

        # standard deviation
        s[n] = np.sqrt(np.sum(wT * (hTau ** 2)))

        # number of tips outside the standard deviation
        tipCt = np.abs(hTau) > s[n]

        # echo density
        echo_dens[n] = np.sum(wT * tipCt)

    # normalize echo density
    echo_dens = echo_dens / scipy.special.erfc(1 / np.sqrt(2))

    echo_dens = np.interp(np.arange(1, len(ir) + 1), sparseInd, echo_dens[sparseInd])

    # determine mixing time
    d = np.argmax(echo_dens > mixingThresh)
    t_abel = (d - preDelay) / fs * 1000

    if t_abel is None:
        t_abel = 0
        print('Mixing time not found within given limits.')

    return t_abel, echo_dens



def rir_onset(rir):
    """
    Calculates the onset of a room impulse response (RIR) signal.

    Parameters:
    rir (ndarray): The room impulse response signal.

    Returns:
    int: The onset of the RIR signal.

    """
    spectrogram = np.abs(scipy.signal.stft(rir, nperseg=64, noverlap=60)[2])
    windowed_energy = np.sum(spectrogram, axis=-2)
    delta_energy = windowed_energy[..., 1:] / (windowed_energy[..., 0:-1] + 1e-16)
    highest_energy_change_window_idx = np.argmax(delta_energy)
    onset = int((highest_energy_change_window_idx - 2) * 4 + 64)
    return onset

def get_device():
    '''set device according to cuda availablilty'''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device

def window2d(window):
    '''create a 2D window from a given 1D window'''
    return window[:, None] * window[None, :]


def find_file(name, path):
    """
    Recursively searches for a file with the given name in the specified path and its subfolders.

    Parameters:
    name (str): The name of the file to search for.
    path (str): The path to the main folder to start the search from.

    Returns:
    str: The full path of the file if found, or None if the file is not found.
    """
    for root, dirs, files in os.walk(path):
        for dir in dirs:
            for root2, dirs2, files2 in os.walk(os.path.join(path, dir)): 
                if name in files2:
                    return os.path.join(root2, name)
    return None