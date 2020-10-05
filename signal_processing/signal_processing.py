import math

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.integrate as sp_int
import scipy.signal as sp_sig


def detect_pulses_in_signal(x, max_scale, fs):
    dx = sp_sig.detrend(x)
    Wc1 = 5 / (fs/2)
    b1, a1 = sp_sig.iirfilter(N=6 ,Wn=Wc1, btype='lowpass', rs=60, rp=1, ftype='cheby1')
    fx = sp_sig.filtfilt(b1, a1, dx)

    signal_peaks, signal_troughs = detect_peaks_troughs(-fx, max_scale=max_scale)
    pulse_onset_inds = signal_peaks[:, 0]

    return pulse_onset_inds


def detect_peaks_troughs(signal, max_scale=0):
    N = len(signal)
    if max_scale != 0:
        L = math.ceil(max_scale / 2) - 1
    else:
        L = math.ceil(N / 2) - 1

    mean_sig = np.nanmean(signal)
    signal[np.argwhere(np.isnan(signal))] = mean_sig
    d_signal = sp_sig.detrend(signal, type='linear')

    Mx = np.zeros((N, L))
    Mn = np.zeros((N, L))
    for kk in range(1, L + 1):
        for ii in range(kk + 1, N - kk):
            if d_signal[ii - 1] > d_signal[ii - 1 - kk] and d_signal[ii - 1] > d_signal[ii - 1 + kk]:
                Mx[ii - 1, kk - 1] = 1
            if d_signal[ii - 1] < d_signal[ii - 1 - kk] and d_signal[ii - 1] < d_signal[ii - 1 + kk]:
                Mn[ii - 1, kk - 1] = 1

    dx = np.argmax(np.sum(Mx, 0))
    Mx = Mx[:, :dx + 1]
    dn = np.argmax(np.sum(Mn, 0))
    Mn = Mn[:, :dn + 1]

    row_count, col_count = Mx.shape

    Zx = np.zeros((row_count, 1))
    Zn = np.zeros((row_count, 1))
    for row in range(0, row_count):
        Zx[row] = col_count - np.count_nonzero(Mx[row, :])
        Zn[row] = col_count - np.count_nonzero(Mn[row, :])

    peaks = np.argwhere(Zx == 0)
    troughs = np.argwhere(Zn == 0)
    return peaks, troughs


def detrend_pulse(t, signal):
    a = (signal[0] - signal[-1]) / (t[0] - t[-1])
    b = signal[0] - a * t[0]
    signal_corr = []
    for tx, icpx in zip(t, signal):
        icpc = a * tx + b
        signal_corr.append(icpx - icpc)
    signal_corr = np.asarray(signal_corr)
    return signal_corr


def find_signal_offset_by_min(x, y, max_offset=30):
    d_x = np.diff(x)
    d_x = np.insert(d_x, 0, 0)
    d_y = np.diff(y)
    d_y = np.insert(d_y, 0, 0)

    offset = np.argmax(sp_sig.correlate(d_x, d_y)) - len(d_y)
    if abs(offset) > max_offset:
        offset = max_offset if offset < 0 else -max_offset
    else:
        offset = -offset
    return offset


def shift_signal(signal, offset):
    if offset == 0:
        new_signal = signal
    else:
        new_signal = np.zeros(len(signal))
        if offset > 0:
            new_signal[:-offset] = signal[offset:]
        elif offset < 0:
            new_signal[abs(offset):] = signal[:-abs(offset)]
    return new_signal


def normalize(x):
    x_norm = (x - np.min(x)) / (np.max(x) - np.min(x))
    return x_norm


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def find_nearest_index(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx
