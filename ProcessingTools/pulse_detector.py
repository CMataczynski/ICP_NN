# Generates pickle file with list of segmented pulses for each of the data file

import math
from typing import List

import numpy as np
import scipy.signal as sp_sig

class Segmenter:
    def __init__(self) -> None:
        pass

    def detect_pulses_in_signal(self,
        x: np.ndarray,
        fs: float
    ) -> np.ndarray:
        dx = sp_sig.detrend(x)

        filter_critical_coefficient = 10
        filter_order = 6
        filter_max_ripple = 60
        filter_min_attenuation = 1
        filter_btype = 'lowpass'
        filter_ftype = 'cheby1'

        Wc1 = filter_critical_coefficient / fs

        b1, a1 = sp_sig.iirfilter(
            N=filter_order,
            Wn=Wc1,
            btype=filter_btype,
            rs=filter_max_ripple,
            rp=filter_min_attenuation,
            ftype=filter_ftype,
        )
        fx = sp_sig.filtfilt(b1, a1, dx)

        signal_peaks = self.detect_peaks_troughs(-fx, max_scale=fs)
        pulse_onset_inds = signal_peaks[:, 0]

        return pulse_onset_inds


    def detect_peaks_troughs(self,
        signal: np.ndarray,
        max_scale: float = 0,
    ) -> np.ndarray:
        N = len(signal)
        # Max scale - pulse length
        if max_scale != 0:
            L = math.ceil(max_scale / 2) - 1
        else:
            L = math.ceil(N / 2) - 1

        # Replace nans with global mean
        mean_sig = np.nanmean(signal)
        signal[np.argwhere(np.isnan(signal))] = mean_sig

        # Linear detrending
        d_signal = sp_sig.detrend(signal)

        # N - signal length
        # L - Pulse length / 2 - 1
        # Check if point is surrounded by smaller values for in distance 1-L
        # Mx - if row has only 1 - point is local max
        Mx = np.zeros((N, L), dtype=np.float32)
        for kk in range(1, L + 1):
            # Last location skipped because of frequent anomalies
            right = d_signal[kk : -kk - 1] > d_signal[2 * kk : -1]
            left = d_signal[kk : -kk - 1] > d_signal[: -2 * kk - 1]
            Mx[kk : -kk - 1, kk - 1] = np.logical_and(left, right)

        # Cut out too big surroundings
        dx = np.argmax(np.sum(Mx, 0))
        Mx = Mx[:, : dx + 1]

        # Find maxs - peaks
        _, col_count = Mx.shape
        Zx = col_count - np.count_nonzero(Mx, axis=1, keepdims=True)
        peaks = np.argwhere(Zx == 0)

        return peaks


    def split_pulses(self, icp: np.ndarray, time: np.ndarray, fs_hat: float, mean_time: bool = False) -> List[np.ndarray]:
        peks = self.detect_pulses_in_signal(icp, fs_hat)
        pulses = [icp[peks[i] : peks[i + 1]] for i in range(len(peks) - 1)]
        
        if mean_time:
            times = [np.mean(time[peks[i] : peks[i + 1]]) for i in range(len(peks) - 1)]
        else:
            times = [time[peks[i] : peks[i + 1]] for i in range(len(peks) - 1)]

        return pulses, times, peks