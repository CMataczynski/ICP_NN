import csv
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.signal as sp_sig
import scipy.interpolate as sp_int


class FileReadError(Exception):
    pass


def read_signals_from_csv_with_datetime(file_path: Path, should_resample=False, resampling_freuqency=0):
    with open(file_path) as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        headers = next(reader)
        data_cols = get_signal_file_headers(headers)
    with open(file_path) as csv_file:
        raw_data = pd.read_csv(csv_file, sep=',', index_col=False)

    datetime = raw_data[data_cols['datetime']].to_numpy()
    t0 = (datetime[0] - np.floor(datetime[0]))*24*3600
    raw_t = np.squeeze((datetime - np.floor(datetime))*24*3600 - t0)
    fs_hat = round(1 / (raw_t[5] - raw_t[4]), -1)

    n = len(raw_t)
    t_hat = np.linspace(0, (n - 1) * (1 / fs_hat), int(n))

    raw_icp = np.squeeze(raw_data[data_cols['icp']].to_numpy())
    n_icp = remove_nans(raw_icp)
    if data_cols['abp']:
        raw_abp = np.squeeze(raw_data[data_cols['abp']].to_numpy())
        n_abp = remove_nans(raw_abp)
    else:
        n_abp = None
    if data_cols['fvx']:
        raw_fv = np.squeeze(raw_data[data_cols['fvx']].to_numpy())
        n_fv = remove_nans(raw_fv)
    else:
        n_fv = None

    if should_resample:
        new_fs = resampling_freuqency
        fs_ratio = new_fs / fs_hat
        new_t = np.linspace(0, t_hat[-1], int(n * fs_ratio))
        fun_icp = sp_int.interp1d(t_hat, n_icp)
        new_icp = fun_icp(new_t)
        if n_abp is not None:
            fun_abp = sp_int.interp1d(t_hat, n_abp)
            new_abp = fun_abp(new_t)
        else:
            new_abp = None
        if n_fv is not None:
            fun_fv = sp_int.interp1d(t_hat, n_fv)
            new_fv = fun_fv(new_t)
        else:
            new_fv = None
        fs_hat = resampling_freuqency
        t_hat = new_t
        n_icp, n_abp, n_fv = new_icp, new_abp, new_fv

    Wn1 = 10 / (fs_hat / 2)
    b1, a1 = sp_sig.iirfilter(N=8, Wn=Wn1, btype='lowpass', rs=60, rp=1, ftype='cheby1')
    f_icp = sp_sig.filtfilt(b1, a1, n_icp)
    if n_abp is not None:
        f_abp = sp_sig.filtfilt(b1, a1, n_abp)
    else:
        f_abp = None
    if n_fv is not None:
        f_fv = sp_sig.filtfilt(b1, a1, n_fv)
    else:
        f_fv = None
    return fs_hat, t_hat, n_icp, n_abp, f_icp, f_abp, f_fv


def get_signal_file_headers(all_headers):
    data_types = ['datetime', 'icp', 'abp', 'fvx']
    data_headers = {}
    for c_type in data_types:
        c_hd = [hd for hd in all_headers if c_type in hd.lower()]
        if c_type == 'fvx' and not c_hd:
            c_hd = [hd for hd in all_headers if 'fvr' in hd.lower()]
        if c_type == 'abp' and not c_hd:
            c_hd = [hd for hd in all_headers if 'art' in hd.lower()]
        data_headers[c_type] = c_hd
    return data_headers


def remove_nans(data, fill='mean', fill_constant=None):
    if fill == 'mean':
        fill_value = np.nanmean(data)
    elif fill == 'zero':
        fill_value = 0
    elif fill == 'constant':
        fill_value = fill_constant
    else:
        raise ValueError('Invalid fill mode: {}. Available: mean, zero, constant'.format(fill))
    nan_inds = np.argwhere(np.isnan(data))
    data[nan_inds] = fill_value
    return data


def save_pulse_to_csv_with_abp(file_path, t, icp, abp):
    headers = ['t[s]', 'icp[mmhg]', 'abp[mmhg]']
    with open(file_path, 'w', newline='') as cf:
        writer = csv.writer(cf, delimiter=',')
        writer.writerow(headers)
        for c_t, c_icp, c_abp in zip(t, icp, abp):
            writer.writerow([c_t, c_icp, c_abp])
