from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import file_operations as fo
import signal_processing as sproc

def find_patient(string):
    splitted = string.split('\\')
    for split in splitted:
        if "PAC_" in split:
            return split
    return "Other"


data_folder = Path(r'E:\Projekty\ICPPRoject\icp_21.07\ICP_NN\datasets')
csv_folder = Path(r'E:\Projekty\ICPPRoject\icp_21.07\ICP_NN\datasets\aSAH')
data_files = list(csv_folder.rglob('*.csv'))
file_count = len(data_files)

resave_folder = data_folder.joinpath('SAH')
if not resave_folder.exists():
    resave_folder.mkdir()

for signal_file in tqdm(data_files):
    # try:
    #   print('{}/{}: {}'.format(idx+1, file_count, signal_file.stem))

    patient_folder = resave_folder.joinpath(find_patient(str(signal_file)))
    if not patient_folder.exists():
        patient_folder.mkdir()
    fs_hat, t_hat, raw_icp, raw_abp, f_icp, f_abp, f_fv = fo.\
        read_signals_from_csv_with_datetime(signal_file, should_resample=True, resampling_freuqency=100)
    pulse_onset_inds = sproc.detect_pulses_in_signal(f_icp, max_scale=fs_hat, fs=fs_hat)

    for idp in range(0, len(pulse_onset_inds) - 1):
        c_pulse_start, c_pulse_end = pulse_onset_inds[idp], pulse_onset_inds[idp + 1]
        c_icp = f_icp[c_pulse_start:c_pulse_end]
        c_abp = f_abp[c_pulse_start:c_pulse_end]
        n = len(c_icp)
        c_t = np.linspace(0, (n - 1) * (1 / fs_hat), int(n))

        c_icp = sproc.detrend_pulse(c_t, c_icp)
        c_abp = sproc.detrend_pulse(c_t, c_abp)

        offset_icp_abp = sproc.find_signal_offset_by_min(c_icp, c_abp)
        c_abp_shifted = sproc.shift_signal(c_abp, offset_icp_abp)

        save_file_name = '{}_pulse_{}.csv'.format(signal_file.stem, idp)
        save_file = patient_folder.joinpath(save_file_name)
        fo.save_pulse_to_csv_with_abp(save_file, c_t, c_icp, c_abp_shifted)
    #
    # except Exception:
    #     print('Could not process file: {}'.format(signal_file.stem))
