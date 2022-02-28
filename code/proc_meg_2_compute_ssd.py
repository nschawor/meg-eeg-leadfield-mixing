""" Data: compute SSD filters for all subjects and save them.
"""
import pandas as pd
import numpy as np
import os

import ssd
from helper import print_progress, load_meg_data
from params import SSD_BANDWIDTH, SNR_THRESHOLD, SSD_MEG_DIR, SPEC_PARAM_CSV_MEG

# %% specify participants and folders
df = pd.read_csv(SPEC_PARAM_CSV_MEG)
df = df[df.peak_amplitude > SNR_THRESHOLD]
df = df.set_index("subject")
subjects = df.index

os.makedirs(SSD_MEG_DIR, exist_ok=True)

# %% compute for all participants
for i_sub, subject in enumerate(subjects):
    print_progress(i_sub, subject, subjects)

    ssd_file_name = f"{SSD_MEG_DIR}/{subject}_ssd.npy"

    if os.path.exists(ssd_file_name):
        continue

    # load data
    raw = load_meg_data(subject)

    # compute SSD in narrow band
    peak = df.loc[subject]["peak_frequency"]
    signal_bp = [peak - SSD_BANDWIDTH, peak + SSD_BANDWIDTH]
    noise_bp = [peak - (SSD_BANDWIDTH + 2), peak + (SSD_BANDWIDTH + 2)]
    noise_bs = [peak - (SSD_BANDWIDTH + 1), peak + (SSD_BANDWIDTH + 1)]
    filters, patterns = ssd.compute_ssd(raw, signal_bp, noise_bp, noise_bs)

    # save patterns and filters
    results = dict(filters=filters, patterns=patterns, peak=peak)
    np.save(ssd_file_name, results)
