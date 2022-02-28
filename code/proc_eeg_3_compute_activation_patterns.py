""" Data: compute weighted activation pattern for each subject.
"""
import os
import pandas as pd
import numpy as np
import mne

import ssd
from helper import print_progress, load_ssd
from params import EEG_DATA_FOLDER, SSD_BANDWIDTH, SSD_EEG_DIR, \
                    SPEC_PARAM_CSV_EEG, PATTERN_EEG_DIR

# %% specify participants and folders
subjects = np.unique([s.split("_")[0] for s in os.listdir(SSD_EEG_DIR)])
df = pd.read_csv(SPEC_PARAM_CSV_EEG)
df = df.set_index("subject")

os.makedirs(PATTERN_EEG_DIR, exist_ok=True)
conditions = ["eo", "ec"]

# %% compute for all participants
for i_sub, subject in enumerate(subjects):
    print_progress(i_sub, subject, subjects)

    for condition in conditions:

        # compute weighted spatial pattern cofficients
        df_file_name = f"{PATTERN_EEG_DIR}/{subject}_{condition}_patterns.csv"
        if os.path.exists(df_file_name):
            continue

        # load raw file for computing band-power
        file_name = f"{EEG_DATA_FOLDER}/{subject}_{condition}-raw.fif"
        raw = mne.io.read_raw_fif(file_name, verbose=False)
        raw.load_data()
        raw.pick_types(eeg=True)
        raw.set_eeg_reference("average")

        filters, patterns = load_ssd(subject, "eeg", condition)
        raw_ssd = ssd.apply_filters(raw, filters)

        # compute band power in narrow band around peak
        peak = df.loc[subject]["peak_frequency"]
        raw_ssd.filter(peak - SSD_BANDWIDTH, peak + SSD_BANDWIDTH, verbose=False)

        # weight patterns by filtered band-power
        std_comp = np.std(raw_ssd._data, axis=1)
        assert np.all(std_comp[1] * patterns[:, 1] == (std_comp * patterns)[:, 1])
        weighted_patterns = std_comp * patterns

        # save weighted patterns
        df_patterns = pd.DataFrame(weighted_patterns.T, columns=raw.ch_names)
        df_patterns.to_csv(df_file_name, index=False)
