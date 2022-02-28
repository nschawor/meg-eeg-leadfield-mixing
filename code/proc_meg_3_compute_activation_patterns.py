""" Data: compute weighted activation pattern for each subject.
"""
import os
import pandas as pd
import numpy as np

import ssd
from helper import print_progress, load_ssd, load_meg_data
from params import SSD_BANDWIDTH, SSD_MEG_DIR, SPEC_PARAM_CSV_MEG, PATTERN_MEG_DIR

# %% specify participants and folders
subjects = np.unique([s.split("_")[0] for s in os.listdir(SSD_MEG_DIR) if 'ssd' in s])
df = pd.read_csv(SPEC_PARAM_CSV_MEG)
df = df.set_index("subject")

os.makedirs(PATTERN_MEG_DIR, exist_ok=True)

# %% compute for all participants
for i_sub, subject in enumerate(subjects):
    print_progress(i_sub, subject, subjects)

    # compute weighted spatial pattern cofficients
    df_file_name = f"{PATTERN_MEG_DIR}/{subject}_patterns.csv"
    if os.path.exists(df_file_name):
        continue

    # load raw file for computing band-power
    raw = load_meg_data(subject)

    filters, patterns = load_ssd(subject, "meg")
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
