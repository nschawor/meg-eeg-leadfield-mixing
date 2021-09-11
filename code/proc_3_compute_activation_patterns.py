# %% compute weighted activation pattern for each subject
import os
import pandas as pd
import numpy as np
import mne
import ssd
import helper

subjects = pd.read_csv("../csv/name_match.csv")
df = pd.read_csv("../results/center_frequencies.csv")
df = df.set_index("subject")

folder = "../working/"
results_dir = "../results/df/"
os.makedirs(results_dir, exist_ok=True)
ssd_dir = "../results/ssd/"
conditions = ["eo", "ec"]

# %%
for i_sub, subject in enumerate(df.index):
    print("%03i/%03i: %s" % (i_sub, len(df), subject))

    for condition in conditions:

        # load raw file for computing band-power
        file_name = "%s/%s_%s-raw.fif" % (folder, subject, condition)
        raw = mne.io.read_raw_fif(file_name, verbose=False)
        raw.load_data()
        raw.pick_types(eeg=True)
        raw.set_eeg_reference("average")

        # compute weighted spatial pattern cofficients
        df_file_name = "%s/%s_%s_patterns.csv" % (results_dir, subject, condition)
        # if os.path.exists(df_file_name):
        #     continue

        # if there is no peak, move on
        peak = df.loc[subject]["alpha_peak"]
        if np.isnan(peak):
            continue

        filters, patterns = helper.load_ssd(subject, condition)
        raw_ssd = ssd.apply_filters(raw, filters)

        # compute band power in narrow band around peak
        raw_ssd.filter(peak - 2, peak + 2, verbose=False)

        # weight patterns by filtered band-power
        std_comp = np.std(raw_ssd._data, axis=1)
        assert np.all(std_comp[1] * patterns[:, 1]
                      == (std_comp * patterns)[:, 1])
        weighted_patterns = std_comp * patterns

        # save weighted patterns
        df_patterns = pd.DataFrame(weighted_patterns.T, columns=raw.ch_names)
        df_patterns.to_csv(df_file_name, index=False)
