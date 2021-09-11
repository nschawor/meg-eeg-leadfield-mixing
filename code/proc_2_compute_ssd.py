# %% compute SSD filters for all subjects and save them
import pandas as pd
import mne
import numpy as np
import os
import ssd

mne.set_log_level(verbose=False)

df = pd.read_csv("../results/center_frequencies.csv")
df = df.set_index("subject")
folder = "../working/"

results_dir = "../results/ssd/"
os.makedirs(results_dir, exist_ok=True)

peak_cond = "alpha_peak"
conditions = ("eo", "ec")
bin_width = 2

for i_sub, subject in enumerate(df.index):
    print("%i/%i: %s" % (i_sub, len(df), subject))

    for i_cond, condition in enumerate(conditions):

        ssd_file_name = "%s/%s_ssd_%s.npy" % (
            results_dir,
            subject,
            condition,
        )

        if os.path.exists(ssd_file_name):
            continue

        file_name = "%s/%s_%s-raw.fif" % (folder, subject, condition)
        raw = mne.io.read_raw_fif(file_name)
        raw.load_data()
        raw.pick_types(eeg=True)
        raw.set_eeg_reference("average")

        # only do it for subjects which have a peak
        peak = df.loc[subject][peak_cond]
        if np.isnan(peak):
            continue

        # compute SSD in narrow band
        signal_bp = [peak - bin_width, peak + bin_width]
        noise_bp = [peak - (bin_width + 2), peak + (bin_width + 2)]
        noise_bs = [peak - (bin_width + 1), peak + (bin_width + 1)]
        filters, patterns = ssd.compute_ssd(raw, signal_bp, noise_bp, noise_bs)

        # save patterns and filters
        results = dict(filters=filters, patterns=patterns, peak=peak)
        np.save(ssd_file_name, results)
