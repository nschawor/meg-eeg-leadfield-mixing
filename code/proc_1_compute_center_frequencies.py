# compute center frequency for each LEMON subject
import os
import mne
import numpy as np
import fooof
import pandas as pd

mne.set_log_level(verbose=False)

folder = "../working/"
subjects = pd.read_csv("../csv/name_match.csv")
conditions = ("eo", "ec")
min_freq = 8
max_freq = 13

dfs = []
for i_subj, subject in enumerate(subjects.INDI_ID):

    # combine eyes open and closed condition
    raws = []
    for condition in conditions:
        file_name = "%s/%s_%s-raw.fif" % (folder, subject, condition)
        if not (os.path.exists(file_name)):
            continue
        raw = mne.io.read_raw_fif(file_name, preload=True)
        raw.set_eeg_reference("average")
        raws.append(raw)

    # only do it for subjects that have both recording types available
    if len(raws) < 2:
        continue

    # use all channels common to both datasets
    all_chan = set.union(set(raws[0].info["ch_names"]), set(raws[1].info["ch_names"]))
    missing1 = all_chan - set(raws[0].info["ch_names"])
    missing2 = all_chan - set(raws[1].info["ch_names"])
    if missing1 or missing2:
        raws[1].drop_channels(list(missing1))
        raws[0].drop_channels(list(missing2))

    raw = mne.concatenate_raws(raws)

    # use only data sets for which event markers are available
    events = mne.find_events(raw)
    if len(events) < 3:
        continue

    # discard the stim channel
    raw.pick_types(eeg=True)
    midline_channels = [ch for ch in raw.ch_names if "z" in ch]
    raw.pick_channels(midline_channels)

    psd, freqs = mne.time_frequency.psd_welch(raw, fmin=0.1, fmax=35, n_fft=2 * 1024)

    # fit FOOOF
    fm = fooof.FOOOFGroup(max_n_peaks=5)
    fm.fit(freqs, psd)
    alpha_bands = fooof.analysis.get_band_peak_fg(fm, [min_freq, max_freq])

    peak = np.nanmean(alpha_bands[:, 0])
    amp = np.nanmean(alpha_bands[:, 1])

    # create dataframe with data
    df_subject = pd.Series(
        data={"subject": subject, "alpha_peak": peak, "alpha_amp": amp}
    )
    dfs.append(df_subject)
    df_subjects = pd.concat(dfs, axis=1).T
    df_subjects.to_csv("../results/center_frequencies.csv")
    print("%i/%i, %s, alpha: %.2f Hz" % (i_subj, len(subjects), subject, peak))

# select subjects according to amplitude?
threshold = 0.5
print("number of subjects exceeding the threshold")
print(len(df_subjects[df_subjects.alpha_amp > threshold]))
