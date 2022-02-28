""" Data: compute center frequency for each EEG subject."""
import os
import mne
import numpy as np
import fooof
import pandas as pd

from helper import print_progress
from params import CSV_FOLDER, EEG_DATA_FOLDER, \
    ALPHA_FMIN, ALPHA_FMAX, SPEC_FMIN, SPEC_FMAX, \
    SPEC_NR_PEAKS, SPEC_NR_SECONDS, SNR_THRESHOLD, \
    SPEC_PARAM_CSV_EEG, SPEC_PARAM_DIR_EEG

# %% specify participants and folders
os.makedirs(SPEC_PARAM_DIR_EEG, exist_ok=True)
subjects = pd.read_csv(f"{CSV_FOLDER}/name_match.csv")
subjects = subjects.INDI_ID
conditions = ("eo", "ec")

# %% compute for all participants
for i_sub, subject in enumerate(subjects):
    print_progress(i_sub, subject, subjects)

    spec_file = f"{SPEC_PARAM_DIR_EEG}/{subject}.csv"
    if os.path.exists(spec_file):
        continue

    # combine eyes open and closed condition
    raws = []
    for condition in conditions:
        file_name = f"{EEG_DATA_FOLDER}/{subject}_{condition}-raw.fif"
        if not (os.path.exists(file_name)):
            continue
        raw = mne.io.read_raw_fif(file_name, preload=True)
        raw.set_eeg_reference("average")
        raws.append(raw)

    # only do it for subjects that have both recording types available
    if len(raws) < 2:
        continue

    if subject == "sub-032478":
        # this participants has potentially wrong-labeled channel names
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

    # pick midline channels
    raw.pick_types(eeg=True)
    midline_channels = [ch for ch in raw.ch_names if "z" in ch]
    raw.pick_channels(midline_channels)

    psd, freqs = mne.time_frequency.psd_welch(
        raw,
        fmin=SPEC_FMIN,
        fmax=SPEC_FMAX,
        n_fft=int(SPEC_NR_SECONDS * raw.info["sfreq"]),
        n_overlap=raw.info["sfreq"],
    )

    # fit FOOOF
    fm = fooof.FOOOFGroup(max_n_peaks=SPEC_NR_PEAKS)
    fm.fit(freqs, psd)
    alpha_bands = fooof.analysis.get_band_peak_fg(fm, [ALPHA_FMIN, ALPHA_FMAX])

    peak = np.nanmean(alpha_bands[:, 0])
    amp = np.nanmean(alpha_bands[:, 1])

    # create dataframe with data
    df_subject = pd.Series(
        data={"subject": subject, "peak_frequency": peak, "peak_amplitude": amp}
    )
    df_subject.to_csv(spec_file, index=False)

# %% compile files across subjects
dfs = []
subjects = [s.split(".")[0] for s in os.listdir(SPEC_PARAM_DIR_EEG)]
for i_sub, subject in enumerate(subjects):
    spec_file = f"{SPEC_PARAM_DIR_EEG}/{subject}.csv"
    df_subject = pd.read_csv(spec_file).T
    df_subject.columns = ["subject", "peak_frequency", "peak_amplitude"]
    dfs.append(df_subject)
    peak = df_subject.peak_frequency.values[0]
    print(f"{i_sub:03}/{len(subjects):03}: {subject}: {peak:.2} Hz")

df_subjects = pd.concat(dfs, axis=0)
types_dict = {"subject": "str", "peak_frequency": "float32", "peak_amplitude": "float32"}
df_subjects = df_subjects.astype(types_dict)
df_subjects.to_csv(SPEC_PARAM_CSV_EEG, index=False)

print("number of subjects exceeding the threshold")
print(len(df_subjects[df_subjects.peak_amplitude > SNR_THRESHOLD]))
