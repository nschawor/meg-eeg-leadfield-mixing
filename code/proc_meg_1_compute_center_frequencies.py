""" Data: compute center frequency for each MEG subject."""
import os
import mne
import numpy as np
import fooof
import pandas as pd

from helper import get_meg_subjects, print_progress, load_meg_data
from params import RESULTS_FOLDER, \
        ALPHA_FMIN, ALPHA_FMAX, SPEC_FMIN, SPEC_FMAX, \
        SPEC_NR_PEAKS, SPEC_NR_SECONDS, SNR_THRESHOLD, \
        SPEC_PARAM_CSV_MEG, SPEC_PARAM_DIR_MEG

# %% specify participants and folders
os.makedirs(SPEC_PARAM_DIR_MEG, exist_ok=True)
name_results_all_subjects = f"{RESULTS_FOLDER}/meg_center_frequencies.csv"
subjects = get_meg_subjects()

# %% compute for all participants
for i_sub, subject in enumerate(subjects):
    print_progress(i_sub, subject, subjects)

    spec_file = f"{SPEC_PARAM_DIR_MEG}/{subject}.csv"
    if os.path.exists(spec_file):
        continue

    if subject == "V1025":
        # this subject has a read-in error
        continue
    if subject in ["A2119", "V1001", "V1002", "V1003", "V1005"]:
        # no resting data is available according to the companion paper
        continue

    raw = load_meg_data(subject)

    # pick midline channels
    raw.pick_types(meg=True)
    midline_channels = [ch for ch in raw.ch_names if "Z" in ch]
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
    data = {"subject": subject, "peak_frequency": peak, "peak_amplitude": amp}
    df_subject = pd.Series(data)
    df_subject.to_csv(spec_file, index=False)

# %% compile files across subjects
dfs = []
subjects = [s.split(".")[0] for s in os.listdir(SPEC_PARAM_DIR_MEG)]
for i_sub, subject in enumerate(subjects):
    spec_file = f"{SPEC_PARAM_DIR_MEG}/{subject}.csv"
    df_subject = pd.read_csv(spec_file).T
    df_subject.columns = ["subject", "peak_frequency", "peak_amplitude"]
    dfs.append(df_subject)
    peak = df_subject.peak_frequency.values[0]
    print(f"{i_sub:03}/{len(subjects):03}: {subject}: {peak:.02} Hz")

df_subjects = pd.concat(dfs, axis=0)
types_dict = {"subject": "str",
              "peak_frequency": "float32",
              "peak_amplitude": "float32"}
df_subjects = df_subjects.astype(types_dict)
df_subjects.to_csv(SPEC_PARAM_CSV_MEG, index=False)

print("number of subjects exceeding the threshold")
print(len(df_subjects[df_subjects.peak_amplitude > SNR_THRESHOLD]))
