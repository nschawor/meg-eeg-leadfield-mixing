"""Data: compute weighted activation pattern time course for 1 subject."""
import pandas as pd
import mne
import numpy as np
import os
import helper
import ssd
import matplotlib.pyplot as plt
from matplotlib import rc
from helper import get_rainbow_colors, load_ssd
from params import FIG_FOLDER, CSV_FOLDER, RESULTS_FOLDER, EEG_DATA_FOLDER, SSD_NR_COMPONENTS, \
        SSD_BANDWIDTH

rc("font", **{"family": "sans-serif", "sans-serif": ["Arial"]})

df_complete = pd.read_csv(f"{CSV_FOLDER}/complete_datasets.csv")
df = pd.read_csv(f"{RESULTS_FOLDER}/eeg_center_frequencies.csv")
df = df.set_index("subject")


# rainbow colors 
colors = get_rainbow_colors()

duration = 0.5
condition = "eo"
subjects = ["sub-032370"]
# %% make a plot for all selected participants
for i_sub, subject in enumerate(subjects):

    print(subject)
    complete = df_complete[df_complete.INDI_ID == subject][condition]
    if ~np.all(complete.values):
        continue

    file_name = f"{EEG_DATA_FOLDER}/{subject}_{condition}-raw.fif"
    raw = mne.io.read_raw_fif(file_name, verbose=False)
    raw.load_data()
    events = mne.find_events(raw)
    raw.pick_types(eeg=True)

    # compute SSD in narrow band
    peak = df.iloc[i_sub]["peak_frequency"]
    filters, patterns = load_ssd(subject, "eeg", condition)
    raw_ssd = ssd.apply_filters(raw, filters)

    # compute band power in narrow band around peak or fixed
    raw_ssd.filter(peak - SSD_BANDWIDTH, peak + SSD_BANDWIDTH, verbose=False)
    raw_ssd.apply_hilbert(envelope=True)

    # cut into segments of 10s and average
    epo = mne.make_fixed_length_epochs(raw_ssd, duration=duration)
    epo.load_data()
    std_time = np.std(epo._data, axis=-1)
    time = np.linspace(0, raw.times[-1], std_time.shape[0]) / 60

    ch_names = ["PO8", "C3"]
    picks = mne.pick_channels(raw.ch_names, ch_names, ordered=True)
    if len(picks) < 2:
        continue

    fig, ax = plt.subplots(2, 1)
    for i in range(2):
        idx = picks[i]
        w = (std_time * np.abs(patterns[idx, :]))[:, :SSD_NR_COMPONENTS]
        w = w / np.sum(w, axis=1).max()  # w.max()#(axis=1, keepdims=True)
        ax[i].stackplot(time, 100 * w.T, colors=colors)
        ax[i].set(
            xlabel="time [min]",
            xlim=(0, time[-1]),
            ylabel="power in the alpha-band\n[% maximum]",
            title=f"electrode {raw.ch_names[idx]}",
            ylim=(0, 60),
        )

        boundaries = events[1:, 0] / raw.info["sfreq"] / 60
        for bound in boundaries:
            ax[i].axvline(bound, color="k")

    fig.set_size_inches(5, 5)
    fig.tight_layout()
    fig.savefig(f"{FIG_FOLDER}/fig9_over_time_time_course.png", dpi=200)
    fig.show()

    fig, ax = plt.subplots(1, 2)

    nr_bins = 100
    ax[0].hist(std_time[:, 1] / std_time[:, 0], nr_bins, color="k")
    ax[0].axvline(1, color="r")
    ax[0].set(
        xlabel="power ratio component #2/\ncomponent #1",
        ylabel="number of segments",
        xlim=(0, 5),
    )
    ax[1].hist(std_time[:, 2] / std_time[:, 0], nr_bins, color="k")
    ax[1].axvline(1, color="r")
    ax[1].set(
        xlabel="power ratio component #3/\ncomponent #1",
        ylabel="number of segments",
        xlim=(0, 5),
    )

    fig.set_size_inches(6, 3)
    fig.tight_layout()
    fig.savefig(f"{FIG_FOLDER}/fig9_over_time_ratio.png", dpi=200)
    fig.show()

    nr_patterns = 5
    mask = np.zeros_like(patterns[:, 0], dtype="bool")
    mask[picks] = True
    fig, ax = plt.subplots(nr_patterns, 1)
    for i in range(nr_patterns):
        p = patterns[:, i]
        idx = np.argmax(np.abs(p))
        sign = np.sign(p[idx])
        mne.viz.plot_topomap(
            sign * patterns[:, i],
            raw.info,
            axes=ax[i],
            mask=mask,
        )
        ax[i].set_title(
            "      ",  # ,\,\,\,       ",
            backgroundcolor=colors[i],
            fontsize=8,
        )
    fig.set_size_inches(2, 8)
    fig.tight_layout()
    fig.savefig(f"{FIG_FOLDER}/fig9_over_time_patterns.png", dpi=200)
    fig.show()
