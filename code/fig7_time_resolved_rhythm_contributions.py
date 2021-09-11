"""Data: compute weighted activation pattern time course for 1 subject."""
import pandas as pd
import mne
import numpy as np
import os
import helper
import ssd
import matplotlib.pyplot as plt

from matplotlib import rc

rc("font", **{"family": "sans-serif", "sans-serif": ["Arial"]})

df_complete = pd.read_csv("../csv/complete_datasets.csv")
df = pd.read_csv("../results/center_frequencies.csv")
df = df.set_index("subject")

folder = "../working/"
results_dir = "../results/individual/"
os.makedirs(results_dir, exist_ok=True)

ssd_dir = "../results/ssd/"
peak_cond = "alpha_peak"
band_condition = "ipf"
condition = "eo"

# rainbow colors + gray scales
colors = [
    "#482878",
    "#3182BD",
    "#35B779",
    "#2E975A",
    "#FDF224",
    "#FDE725",
    "#FDC325",
    "#FD9E24",
    "#EF7621",
    "#EF4E20",
]

nr_components = 10
duration = 0.5
subjects = ["sub-032370"]

for i_sub, subject in enumerate(subjects):

    plt.close("all")
    print(subject)
    test = df_complete[df_complete.INDI_ID == subject][condition]
    print(test)
    if ~np.all(test.values):
        continue

    file_name = "%s/%s_%s-raw.fif" % (folder, subject, condition)
    raw = mne.io.read_raw_fif(file_name, verbose=False)
    raw.load_data()

    events = mne.find_events(raw)
    raw.pick_types(eeg=True)

    df_file_name = "../results/df/%s_%s_patterns.csv" % (subject, condition)

    # compute SSD in narrow band
    peak = df.iloc[i_sub]["alpha_peak"]
    if np.isnan(peak):
        continue

    filters, patterns = helper.load_ssd(subject, condition)
    raw_ssd = ssd.apply_filters(raw, filters)

    # compute band power in narrow band around peak or fixed
    raw_ssd.filter(peak - 2, peak + 2, verbose=False)
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
        w = (std_time * np.abs(patterns[idx, :]))[:, :nr_components]
        w = w / np.sum(w, axis=1).max()  # w.max()#(axis=1, keepdims=True)
        ax[i].stackplot(time, 100 * w.T, colors=colors)
        ax[i].set(
            xlabel="time [min]",
            xlim=(0, time[-1]),
            ylabel="power in the alpha-band\n[% maximum]",
            title="electrode %s" % raw.ch_names[idx],
            ylim=(0, 60),
        )

        boundaries = events[1:, 0] / raw.info["sfreq"] / 60
        for bound in boundaries:
            ax[i].axvline(bound, color="k")

    fig.set_size_inches(5, 5)
    fig.tight_layout()
    fig.savefig("../figures/fig7_over_time_time_course.png", dpi=200)
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
    fig.savefig("../figures/fig7_over_time_ratio.png", dpi=200)
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
    fig.savefig("../figures/fig7_over_time_patterns.png", dpi=200)
    fig.show()
