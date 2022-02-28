"""Spatial pattern demo #1: Signal spread for different references."""
import mne
import numpy as np
import ssd
import matplotlib.pyplot as plt
from helper import make_topoplot, get_electrodes
from params import EEG_DATA_FOLDER, FIG_FOLDER

condition = "eo"
subject = "sub-032406"

# %% load data and select Laplacian around C3
file_name = f"{EEG_DATA_FOLDER}/{subject}_{condition}-raw.fif"
raw = mne.io.read_raw_fif(file_name, verbose=False)
raw.load_data()
raw.pick_types(eeg=True)

raw_filt = raw.copy().filter(8, 13)
cov_signal = np.cov(raw._data)
sensors = ["C3", "FC5", "FC1", "CP5", "CP1"]
picks = mne.pick_channels(raw.ch_names, sensors, ordered=True)

# %% define three different types of spatial filters

# FCz reference
nr_spatial_filters = 3
W = np.zeros((len(raw.ch_names), nr_spatial_filters))
W[picks[0], 0] = 1

# common average reference
W[:, 1] = -1 / len(raw.ch_names)
W[picks[0], 1] = 1

# laplacian
W[picks[0], 2] = 1
W[picks[1:], 2] = -0.25

# %% plot spatial filters
cov_signal = np.cov(raw_filt._data)
raw2 = get_electrodes()

vmins = [-1, -0.8, -1]
fig, ax = plt.subplots(1, nr_spatial_filters, squeeze=False)
for i in range(nr_spatial_filters):
    make_topoplot(
        W[:, i],
        raw2.info,
        ax[0, i],
        size=35,
        cmap="bwr",
        vmin=vmins[i],
        vmax=1,
    )
labels = ["FCz-reference", "CAR", "Laplacian"]

fig.set_size_inches(7.4, 4)
fig.tight_layout()
fig.savefig(f"{FIG_FOLDER}/fig2_ref_choice_demo_filters.png", dpi=200)

# %% plot spatial patterns
mask = np.zeros((len(raw.ch_names),), dtype="bool")
mask[picks[0]] = 1

mask_params = dict(
    marker="o",
    markerfacecolor="w",
    markeredgecolor="k",
    linewidth=0,
    markersize=6,
)

fig, ax = plt.subplots(1, nr_spatial_filters, squeeze=False)

for i in range(nr_spatial_filters):
    patterns = ssd.compute_patterns(cov_signal, W[:, i][:, np.newaxis])
    p = patterns[:, 0]
    im = mne.viz.plot_topomap(
        p,
        raw2.info,
        axes=ax[0, i],
        mask=mask,
        mask_params=mask_params,
        cmap="PiYG",
        show=False,
    )

fig.set_size_inches(7.4, 4)
fig.tight_layout()
fig.savefig(f"{FIG_FOLDER}/fig2_ref_choice_demo_patterns.png", dpi=200)
