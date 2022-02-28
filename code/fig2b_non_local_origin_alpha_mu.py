"""Spatial pattern demo #2: Non-local origin of Laplacian C3 signal"""
import mne
import numpy as np
import matplotlib.pyplot as plt
import ssd

from helper import get_electrodes
from params import EEG_DATA_FOLDER, FIG_FOLDER

condition = "ec"
subject = "sub-032327"

# load raw file for computing band-power
file_name = f"{EEG_DATA_FOLDER}/{subject}_{condition}-raw.fif"
raw = mne.io.read_raw_fif(file_name, verbose=False)
raw.load_data()
raw.pick_types(eeg=True)
raw_filt = raw.copy().filter(8, 13)

sensors = ["C3", "FC5", "FC1", "CP5", "CP1"]
picks = mne.pick_channels(raw.ch_names, sensors, ordered=True)

# define laplacian spatial filter around C3
W = np.zeros((len(raw.ch_names),))
W[picks[0]] = 1
W[picks[1:]] = -0.25

# compute spatial pattern
W = W[:, np.newaxis]
cov_signal = np.cov(raw_filt._data)
patterns = ssd.compute_patterns(cov_signal, W)

# %% plot topo map
mask = np.zeros((len(raw.ch_names),), dtype="bool")
mask[picks[0]] = 1

raw2 = get_electrodes()
fig, ax = plt.subplots(1, 1)

mask_params = dict(
    marker="o",
    markerfacecolor="w",
    markeredgecolor="k",
    linewidth=0,
    markersize=6,
)

im = mne.viz.plot_topomap(
    patterns[:, 0],
    raw2.info,
    axes=ax,
    mask=mask,
    mask_params=mask_params,
    cmap="PiYG",
    show=False,
)

fig.set_size_inches(3, 3)
fig.tight_layout()
fig.savefig(f"{FIG_FOLDER}/fig2_demo_non_local_patterns.png", dpi=200)
fig.show()
