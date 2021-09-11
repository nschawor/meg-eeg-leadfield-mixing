"""Spatial pattern demo #3: Not sufficient data cleaning."""
import mne
import numpy as np
import helper
import matplotlib.pyplot as plt
import ssd

plt.close("all")
folder = "../data/raw/"

# subject ="sub-032305"
subject = "sub-010006"
condition = "ec"

file_name = "%s/%s/RSEEG/%s.vhdr" % (folder, subject, subject)
raw = mne.io.read_raw_brainvision(file_name, preload=True)
raw.drop_channels(["VEOG"])

raw.load_data()
raw.pick_types(eeg=True)
raw.filter(4, 7)

sensors = ["Fz"]
picks = mne.pick_channels(raw.ch_names, sensors, ordered=True)

# create spatial filter
W = np.zeros((len(raw.ch_names),))
W[picks[0]] = 1
W = W[:, np.newaxis]
cov_signal = np.cov(raw._data)
patterns = ssd.compute_patterns(cov_signal, W)

# better electrode positions
raw2 = helper.get_electrodes()
remove = set(raw2.ch_names).difference(raw.ch_names)
raw2.drop_channels(list(remove))
raw2.reorder_channels(raw.ch_names)

# create plot
fig, ax = plt.subplots(1, 1)

mask = np.zeros((len(raw.ch_names),), dtype="bool")
mask[picks] = 1

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
    show=False
)

fig.set_size_inches(3, 3)
fig.tight_layout()
fig.savefig("../figures/fig2_theta_eye_movements.png", dpi=200)
fig.show()
