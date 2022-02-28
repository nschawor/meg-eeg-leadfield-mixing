"""Spatial pattern demo #3: Not sufficient data cleaning."""
import os
import mne
import numpy as np
import matplotlib.pyplot as plt

import ssd
from helper import get_electrodes
from params import FIG_FOLDER

# download sub-32305 Raw Data from here:
# http://fcon_1000.projects.nitrc.org/indi/retro/MPI_LEMON/downloads/download_EEG.html
# and put it in the specified data folder
data_folder = "../data/"

# this renaming has to take place because otherwise there is a mismatch 
# between subject identifiers in the header files
old_sub_name = "sub-032305"
subject = "sub-010006"
os.makedirs(f"{data_folder}/{subject}/RSEEG", exist_ok=True)
for file_type in ["eeg", "vhdr", "vmrk"]:
    old_file = f"{data_folder}/{old_sub_name}/RSEEG/{old_sub_name}.{file_type}"
    new_file = old_file.replace(old_sub_name, subject)
    if not(os.path.exists(new_file)):
        os.rename(old_file, new_file)

# load data and filter in theta range
file_name = f"{data_folder}/{subject}/RSEEG/{subject}.vhdr"
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

# load electrode positions
raw2 = get_electrodes()
remove = set(raw2.ch_names).difference(raw.ch_names)
raw2.drop_channels(list(remove))
raw2.reorder_channels(raw.ch_names)

# plot spatial patterns
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
fig.savefig(f"{FIG_FOLDER}/fig2_theta_eye_movements.png", dpi=200)
fig.show()
