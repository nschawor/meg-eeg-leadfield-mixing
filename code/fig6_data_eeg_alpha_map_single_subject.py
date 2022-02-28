"""Data: show 2 participant alpha maps as an example."""
import pandas as pd
import mne
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cycler

from helper import plot_patterns, compute_coordinates_tiny_topos, get_rainbow_colors
from complexity import compute_norm_coefficients
from params import FIG_FOLDER, RESULTS_FOLDER, EEG_DATA_FOLDER, SSD_NR_COMPONENTS, PATTERN_EEG_DIR

df = pd.read_csv(f"{RESULTS_FOLDER}/eeg_center_frequencies.csv")

# set colors
colors = get_rainbow_colors()

subjects = ["sub-032348", "sub-032302"]
condition = "eo"
topo_size = 0.10

for i_sub, subject in enumerate(subjects):

    fig = plt.figure()
    outer_grid = gridspec.GridSpec(
        1, 2, width_ratios=[1, 4], left=0.02, top=0.98, bottom=0.02
    )

    top_cell = outer_grid[0, 0]
    gs = gridspec.GridSpecFromSubplotSpec(5, 2, top_cell, hspace=.0, wspace=.0)

    df_file_name = f"{PATTERN_EEG_DIR}/{subject}_{condition}_patterns.csv"
    df_patterns = pd.read_csv(df_file_name)
    weighted_patterns = df_patterns.values.T

    file_name = f"{EEG_DATA_FOLDER}/{subject}_{condition}-raw.fif"
    raw = mne.io.read_raw_fif(file_name, verbose=False)
    raw.pick_types(eeg=True)

    # check that the channels are in the correct order
    assert np.all(raw.ch_names == df_patterns.columns)

    plot_patterns(
        weighted_patterns,
        raw,
        SSD_NR_COMPONENTS,
        gs=gs,
        colors=colors,
    )

    # get coordinates
    lay = mne.channels.make_eeg_layout(raw.info)
    pos = lay.pos[:, :2]
    YY = compute_coordinates_tiny_topos(pos, x_offset=0.3)

    # compute absolute patterns
    M = compute_norm_coefficients(weighted_patterns, SSD_NR_COMPONENTS)

    mpl.rcParams["axes.prop_cycle"] = cycler.cycler("color", colors)

    # plot pie chart plot
    right_cell = outer_grid[0, 1]
    gs = gridspec.GridSpecFromSubplotSpec(1, 1, right_cell)
    ax1 = plt.subplot(gs[0, 0])
    ax1.axis("off")

    nr_channels = len(raw.ch_names)
    for i in range(nr_channels):
        idx = np.nonzero(np.array(raw.ch_names) == raw.ch_names[i])[0][0]
        ax1 = plt.Axes(fig, rect=[YY[idx, 0], YY[idx, 1], topo_size, topo_size])
        ax1 = fig.add_axes(ax1)
        ax1.pie(M[i, :SSD_NR_COMPONENTS], normalize=True)

    fig.set_size_inches(7, 5)
    fig.savefig(
        f"{FIG_FOLDER}/fig6_eeg_single_subject_example_{subject}.png",
        dpi=200,
        transparent=True,
    )
    fig.show()
