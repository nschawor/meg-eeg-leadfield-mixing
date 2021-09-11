"""Data: show 2 participant alpha maps as an example."""
import pandas as pd
import mne
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cycler
import helper

subjects = pd.read_csv("../csv/name_match.csv")
df = pd.read_csv("../results/center_frequencies.csv")

# set colors
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

folder = "../working/"
subjects = ["sub-032332", "sub-032302"]
nr_components = 10
condition = "eo"
topo_size = 0.10


for i_sub, subject in enumerate(subjects):

    fig = plt.figure()
    outer_grid = gridspec.GridSpec(
        1, 2, width_ratios=[1, 4], left=0.02, top=0.98, bottom=0.02
    )

    top_cell = outer_grid[0, 0]
    gs = gridspec.GridSpecFromSubplotSpec(5, 2, top_cell)

    df_file_name = "../results/df/%s_%s_patterns.csv" % (subject, condition)
    df_patterns = pd.read_csv(df_file_name)
    weighted_patterns = df_patterns.values.T

    file_name = "%s/%s_%s-raw.fif" % (folder, subject, condition)
    raw = mne.io.read_raw_fif(file_name, verbose=False)
    raw.pick_types(eeg=True)

    # check that the channels are in the correct order
    assert np.all(raw.ch_names == df_patterns.columns)

    helper.plot_patterns(
        weighted_patterns,
        raw,
        nr_components,
        gs=gs,
        colors=colors,
    )

    # get coordinates
    lay = mne.channels.make_eeg_layout(raw.info)
    pos = lay.pos[:, :2]
    YY = helper.compute_coordinates_tiny_topos(pos, x_offset=0.3)

    # compute absolute patterns
    M = np.abs(weighted_patterns)
    M = (M[:, :nr_components].T / np.sum(M[:, :nr_components], axis=1)).T

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
        ax1.pie(M[i, :nr_components])

    fig.set_size_inches(6, 5)
    fig.savefig(
        "../figures/fig5_single_subject_example_%s.png" % subject,
        dpi=200,
        transparent=True,
    )
    fig.show()
