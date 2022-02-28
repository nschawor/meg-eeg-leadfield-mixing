"""Data: show 2 participant alpha maps as an example."""
from code.helper import get_rainbow_colors
import pandas as pd
import mne
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cycler
import os

from helper import get_rainbow_colors, load_meg_data, \
        plot_patterns, compute_coordinates_tiny_topos
from complexity import compute_norm_coefficients
from params import FIG_FOLDER, SSD_MEG_DIR, SSD_NR_COMPONENTS, PATTERN_MEG_DIR

subjects = np.unique([s.split("_")[0] for s in os.listdir(SSD_MEG_DIR) if 'ssd' in s])

# set colors
colors = get_rainbow_colors()

subject = "A2007"
topo_size = 0.05
task = "rest"
suffix = "meg"
datatype = "meg"

fig = plt.figure()
outer_grid = gridspec.GridSpec(
    1, 2, width_ratios=[1, 4], left=0.02, top=0.98, bottom=0.02
)

top_cell = outer_grid[0, 0]
gs = gridspec.GridSpecFromSubplotSpec(5, 2, top_cell)

df_file_name = f"{PATTERN_MEG_DIR}/{subject}_patterns.csv"
df_patterns = pd.read_csv(df_file_name)
weighted_patterns = df_patterns.values.T

# load raw file for computing band-power
raw = load_meg_data(subject)

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
pos = mne.channels.layout._find_topomap_coords(raw.info, picks=None)
YY = compute_coordinates_tiny_topos(pos, x_offset=0.1)

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

fig.set_size_inches(11.68,  7.42)
plot_name = f"{FIG_FOLDER}/fig7_meg_single_subject_example_{subject}.png"
fig.savefig(plot_name, dpi=200, transparent=True)
fig.show()
