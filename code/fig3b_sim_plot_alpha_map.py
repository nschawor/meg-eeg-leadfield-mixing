"""Simulation, step #2: create alpha-maps."""
import numpy as np
import mne
import pandas as pd
import matplotlib.pyplot as plt
import helper
import cycler


def plot_alpha_map(df, alpha_scale, colors):

    scalings = df.source_type.replace(
        {
            "parietal": parietal_scale,
            "somatosensory": mu_scale,
            "occipital": alpha_scale,
            "temporal": temporal_scale,
        }
    ).values

    scaled_leadfield = scalings * LF.T
    raw = mne.io.RawArray(scaled_leadfield, info)
    raw.set_montage(montage)

    # take only channels present in the real data for easier comparision
    raw.pick_channels(raw_real.ch_names)
    nr_channels = len(raw.ch_names)

    fig = plt.figure(figsize=(5, 5))
    lay = mne.channels.make_eeg_layout(raw.info)
    pos = lay.pos[:, :2]
    topo_pos = helper.compute_coordinates_tiny_topos(pos)

    # compute contribution
    weighted_patterns = raw._data
    nr_components = raw._data.shape[1]
    M = np.abs(weighted_patterns)
    M = (M[:, :nr_components].T / np.sum(M[:, :nr_components], axis=1)).T

    for i in range(nr_channels):
        idx = np.nonzero(np.array(raw.ch_names) == raw.ch_names[i])[0][0]
        ax1 = plt.Axes(
            fig, rect=[topo_pos[idx, 0], topo_pos[idx, 1], pie_size, pie_size]
        )
        ax1 = fig.add_axes(ax1)
        ax1.set_prop_cycle(cycler.cycler("color", colors[idx_wedges]))
        wedges, texts = ax1.pie(M[i, idx_wedges], startangle=90)
        for w in wedges:
            w.set_linewidth(0.005)
            w.set_edgecolor("w")

    fig.show()
    plot_file = "../figures/fig3_simulation_alpha_scale_%.2f.png" % alpha_scale
    fig.savefig(plot_file, dpi=200)
    return fig


folder = "../working/"
raw_real = helper.get_electrodes()
plt.close("all")

# %% load required files + create MNE info + montage structure for plotting
electrodes, ch_names, _, file = helper.load_leadfield()
ch_pos = dict(zip(ch_names, electrodes[:, :3]))
info = mne.create_info(ch_names, sfreq=1000, ch_types="eeg")
montage = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame="head")

# load dipoles
dipole_fname = "../csv/leadfield_selected_dipoles.npy"
data = np.load(dipole_fname, allow_pickle=True).item()
LF = data["LF"]
dipole_coord = data["dipole_coord"]
df = pd.read_csv("../csv/source_locations_all_hemispheres.csv")


# %% define gain factors to scale leadfield according to the source types
mu_scale = 0.5
parietal_scale = 0.5
temporal_scale = 0.25

pie_size = 0.11

# set colors for left and right hemisphere
df_left = pd.read_csv("../csv/source_locations.csv")
colors_left = {
    "occipital": "#482878",
    "parietal": "#F6CA44",
    "somatosensory": "#CA4754",
    "temporal": "#37A262",
}
colors_left = df_left.source_type.replace(colors_left).values

df_right = pd.read_csv("../csv/source_locations.csv")
colors_right = {
    "occipital": "#A498C0",
    "parietal": "#FAE4A1",
    "somatosensory": "#DDA7AA",
    "temporal": "#AACFB3",
}
colors_right = df_right.source_type.replace(colors_right).values
colors = np.hstack((colors_left, colors_right))

# re-ordering pie wedges to make left and right hemisphere symmetric
idx_wedges = np.hstack(
    [np.arange(len(df_left)), len(df_left) + np.arange(len(df_right))[::-1]]
)

# make alpha pie figure
fig = plot_alpha_map(df, alpha_scale=1, colors=colors)
fig = plot_alpha_map(df, alpha_scale=4, colors=colors)
