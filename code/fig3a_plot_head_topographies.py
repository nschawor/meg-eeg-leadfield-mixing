"""Simulation, step #1: place alpha-sources and plot topographies."""
import pyvista as pv
import numpy as np
import mne
import matplotlib.pyplot as plt
from helper import plot_mesh, plot_electrodes, load_leadfield
import pandas as pd


# %% plot topomaps: define generators and put them into a dataframe
def plot_topoplots(ch_names, electrodes, pos_brain, df, LF, colors_dict):

    data = np.zeros((len(ch_names), 10))
    ch_pos = dict(zip(ch_names, electrodes[:, :3] / 1000))
    info = mne.create_info(ch_names, sfreq=1000, ch_types="eeg")
    raw = mne.io.RawArray(data, info)
    montage = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame="head")
    raw.set_montage(montage)

    colors = df.source_type.replace(colors_dict).values
    dipole_coordinates = df[["mni-x", "mni-y", "mni-z"]].values

    fig, ax = plt.subplots(2, 8)
    for i, dipole in enumerate(dipole_coordinates):

        # find vertex closest to MNI-location
        idx_source = np.argmin(np.sum((pos_brain - dipole) ** 2, axis=1))
        LF_entry = LF[idx_source]

        ax1 = ax.flatten()[i]
        ax1.set_title("          ", backgroundcolor=colors[i], fontsize=6)
        mne.viz.plot_topomap(LF_entry, raw.info, axes=ax1, show=False)

    fig.set_size_inches(12, 3)
    fig.savefig("../figures/fig3_topos.png", dpi=200)
    fig.show()

    return fig


# %% load required files
electrodes, ch_names, LF, file = load_leadfield()
df_left = pd.read_csv("../csv/source_locations.csv")

colors_dict = {
    "occipital": "#482878",
    "parietal": "#F6CA44",
    "somatosensory": "#CA4754",
    "temporal": "#37A262",
}

# %% plot 3d brain
pv.set_plot_theme("document")
plotter = pv.Plotter(off_screen=False, window_size=(800, 800))
plot_mesh(file, field="head", plotter=plotter, color="orange", opacity=0.15)
pos_brain = plot_mesh(
    file=file, field="cortex75K", plotter=plotter, color="#AAAAAA", opacity=1
)

fig = plot_topoplots(ch_names, electrodes, pos_brain, df_left, LF, colors_dict)

# create dipoles in right hemisphere
df_right = df_left.copy()
df_right.loc[:, "hemisphere"] = "right"
df_right.loc[:, "mni-x"] *= -1

# save all sources into csv-file
df = pd.concat((df_left, df_right))
df.to_csv("../csv/source_locations_all_hemispheres.csv", index=False)


# %% plot alpha sources as small sphere
dipole_coordinates = df[["mni-x", "mni-y", "mni-z"]].values
node_colors = df.source_type.replace(colors_dict).values

for i in range(len(df)):
    source = dipole_coordinates[i]
    plotter.add_mesh(
        source,
        render_points_as_spheres=True,
        point_size=30,
        color=node_colors[i],
    )

plot_electrodes(electrodes, plotter, color="w")

# save figure for display with mpimg later
plot_3d_filename = "../figures/fig3_mesh_nyhead.png"
plotter.set_background(None)
cpos = [(-647.5069957432964, 56.87433923564016, 50), (0, 0, 0), (0, 0, 1)]
plotter.show(
    cpos=cpos,
    interactive_update=True,
    screenshot=plot_3d_filename,
)

# %% for all channels, save lead field entries for selected dipoles
nr_dipoles = len(dipole_coordinates)
nr_channels = LF.shape[1]
LF_selected = np.zeros((nr_dipoles, nr_channels))

for i in range(nr_dipoles):
    chan = dipole_coordinates[i]
    idx_source = np.argmin(np.sum((pos_brain - chan) ** 2, axis=1))
    LF_selected[i] = LF[idx_source]
    dipole_coordinates[i] = pos_brain[idx_source]

data = {"LF": LF_selected, "dipole_coord": dipole_coordinates}
np.save("../csv/leadfield_selected_dipoles.npy", data)
