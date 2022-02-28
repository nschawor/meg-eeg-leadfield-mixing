"""Simulation, step #1: place alpha-sources and plot topographies."""
import pyvista as pv
import numpy as np
import mne
import matplotlib.pyplot as plt
import pandas as pd

from helper import plot_mesh, plot_electrodes, load_leadfield, plot_topoplots, get_colors_dict
from params import FIG_FOLDER, CSV_FOLDER


# %% load required files
pv.set_plot_theme("document")
electrodes, ch_names, LF, file = load_leadfield()
df_left = pd.read_csv(f"{CSV_FOLDER}/source_locations.csv")

# %% plot 3d brain
plotter = pv.Plotter(off_screen=False, window_size=(800, 800))
plot_mesh(file, field="head", plotter=plotter, color="orange", opacity=0.15)
pos_brain = plot_mesh(
    file=file, field="cortex75K", plotter=plotter, color="#AAAAAA", opacity=1
)

# create dipoles in right hemisphere
df_right = df_left.copy()
df_right.loc[:, "hemisphere"] = "right"
df_right.loc[:, "mni-x"] *= -1

# save all sources into csv-file
df = pd.concat((df_left, df_right))
df.to_csv(f"{CSV_FOLDER}/source_locations_all_hemispheres.csv", index=False)


# %% plot alpha sources as small spheres
colors_dict = get_colors_dict()
node_colors = df.source_type.replace(colors_dict).values
dipole_coordinates = df[["mni-x", "mni-y", "mni-z"]].values

for i in range(len(df)):
    source = dipole_coordinates[i]
    plotter.add_mesh(
        source,
        render_points_as_spheres=True,
        point_size=30,
        color=node_colors[i],
    )

plot_electrodes(electrodes, plotter, color="w")

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
np.save(f"{CSV_FOLDER}/leadfield_selected_dipoles_eeg.npy", data)

# save figure for display with mpimg later
plot_3d_filename = f"{FIG_FOLDER}/fig3_mesh_nyhead_eeg.png"
plotter.set_background(None)
cpos = [(-647.5069957432964, 56.87433923564016, 50), (0, 0, 0), (0, 0, 1)]
plotter.show(
    cpos=cpos,
    interactive_update=True,
    screenshot=plot_3d_filename,
)


# %% make topo plots
fig = plot_topoplots(
    ch_names,
    electrodes,
    df_left,
    LF_selected,
    colors_dict,
    plot_fname=f"{FIG_FOLDER}/fig3_topos_eeg.png",
)
