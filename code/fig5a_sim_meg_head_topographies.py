"""Simulation, step #1: place alpha-sources and plot topographies."""
import pyvista as pv
import numpy as np
import mne
import scipy.linalg
import pandas as pd

from helper import plot_mesh, plot_meg_sensors, load_leadfield, get_colors_dict, plot_topoplots
from params import FIG_FOLDER, CSV_FOLDER, LEADFIELD_DIR



# %% load required files
pv.set_plot_theme("document")
electrodes, ch_names, LF, file = load_leadfield()
df_left = pd.read_csv(f"{CSV_FOLDER}/source_locations.csv")

# load leadfield
fwd = mne.read_forward_solution(f"{LEADFIELD_DIR}/meg_fwd.fif")
fwd = mne.convert_forward_solution(fwd, force_fixed=True)
LF = fwd["sol"]["data"].T

raw = mne.io.read_raw_fif(f"{LEADFIELD_DIR}/meg_raw.fif")
electrodes = np.array([ch["loc"] for ch in raw.info["chs"]])*1000
ch_names = raw.ch_names

df = pd.read_csv(f"{CSV_FOLDER}/source_locations_all_hemispheres.csv")
colors_dict = get_colors_dict()

# %% plot 3d brain
plotter = pv.Plotter(off_screen=False, window_size=(800, 800))
plot_mesh(file, field="head", plotter=plotter, color="orange", opacity=0.15, scaling=1)

src_fname = f"{LEADFIELD_DIR}/meg_src.fif"
src = mne.read_source_spaces(src_fname)

trans_fname = f"{LEADFIELD_DIR}/nyICBM-trans.fif"
trans = mne.read_trans(trans_fname)

for i in range(2):
    pos = src[i]["rr"]
    pos = mne.transforms.apply_trans(scipy.linalg.pinv(trans["trans"]), pos)
    # structure for triangular faces as required by pyivsta
    tri = src[i]["tris"]
    faces = np.hstack((3 * np.ones((tri.shape[0], 1)), tri))
    faces = faces.astype("int")

    cloud = pv.PolyData(pos*1000, faces)
    plotter.add_mesh(cloud, color="#DDDDDD", opacity=1)

for i in range(0, fwd["source_rr"].shape[0], 100):
    source = mne.transforms.apply_trans(trans["trans"], fwd["source_rr"][i])
    plotter.add_mesh(
        source,
        render_points_as_spheres=True,
        # point_size=30,
        color="b",
    )

# %% plot alpha sources as small spheres
node_colors = df.source_type.replace(colors_dict).values

fname_dipoles = f"{CSV_FOLDER}/leadfield_selected_dipoles_meg.npy"
data = np.load(fname_dipoles, allow_pickle=True).item()
dipole_coordinates_meg = data["dipole_coord"]
LF_selected = data["LF"]

for i in range(len(df)):
    source = dipole_coordinates_meg[i]
    plotter.add_mesh(
        source,
        render_points_as_spheres=True,
        point_size=30,
        color=node_colors[i],
    )

# %% save figure for display with mpimg later
plot_3d_filename = f"{FIG_FOLDER}/meg_fig3_mesh_nyhead.png"
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
    plot_fname=f"{FIG_FOLDER}/fig3_topos_meg.png",
)





