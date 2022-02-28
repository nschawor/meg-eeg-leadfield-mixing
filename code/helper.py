import os
import pandas as pd
import numpy as np
import mne
import matplotlib.pyplot as plt
from mne.channels.layout import _find_topomap_coords
from mne.viz.topomap import _make_head_outlines
import pyvista as pv
from h5py import File
from mne_bids import BIDSPath, read_raw_bids

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

from params import CSV_FOLDER, EEG_DATA_FOLDER, MEG_DATA_FOLDER, \
        SSD_EEG_DIR, SSD_MEG_DIR, LEADFIELD_DIR


def get_rainbow_colors():
    """Return color scheme for individual participant plots.

    Returns
    -------
        colors : list, hexcode of colors
    """
    
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

    return colors


def get_colors_dict(hemisphere='left'):
    """ Geturn colors for simulated plots.

    Parameters
    ----------
        hemisphere (str, optional): Colors for which hemisphere. Defaults to 'left'.

    Returns
    -------
        colors_dict : dict, colors for each rhythm type
    """

    if hemisphere == "left":
        colors_dict = {
            "occipital": "#482878",
            "parietal": "#F6CA44",
            "somatosensory": "#CA4754",
            "temporal": "#37A262",
        }
    elif hemisphere == "right":
        colors_dict = {
           "occipital": "#A498C0",
           "parietal": "#FAE4A1",
           "somatosensory": "#DDA7AA",
           "temporal": "#AACFB3",
        }

    return colors_dict


def print_progress(i_sub, subject, subjects):
    print(f"{i_sub+1:03}/{len(subjects):03}: {subject}")


def get_meg_subjects():
    """Get list of MEG participants.

    Returns
    -------
        subject : list of available participants.
    """
    df = pd.read_csv(f"{MEG_DATA_FOLDER}/participants.tsv", delimiter="\t")
    subjects = [s for s in df.participant_id if s in os.listdir(MEG_DATA_FOLDER)]
    subjects = [s.split("-")[1] for s in subjects]
    return subjects


def load_meg_data(subject):
    """Load MEG data for a participant ID with mne_bids.

    Parameters
    ----------
        subject (str): participant ID for Schoffelen data set.

    Returns
    -------
        raw: MNE.raw data structure
    """

    datatype = "meg"
    task = "rest"
    suffix = "meg"

    bids_path = BIDSPath(subject=subject, task=task,
                         suffix=suffix, datatype=datatype, root=MEG_DATA_FOLDER)
    raw = read_raw_bids(bids_path=bids_path, verbose=False)
    raw.load_data()
    raw.pick_types(meg="mag", ref_meg=False)

    return raw


def plot_topoplots(ch_names, electrodes, df, LF, colors_dict, plot_fname):
    """Plot lead field topomaps with color in title. 

    Parameters
    ----------
        ch_names (str): sensor names.
        electrodes (array): xyz-sensor positions.
        df (pandas.DataFrame): dataframe with specified sources
        LF (array): lead field entries to plot as a topography.
        colors_dict (dict): color map for titles, e.g. from get_colors_dict
        plot_fname (str): Plot file name for saving the created figure.

    Returns
    -------
        fig: Figure containing nr_dipoles topomaps.
    """

    data = np.zeros((len(ch_names), 10))
    ch_pos = dict(zip(ch_names, electrodes[:, :3] / 1000))
    info = mne.create_info(ch_names, sfreq=1000, ch_types="eeg")
    raw = mne.io.RawArray(data, info)
    montage = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame="head")
    raw.set_montage(montage)

    colors = df.source_type.replace(colors_dict).values
    nr_dipoles = len(df)

    fig, ax = plt.subplots(2, 8)
    for i in range(nr_dipoles):

        # find vertex closest to MNI-location
        LF_entry = LF[i]

        ax1 = ax.flatten()[i]
        ax1.set_title("          ", backgroundcolor=colors[i], fontsize=6)
        mne.viz.plot_topomap(LF_entry, raw.info, axes=ax1, show=False)

    fig.tight_layout()
    fig.set_size_inches(2*4, 3)
    fig.savefig(plot_fname, dpi=200, transparent=True)
    fig.show()

    return fig


def compute_coordinates_tiny_topos(pos, scale=0.9, x_offset=0.0, y_offset=0.0):
    """ Generates coordinates for pie plots according to electrode positions.

    Parameters
    ----------
        pos : array, (n_electrodes x 2) x-y EEG electrode positions
        scale (float, optional): Global scaling factor for figure. Defaults to 0.9.
        x_offset (float, optional): Shifts coordinates right. Defaults to 0.0.
        y_offset (float, optional): Shifts coordinates up. Defaults to 0.0.

    Returns
    -------
        topo_pos: x-y-coordinates for defining axes
    """
    topo_pos = pos.copy()  #
    topo_pos[:, 0] = (
        scale
        * (pos[:, 0] - np.min(pos[:, 0]) + x_offset)
        / (np.max(pos[:, 0]) - np.min(pos[:, 0]) + x_offset)
    )
    topo_pos[:, 1] = (
        scale * (pos[:, 1] - np.min(pos[:, 1])) / (np.max(pos[:, 1]) - np.min(pos[:, 1]))
        + y_offset
    )
    return topo_pos


def load_ssd(subject, modality, condition=None):
    """Loads spatial filters and patterns generated with SSD.

    Parameters
    ----------
        subject (str): Participant ID
        condition (str): "eo" for eyes open or "ec" for eyes closed condition

    Returns
    -------
        filters: array, spatial filters
        patterns: array, spatial patterns
    """

    if modality == "meg":
        ssd_dir = SSD_MEG_DIR
        ssd_file_name = f"{ssd_dir}/{subject}_ssd.npy"
        results = np.load(ssd_file_name, allow_pickle=True).item()        
    else:
        ssd_dir = SSD_EEG_DIR
        ssd_file_name = f"{ssd_dir}/{subject}_ssd_{condition}.npy"
        results = np.load(ssd_file_name, allow_pickle=True).item()
    filters = results["filters"]
    patterns = results["patterns"]

    return filters, patterns


def get_electrodes(subject="sub-032303"):
    """ Loads subject with all electrodes present for extracting electrode coordinates.

    Returns
    -------
        raw : mne.io.Raw, raw-file with electrode position.
    """
    subjects = pd.read_csv(f"{CSV_FOLDER}/name_match.csv")
    subject = subjects.INDI_ID.iloc[2]
    file_name = f"{EEG_DATA_FOLDER}/{subject}_eo-raw.fif"
    raw = mne.io.read_raw_fif(file_name, preload=True)
    raw.crop(0, 1)
    raw.pick_types(eeg=True)
    return raw


# plot patterns
def plot_patterns(patterns, raw, nr_components, colors, gs, cmap="RdBu_r"):
    """Plot a number of spatial patterns as a topography.

    Parameters
    ----------
        patterns (array): spatial patterns to plot.
        raw (mne.io.Raw): raw-file for electrode positions.
        nr_components (int): Number of components that will be plotted.
        colors (list): List of identifying colors.
        gs (matplotlib.gridspec.GridSpec): grid for plotting the topographies.
        cmap (str, optional): Colormap for topographies. Defaults to "RdBu_r".
    """

    dd = 0
    cc = 0
    for i in range(nr_components):
        ax1 = plt.subplot(gs[dd, cc])

        idx1 = np.argmax(np.abs(patterns[:, i]))
        patterns[:, i] = np.sign(patterns[idx1, i]) * patterns[:, i]
        mne.viz.plot_topomap(patterns[:, i], raw.info, axes=ax1, cmap=cmap, show=False)
        ax1.set_title("      ", backgroundcolor=colors[i], fontsize=8)

        cc += 1
        if cc == 2:
            dd += 1
            cc = 0


def load_leadfield():
    """Extract useful fields from matlab-structure.

    Returns
    -------
        electrodes : array, xyz-coordinates of electrodes + normals
        ch_names : list, names of EEG electrodes
        LF : array, lead field entries
        file : H5file of lead field.
    """

    # load lead field matrix
    file = File(f"{LEADFIELD_DIR}/sa_nyhead.mat", "r")

    # extract channel names
    ch_names = get_channel_names(file)

    # 3d electrode positions
    electrodes = file["sa"]["locs_3D"][:].T

    # lead field
    LF = file["sa"]["cortex75K"]["V_fem_normal"][:]

    # remove lower electrodes for aesthetical reasons
    idx_select = electrodes[:, 2] > -50
    electrodes = electrodes[idx_select, :]
    LF = LF[:, idx_select]
    ch_names = list(np.array(ch_names)[idx_select])

    return electrodes, ch_names, LF, file


def plot_meg_sensors(electrodes, plotter, color="w"):
    """Plot electrodes as small cyclinders into a specific pyvista plotter.

    Parameters
    ----------
    electrodes : array, (n_electrodes x 6) electrode coordinates and normals.
    plotter : pyvista plotter
    color : str, color of electrodes.
    """
    for i in range(len(electrodes)):
        cylinder = pv.Plane(
            center=electrodes[i, :3],
            direction=electrodes[i,6:9], #np.reshape(electrodes[i,3:], [3,3]),
            # radius=3.5,
            i_size=3.5,
            j_size=3.5,
            # height=2.0,
        )
        plotter.add_mesh(cylinder, color=color)


def plot_electrodes(electrodes, plotter, color="w"):
    """Plot electrodes as small cyclinders into a specific pyvista plotter.

    Parameters
    ----------
    electrodes : array, (n_electrodes x 6) electrode coordinates and normals.
    plotter : pyvista plotter
    color : str, color of electrodes.
    """
    for i in range(len(electrodes)):
        cylinder = pv.Cylinder(
            center=electrodes[i, :3],
            direction=electrodes[i, 3:],
            radius=3.5,
            height=2.0,
        )
        plotter.add_mesh(cylinder, color=color)


def plot_mesh(file, field, plotter, color, opacity=1, scaling=1):
    """Visualizes a specific field from NYhead with pyvista and returns
        corresponding node positions.

    Parameters
    ----------
    field : str, field in NYhead matfile structure which should be visualized.
    plotter : pyvista plotter
    color : str, color of mesh.
    opacity : float, regulate opacity of mesh.

    Returns
    -------
    pos : array, xyz-coordinates of nodes in mesh.

    """

    pos = file["sa"][field]["vc"][:].T/scaling

    # structure for triangular faces as required by pyivsta
    tri = file["sa"][field]["tri"][:].T - 1
    faces = np.hstack((3 * np.ones((tri.shape[0], 1)), tri))
    faces = faces.astype("int")

    cloud = pv.PolyData(pos, faces)
    plotter.add_mesh(cloud, color=color, opacity=opacity)
    return pos


def get_channel_names(h5_file):
    """Flatten electrode names into list.
    Parameters
    ----------
    h5_file : H5 file object from which electrode names should be extracted.
    """
    ch_ref = [ch[0] for ch in h5_file["sa"]["clab_electrodes"][:]]
    ch_names = []
    for ch in ch_ref:
        ch_name = "".join([chr(i[0]) for i in h5_file[ch][:]])
        ch_names.append(ch_name)

    return ch_names


def make_topoplot(
    values,
    info,
    ax,
    vmin=None,
    vmax=None,
    plot_head=True,
    cmap="RdBu_r",
    size=30,
    hemisphere="all",
    picks=None,
    pick_color=["#2d004f", "#254f00", "#000000"],
):
    """Makes an topo plot with electrodes circles, without interpolation.
    Modified from MNE plot_topomap to plot electrodes without interpolation.
    Parameters
    ----------
    values : array, 1-D
        Values to plot as color-coded circles.
    info : instance of Info
        The x/y-coordinates of the electrodes will be infered from this object.
    ax : instance of Axes
        The axes to plot to.
    vmin : float | None
         Lower bound of the color range. If None: - maximum absolute value.
    vmax : float | None
        Upper bounds of the color range. If None: maximum absolute value.
    plot_head : True | False
        Whether to plot the outline for the head.
    cmap : matplotlib colormap | None
        Colormap to use for values, if None, defaults to RdBu_r.
    size : int
        Size of electrode circles.
    picks : list | None
        Which electrodes should be highlighted with by drawing a thicker edge.
    pick_color : list
        Edgecolor for highlighted electrodes.
    hemisphere : string ("left", "right", "all")
        Restrict which hemisphere of head outlines coordinates to plot.
    Returns
    -------
    sc : matplotlib PathCollection
        The colored electrode circles.
    """

    pos = _find_topomap_coords(info, picks=None)
    sphere = np.array([0.0, 0.0, 0.0, 0.095])
    outlines = _make_head_outlines(
        sphere=sphere, pos=pos, outlines="head", clip_origin=(0.0, 0.0)
    )

    if plot_head:
        outlines_ = {
            k: v for k, v in outlines.items() if k not in ["patch", "mask_pos"]
        }
        for key, (x_coord, y_coord) in outlines_.items():
            if hemisphere == "left":
                if type(x_coord) == np.ndarray:
                    idx = x_coord <= 0
                    x_coord = x_coord[idx]
                    y_coord = y_coord[idx]
                ax.plot(x_coord, y_coord, color="k", linewidth=1, clip_on=False)
            elif hemisphere == "right":
                if type(x_coord) == np.ndarray:
                    idx = x_coord >= 0
                    x_coord = x_coord[idx]
                    y_coord = y_coord[idx]
                ax.plot(x_coord, y_coord, color="k", linewidth=1, clip_on=False)
            else:
                ax.plot(x_coord, y_coord, color="k", linewidth=1, clip_on=False)

    if not (vmin) and not (vmax):
        vmin = -values.max()
        vmax = values.max()

    sc = ax.scatter(
        pos[:, 0],
        pos[:, 1],
        s=size,
        edgecolors="grey",
        c=values,
        vmin=vmin,
        vmax=vmax,
        cmap=plt.get_cmap(cmap),
    )

    if np.any(picks):
        picks = np.array(picks)
        if picks.ndim > 0:
            if len(pick_color) == 1:
                pick_color = [pick_color] * len(picks)
            for i, idxx in enumerate(picks):
                ax.scatter(
                    pos[idxx, 0],
                    pos[idxx, 1],
                    s=size,
                    edgecolors=pick_color[i],
                    facecolors="None",
                    linewidths=1.5,
                    c=None,
                    vmin=vmin,
                    vmax=vmax,
                    cmap=plt.get_cmap(cmap),
                )

    ax.axis("square")
    ax.axis("off")
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    return sc
