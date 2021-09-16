import pandas as pd
import numpy as np
import mne
import matplotlib.pyplot as plt
from mne.channels.layout import _find_topomap_coords
from mne.viz.topomap import _make_head_outlines
import pyvista as pv
from h5py import File


def compute_sensor_complexity(weighted_patterns, nr_components):
    """Computes sensor complexity as a proxy for assessing spatial mixing.

    Parameters
    ----------
        weighted_patterns : array, patterns weighted by amplitude.
        nr_components : measure is calcualted using this many components.

    Returns
    -------
        sensor_complexity : array, sensor complexity for each sensor.
    """

    weighted_patterns = weighted_patterns.astype("float32")

    # take absolute value
    M = np.abs(weighted_patterns)
    M = M[:, :nr_components]

    # normalize across dipoles
    norm_M = np.sum(M, axis=1)
    M = (M.T / norm_M).T
    sensor_complexity = -np.sum(M * np.log(M), axis=1)

    return sensor_complexity


def compute_coordinates_tiny_topos(pos, scale=0.9, x_offset=0.0, y_offset=0.0):
    """ Generates coordinates for pie plots according to electrode positions.

    Parameters:
    -----------
        pos : array, (n_electrodes x 2) x-y EEG electrode positions
        scale (float, optional): Global scaling factor for figure. Defaults to 0.9.
        x_offset (float, optional): Shifts coordinates right. Defaults to 0.0.
        y_offset (float, optional): Shifts coordinates up. Defaults to 0.0.

    Returns:
    --------
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


def load_ssd(subject, condition):
    """Loads spatial filters and patterns generated with SSD.

    Parameters
    ----------
        subject (str): Participant ID
        condition (str): 'eo' for eyes open or 'ec' for eyes closed condition

    Returns
    -------
        filters: array, spatial filters
        patterns: array, spatial patterns
    """
    ssd_dir = "../results/ssd/"
    ssd_file_name = "%s/%s_ssd_%s.npy" % (ssd_dir, subject, condition)
    results = np.load(ssd_file_name, allow_pickle=True).item()
    filters = results["filters"]
    patterns = results["patterns"]

    return filters, patterns


def get_electrodes():
    """ Loads subject with all electrodes present for extracting electrode coordinates.

    Returns
    -------
        raw : mne.io.Raw, raw-file with electrode position.
    """
    subjects = pd.read_csv("../csv/name_match.csv")
    subject = subjects.INDI_ID.iloc[2]
    folder = "../working/"
    file_name = "%s/%s_eo-raw.fif" % (folder, subject)
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
    file = File("../data/leadfields/sa_nyhead.mat", "r")

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


def plot_mesh(file, field, plotter, color, opacity=1):
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

    pos = file["sa"][field]["vc"][:].T

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
