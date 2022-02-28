"""Simulation, step #2: create alpha-maps."""
import numpy as np
import mne
import pandas as pd
import matplotlib.pyplot as plt
import cycler

from helper import get_colors_dict, compute_coordinates_tiny_topos
from complexity import compute_norm_coefficients
from params import FIG_FOLDER, CSV_FOLDER, LEADFIELD_DIR


def plot_alpha_map(df, raw, colors, gain_factors, plot_file, pie_size=0.11):

    # re-ordering pie wedges to make left and right hemisphere symmetric
    nr_sources = int(len(df) / 2)  # per hemisphere
    idx_wedges = np.hstack(
        [np.arange(nr_sources), nr_sources + np.arange(nr_sources)[::-1]]
    )

    scalings = df.source_type.replace(gain_factors).values

    fig_size = (5, 5)
    fig = plt.figure()
    pos = mne.channels.layout._find_topomap_coords(raw.info, picks=None)
    topo_pos = compute_coordinates_tiny_topos(pos)

    # compute contribution
    weighted_patterns = raw._data * scalings
    nr_components = raw._data.shape[1]
    M = compute_norm_coefficients(weighted_patterns, nr_components)

    nr_channels = len(raw.ch_names)
    for i in range(nr_channels):
        idx = np.nonzero(np.array(raw.ch_names) == raw.ch_names[i])[0][0]
        ax1 = plt.Axes(
            fig, rect=[topo_pos[idx, 0], topo_pos[idx, 1], pie_size, pie_size]
        )
        ax1 = fig.add_axes(ax1)
        ax1.set_prop_cycle(cycler.cycler("color", colors[idx_wedges]))
        wedges, texts = ax1.pie(M[i, idx_wedges], startangle=90, normalize=True)
        for w in wedges:
            w.set_linewidth(0.005)
            w.set_edgecolor("w")

    fig.set_size_inches(fig_size[0], fig_size[1])
    fig.show()
    fig.savefig(plot_file, dpi=200, transparent=True)
    return fig


def define_scaling(
    alpha_scale=1, mu_scale=0.5, parietal_scale=0.5, temporal_scale=0.25
):
    scalings_dict = {
        "parietal": parietal_scale,
        "somatosensory": mu_scale,
        "occipital": alpha_scale,
        "temporal": temporal_scale,
    }
    return scalings_dict


def main():

    raw = mne.io.read_raw_fif(f"{LEADFIELD_DIR}/meg_raw.fif")

    # load leadfield entries for dipoles
    dipole_fname = f"{CSV_FOLDER}/leadfield_selected_dipoles_meg.npy"
    data = np.load(dipole_fname, allow_pickle=True).item()
    raw._data = data["LF"].T
    df = pd.read_csv(f"{CSV_FOLDER}/source_locations_all_hemispheres.csv")

    # set colors for left and right hemisphere
    df_left = pd.read_csv(f"{CSV_FOLDER}/source_locations.csv")
    colors_left = get_colors_dict()
    colors_left = df_left.source_type.replace(colors_left).values

    df_right = pd.read_csv(f"{CSV_FOLDER}/source_locations.csv")
    colors_right = get_colors_dict("right")
    colors_right = df_right.source_type.replace(colors_right).values
    colors = np.hstack((colors_left, colors_right))

    # make alpha pie figures
    for alpha_scale in (0.5, 2):
        gain_factors = define_scaling(alpha_scale=alpha_scale)
        plot_file = f"{FIG_FOLDER}/fig3_meg_sim_alpha_scale_{alpha_scale}.png"
        plot_alpha_map(
            df,
            raw,
            colors=colors,
            gain_factors=gain_factors,
            pie_size=0.05,
            plot_file=plot_file,
            fig_size=(6, 5)
        )


if __name__ == "__main__":
    main()
