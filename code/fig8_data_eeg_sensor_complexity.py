"""Data: compute sensor complexity measure for all participants."""
import os
import pandas as pd
import mne
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import matplotlib.gridspec as gridspec
import seaborn as sns

from helper import get_electrodes
from complexity import compute_sensor_complexity
from params import FIG_FOLDER, RESULTS_FOLDER, SSD_NR_COMPONENTS, PATTERN_EEG_DIR

from matplotlib import rc
rc("font", **{"family": "sans-serif", "sans-serif": ["Arial"]})

# load one subject for electrode locations
raw = get_electrodes()
conditions = ["ec", "eo"]

subjects = [s.split("_")[0] for s in os.listdir(PATTERN_EEG_DIR)]
subjects = np.sort(np.unique(subjects))

# %%
fig = plt.figure()
gs0 = gridspec.GridSpec(1, 2, figure=fig, top=0.85, bottom=0.25, wspace=0.35)
gs00 = gridspec.GridSpecFromSubplotSpec(
    2, 2, subplot_spec=gs0[0], height_ratios=[10, 1]
)

for i_cond, condition in enumerate(conditions):

    dfs = []
    for i_sub, subject in enumerate(subjects):

        df_file_name = f"{PATTERN_EEG_DIR}/{subject}_{condition}_patterns.csv"

        # compute complexity
        df_patterns = pd.read_csv(df_file_name)
        weighted_patterns = df_patterns.values.T
        metric = compute_sensor_complexity(weighted_patterns, SSD_NR_COMPONENTS)

        channels = df_patterns.columns
        dff = pd.DataFrame(metric[:, np.newaxis].T, columns=channels)
        dff["subject"] = subject
        dfs.append(dff)

    print(condition, len(dfs))
    df = pd.concat(dfs, sort=False)

    metric_file = f"{RESULTS_FOLDER}/eeg_sensor_complexity_{condition}.csv"
    df.to_csv(metric_file, index=False)
    df = df.drop("subject", axis=1)

    # assert(np.all(raw.ch_names == df.columns))
    raw.reorder_channels(df.columns.to_list())

    ax1 = fig.add_subplot(gs00[0, i_cond])
    mean_metric = df.apply("mean")
    im = mne.viz.plot_topomap(
        mean_metric, raw.info, axes=ax1, vmin=1.8, vmax=2.05, show=False
    )
    ax2 = fig.add_subplot(gs00[1, :])
    cb = plt.colorbar(im[0], orientation="horizontal", cax=ax2)
    cb.set_label("mean sensor complexity\n across participants")

    if i_cond == 0:
        ax1.set_title("sensor complexity\n eyes closed")
    if i_cond == 1:
        ax1.set_title("sensor complexity\n eyes open")


# %%

raw = get_electrodes()

df_ec = pd.read_csv(f"{RESULTS_FOLDER}/eeg_sensor_complexity_ec.csv")
df_eo = pd.read_csv(f"{RESULTS_FOLDER}/eeg_sensor_complexity_eo.csv")
df_eo.drop(["subject"], axis=1, inplace=True)
df_ec.drop(["subject"], axis=1, inplace=True)
p = np.zeros((len(df_eo.columns),))
w = np.zeros((len(df_eo.columns),))
for i_chan, chan in enumerate(df_eo.columns):
    w[i_chan], p[i_chan] = scipy.stats.wilcoxon(df_eo[chan], df_ec[chan])
    if chan in ["Cz", "Oz"]:
        print(chan, p[i_chan] * len(raw.ch_names))

raw.reorder_channels(list(df_eo.columns))
mask = np.zeros_like(p, dtype="bool")
mask[p < 0.05 / len(p)] = True


# %%
gs01 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs0[1], wspace=0.35)

chan = "Oz"
ax0 = fig.add_subplot(gs01[0])
df = pd.concat((df_eo[chan], df_ec[chan]), axis=1)
df.columns = ("eo", "ec")
sns.swarmplot(data=df, ax=ax0, size=3)
sns.boxplot(data=df, ax=ax0, zorder=20, boxprops=dict(alpha=0.5))

chan = "Cz"
ax1 = fig.add_subplot(gs01[1])
df = pd.concat((df_eo[chan], df_ec[chan]), axis=1)
df.columns = ("eo", "ec")
sns.swarmplot(data=df, ax=ax1, size=3)
sns.boxplot(data=df, ax=ax1, zorder=20, boxprops=dict(alpha=0.5))
ax0.set_xticklabels(["eyes\nopen", "eyes\nclosed"])

x1, x2 = 0, 1
y, h, col = 2.3, 0.25, "k"
ax0.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c=col)
ax0.text((x1 + x2) * 0.5, y + h, "***", ha="center", va="bottom", color=col)

ax0.set(ylabel="sensor complexity", title="occipital \nelectrode Oz")
ax1.sharey(ax0)
ax1.set(
    title="sensorimotor \nelectrode Cz",
)
ax1.set_xticklabels(["eyes\nopen", "eyes\nclosed"])

x1, x2 = 0, 1
y, h, col = 2.3, 0.25, "k"
ax1.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c=col)
ax1.text((x1 + x2) * 0.5, y + h, "***", ha="center", va="bottom", color=col)

fig.set_size_inches(8.9, 3.6)
fig.tight_layout()
plot_file = f"{FIG_FOLDER}/fig8_mean_sensor_complexity_eeg.png"
fig.savefig(plot_file, dpi=200)
fig.show()
