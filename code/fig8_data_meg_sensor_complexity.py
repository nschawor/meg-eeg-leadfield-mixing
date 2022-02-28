"""Data: compute sensor complexity measure for all participants."""
import os
import pandas as pd
import mne
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import matplotlib.gridspec as gridspec
import seaborn as sns

from helper import load_meg_data
from complexity import compute_sensor_complexity
from params import FIG_FOLDER, RESULTS_FOLDER, SSD_NR_COMPONENTS, PATTERN_MEG_DIR

from matplotlib import rc
rc("font", **{"family": "sans-serif", "sans-serif": ["Arial"]})

# %% specify participants
subjects = [s.split("_")[0] for s in os.listdir(PATTERN_MEG_DIR)]
subjects = np.sort(subjects)

# %% collect measure for all participants
dfs = []
for i_sub, subject in enumerate(subjects):
    print(subject)
    df_file_name = f"{PATTERN_MEG_DIR}/{subject}_patterns.csv"

    # sensor complexity
    df_patterns = pd.read_csv(df_file_name)
    weighted_patterns = df_patterns.values.T
    metric = compute_sensor_complexity(weighted_patterns, SSD_NR_COMPONENTS)

    channels = df_patterns.columns
    dff = pd.DataFrame(metric[:, np.newaxis].T, columns=channels)
    dff["subject"] = subject
    dfs.append(dff)

print(f"total number of participants = {len(dfs)}")
df = pd.concat(dfs, sort=False)

metric_file = f"{RESULTS_FOLDER}/meg_sensor_complexity.csv" 
df.to_csv(metric_file, index=False)
df = df.drop("subject", axis=1)

raw = load_meg_data(subject="V1006")
df = df.drop(["MLF62", "MLT37"], axis=1)
raw.reorder_channels(df.columns.to_list())
assert(np.all(raw.ch_names == df.columns))

# %% create figure
fig = plt.figure()
gs0 = gridspec.GridSpec(1, 1, figure=fig, top=0.85, bottom=0.05, wspace=0.35)
ax1 = fig.add_subplot(gs0[0, 0])
mean_metric = df.apply("mean")
im = mne.viz.plot_topomap(
    mean_metric, raw.info, axes=ax1, vmin=1.8, vmax=2.05, show=False
)
ax1.set_title("MEG sensor complexity\n eyes open")

fig.set_size_inches(3, 3)
fig.tight_layout()
fig.show()
plot_file = f"{FIG_FOLDER}/fig8_mean_sensor_complexity_meg.png"
fig.savefig(plot_file, dpi=200, transparent=True)
