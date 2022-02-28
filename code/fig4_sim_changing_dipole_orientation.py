"""Simulation: impact of changing dipole orientation on contributions to frontal channels."""
import numpy as np
import os
import matplotlib.pyplot as plt
import mne
import matplotlib.gridspec as gridspec
from scipy.spatial.transform import Rotation as R
import seaborn as sns
from matplotlib import rc

from params import FIG_FOLDER, LEADFIELD_DIR

rc("font", **{"family": "sans-serif", "sans-serif": ["Arial"]})

# %% set parameters and load sample data
data_path = mne.datasets.sample.data_path()
subject = "sample"
subjects_dir = data_path + "/subjects"
os.environ["SUBJECTS_DIR"] = subjects_dir

trans = mne.read_trans(data_path + "/MEG/sample/sample_audvis_raw-trans.fif")
raw = mne.io.read_raw_fif(data_path + "/MEG/sample/sample_audvis_raw.fif")
raw.load_data()
raw.pick_types(eeg=True, meg=False)

# %% compute source space
src_fname = f"{LEADFIELD_DIR}/eeg_demo_src.fif"
if not (os.path.exists(src_fname)):
    src = mne.setup_source_space(
        subject, spacing="oct6", add_dist=False, subjects_dir=subjects_dir
    )
    src.save(src_fname)
else:
    src = mne.read_source_spaces(src_fname)

# %% compute boundary element model
bem_fname = f"{LEADFIELD_DIR}/eeg_demo_bem.fif"
if not (os.path.exists(bem_fname)):
    conductivity = (0.3, 0.006, 0.3)  # three layers
    model = mne.make_bem_model(
        subject=subject, ico=4, subjects_dir=subjects_dir, conductivity=conductivity
    )
    bem = mne.make_bem_solution(model)
    mne.write_bem_solution(bem_fname, bem)
else:
    bem = mne.read_bem_solution(bem_fname)

# %% compute lead field
fwd_fname = f"{LEADFIELD_DIR}/eeg_demo_fwd.fif"
if not (os.path.exists(fwd_fname)):
    fwd = mne.make_forward_solution(
        raw.info, trans, src, bem, mindist=5.0, meg=False, eeg=True
    )
    mne.write_forward_solution(fwd_fname, fwd)
else:
    fwd = mne.read_forward_solution(fwd_fname)

# plot dipoles in 3d
lh = fwd["src"][0]  # Visualize the left hemisphere

# create dipole
dip_ori = -np.array([[0.68278979, 0.72773717, 0.06478193]])
dip_pos = np.array([[-0.04212194, 0.02041447, 0.11716272]])
dip_len = len(dip_pos)
dip_times = [0]
actual_amp = np.ones(dip_len)
actual_gof = np.ones(dip_len)

colors = ("m", "r", "y")

# create topomaps
dip_ori = np.array([[0.68278979, 0.72773717, 0.06478193]])
around = np.array([[0.0, -0.72773717, -0.06478193]])

nr_steps = 16
leadfields = np.zeros((len(raw.ch_names), nr_steps))
steps = np.arange(nr_steps)
for i in steps:
    n = np.array([0.08394118, -0.08272625, 0.04459249])
    r = R.from_rotvec(i * np.pi / 2 * n)
    dip_ori1 = r.apply(dip_ori)
    dipoles = mne.Dipole(dip_times, dip_pos, actual_amp, dip_ori1, actual_gof)
    fwd1, stc = mne.make_forward_dipole(dipoles, bem, raw.info, trans=trans)
    fwd1 = mne.forward.convert_forward_solution(fwd1, force_fixed=True)
    leadfield = fwd1["sol"]["data"]
    leadfields[:, i] = leadfield[:, 0]

# create electrode positions
exclude = ["EEG 017", "EEG 025", "EEG 036"]
picks = mne.pick_channels(ch_names=raw.ch_names, include=[], exclude=exclude)
pos = mne.channels.layout._auto_topomap_coords(
    raw.info, picks=picks, ignore_overlap=True, to_sphere=True, sphere=None
)
pos[:, 1] -= 0.01  # move slightly upwards for visualization

# %% create figure
fig = plt.figure()

gs = gridspec.GridSpec(6, 2, hspace=0.5, wspace=0.5, top=0.95)
fig = plt.figure()
fig.set_size_inches(6.7, 6.5)

picks2 = [1, 18]
mask = np.zeros((len(picks),), dtype="bool")
mask[picks2] = True
labels = ["tangential1", "radial", "tangential2"]
leadfields_picked = leadfields[picks, :]
steps_selected = [2, 10, 15]
for ii, i_angle in enumerate(steps_selected):

    ax1 = plt.subplot(gs[2*ii:2*ii+2, 0])
    vmin = np.min(leadfields)
    vmax = np.max(leadfields)
    mne.viz.plot_topomap(
        leadfields_picked[:, i_angle],
        pos,
        axes=ax1,
        outlines="head",
        mask=mask,
    )
    ax1.set_ylabel(labels[ii], color=colors[ii])


ax = plt.subplot(gs[1:3, 1])
angle = np.linspace(0, np.pi, 16)
angle_ticks = (0, np.pi / 2, np.pi)
angle_labels = (0, "0.5$\pi$", "$\pi$")
LF_norm = leadfields_picked / leadfields_picked[picks2[1]].max()
ax.plot(angle, 100 * np.abs(LF_norm[picks2[1], :]), color="k")
for i in range(3):
    ax.plot(
        angle[steps_selected[i]],
        100 * np.abs(LF_norm[picks2[1], steps_selected[i]]),
        ".",
        markersize=10,
        color=colors[i],
    )

ax.set(
    title="sensorimotor electrode",
    xlabel="dipole angle",
    ylabel="absolute lead field \ncontribution [% of maximum]",
    xticks=angle_ticks,
    xticklabels=angle_labels,
)
ax = plt.subplot(gs[4:, 1])
ax.plot(angle, 100 * np.abs(LF_norm[picks2[0], :]), color="k")
ax.set(
    title="frontal electrode",
    xlabel="dipole angle",
    ylabel="absolute lead field \ncontribution [% of maximum]",

    ylim=(0, 20),
    xticks=angle_ticks,
    xticklabels=angle_labels,
)

for i in range(3):
    ax.plot(
        angle[steps_selected[i]],
        100 * np.abs(LF_norm[picks2[0], steps_selected[i]]),
        ".",
        markersize=10,
        color=colors[i],
    )
sns.despine(fig=fig)

fig.set_size_inches(6, 6)
fig.tight_layout()
fig.savefig(f"{FIG_FOLDER}/fig4_dipole_demo.png", dpi=200, transparent=True)
fig.show()
