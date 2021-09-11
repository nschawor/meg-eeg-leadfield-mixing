"""Simulation: impact of changing dipole orientation on contributions to frontal channels."""
import numpy as np
import os
import matplotlib.pyplot as plt
import mne
from mne import read_proj
from mne.io import read_raw_fif
from mne.viz import plot_alignment, set_3d_view
import matplotlib.gridspec as gridspec
from scipy.spatial.transform import Rotation as R

mne.set_config("MNE_3D_BACKEND", "pyvista")
cpos_top = [
    (0.0017779221466052838, 0.12519202230017865, 0.3186250725973344),
    (-0.0037940865157990097, 0.02049481965752055, 0.07171139167059473),
    (0.9993059051070611, -0.03658070608518725, -0.007039883554899845),
]

cpos_side = [
    (-0.31808730896405196, 0.0921716482400515, 0.166409509474727),
    (-0.0037940865157990097, 0.02049481965752055, 0.07171139167059473),
    (0.33440965034380143, 0.27568906175017055, 0.9012023784856801),
]

data_path = mne.datasets.sample.data_path()
subject = "sample"
subjects_dir = data_path + "/subjects"
os.environ["SUBJECTS_DIR"] = subjects_dir

trans = mne.read_trans(data_path + "/MEG/sample/sample_audvis_raw-trans.fif")
raw = mne.io.read_raw_fif(data_path + "/MEG/sample/sample_audvis_raw.fif")
raw.load_data()
raw.pick_types(eeg=True, meg=False)

#
fig = plot_alignment(
    raw.info,
    trans,
    subject=subject,
    dig=False,
    eeg=["original", "projected"],
    meg=[],
    coord_frame="head",
    subjects_dir=subjects_dir,
)

# compute source space
src_fname = "../data/src.fif"
if not (os.path.exists(src_fname)):
    src = mne.setup_source_space(
        subject, spacing="oct6", add_dist=False, subjects_dir=subjects_dir
    )
    src.save()
else:
    src = mne.read_source_spaces("../data/src.fif")

# compute boundary element model
bem_fname = "../data/bem.fif"
if not (os.path.exists(bem_fname)):
    conductivity = (0.3, 0.006, 0.3)  # three layers
    model = mne.make_bem_model(
        subject=subject, ico=4, subjects_dir=subjects_dir, conductivity=conductivity
    )
    bem = mne.make_bem_solution(model)
    mne.write_bem_solution(bem_fname, bem)
else:
    bem = mne.read_bem_solution(bem_fname)

# compute lead field
fwd_fname = "../data/fwd.fif"
if not (os.path.exists(fwd_fname)):
    fwd = mne.make_forward_solution(
        raw.info, trans, src, bem, mindist=5.0, meg=False, eeg=True
    )
    mne.write_forward_solution(fwd_fname, fwd)
else:
    fwd = mne.read_forward_solution(fwd_fname)

# plot dipoles in 3d
lh = fwd["src"][0]  # Visualize the left hemisphere
plt.ion()
images = []
steps = np.arange(32)

# create dipole
dip_ori = -np.array([[0.68278979, 0.72773717, 0.06478193]])
dip_pos = np.array([[-0.04212194, 0.02041447, 0.11716272]])
dip_len = len(dip_pos)
dip_times = [0]
actual_amp = np.ones(dip_len)
actual_gof = np.ones(dip_len)

fig = mne.viz.create_3d_figure(size=(500, 500))
fig = mne.viz.plot_alignment(
    subject=subject,
    subjects_dir=subjects_dir,
    trans=trans,
    surfaces="white",
    coord_frame="head",
    fig=fig,
)

# plot for a few selected dipole orientations
steps_selected = [2, 10, 15]
colors = ("m", "r", "y")
for ii, i in enumerate(steps_selected):

    n = np.array([0.08394118, -0.08272625, 0.04459249])
    r = R.from_rotvec(i * np.pi / 2 * n)
    dip_ori1 = r.apply(dip_ori)
    dipole = mne.Dipole(dip_times, dip_pos, actual_amp, dip_ori1, actual_gof)

    fig = mne.viz.plot_dipole_locations(
        dipoles=dipole,
        trans=trans,
        mode="arrow",
        subject=subject,
        subjects_dir=subjects_dir,
        coord_frame="head",
        scale=7e-3,
        fig=fig,
        color=colors[ii],
    )
fig.plotter.camera_position = cpos_side
im = fig.plotter.screenshot()


# plot topomaps
dip_ori = np.array([[0.68278979, 0.72773717, 0.06478193]])
around = np.array([[0.0, -0.72773717, -0.06478193]])

leadfields = np.zeros((59, 16))
steps = np.arange(16)
for i in steps:
    n = np.array([0.08394118, -0.08272625, 0.04459249])
    r = R.from_rotvec(i * np.pi / 2 * n)
    dip_ori1 = r.apply(dip_ori)
    dipoles = mne.Dipole(dip_times, dip_pos, actual_amp, dip_ori1, actual_gof)
    fwd1, stc = mne.make_forward_dipole(dipoles, bem, raw.info, trans=trans)
    fwd1 = mne.forward.convert_forward_solution(fwd1, force_fixed=True)
    leadfield = fwd1["sol"]["data"]
    leadfields[:, i] = leadfield[:, 0]

plt.ion()

exclude = ["EEG 017", "EEG 025", "EEG 036"]
picks = mne.pick_channels(ch_names=raw.ch_names, include=[], exclude=exclude)
pos = mne.channels.layout._auto_topomap_coords(
    raw.info, picks=picks, ignore_overlap=True, to_sphere=True, sphere=None
)

pos[:, 1] -= 0.01 # move slightly upwards for visualization

fig = plt.figure()
leadfields_picked = leadfields[picks, :]

plt.close("all")
gs = gridspec.GridSpec(4, 4, hspace=0.4, wspace=0.4, top=0.95)
fig = plt.figure()
fig.set_size_inches(6.7, 6.5)

# 3d brain
ax0 = plt.subplot(gs[0:3, :3])
ax0.imshow(im)
xlim = ax0.get_xlim()
ax0.set_xlim(xlim[0], xlim[1] - 50)
ax0.axis("off")

picks2 = [1, 18]
mask = np.zeros((len(picks),), dtype="bool")
mask[picks2] = True
labels = ["tangential1", "radial", "tangential2"]
for ii, i_angle in enumerate(steps_selected):

    ax1 = plt.subplot(gs[ii, -1])
    vmin = np.min(leadfields)
    vmax = np.max(leadfields)
    mne.viz.plot_topomap(
        leadfields_picked[:, i_angle],
        pos,
        axes=ax1,
        outlines="head",
        mask=mask,
    )
    ax1.set_title(labels[ii], color=colors[ii])

ax = plt.subplot(gs[3, :2])
angle = np.linspace(0, np.pi, 16)
angle_ticks = (0, np.pi / 2, np.pi)
angle_labels = (0, "0.5$pi$", "$pi$")
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
    ylabel="absolute lead field contribution % of maximum]",
    xticks=angle_ticks,
    xticklabels=angle_labels,
)
ax = plt.subplot(gs[3, 2:])
ax.plot(angle, 100 * np.abs(LF_norm[picks2[0], :]), color="k")
ax.set(
    title="frontal electrode",
    xlabel="dipole angle",
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

fig.set_size_inches(7.8, 7)
fig.savefig("../figures/fig4_dipole_demo.png", dpi=200)
fig.show()
