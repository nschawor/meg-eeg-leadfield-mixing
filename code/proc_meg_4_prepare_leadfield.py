""" Simulations: compute leadfield entries for MEG.
"""

import os
import numpy as np
import pandas as pd
import mne
import scipy.linalg

from params import CSV_FOLDER, LEADFIELD_DIR


# %% load required files
subjects_dir = os.path.dirname(LEADFIELD_DIR)
subject = "icbm152"

trans_fname = f"{LEADFIELD_DIR}/nyICBM-trans.fif"
trans_to_mni = mne.read_trans(trans_fname)

trans_meg_fname = f"{LEADFIELD_DIR}/nyICBM_meg-trans.fif"
trans_to_meg = mne.read_trans(trans_meg_fname)

trans_all = scipy.linalg.pinv(trans_to_mni["trans"]) @ trans_to_meg["trans"]
trans_all = mne.transforms.Transform("mri", "head", trans_all)
trans_all.save(f"{LEADFIELD_DIR}/nyICBM_allinone-trans.fif")

# sensor positions
raw = mne.io.read_raw_fif(f"{LEADFIELD_DIR}/meg_raw.fif")

# %% set up MEG source space
src_fname = f"{LEADFIELD_DIR}/meg_src.fif"
if not (os.path.exists(src_fname)):
    src = mne.setup_source_space(
        subject, spacing="oct7", add_dist=False, subjects_dir=subjects_dir
    )
    src.save(src_fname, overwrite=True)
else:
    src = mne.read_source_spaces(src_fname)

# %% set up MEG boundary element model
bem_fname = f"{LEADFIELD_DIR}/meg_bem.fif"
if not (os.path.exists(bem_fname)):
    conductivity = (0.3,)
    model = mne.make_bem_model(
        subject=subject, ico=4, subjects_dir=subjects_dir, conductivity=conductivity
    )
    bem = mne.make_bem_solution(model)
    mne.write_bem_solution(bem_fname, bem, overwrite=True)
else:
    bem = mne.read_bem_solution(bem_fname)

# %% compute lead field
fwd_fname = f"{LEADFIELD_DIR}/meg_fwd.fif"
fwd = mne.make_forward_solution(
    raw.info,
    trans=trans_to_meg,
    src=src,
    bem=bem,
    mindist=5.0,
    meg=True,
    eeg=False,
)
mne.write_forward_solution(fwd_fname, fwd, overwrite=True)

fwd_fixed = mne.convert_forward_solution(fwd, force_fixed=True)
LF = fwd_fixed["sol"]["data"].T
pos = fwd["source_rr"]

# %% load dipole location and find closest vertex
df = pd.read_csv(f"{CSV_FOLDER}/source_locations_all_hemispheres.csv")
dipole_coordinates = df[["mni-x", "mni-y", "mni-z"]].values


# transform source space to alpha location space and find closest vertex
pos_mni = mne.transforms.apply_trans(trans_all["trans"], pos)

nr_dipoles = len(dipole_coordinates)
nr_channels = LF.shape[1]
LF_selected = np.zeros((nr_dipoles, nr_channels))
dipole_coordinates_meg = np.zeros_like(dipole_coordinates, dtype="float32")

for i in range(nr_dipoles):
    chan = dipole_coordinates[i]/1000
    idx_source = np.argmin(np.sum((pos_mni - chan) ** 2, axis=1))
    LF_selected[i] = LF[idx_source]
    dipole_coordinates_meg[i] = pos[idx_source]

# save selected dipoles
data = {"LF": LF_selected, "dipole_coord": dipole_coordinates}
np.save(f"{CSV_FOLDER}/leadfield_selected_dipoles_meg.npy", data)


