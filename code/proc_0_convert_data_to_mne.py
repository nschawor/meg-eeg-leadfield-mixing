import os
import mne
import pandas as pd
import numpy as np

from params import EEG_DATA_FOLDER, CSV_FOLDER

os.makedirs(EEG_DATA_FOLDER, exist_ok=True)
data_folder = '../data/preprocessed/'
subjects = [s for s in os.listdir(data_folder) if 'sub' in s]
subjects = np.sort(subjects)

df = pd.read_csv(f'{CSV_FOLDER}/name_match.csv')
print(len(df[df.INDI_ID.isin(subjects)])/len(df))
df[~df.INDI_ID.isin(subjects)].INDI_ID

missing = df[~df.INDI_ID.isin(subjects)].INDI_ID.to_list()

for subject in subjects:

    ec_file = f'{EEG_DATA_FOLDER}/%s_ec-raw.fif' % subject
    if not(os.path.exists(ec_file)):
        f_name = '%s/%s/%s_EC.set' % (data_folder, subject, subject)
        raw = mne.io.read_raw_eeglab(f_name)
        raw.save(ec_file, overwrite=True)

    eo_file = f'{EEG_DATA_FOLDER}/%s_eo-raw.fif'
    if not(os.path.exists(eo_file)):
        f_name = '%s/%s/%s_EO.set' % (data_folder, subject, subject)
        raw = mne.io.read_raw_eeglab(f_name)
        raw.save(eo_file % subject, overwrite=True)
