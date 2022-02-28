# parameters and paths

EEG_DATA_FOLDER = "/cs/department2/data/eeg_lemon/"
MEG_DATA_FOLDER = "/cs/department2/data/meg_schoffelen/"
FIG_FOLDER = "../figures/"
CSV_FOLDER = "../csv/"
RESULTS_FOLDER = "../results/"
LEADFIELD_DIR = "../leadfields/"


# spectral parametrization
SPEC_PARAM_DIR_EEG = f"{RESULTS_FOLDER}/eeg/spec_param/"
SPEC_PARAM_DIR_MEG = f"{RESULTS_FOLDER}/meg/spec_param/"

SPEC_PARAM_CSV_EEG = f"{RESULTS_FOLDER}/eeg_center_frequencies.csv"
SPEC_PARAM_CSV_MEG = f"{RESULTS_FOLDER}/meg_center_frequencies.csv"

ALPHA_FMIN = 8
ALPHA_FMAX = 13
SPEC_FMIN = 2
SPEC_FMAX = 35
SPEC_NR_PEAKS = 5
SPEC_NR_SECONDS = 2
SNR_THRESHOLD = 0.5

# SSD
SSD_BANDWIDTH = 2
SSD_NR_COMPONENTS = 10
SSD_EEG_DIR = f"{RESULTS_FOLDER}/eeg/ssd/"
SSD_MEG_DIR = f"{RESULTS_FOLDER}/meg/ssd/"

# patterns
PATTERN_EEG_DIR = f"{RESULTS_FOLDER}/eeg/patterns/"
PATTERN_MEG_DIR = f"{RESULTS_FOLDER}/meg/patterns/"
