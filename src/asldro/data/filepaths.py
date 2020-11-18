""" Constants with data file paths """
import os

# The data directory for the asldro module
DATA_DIR = os.path.dirname(os.path.realpath(__file__))


GROUND_TRUTH_DATA = {
    "hrgt_icbm_2009a_nls_v3": {
        "json": os.path.join(DATA_DIR, "hrgt_ICBM_2009a_NLS_v3.json"),
        "nii": os.path.join(DATA_DIR, "hrgt_ICBM_2009a_NLS_v3.nii.gz"),
    },
    "hrgt_icbm_2009a_nls_v4": {
        "json": os.path.join(DATA_DIR, "hrgt_ICBM_2009a_NLS_v4.json"),
        "nii": os.path.join(DATA_DIR, "hrgt_ICBM_2009a_NLS_v4.nii.gz"),
    },
}

ASL_BIDS_SCHEMA = os.path.join(DATA_DIR, "asl_bids_validator.json")

M0SCAN_BIDS_SCHEMA = os.path.join(DATA_DIR, "m0scan_bids_validator.json")
