""" Constants with data file paths """
import os

# The data directory for the asldro module
DATA_DIR = os.path.dirname(os.path.realpath(__file__))


GROUND_TRUTH_DATA = {
    "hrgt_icbm_2009a_nls_3t": {
        "json": os.path.join(DATA_DIR, "hrgt_icbm_2009a_nls_3t.json"),
        "nii": os.path.join(DATA_DIR, "hrgt_icbm_2009a_nls_3t.nii.gz"),
    },
    "hrgt_icbm_2009a_nls_1.5t": {
        "json": os.path.join(DATA_DIR, "hrgt_icbm_2009a_nls_1.5t.json"),
        "nii": os.path.join(DATA_DIR, "hrgt_icbm_2009a_nls_1.5t.nii.gz"),
    },
}

ASL_BIDS_SCHEMA = os.path.join(DATA_DIR, "asl_bids_validator.json")

M0SCAN_BIDS_SCHEMA = os.path.join(DATA_DIR, "m0scan_bids_validator.json")
