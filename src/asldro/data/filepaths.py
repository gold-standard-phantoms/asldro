""" Constants with data file paths """
import os

# The data directory for the asldro module
DATA_DIR = os.path.dirname(os.path.realpath(__file__))


HRGT_ICBM_2009A_NLS_V3_NIFTI = os.path.join(DATA_DIR, "hrgt_ICBM_2009a_NLS_v3.nii.gz")

HRGT_ICBM_2009A_NLS_V3_JSON = os.path.join(DATA_DIR, "hrgt_ICBM_2009a_NLS_v3.json")
