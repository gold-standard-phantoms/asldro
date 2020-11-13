""" Test the data here is valid """
import json
from asldro.validators.ground_truth_json import validate_input
from asldro.data.filepaths import GROUND_TRUTH_DATA


def test_hrgt_icbm_2009a_nls_v3():
    """The high resolution ground truth json
    must be valid as per the schema"""

    with open(GROUND_TRUTH_DATA["hrgt_icbm_2009a_nls_v3"]["json"]) as file:
        validate_input(json.load(file))


def test_hrgt_icbm_2009a_nls_v4():
    """The high resolution ground truth json
    must be valid as per the schema"""

    with open(GROUND_TRUTH_DATA["hrgt_icbm_2009a_nls_v4"]["json"]) as file:
        validate_input(json.load(file))
