""" Test the data here is valid """
import json
from asldro.validators.ground_truth_json import validate_input
from asldro.data.filepaths import (
    HRGT_ICBM_2009A_NLS_V3_JSON,
    HRGT_ICBM_2009A_NLS_V4_JSON,
)


def test_hrgt_icbm_2009a_nls_v3():
    """ The high resolution ground truth json
    must be valid as per the schema """

    with open(HRGT_ICBM_2009A_NLS_V3_JSON) as file:
        validate_input(json.load(file))


def test_hrgt_icbm_2009a_nls_v4():
    """ The high resolution ground truth json
    must be valid as per the schema """

    with open(HRGT_ICBM_2009A_NLS_V4_JSON) as file:
        validate_input(json.load(file))
