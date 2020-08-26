""" GkmFilter tests """

from copy import deepcopy
import pytest
import numpy as np
import numpy.testing
from asldro.containers.image import BaseImageContainer, NumpyImageContainer
from asldro.filters.basefilter import BaseFilter, FilterInputValidationError
from asldro.filters.gkm_filter import GkmFilter

TEST_VOLUME_DIMENSIONS = (32, 32, 32)
TEST_IMAGE_ONES = NumpyImageContainer(image=np.ones(TEST_VOLUME_DIMENSIONS))
TEST_IMAGE_NEG = NumpyImageContainer(image=-1.0 * np.ones(TEST_VOLUME_DIMENSIONS))
TEST_IMAGE_101 = NumpyImageContainer(image=100.0 + np.ones(TEST_VOLUME_DIMENSIONS))

# test data dictionary, [0] in each tuple passes, after that should fail validation
TEST_DATA_DICT = {
    "perfusion_rate": (TEST_IMAGE_ONES, TEST_IMAGE_NEG),
    "transit_time": (TEST_IMAGE_ONES, TEST_IMAGE_NEG),
    "m0": (TEST_IMAGE_ONES, TEST_IMAGE_NEG),
    "label_type": ("pCASL", "casl", "PSL", "str"),
    "label_duration": (1.8, -0.1, 101.0),
    "signal_time": (3.6, -0.1, 101.0),
    "label_efficiency": (0.85, -0.1, 1.01),
    "lambda_blood_brain": (0.9, -0.1, 1.01),
    "t1_arterial_blood": (1.6, -0.1, 101.0),
}
CASL = "CASL"
PCASL = "PCASL"
PASL = "PASL"


def test_gkm_filter_validate_inputs():
    """ Check a FilterInputValidationError is raised when the
    inputs to the GkmFilter are incorrect or missing
    """

    for inputs_key in TEST_DATA_DICT:
        gkm_filter = GkmFilter()
        test_data = deepcopy(TEST_DATA_DICT)
        # remove the corresponding key from test_data
        test_data.pop(inputs_key)

        for data_key in test_data:
            gkm_filter.add_input(data_key, test_data[data_key][0])

        # Key not defined
        with pytest.raises(FilterInputValidationError):
            gkm_filter.run()

        # Key has wrong data type
        gkm_filter.add_input(inputs_key, None)
        with pytest.raises(FilterInputValidationError):
            gkm_filter.run()

        # Data not in the valid range
        for test_value in TEST_DATA_DICT[inputs_key][1:]:
            # re-initialise filter
            gkm_filter = GkmFilter()

            # add valid inputs
            for data_key in test_data:
                gkm_filter.add_input(data_key, test_data[data_key][0])

            # add invalid input and check a FilterInputValidationError is raised
            gkm_filter.add_input(inputs_key, test_value)
            with pytest.raises(FilterInputValidationError):
                gkm_filter.run()


if __name__ == "__main__":
    test_gkm_filter_validate_inputs()
