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
TEST_IMAGE_SMALL = NumpyImageContainer(image=np.ones((8, 8, 8)))

# test data dictionary, [0] in each tuple passes, after that should fail validation
TEST_DATA_DICT_M0_IM = {
    "perfusion_rate": (TEST_IMAGE_ONES, TEST_IMAGE_NEG, TEST_IMAGE_SMALL),
    "transit_time": (TEST_IMAGE_ONES, TEST_IMAGE_NEG, TEST_IMAGE_SMALL),
    "m0": (TEST_IMAGE_ONES, TEST_IMAGE_NEG, TEST_IMAGE_SMALL),
    "label_type": ("pCASL", "casl", "PSL", "str"),
    "label_duration": (1.8, -0.1, 101.0),
    "signal_time": (3.6, -0.1, 101.0),
    "label_efficiency": (0.85, -0.1, 1.01),
    "lambda_blood_brain": (0.9, -0.1, 1.01),
    "t1_arterial_blood": (1.6, -0.1, 101.0),
}

TEST_DATA_DICT_M0_FLOAT = {
    "perfusion_rate": (TEST_IMAGE_ONES, TEST_IMAGE_NEG, TEST_IMAGE_SMALL),
    "transit_time": (TEST_IMAGE_ONES, TEST_IMAGE_NEG, TEST_IMAGE_SMALL),
    "m0": (1.0, -1.0, int(1)),
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


def add_multiple_inputs_to_filter(filter: BaseFilter, input_data: dict):
    """ Adds the data held within the input_data dictionary to the filter's inputs """
    for key in input_data:
        filter.add_input(key, input_data[key])

    return filter


@pytest.mark.parametrize(
    "validation_data", [TEST_DATA_DICT_M0_IM, TEST_DATA_DICT_M0_FLOAT]
)
def test_gkm_filter_validate_inputs(validation_data: dict):
    """ Check a FilterInputValidationError is raised when the
    inputs to the GkmFilter are incorrect or missing
    """

    for inputs_key in validation_data:
        gkm_filter = GkmFilter()
        test_data = deepcopy(validation_data)
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
        for test_value in validation_data[inputs_key][1:]:
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
    test_gkm_filter_validate_inputs(TEST_DATA_DICT_M0_IM)
