""" GkmFilter tests """

from copy import deepcopy
import pytest
import numpy as np
import numpy.testing
from asldro.containers.image import BaseImageContainer, NumpyImageContainer
from asldro.filters.basefilter import BaseFilter, FilterInputValidationError
from asldro.filters.gkm_filter import (
    GkmFilter,
    KEY_PERFUSION_RATE,
    KEY_TRANSIT_TIME,
    KEY_M0,
    KEY_LABEL_TYPE,
    KEY_LABEL_DURATION,
    KEY_SIGNAL_TIME,
    KEY_LABEL_EFFICIENCY,
    KEY_LAMBDA_BLOOD_BRAIN,
    KEY_T1_ARTERIAL_BLOOD,
    KEYS_TUPLE,
    CASL,
    PCASL,
    PASL,
)

TEST_VOLUME_DIMENSIONS = (32, 32, 32)

TEST_DATA_DICT = {
    KEY_PERFUSION_RATE: NumpyImageContainer(image=np.zeros(TEST_VOLUME_DIMENSIONS)),
    KEY_TRANSIT_TIME: NumpyImageContainer(image=np.zeros(TEST_VOLUME_DIMENSIONS)),
    KEY_M0: NumpyImageContainer(image=np.zeros(TEST_VOLUME_DIMENSIONS)),
    KEY_LABEL_TYPE: "pCASL",
    KEY_LABEL_DURATION: 1.8,
    KEY_SIGNAL_TIME: 3.6,
    KEY_LABEL_EFFICIENCY: 0.85,
    KEY_LAMBDA_BLOOD_BRAIN: 0.9,
    KEY_T1_ARTERIAL_BLOOD: 1.6,
}


def test_gkm_filter_validate_inputs():
    """ Check a FilterInputValidationError is raised when the
    inputs to the GkmFilter are incorrect or missing
    """

    for inputs_key in KEYS_TUPLE:
        gkm_filter = GkmFilter()
        test_data = deepcopy(TEST_DATA_DICT)
        # remove the corresponding key from test_data
        test_data.pop(inputs_key)

        for data_key in test_data:
            gkm_filter.add_input(data_key, test_data[data_key])

        # Key not defined
        with pytest.raises(FilterInputValidationError):
            gkm_filter.run()

        # Key has wrong data
        gkm_filter.add_input(inputs_key, None)
        with pytest.raises(FilterInputValidationError):
            gkm_filter.run()

