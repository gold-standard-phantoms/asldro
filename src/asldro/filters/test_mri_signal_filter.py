"""  MriSignalFilter tests """

from copy import deepcopy
import pytest
import numpy as np
import numpy.testing
from asldro.containers.image import NumpyImageContainer
from asldro.filters.basefilter import BaseFilter, FilterInputValidationError
from asldro.filters.mri_signal_filter import MriSignalFilter

TEST_VOLUME_DIMENSIONS = (32, 32, 32)
TEST_IMAGE_ONES = NumpyImageContainer(image=np.ones(TEST_VOLUME_DIMENSIONS))
TEST_IMAGE_101 = NumpyImageContainer(image=101 * np.ones(TEST_VOLUME_DIMENSIONS))
TEST_IMAGE_NEG = NumpyImageContainer(image=-1.0 * np.ones(TEST_VOLUME_DIMENSIONS))
TEST_IMAGE_SMALL = NumpyImageContainer(image=np.ones((8, 8, 8)))
TEST_IMAGE_COMPLEX = NumpyImageContainer(
    image=np.full(TEST_VOLUME_DIMENSIONS, 1 + 1j, dtype=np.complex128)
)

# test data dictionary, [0] in each tuple passes, after that should fail validation
TEST_DATA_DICT_GE = {
    "t1": (TEST_IMAGE_ONES, TEST_IMAGE_NEG, TEST_IMAGE_COMPLEX),
    "t2": (TEST_IMAGE_ONES, TEST_IMAGE_NEG, TEST_IMAGE_COMPLEX),
    "t2_star": (TEST_IMAGE_ONES, TEST_IMAGE_NEG, TEST_IMAGE_COMPLEX),
    "m0": (TEST_IMAGE_ONES, TEST_IMAGE_NEG, TEST_IMAGE_COMPLEX),
    "acq_contrast": ("ge", "gradient echo", "str", 435),
    "echo_time": (0.01, -0.01, "echo"),
    "repetition_time": (1.0, -1.0, "repeat"),
}

TEST_DATA_DICT_SE = {
    "t1": (TEST_IMAGE_ONES, TEST_IMAGE_NEG, TEST_IMAGE_COMPLEX),
    "t2": (TEST_IMAGE_ONES, TEST_IMAGE_NEG, TEST_IMAGE_COMPLEX),
    "m0": (TEST_IMAGE_ONES, TEST_IMAGE_NEG, TEST_IMAGE_COMPLEX),
    "acq_contrast": ("se", "spin echo", "str", 435),
    "echo_time": (0.01, -0.01, "echo"),
    "repetition_time": (1.0, -1.0, "repeat"),
}

# t1, t2, m0, t2_star, acq_contrast, acq_type, echo_time, repetition_time, expected
TIMECOURSE_PARAMS = (
    (
        1.4,
        0.1,
        1,
        0.07,
        "ge",
        np.linspace(0, 0.1, 11),
        1.0,
        [
            5.10458340443e-01,
            4.42505054073e-01,
            3.83597851904e-01,
            3.32532500207e-01,
            2.88265075378e-01,
            2.49890623115e-01,
            2.16624658533e-01,
            1.87787129023e-01,
            1.62788512008e-01,
            1.41117763393e-01,
            1.22331870348e-01,
        ],
    ),
    (
        2.0,
        0.5,
        1,
        0.35,
        "se",
        np.linspace(0, 1.0, 11),
        6.0,
        [
            9.50212931632e-01,
            7.77968549100e-01,
            6.36946776075e-01,
            5.21487913647e-01,
            4.26958192261e-01,
            3.49563802283e-01,
            2.86198635092e-01,
            2.34319624039e-01,
            1.91844682250e-01,
            1.57069141173e-01,
            1.28597336238e-01,
        ],
    ),
)


@pytest.fixture(name="mock_data")
def mock_data_fixture() -> dict:
    """ creates valid mock test data  """
    np.random.seed(0)
    return {
        "t1": NumpyImageContainer(
            image=np.random.normal(1.4, 0.1, TEST_VOLUME_DIMENSIONS)
        ),
        "t2": NumpyImageContainer(
            image=np.random.normal(0.1, 0.01, TEST_VOLUME_DIMENSIONS)
        ),
        "t2_star": NumpyImageContainer(
            image=np.random.normal(0.7, 0.01, TEST_VOLUME_DIMENSIONS)
        ),
        "m0": NumpyImageContainer(
            image=np.random.normal(100, 1, TEST_VOLUME_DIMENSIONS)
        ),
        "mag_enc": NumpyImageContainer(
            image=np.random.normal(1, 0.1, TEST_VOLUME_DIMENSIONS)
        ),
        "acq_contrast": "ge",
        "echo_time": 0.01,
        "repetition_time": 1.0,
    }


@pytest.mark.parametrize("validation_data", [TEST_DATA_DICT_GE, TEST_DATA_DICT_SE])
def test_mri_signal_filter_validate_inputs(validation_data: dict):
    """ Check a FilterInputValidationError is raised when the
    inputs to the MriSignalFilter are incorrect or missing
    """
    for inputs_key in validation_data:
        mri_signal_filter = MriSignalFilter()
        test_data = deepcopy(validation_data)
        # remove the corresponding key from test_data
        test_data.pop(inputs_key)

        for data_key in test_data:
            mri_signal_filter.add_input(data_key, test_data[data_key][0])

        # Key not defined

        with pytest.raises(FilterInputValidationError):
            mri_signal_filter.run()

        # Key has wrong data type
        mri_signal_filter.add_input(inputs_key, None)
        with pytest.raises(FilterInputValidationError):
            mri_signal_filter.run()

        # Data not in the valid range
        for test_value in validation_data[inputs_key][1:]:
            # re-initialise filter
            mri_signal_filter = MriSignalFilter()

            # add valid inputs
            for data_key in test_data:
                mri_signal_filter.add_input(data_key, test_data[data_key][0])

            # add invalid input and check a FilterInputValidationError is raised
            mri_signal_filter.add_input(inputs_key, test_value)
            with pytest.raises(FilterInputValidationError):
                mri_signal_filter.run()

    # Check optional parameters
    # 'mag_enc': optional
    test_data = deepcopy(validation_data)
    mri_signal_filter = MriSignalFilter()
    # add passing data
    for data_key in test_data:
        mri_signal_filter.add_input(data_key, test_data[data_key][0])

    # Wrong data type, should fail
    mri_signal_filter.add_input("mag_enc", "str")
    with pytest.raises(FilterInputValidationError):
        mri_signal_filter.run()

    # Numerically out-of-bounds
    mri_signal_filter = MriSignalFilter()
    # add passing data
    for data_key in test_data:
        mri_signal_filter.add_input(data_key, test_data[data_key][0])

    mri_signal_filter.add_input("mag_enc", TEST_IMAGE_COMPLEX)
    # negative values not allowed, so should fail
    with pytest.raises(FilterInputValidationError):
        mri_signal_filter.run()

    # Check correct use
    mri_signal_filter = MriSignalFilter()
    # add passing data
    for data_key in test_data:
        mri_signal_filter.add_input(data_key, test_data[data_key][0])

    mri_signal_filter.add_input("mag_enc", TEST_IMAGE_ONES)
    # Should run normally
    mri_signal_filter.run()


def test_mri_signal_filter_validate_inputs_ge_no_t2_star():
    """ Checks a FilterInputValidationError is raised when 
    'acq_contrast' == 'ge' and 't2_star' is not supplied """
    test_data = deepcopy(TEST_DATA_DICT_GE)
    # remove the 't2_star' entry
    test_data.pop("t2_star")
    mri_signal_filter = MriSignalFilter()
    for data_key in test_data:
        mri_signal_filter.add_input(data_key, test_data[data_key][0])

    with pytest.raises(FilterInputValidationError):
        mri_signal_filter.run()


def add_multiple_inputs_to_filter(input_filter: BaseFilter, input_data: dict):
    """ Adds the data held within the input_data dictionary to the filter's inputs """
    for key in input_data:
        input_filter.add_input(key, input_data[key])

    return input_filter


def mri_signal_gradient_echo_function(input_data: dict) -> np.ndarray:
    """ Function that calculates the gradient echo signal """
    t1: np.ndarray = input_data["t1"].image
    m0: np.ndarray = input_data["m0"].image
    t2_star: np.ndarray = input_data["t2_star"].image
    mag_enc: np.ndarray = input_data["mag_enc"].image
    echo_time: float = input_data["echo_time"]
    repetition_time: float = input_data["repetition_time"]
    return (m0 * (1 - np.exp(-repetition_time / t1)) + mag_enc) * np.exp(
        -echo_time / t2_star
    )


def mri_signal_spin_echo_function(input_data: dict) -> np.ndarray:
    """ Function that calculates the spin echo signal """
    t1: np.ndarray = input_data["t1"].image
    t2: np.ndarray = input_data["t2"].image
    m0: np.ndarray = input_data["m0"].image
    mag_enc: np.ndarray = input_data["mag_enc"].image
    echo_time: float = input_data["echo_time"]
    repetition_time: float = input_data["repetition_time"]
    return (m0 * (1 - np.exp(-repetition_time / t1)) + mag_enc) * np.exp(
        -echo_time / t2
    )


def test_mri_signal_filter_gradient_echo(mock_data):
    """ Tests the MriSignalFilter for 'acq_contrast' == 'ge':
    Gradient Echo """
    mock_data["acq_contrast"] = "ge"
    mri_signal_filter = MriSignalFilter()
    mri_signal_filter = add_multiple_inputs_to_filter(mri_signal_filter, mock_data)
    mri_signal_filter.run()

    ge_signal = mri_signal_gradient_echo_function(mock_data)
    numpy.testing.assert_array_equal(
        ge_signal, mri_signal_filter.outputs["image"].image
    )


def test_mri_signal_filter_spin_echo(mock_data):
    """ Tests the MriSignalFilter for 'acq_contrast' == 'se':
    Spin Echo """
    mock_data["acq_contrast"] = "se"
    mri_signal_filter = MriSignalFilter()
    mri_signal_filter = add_multiple_inputs_to_filter(mri_signal_filter, mock_data)
    mri_signal_filter.run()

    se_signal = mri_signal_spin_echo_function(mock_data)
    numpy.testing.assert_array_equal(
        se_signal, mri_signal_filter.outputs["image"].image
    )


@pytest.mark.parametrize(
    "t1, t2, m0, t2_star, acq_contrast, echo_time, repetition_time, expected",
    TIMECOURSE_PARAMS,
)
def test_mri_signal_timecourse(
    t1: float,
    t2: float,
    m0: float,
    t2_star: float,
    acq_contrast: str,
    echo_time: float,
    repetition_time: float,
    expected: float,
):
    """Tests the MriSignalFilter with timecourse data that is generated at multiple echo times

    Args:
        t1 (float): longitudinal relaxation time, s
        t2 (float): transverse relaxation time, s
        m0 (float): equilibrium magnetisation
        t2_star (float): transverse relaxation time inc. time invariant fields, s
        acq_contrast (str): signal model to use: 'ge' or 'se'
        echo_time (float): array of echo times, s
        repetition_time (float): repeat time, s
        expected (float): Array of expected values that the MriSignalFilter should generate
        Should be the same size and shape as 'echo_time'
    """
    mri_signal_timecourse = np.ndarray(echo_time.shape)
    for idx, te in np.ndenumerate(echo_time):
        params = {
            "t1": NumpyImageContainer(image=np.full((1, 1, 1), t1)),
            "t2": NumpyImageContainer(image=np.full((1, 1, 1), t2)),
            "t2_star": NumpyImageContainer(image=np.full((1, 1, 1), t2_star)),
            "m0": NumpyImageContainer(image=np.full((1, 1, 1), m0)),
            "mag_enc": NumpyImageContainer(image=np.zeros((1, 1, 1))),
            "acq_contrast": acq_contrast,
            "echo_time": te,
            "repetition_time": repetition_time,
        }

        mri_signal_filter = MriSignalFilter()
        mri_signal_filter = add_multiple_inputs_to_filter(mri_signal_filter, params)
        mri_signal_filter.run()
        mri_signal_timecourse[idx] = mri_signal_filter.outputs["image"].image
    # arrays should be equal to 9 decimal places
    numpy.testing.assert_array_almost_equal(mri_signal_timecourse, expected, 9)

