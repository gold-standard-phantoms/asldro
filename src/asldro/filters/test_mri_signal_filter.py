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
    "repetition_time": (1.0, -1.0, 0.009, "repeat"),
    "excitation_flip_angle": (90.0, "str", 90),
}

TEST_DATA_DICT_SE = {
    "t1": (TEST_IMAGE_ONES, TEST_IMAGE_NEG, TEST_IMAGE_COMPLEX),
    "t2": (TEST_IMAGE_ONES, TEST_IMAGE_NEG, TEST_IMAGE_COMPLEX),
    "m0": (TEST_IMAGE_ONES, TEST_IMAGE_NEG, TEST_IMAGE_COMPLEX),
    "acq_contrast": ("se", "spin echo", "str", 435),
    "echo_time": (0.01, -0.01, "echo"),
    "repetition_time": (1.0, -1.0, 0.009, "repeat"),
}

TEST_DATA_DICT_IR = {
    "t1": (TEST_IMAGE_ONES, TEST_IMAGE_NEG, TEST_IMAGE_COMPLEX),
    "t2": (TEST_IMAGE_ONES, TEST_IMAGE_NEG, TEST_IMAGE_COMPLEX),
    "m0": (TEST_IMAGE_ONES, TEST_IMAGE_NEG, TEST_IMAGE_COMPLEX),
    "acq_contrast": ("ir", "inversion recovery", "str", 435),
    "echo_time": (0.01, -0.01, "echo"),
    "repetition_time": (1.5, -1.0, 1.0, "repeat"),
    "excitation_flip_angle": (90.0, "str", 90),
    "inversion_flip_angle": (180.0, "str", 90),
    "inversion_time": (1.0, -1.0, 1, "str"),
}

# t1, t2, m0, t2_star, acq_contrast, echo_time, repetition_time, expected
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
            5.10469685712e-01,
            4.42514889036e-01,
            3.83606377616e-01,
            3.32539890958e-01,
            2.88271482257e-01,
            2.49896177097e-01,
            2.16629473157e-01,
            1.87791302715e-01,
            1.62792130089e-01,
            1.41120899827e-01,
            1.22334589253e-01,
        ],
    ),
    (
        2.0,
        0.5,
        1,
        0.07,
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

# t1, t2, m0, t2_star,  echo_time, repetition_time,
# excitation_angle, inversion_angle, inversion_time, expected
IR_TIMECOURSE_PARAMS = (
    (
        1.4,
        0.1,
        1.0,
        0.07,
        0.01,
        10.0,
        90.0,
        180.0,
        np.concatenate([np.linspace(0, 1.0, 11), np.linspace(2.0, 8.0, 7)]),
        [
            -9.04122152813005e-01,
            -7.79368199974629e-01,
            -6.63214437865866e-01,
            -5.55067993243801e-01,
            -4.54376863898874e-01,
            -3.60627101119449e-01,
            -2.73340186389445e-01,
            -1.92070588929139e-01,
            -1.16403491612320e-01,
            -4.59526736523874e-02,
            1.96414607498095e-02,
            4.71862233171678e-01,
            6.93243140589206e-01,
            8.01618317400608e-01,
            8.54672481311639e-01,
            8.80644704759052e-01,
            8.93359190127883e-01,
            8.99583460395753e-01,
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
        "excitation_flip_angle": 90.0,
    }


@pytest.mark.parametrize(
    "validation_data", [TEST_DATA_DICT_GE, TEST_DATA_DICT_SE, TEST_DATA_DICT_IR]
)
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

    # 'image_flavour': optional
    test_data = deepcopy(validation_data)
    mri_signal_filter = MriSignalFilter()
    # add passing data
    for data_key in test_data:
        mri_signal_filter.add_input(data_key, test_data[data_key][0])

    # Wrong data type, should fail
    mri_signal_filter.add_input("image_flavour", 0)
    with pytest.raises(FilterInputValidationError):
        mri_signal_filter.run()

    # Check correct use
    mri_signal_filter = MriSignalFilter()
    # add passing data
    for data_key in test_data:
        mri_signal_filter.add_input(data_key, test_data[data_key][0])

    mri_signal_filter.add_input("image_flavour", "PERFUSION")
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
    t2: np.ndarray = input_data["t2"].image
    m0: np.ndarray = input_data["m0"].image
    t2_star: np.ndarray = input_data["t2_star"].image
    mag_enc: np.ndarray = input_data["mag_enc"].image
    echo_time: float = input_data["echo_time"]
    repetition_time: float = input_data["repetition_time"]
    flip_angle: float = np.radians(input_data["excitation_flip_angle"])

    return (
        np.sin(flip_angle)
        * (
            (m0 * (1 - np.exp(-repetition_time / t1)))
            / (
                1
                - np.cos(flip_angle) * np.exp(-repetition_time / t1)
                - np.exp(-repetition_time / t2)
                * (np.exp(-repetition_time / t1) - np.cos(flip_angle))
            )
            + mag_enc
        )
        * np.exp(-echo_time / t2_star)
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


def mri_signal_inversion_recovery_function(input_data: dict) -> np.ndarray:
    """ Function that calculates the inversion recovery signal """
    t1: np.ndarray = input_data["t1"].image
    t2: np.ndarray = input_data["t2"].image
    m0: np.ndarray = input_data["m0"].image

    mag_enc: np.ndarray = input_data["mag_enc"].image
    echo_time: float = input_data["echo_time"]
    repetition_time: float = input_data["repetition_time"]
    flip_angle: float = np.radians(input_data["excitation_flip_angle"])
    inversion_angle: float = np.radians(input_data["inversion_flip_angle"])
    inversion_time: float = input_data["inversion_time"]

    return (
        np.sin(flip_angle)
        * (
            (
                m0
                * (
                    1
                    - (1 - np.cos(inversion_angle)) * np.exp(-inversion_time / t1)
                    - np.cos(inversion_angle) * np.exp(-repetition_time / t1)
                )
                / (
                    1
                    - np.cos(flip_angle)
                    * np.cos(inversion_angle)
                    * np.exp(-repetition_time / t1)
                )
            )
            + mag_enc
        )
        * np.exp(-echo_time / t2)
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

    assert mri_signal_filter.outputs["image"].metadata == {
        "acq_contrast": mock_data["acq_contrast"],
        "echo_time": mock_data["echo_time"],
        "repetition_time": mock_data["repetition_time"],
        "excitation_flip_angle": mock_data["excitation_flip_angle"],
        "image_flavour": "OTHER",
        "mr_acq_type": "3D",
    }

    # edit mock_data["mag_enc"].metadata["image_flavour"] and check
    mock_data["mag_enc"].metadata["image_flavour"] = "PERFUSION"
    mri_signal_filter = MriSignalFilter()
    mri_signal_filter = add_multiple_inputs_to_filter(mri_signal_filter, mock_data)
    mri_signal_filter.run()
    assert mri_signal_filter.outputs["image"].metadata == {
        "acq_contrast": mock_data["acq_contrast"],
        "echo_time": mock_data["echo_time"],
        "repetition_time": mock_data["repetition_time"],
        "excitation_flip_angle": mock_data["excitation_flip_angle"],
        "image_flavour": "PERFUSION",
        "mr_acq_type": "3D",
    }


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

    assert mri_signal_filter.outputs["image"].metadata == {
        "acq_contrast": mock_data["acq_contrast"],
        "echo_time": mock_data["echo_time"],
        "repetition_time": mock_data["repetition_time"],
        "excitation_flip_angle": mock_data["excitation_flip_angle"],
        "image_flavour": "OTHER",
        "mr_acq_type": "3D",
    }

    # edit mock_data["mag_enc"].metadata["image_flavour"] and check
    mock_data["mag_enc"].metadata["image_flavour"] = "PERFUSION"
    mri_signal_filter = MriSignalFilter()
    mri_signal_filter = add_multiple_inputs_to_filter(mri_signal_filter, mock_data)
    mri_signal_filter.run()
    assert mri_signal_filter.outputs["image"].metadata == {
        "acq_contrast": mock_data["acq_contrast"],
        "echo_time": mock_data["echo_time"],
        "repetition_time": mock_data["repetition_time"],
        "excitation_flip_angle": mock_data["excitation_flip_angle"],
        "image_flavour": "PERFUSION",
        "mr_acq_type": "3D",
    }


def test_mri_signal_filter_inversion_recovery(mock_data):
    """ Tests the MriSignalFilter for 'acq_contrast' == 'ir':
    Inversion Recovery """
    mock_data["acq_contrast"] = "ir"
    mock_data["inversion_flip_angle"] = 180.0
    mock_data["inversion_time"] = 1.0
    mock_data["repetition_time"] = 1.1
    mri_signal_filter = MriSignalFilter()
    mri_signal_filter = add_multiple_inputs_to_filter(mri_signal_filter, mock_data)
    mri_signal_filter.run()

    ir_signal = mri_signal_inversion_recovery_function(mock_data)
    numpy.testing.assert_array_equal(
        ir_signal, mri_signal_filter.outputs["image"].image
    )

    assert mri_signal_filter.outputs["image"].metadata == {
        "acq_contrast": mock_data["acq_contrast"],
        "echo_time": mock_data["echo_time"],
        "repetition_time": mock_data["repetition_time"],
        "excitation_flip_angle": mock_data["excitation_flip_angle"],
        "image_flavour": "OTHER",
        "inversion_time": mock_data["inversion_time"],
        "inversion_flip_angle": mock_data["inversion_flip_angle"],
        "mr_acq_type": "3D",
    }

    # edit mock_data["mag_enc"].metadata["image_flavour"] and check
    mock_data["mag_enc"].metadata["image_flavour"] = "PERFUSION"
    mri_signal_filter = MriSignalFilter()
    mri_signal_filter = add_multiple_inputs_to_filter(mri_signal_filter, mock_data)
    mri_signal_filter.run()
    assert mri_signal_filter.outputs["image"].metadata == {
        "acq_contrast": mock_data["acq_contrast"],
        "echo_time": mock_data["echo_time"],
        "repetition_time": mock_data["repetition_time"],
        "excitation_flip_angle": mock_data["excitation_flip_angle"],
        "image_flavour": "PERFUSION",
        "inversion_time": mock_data["inversion_time"],
        "inversion_flip_angle": mock_data["inversion_flip_angle"],
        "mr_acq_type": "3D",
    }


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
            "excitation_flip_angle": 90.0,
        }

        mri_signal_filter = MriSignalFilter()
        mri_signal_filter = add_multiple_inputs_to_filter(mri_signal_filter, params)
        mri_signal_filter.run()
        mri_signal_timecourse[idx] = mri_signal_filter.outputs["image"].image
    # arrays should be equal to 9 decimal places
    numpy.testing.assert_array_almost_equal(mri_signal_timecourse, expected, 9)


# t1, t2, m0, t2_star, echo_time, repetition_time,
# excitation_angle, inversion_angle, inversion_time, expected


@pytest.mark.parametrize(
    "t1, t2, m0, t2_star, echo_time, repetition_time,"
    " flip_angle, inversion_angle, inversion_time, expected",
    IR_TIMECOURSE_PARAMS,
)
def test_mri_signal_timecourse_inversion_recovery(
    t1: float,
    t2: float,
    m0: float,
    t2_star: float,
    echo_time: float,
    repetition_time: float,
    flip_angle: float,
    inversion_angle: float,
    inversion_time: float,
    expected: float,
):
    """Tests the MriSignalFilter inversion recovery signal over a range of TI's.

    :param t1: longitudinal relaxation time, s
    :type t1: float
    :param t2: transverse relaxation time, s
    :type t2: float
    :param m0: equilibrium magnetisation
    :type m0: float
    :param t2_star: transverse relaxation time inc. time invariant fields, s
    :type t2_star: float


    :param echo_time: the echo time, s
    :type echo_time: float
    :param repetition_time: the repetition time, s
    :type repetition_time: float
    :param flip_angle: the excitation pulse flip angle, degrees
    :type flip_angle: float
    :param inversion_angle: the inversion pulse flip angle, degrees
    :type inversion_angle: float
    :param inversion_time: array of durations between the inversion pulse and excitation pulse, s
    :type inversion_time: float
    :param expected: array of expected valuesm, same length as `inversion_time`
    :type expected: float

    """

    mri_signal_timecourse = np.ndarray(inversion_time.shape)
    for idx, ti in np.ndenumerate(inversion_time):
        params = {
            "t1": NumpyImageContainer(image=np.full((1, 1, 1), t1)),
            "t2": NumpyImageContainer(image=np.full((1, 1, 1), t2)),
            "t2_star": NumpyImageContainer(image=np.full((1, 1, 1), t2_star)),
            "m0": NumpyImageContainer(image=np.full((1, 1, 1), m0)),
            "acq_contrast": "ir",
            "echo_time": echo_time,
            "repetition_time": repetition_time,
            "excitation_flip_angle": flip_angle,
            "inversion_flip_angle": inversion_angle,
            "inversion_time": ti,
        }

        mri_signal_filter = MriSignalFilter()
        mri_signal_filter = add_multiple_inputs_to_filter(mri_signal_filter, params)
        mri_signal_filter.run()
        mri_signal_timecourse[idx] = mri_signal_filter.outputs["image"].image
    # arrays should be equal to 9 decimal places
    numpy.testing.assert_array_almost_equal(mri_signal_timecourse, expected, 9)


def test_mri_signal_filter_image_flavour(mock_data):
    """ Tests the MriSignalFilter when the input "image_flavour" is changed """
    # check overrides no supplied mag_enc

    test_data = deepcopy(mock_data)
    test_data.pop("mag_enc")
    mock_data["image_flavour"] = "ABCD"
    mri_signal_filter = MriSignalFilter()
    mri_signal_filter = add_multiple_inputs_to_filter(mri_signal_filter, mock_data)
    mri_signal_filter.run()

    assert mri_signal_filter.outputs["image"].metadata == {
        "acq_contrast": mock_data["acq_contrast"],
        "echo_time": mock_data["echo_time"],
        "repetition_time": mock_data["repetition_time"],
        "excitation_flip_angle": mock_data["excitation_flip_angle"],
        "image_flavour": "ABCD",
        "mr_acq_type": "3D",
    }

    test_data = deepcopy(mock_data)
    mock_data["mag_enc"].metadata["image_flavour"] = "PERFUSION"
    mock_data["image_flavour"] = "ABCD"
    mri_signal_filter = MriSignalFilter()
    mri_signal_filter = add_multiple_inputs_to_filter(mri_signal_filter, mock_data)
    mri_signal_filter.run()

    assert mri_signal_filter.outputs["image"].metadata == {
        "acq_contrast": mock_data["acq_contrast"],
        "echo_time": mock_data["echo_time"],
        "repetition_time": mock_data["repetition_time"],
        "excitation_flip_angle": mock_data["excitation_flip_angle"],
        "image_flavour": "ABCD",
        "mr_acq_type": "3D",
    }
