"""GkmFilter tests"""
# pylint: disable=duplicate-code

from copy import deepcopy
import pytest
import numpy as np
import numpy.testing
from asldro.containers.image import BaseImageContainer, NumpyImageContainer
from asldro.filters.basefilter import BaseFilter, FilterInputValidationError
from asldro.filters.gkm_filter import GkmFilter

TEST_VOLUME_DIMENSIONS = (32, 32, 32)
TEST_IMAGE_ONES = NumpyImageContainer(image=np.ones(TEST_VOLUME_DIMENSIONS))
TEST_IMAGE_101 = NumpyImageContainer(image=101 * np.ones(TEST_VOLUME_DIMENSIONS))
TEST_IMAGE_NEG = NumpyImageContainer(image=-1.0 * np.ones(TEST_VOLUME_DIMENSIONS))
TEST_IMAGE_SMALL = NumpyImageContainer(image=np.ones((8, 8, 8)))

CASL = "CASL"
PCASL = "pCASL"
PASL = "PASL"

# test data dictionary, [0] in each tuple passes, after that should fail validation
TEST_DATA_DICT_PASL_M0_IM = {
    "perfusion_rate": (TEST_IMAGE_ONES, TEST_IMAGE_NEG, TEST_IMAGE_SMALL),
    "transit_time": (TEST_IMAGE_ONES, TEST_IMAGE_NEG, TEST_IMAGE_SMALL),
    "m0": (TEST_IMAGE_ONES, TEST_IMAGE_NEG, TEST_IMAGE_SMALL),
    "t1_tissue": (TEST_IMAGE_ONES, TEST_IMAGE_NEG, TEST_IMAGE_SMALL, TEST_IMAGE_101),
    "label_type": ("pasl", "PSL", "str"),
    "label_duration": (1.8, -0.1, 101.0),
    "signal_time": (3.6, -0.1, 101.0),
    "label_efficiency": (0.85, -0.1, 1.01),
    "lambda_blood_brain": (0.9, -0.1, 1.01),
    "t1_arterial_blood": (1.6, -0.1, 101.0),
}

TEST_DATA_DICT_PASL_M0_FLOAT = {
    "perfusion_rate": (TEST_IMAGE_ONES, TEST_IMAGE_NEG, TEST_IMAGE_SMALL),
    "transit_time": (TEST_IMAGE_ONES, TEST_IMAGE_NEG, TEST_IMAGE_SMALL),
    "m0": (1.0, -1.0, int(1)),
    "t1_tissue": (TEST_IMAGE_ONES, TEST_IMAGE_NEG, TEST_IMAGE_SMALL, TEST_IMAGE_101),
    "label_type": ("pasl", "PSL", "str"),
    "label_duration": (1.8, -0.1, 101.0),
    "signal_time": (3.6, -0.1, 101.0),
    "label_efficiency": (0.85, -0.1, 1.01),
    "lambda_blood_brain": (0.9, -0.1, 1.01),
    "t1_arterial_blood": (1.6, -0.1, 101.0),
}

TEST_DATA_DICT_CASL_M0_IM = {
    "perfusion_rate": (TEST_IMAGE_ONES, TEST_IMAGE_NEG, TEST_IMAGE_SMALL),
    "transit_time": (TEST_IMAGE_ONES, TEST_IMAGE_NEG, TEST_IMAGE_SMALL),
    "m0": (TEST_IMAGE_ONES, TEST_IMAGE_NEG, TEST_IMAGE_SMALL),
    "t1_tissue": (TEST_IMAGE_ONES, TEST_IMAGE_NEG, TEST_IMAGE_SMALL, TEST_IMAGE_101),
    "label_type": ("casl", "CSL", "str"),
    "label_duration": (1.8, -0.1, 101.0),
    "signal_time": (3.6, -0.1, 101.0),
    "label_efficiency": (0.85, -0.1, 1.01),
    "lambda_blood_brain": (0.9, -0.1, 1.01),
    "t1_arterial_blood": (1.6, -0.1, 101.0),
}

TEST_DATA_DICT_CASL_M0_FLOAT = {
    "perfusion_rate": (TEST_IMAGE_ONES, TEST_IMAGE_NEG, TEST_IMAGE_SMALL),
    "transit_time": (TEST_IMAGE_ONES, TEST_IMAGE_NEG, TEST_IMAGE_SMALL),
    "m0": (1.0, -1.0, int(1)),
    "t1_tissue": (TEST_IMAGE_ONES, TEST_IMAGE_NEG, TEST_IMAGE_SMALL, TEST_IMAGE_101),
    "label_type": ("pcasl", "PCSL", "str"),
    "label_duration": (1.8, -0.1, 101.0),
    "signal_time": (3.6, -0.1, 101.0),
    "label_efficiency": (0.85, -0.1, 1.01),
    "lambda_blood_brain": (0.9, -0.1, 1.01),
    "t1_arterial_blood": (1.6, -0.1, 101.0),
}

# f, delta_t, t, label_type, tau, alpha, expected
TIMECOURSE_PARAMS = (
    (
        60,
        0.5,
        np.arange(0, 2.6, 0.1),
        "PASL",
        1.0,
        1.0,
        [
            0.00000000000000e00,
            0.00000000000000e00,
            0.00000000000000e00,
            0.00000000000000e00,
            0.00000000000000e00,
            0.00000000000000e00,
            1.51966825043716e-03,
            2.84093150577538e-03,
            3.98325075686420e-03,
            4.96438731050847e-03,
            5.80054219525963e-03,
            6.50648457218782e-03,
            7.09566999513514e-03,
            7.58034930130601e-03,
            7.97166885413462e-03,
            8.27976280583558e-03,
            7.70041819796574e-03,
            7.16161100433575e-03,
            6.66050477504872e-03,
            6.19446152989728e-03,
            5.76102787120916e-03,
            5.35792206839310e-03,
            4.98302204619408e-03,
            4.63435421342423e-03,
            4.31008307336043e-03,
            4.00850156111438e-03,
        ],
    ),
    (
        60,
        0.5,
        np.arange(0, 2.6, 0.1),
        "CASL",
        1.0,
        1.0,
        [
            0.00000000000000e00,
            0.00000000000000e00,
            0.00000000000000e00,
            0.00000000000000e00,
            0.00000000000000e00,
            0.00000000000000e00,
            1.56824489938627e-03,
            3.02675788403236e-03,
            4.38321702329929e-03,
            5.64476314312850e-03,
            6.81803741758707e-03,
            7.90921633008904e-03,
            8.92404418833934e-03,
            9.86786336416952e-03,
            1.07456424174571e-02,
            1.15620022521814e-02,
            1.07529955429241e-02,
            1.00005959715437e-02,
            9.30084267094010e-03,
            8.65005192047837e-03,
            8.04479775372961e-03,
            7.48189392311000e-03,
            6.95837712647509e-03,
            6.47149140736887e-03,
            6.01867364680540e-03,
            5.59754007020735e-03,
        ],
    ),
)


@pytest.fixture(name="pasl_input")
def pasl_input_fixture() -> dict:
    """creates test data for testing the PASL model"""
    np.random.seed(0)
    return {
        "perfusion_rate": NumpyImageContainer(
            image=np.random.normal(60, 10, TEST_VOLUME_DIMENSIONS)
        ),
        "transit_time": TEST_IMAGE_ONES,
        "t1_tissue": NumpyImageContainer(image=1.4 * np.ones(TEST_VOLUME_DIMENSIONS)),
        "m0": TEST_IMAGE_ONES,
        "label_type": PASL,
        "label_duration": 0.8,
        "signal_time": 1.8,
        "label_efficiency": 0.99,
        "lambda_blood_brain": 0.9,
        "t1_arterial_blood": 1.6,
    }


@pytest.fixture(name="casl_input")
def casl_input_fixture() -> dict:
    """Creates test data for testing the CASL/pCASL model"""
    np.random.seed(0)
    return {
        "perfusion_rate": NumpyImageContainer(
            image=np.random.normal(60, 10, TEST_VOLUME_DIMENSIONS)
        ),
        "transit_time": TEST_IMAGE_ONES,
        "t1_tissue": NumpyImageContainer(image=1.4 * np.ones(TEST_VOLUME_DIMENSIONS)),
        "m0": TEST_IMAGE_ONES,
        "label_type": CASL,
        "label_duration": 1.8,
        "signal_time": 3.6,
        "label_efficiency": 0.85,
        "lambda_blood_brain": 0.9,
        "t1_arterial_blood": 1.6,
    }


def add_multiple_inputs_to_filter(input_filter: BaseFilter, input_data: dict):
    """Adds the data held within the input_data dictionary to the filter's inputs"""
    for key in input_data:
        input_filter.add_input(key, input_data[key])

    return input_filter


def gkm_pasl_function(input_data: dict) -> np.ndarray:
    """Function that calculates the gkm for PASL
    Variable names match those in the GKM equations
    """
    f: np.ndarray = input_data["perfusion_rate"].image / 6000.0
    delta_t: np.ndarray = input_data["transit_time"].image
    tau: float = input_data["label_duration"]
    t: float = input_data["signal_time"]
    alpha: float = input_data["label_efficiency"]
    # _lambda avoids clash with lambda function
    _lambda: float = input_data["lambda_blood_brain"]
    t1b: float = input_data["t1_arterial_blood"]
    t1: np.ndarray = input_data["t1_tissue"].image

    if isinstance(input_data["m0"], BaseImageContainer):
        m0: np.ndarray = input_data["m0"].image
    else:
        m0: np.ndarray = input_data["m0"] * np.ones(f.shape)

    # calculate M0b
    m0 = m0 / _lambda

    t1_prime: np.ndarray = 1 / (1 / t1 + f / _lambda)

    # create boolean masks for each of the states of the delivery curve
    condition_bolus_not_arrived = 0 < t <= delta_t
    condition_bolus_arriving = (delta_t < t) & (t < delta_t + tau)
    condition_bolus_arrived = t >= delta_t + tau

    delta_m = np.zeros(f.shape)

    k: np.ndarray = 1 / t1b - 1 / t1_prime

    q_pasl_arriving = (
        np.exp(k * t) * (np.exp(-k * delta_t) - np.exp(-k * t)) / (k * (t - delta_t))
    )
    q_pasl_arrived = (
        np.exp(k * t)
        * (np.exp(-k * delta_t) - np.exp(-k * (delta_t + tau)))
        / (k * tau)
    )

    delta_m_arriving = (
        2 * m0 * f * (t - delta_t) * alpha * np.exp(-t / t1b) * q_pasl_arriving
    )
    delta_m_arrived = 2 * m0 * f * alpha * tau * np.exp(-t / t1b) * q_pasl_arrived

    # combine the different arrival states into delta_m
    delta_m[condition_bolus_not_arrived] = 0.0
    delta_m[condition_bolus_arriving] = delta_m_arriving[condition_bolus_arriving]
    delta_m[condition_bolus_arrived] = delta_m_arrived[condition_bolus_arrived]

    return delta_m


def gkm_casl_function(input_data: dict) -> np.ndarray:
    # pylint: disable=too-many-locals
    """Function that calculates the gkm for CASL/pCASL
    Variable names match those in the GKM equations
    """
    f: np.ndarray = input_data["perfusion_rate"].image / 6000.0
    delta_t: np.ndarray = input_data["transit_time"].image
    tau: float = input_data["label_duration"]
    t: float = input_data["signal_time"]
    alpha: float = input_data["label_efficiency"]
    # _lambda avoids clash with lambda function
    _lambda: float = input_data["lambda_blood_brain"]
    t1b: float = input_data["t1_arterial_blood"]
    t1: np.ndarray = input_data["t1_tissue"].image

    if isinstance(input_data["m0"], BaseImageContainer):
        m0: np.ndarray = input_data["m0"].image
    else:
        m0: np.ndarray = input_data["m0"] * np.ones(f.shape)

    # calculate M0b
    m0 = m0 / _lambda

    t1_prime: np.ndarray = 1 / (1 / t1 + f / _lambda)

    # create boolean masks for each of the states of the delivery curve
    condition_bolus_not_arrived = 0 < t <= delta_t
    condition_bolus_arriving = (delta_t < t) & (t < delta_t + tau)
    condition_bolus_arrived = t >= delta_t + tau

    delta_m = np.zeros(f.shape)

    q_ss_arriving = 1 - np.exp(-(t - delta_t) / t1_prime)
    q_ss_arrived = 1 - np.exp(-tau / t1_prime)

    delta_m_arriving = (
        2 * m0 * f * t1_prime * alpha * np.exp(-delta_t / t1b) * q_ss_arriving
    )
    delta_m_arrived = (
        2
        * m0
        * f
        * t1_prime
        * alpha
        * np.exp(-delta_t / t1b)
        * np.exp(-(t - tau - delta_t) / t1_prime)
        * q_ss_arrived
    )

    # combine the different arrival states into delta_m
    delta_m[condition_bolus_not_arrived] = 0.0
    delta_m[condition_bolus_arriving] = delta_m_arriving[condition_bolus_arriving]
    delta_m[condition_bolus_arrived] = delta_m_arrived[condition_bolus_arrived]

    return delta_m


@pytest.mark.parametrize(
    "validation_data", [TEST_DATA_DICT_PASL_M0_IM, TEST_DATA_DICT_PASL_M0_FLOAT]
)
def test_gkm_filter_validate_inputs(validation_data: dict):
    """Check a FilterInputValidationError is raised when the
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


def test_gkm_filter_pasl(pasl_input):
    """Test the GkmFilter for Pulsed ASL"""
    gkm_filter = GkmFilter()
    gkm_filter = add_multiple_inputs_to_filter(gkm_filter, pasl_input)
    gkm_filter.run()

    delta_m = gkm_pasl_function(pasl_input)
    numpy.testing.assert_array_equal(delta_m, gkm_filter.outputs["delta_m"].image)

    # Set 'signal_time' to be less than the transit time so that the bolus
    # has not arrived yet
    pasl_input["signal_time"] = 0.5
    assert (pasl_input["signal_time"] < pasl_input["transit_time"].image).all()
    gkm_filter = GkmFilter()
    gkm_filter = add_multiple_inputs_to_filter(gkm_filter, pasl_input)
    gkm_filter.run()

    # check the m0 is added to the metadata, as all values of m0 for the image are the same
    gkm_filter.outputs["delta_m"].metadata["m0"] = 1.0
    # 'delta_m' should be all zero
    numpy.testing.assert_array_equal(
        gkm_filter.outputs["delta_m"].image,
        np.zeros(gkm_filter.outputs["delta_m"].shape),
    )

    # create input images with some zeros in to test that divide-by-zero is not encountered at
    # runtime
    image_with_some_zeros = numpy.concatenate(
        (np.ones((32, 32, 16)), np.zeros((32, 32, 16))), axis=2
    )
    pasl_input["perfusion_rate"] = NumpyImageContainer(image=60 * image_with_some_zeros)
    pasl_input["transit_time"] = NumpyImageContainer(image=image_with_some_zeros)
    pasl_input["m0"] = NumpyImageContainer(image=image_with_some_zeros)
    pasl_input["t1_tissue"] = NumpyImageContainer(image=1.4 * image_with_some_zeros)
    pasl_input["lambda_blood_brain"] = 0.0
    pasl_input["t1_arterial_blood"] = 0.0

    gkm_filter = GkmFilter()
    gkm_filter = add_multiple_inputs_to_filter(gkm_filter, pasl_input)
    gkm_filter.run()


def test_gkm_filter_casl(casl_input):
    """Test the GkmFilter for Continuous ASL"""
    gkm_filter = GkmFilter()
    gkm_filter = add_multiple_inputs_to_filter(gkm_filter, casl_input)
    gkm_filter.run()

    delta_m = gkm_casl_function(casl_input)
    numpy.testing.assert_array_equal(delta_m, gkm_filter.outputs["delta_m"].image)

    # Set 'signal_time' to be less than the transit time so that the bolus
    # has not arrived yet
    casl_input["signal_time"] = 0.5
    assert (casl_input["signal_time"] < casl_input["transit_time"].image).all()
    gkm_filter = GkmFilter()
    gkm_filter = add_multiple_inputs_to_filter(gkm_filter, casl_input)
    gkm_filter.run()
    # 'delta_m' should be all zero
    numpy.testing.assert_array_equal(
        gkm_filter.outputs["delta_m"].image,
        np.zeros(gkm_filter.outputs["delta_m"].shape),
    )

    # create input images with some zeros in to test that divide-by-zero is not encountered at
    # runtime
    image_with_some_zeros = numpy.concatenate(
        (np.ones((32, 32, 16)), np.zeros((32, 32, 16))), axis=2
    )
    casl_input["perfusion_rate"] = NumpyImageContainer(image=60 * image_with_some_zeros)
    casl_input["transit_time"] = NumpyImageContainer(image=image_with_some_zeros)
    casl_input["m0"] = NumpyImageContainer(image=image_with_some_zeros)
    casl_input["t1_tissue"] = NumpyImageContainer(image=1.4 * image_with_some_zeros)
    casl_input["lambda_blood_brain"] = 0.0
    casl_input["t1_arterial_blood"] = 0.0

    gkm_filter = GkmFilter()
    gkm_filter = add_multiple_inputs_to_filter(gkm_filter, casl_input)
    gkm_filter.run()
    # check m0 is NOT added to the metadata, as the values are not all the same
    assert "m0" not in gkm_filter.outputs["delta_m"].metadata


@pytest.mark.parametrize(
    "f, delta_t, timepoints, label_type, tau, alpha, expected", TIMECOURSE_PARAMS
)
def test_gkm_timecourse(
    f: float,
    delta_t: float,
    timepoints: np.array,
    label_type: str,
    tau: float,
    alpha: str,
    expected: np.ndarray,
):
    # pylint: disable=too-many-arguments
    """Tests the GkmFilter with timecourse data that is generated at multiple 'signal_time's.

    Arguments:
        f (float): Perfusion Rate, ml/100g/min
        delta_t (float): Bolus arrival time/transit time, seconds
        timepoints (np.array): Array of time points that the signal is generated at, seconds
        label_type (str): GKM model to use: 'PASL' or 'CASL'/'pCASL'
        tau (float): Label duration, seconds
        alpha (str): Label efficiency, 0 to 1
        expected (np.ndarray): Array of expected values that the GkmFilter should generate. Should
        be the same size and shape as `timepoints`.
    """
    delta_m_timecourse = np.ndarray(timepoints.shape)
    for idx, t in np.ndenumerate(timepoints):
        params = {
            "perfusion_rate": NumpyImageContainer(image=f * np.ones((1, 1, 1))),
            "transit_time": NumpyImageContainer(image=delta_t * np.ones((1, 1, 1))),
            "t1_tissue": NumpyImageContainer(image=1.4 * np.ones((1, 1, 1))),
            "m0": 1.0,
            "label_type": label_type,
            "label_duration": tau,
            "signal_time": t,
            "label_efficiency": alpha,
            "lambda_blood_brain": 0.9,
            "t1_arterial_blood": 1.6,
        }
        gkm_filter = GkmFilter()
        gkm_filter = add_multiple_inputs_to_filter(gkm_filter, params)
        gkm_filter.run()
        delta_m_timecourse[idx] = gkm_filter.outputs["delta_m"].image
    # arrays should be equal to 9 decimal places
    numpy.testing.assert_array_almost_equal(delta_m_timecourse, expected, 10)


def test_gkm_filter_metadata(casl_input):
    """Test the metadata output from GkmFilter"""
    gkm_filter = GkmFilter()
    casl_input["perfusion_rate"].metadata = {
        "units": "ml/100g/min",
        "quantity": "perfusion_rate",
    }
    gkm_filter = add_multiple_inputs_to_filter(gkm_filter, casl_input)
    gkm_filter.run()

    assert gkm_filter.outputs["delta_m"].metadata == {
        "label_type": casl_input["label_type"].lower(),
        "label_duration": casl_input["label_duration"],
        "post_label_delay": casl_input["signal_time"] - casl_input["label_duration"],
        "label_efficiency": casl_input["label_efficiency"],
        "lambda_blood_brain": casl_input["lambda_blood_brain"],
        "t1_arterial_blood": casl_input["t1_arterial_blood"],
        "image_flavour": "PERFUSION",
        "m0": 1.0,
    }


def test_gkm_filter_m0_float(casl_input):
    """Test the GkmFilter when m0 is supplied as a number and not an image"""
    # set m0 to a float
    casl_input["m0"] = 100.0
    casl_input["perfusion_rate"].metadata = {
        "units": "ml/100g/min",
        "quantity": "perfusion_rate",
    }
    gkm_filter = GkmFilter()
    gkm_filter = add_multiple_inputs_to_filter(gkm_filter, casl_input)
    gkm_filter.run()
    assert gkm_filter.outputs["delta_m"].metadata == {
        "label_type": casl_input["label_type"].lower(),
        "label_duration": casl_input["label_duration"],
        "post_label_delay": casl_input["signal_time"] - casl_input["label_duration"],
        "label_efficiency": casl_input["label_efficiency"],
        "lambda_blood_brain": casl_input["lambda_blood_brain"],
        "t1_arterial_blood": casl_input["t1_arterial_blood"],
        "image_flavour": "PERFUSION",
        "m0": 100.0,
    }
