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

CASL = "CASL"
PCASL = "PCASL"
PASL = "PASL"

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
            1.37381518558427e-03,
            2.57972668111671e-03,
            3.63312538529594e-03,
            4.54814811537416e-03,
            5.33777340800020e-03,
            6.01391029427834e-03,
            6.58748054999844e-03,
            7.06849488699720e-03,
            7.46612351902566e-03,
            7.78876150515868e-03,
            7.30873896677194e-03,
            6.85830028933751e-03,
            6.43562221507296e-03,
            6.03899385384618e-03,
            5.66680975794015e-03,
            5.31756342362121e-03,
            4.98984119320646e-03,
            4.68231653294780e-03,
            4.39374466357073e-03,
            4.12295752173393e-03,
        ],
    ),
)


@pytest.fixture(name="pasl_input")
def pasl_input_fixture() -> dict:
    np.random.seed(0)
    return {
        "perfusion_rate": NumpyImageContainer(
            image=np.random.normal(60, 10, TEST_VOLUME_DIMENSIONS)
        ),
        "transit_time": TEST_IMAGE_ONES,
        "m0": TEST_IMAGE_ONES,
        "label_type": PASL,
        "label_duration": 0.8,
        "signal_time": 1.8,
        "label_efficiency": 0.99,
        "lambda_blood_brain": 0.9,
        "t1_arterial_blood": 1.6,
    }


def add_multiple_inputs_to_filter(input_filter: BaseFilter, input_data: dict):
    """ Adds the data held within the input_data dictionary to the filter's inputs """
    for key in input_data:
        input_filter.add_input(key, input_data[key])

    return input_filter


def gkm_pasl_function(input_data: dict) -> np.ndarray:
    """ Function that calculates the gkm for PASL
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

    if isinstance(input_data["m0"], BaseImageContainer):
        m0: np.ndarray = input_data["m0"].image
    else:
        m0: np.ndarray = input_data["m0"] * np.ones(f.shape)

    t1_prime: np.ndarray = 1 / (1 / t1b + f / _lambda)

    # create boolean masks for each of the states of the delivery curve
    condition_bolus_not_arrived = 0 < t <= delta_t
    condition_bolus_arriving = (delta_t < t) & (t < delta_t + tau)
    condition_bolus_arrived = t >= delta_t + tau

    delta_m = np.zeros(f.shape)

    k: np.ndarray = (1 / t1b - 1 / t1_prime)

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


def test_gkm_filter_pasl(pasl_input):
    """ Test the GkmFilter for Pulsed ASL """
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
    # 'delta_m' should be all zero
    numpy.testing.assert_array_equal(
        gkm_filter.outputs["delta_m"].image,
        np.zeros(gkm_filter.outputs["delta_m"].shape),
    )


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
    delta_m_timecourse = np.ndarray(timepoints.shape)
    for idx, t in np.ndenumerate(timepoints):
        params = {
            "perfusion_rate": NumpyImageContainer(image=f * np.ones((1, 1, 1))),
            "transit_time": NumpyImageContainer(image=delta_t * np.ones((1, 1, 1))),
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


if __name__ == "__main__":
    test_gkm_filter_validate_inputs(TEST_DATA_DICT_M0_IM)
