"""Tests for background_suppression_filter.py"""
from copy import deepcopy
import pytest
import pdb
import numpy as np
import numpy.testing
import nibabel as nib

from asldro.filters.background_suppression_filter import BackgroundSuppressionFilter
from asldro.containers.image import NiftiImageContainer, COMPLEX_IMAGE_TYPE
from asldro.utils.filter_validation import validate_filter_inputs


@pytest.fixture(name="test_data")
def test_data_fixture():
    """Returns a dictionary with data for testing"""
    test_dims = (4, 4, 1)
    test_seg_mask = np.arange(16).reshape(test_dims)
    mag_z_data = np.ones(test_dims)
    t1_data = test_seg_mask * (4.2 / 15)  # scale T1 between 0.28 and 4.2 s

    return {
        "mag_z": NiftiImageContainer(
            nib.Nifti1Image(mag_z_data, np.eye(4)), metadata={"key_1": 1, "key_2": 2}
        ),
        "t1": NiftiImageContainer(nib.Nifti1Image(t1_data, np.eye(4))),
        "sat_pulse_time": 4.0,
        "inv_pulse_times": [0.5, 2.7],
        "t1_opt": [1.0, 1.4, 4.5],
        "mag_time": 4.1,
        "num_inv_pulses": 2,
        "pulse_efficiency": "realistic",
        "test_seg_mask": test_seg_mask,
    }


@pytest.fixture(name="validation_data")
def input_validation_dict_fixture(test_data):
    """Returns data for the input validation testing"""

    im_wrong_size = NiftiImageContainer(nib.Nifti1Image(np.ones((3, 3, 3)), np.eye(4)))
    im_neg = NiftiImageContainer(nib.Nifti1Image(-1 * np.ones((3, 3, 3)), np.eye(4)))
    im_complex: NiftiImageContainer = test_data["mag_z"].clone()
    im_complex.image_type = COMPLEX_IMAGE_TYPE
    return {
        # for the case where sat_pulse_times are provided
        "validation_dict_sat_pulse_present": {
            "mag_z": [
                False,
                test_data["mag_z"],
                im_complex,
                np.ones((4, 4, 1)),
                1.0,
                str,
            ],
            "t1": [
                False,
                test_data["t1"],
                im_complex,
                im_neg,
                im_wrong_size,
                1.0,
                "str",
            ],
            "sat_pulse_time": [False, 4.0, 4, -1.0, "str"],
            "inv_pulse_times": [False, [0.5, 2.7], [-0.5, -2.7], 0.5, "str"],
            "mag_time": [True, 4.1, 4, "str"],
            "pulse_efficiency": [True, "realistic", "idle", -3.0, 0.99, -1],
        },
        # for the case where sat_pulse_times are not provided
        "validation_dict_sat_pulse_missing": {
            "mag_z": [
                False,
                test_data["mag_z"],
                im_complex,
                np.ones((4, 4, 1)),
                1.0,
                str,
            ],
            "t1": [False, test_data["t1"], im_complex, im_wrong_size, 1.0, "str"],
            "sat_pulse_time": [False, 4.0, 4, -1.0, "str"],
            "mag_time": [True, 4.1],
            "pulse_efficiency": [True, "realistic", "idle", -3.0, 0.99, -1],
            "t1_opt": [
                False,
                [1.0, 1.4, 4.5],
                [-1.0, 1.4, 4.5],
                [1.0, 1.4, 4],
                1.0,
                "str",
            ],
            "num_inv_pulses": [False, 2, -2, "str"],
        },
    }


def calc_mz_function(
    initial_mz, t1, inv_pulse_times, mag_time, inv_eff, sat_eff=1.0
) -> np.ndarray:
    """Calculates the longitudinal magnetisation after a sequence of background
    suppression pulses"""
    num_pulses = np.asarray(inv_pulse_times).size
    if num_pulses == 1:
        inv_pulse_times = [
            inv_pulse_times,
        ]
    mz = 1 + ((1 - sat_eff) - 1) * (inv_eff ** num_pulses) * np.exp(
        -np.divide(mag_time, t1, out=np.zeros_like(t1), where=t1 != 0)
    )
    # pdb.set_trace()
    for m, tm in enumerate(inv_pulse_times):
        mz += ((inv_eff ** (m + 1)) - (inv_eff ** m)) * np.exp(
            -np.divide(tm, t1, out=np.zeros_like(t1), where=t1 != 0)
        )
        # pdb.set_trace()

    return mz * initial_mz


def test_background_suppression_filter_validate_inputs(validation_data, test_data):
    """Check a FilterInputValidationError is raised when the
    inputs to the BackgroundSuppressionFilter are incorrect or mising
    """

    validate_filter_inputs(
        BackgroundSuppressionFilter,
        validation_data["validation_dict_sat_pulse_present"],
    )
    validate_filter_inputs(
        BackgroundSuppressionFilter,
        validation_data["validation_dict_sat_pulse_missing"],
    )

    # test the different options for 'pulse_efficiency'
    data = deepcopy(test_data)
    valid_pulse_efficiencies = ["realistic", "ideal", 0.0, -1.0, -0.5, -0.99, -0.01]
    bsup_filter = BackgroundSuppressionFilter()

    for eff in valid_pulse_efficiencies:
        data["pulse_efficiency"] = eff
        bsup_filter = BackgroundSuppressionFilter()
        bsup_filter.add_inputs(data)
        # should run with no issues
        bsup_filter.run()


def test_background_suppression_filter_calculate_mz(test_data):
    """Tests BackgroundSuppressionFilter.calculate_mz static method"""

    # Test single values
    initial_mz = 1.0
    t1 = 1.4
    inv_pulse_times = [0.5, 2.5]
    sat_time = 4.0
    mag_time = 4.0
    inv_eff = -1.0

    mz = BackgroundSuppressionFilter.calculate_mz(
        initial_mz, t1, inv_pulse_times, sat_time, mag_time, inv_eff
    )

    numpy.testing.assert_array_almost_equal(
        mz, calc_mz_function(initial_mz, t1, inv_pulse_times, mag_time, inv_eff)
    )

    # Test arrays
    initial_mz = test_data["mag_z"].image
    t1 = test_data["t1"].image
    inv_pulse_times = [0.5, 2.5]
    mag_time = 4.0
    sat_time = 4.0
    inv_eff = -1.0 * np.ones_like(initial_mz)

    mz = BackgroundSuppressionFilter.calculate_mz(
        initial_mz, t1, inv_pulse_times, sat_time, mag_time, inv_eff
    )

    numpy.testing.assert_array_almost_equal(
        mz, calc_mz_function(initial_mz, t1, inv_pulse_times, mag_time, inv_eff)
    )

    # Test more pulses
    inv_pulse_times = [0.5, 1.0, 1.5, 2.5]

    mz = BackgroundSuppressionFilter.calculate_mz(
        initial_mz, t1, inv_pulse_times, sat_time, mag_time, inv_eff
    )

    numpy.testing.assert_array_almost_equal(
        mz, calc_mz_function(initial_mz, t1, inv_pulse_times, mag_time, inv_eff)
    )

    # Test where pulses mag_time is shorter than all of the pulses
    mag_time = 2.0
    initial_mz = 1.0
    inv_eff = -1
    t1 = 1.4
    mz = BackgroundSuppressionFilter.calculate_mz(
        initial_mz, t1, inv_pulse_times, sat_time, mag_time, inv_eff
    )
    # should be equivalent to only the last 3 pulses being used
    numpy.testing.assert_array_almost_equal(
        mz, calc_mz_function(initial_mz, t1, inv_pulse_times[3], mag_time, inv_eff)
    )

    # test something equivalent to the optimisation
    initial_mz = np.array([1.0, 1.0, 1.0])
    t1 = np.array([1.0, 1.4, 4.5])
    inv_pulse_times = [0.5, 2.5]
    mag_time = 4.0
    inv_eff = -1.0

    mz = BackgroundSuppressionFilter.calculate_mz(
        initial_mz, t1, inv_pulse_times, sat_time, mag_time, inv_eff
    )

    numpy.testing.assert_array_almost_equal(
        mz, calc_mz_function(initial_mz, t1, inv_pulse_times, mag_time, inv_eff)
    )


def test_background_suppression_filter_calculate_pulse_efficiency(test_data):
    """Tests BackgroundSuppressionFilter.calculate_pulse_efficiency static method"""
    t1 = np.array((0.24, 0.25, 0.449, 0.450, 2.0, 2.001, 4.2, 4.3))
    pulse_eff = BackgroundSuppressionFilter.calculate_pulse_efficiency(t1)
    pe_fun = lambda x: -(
        (-2.245e-15) * (x ** 4)
        + (2.378e-11) * (x ** 3)
        - (8.987e-8) * (x ** 2)
        + (1.442e-4) * x
        + (9.1555e-1)
    )
    numpy.testing.assert_array_equal(
        pulse_eff, (0.0, -0.998, -0.998, pe_fun(450), pe_fun(2000), -0.998, -0.998, 0.0)
    )

    # test with a multidimensional array
    BackgroundSuppressionFilter.calculate_pulse_efficiency(test_data["t1"].image)


def test_background_suppression_filter_optimise_inv_pulse_times():
    """Tests BackgroundSuppressionFilter.optimise_inv_pulse_times static method"""
    sat_time = 4.0
    t1 = [1.0, 1.4, 4.5]
    pulse_eff = -1.0
    num_pulses = 2
    initial_mz = 1.0

    # 0 pulses should raise a ValueError
    with pytest.raises(ValueError):
        BackgroundSuppressionFilter.optimise_inv_pulse_times(
            sat_time, t1, pulse_eff, 0,
        )

    result = BackgroundSuppressionFilter.optimise_inv_pulse_times(
        sat_time, t1, pulse_eff, num_pulses,
    )
    mz = BackgroundSuppressionFilter.calculate_mz(
        initial_mz, t1, result.x, sat_time, sat_time, pulse_eff
    )

    # This should result in a successful optimisation
    assert result.success

    # the value of mz should be less than 0.05 for all cases
    assert np.all(np.abs(mz) < 0.05)


def test_background_suppression_filter_mock_data(test_data: dict):
    """Test the BackgroundSuppressionFilter with some mock data"""
    # run with defaults for the case where the pulse times are provided
    bsup_filter = BackgroundSuppressionFilter()
    bsup_filter.add_inputs(
        {
            key: test_data.get(key)
            for key in ["mag_z", "t1", "sat_pulse_time", "inv_pulse_times"]
        }
    )
    bsup_filter.run()
    mz = calc_mz_function(
        test_data["mag_z"].image,
        test_data["t1"].image,
        test_data["inv_pulse_times"],
        test_data["sat_pulse_time"],
        -1,
    )
    # compare
    numpy.testing.assert_array_almost_equal(
        bsup_filter.outputs[BackgroundSuppressionFilter.KEY_MAG_Z].image, mz
    )

    # check metadata
    assert bsup_filter.outputs[BackgroundSuppressionFilter.KEY_MAG_Z].metadata == {
        "background_suppression": True,
        "background_suppression_inv_pulse_timing": test_data["inv_pulse_times"],
        "background_suppression_sat_pulse_timing": test_data["sat_pulse_time"],
        "background_suppression_num_pulses": len(test_data["inv_pulse_times"]),
        "key_1": 1,
        "key_2": 2,
    }

    # run where pulse times need to be optimised
    bsup_filter = BackgroundSuppressionFilter()
    bsup_filter.add_inputs(
        {
            key: test_data.get(key)
            for key in ["mag_z", "t1", "sat_pulse_time", "inv_pulse_times"]
        }
    )
    bsup_filter.run()
    mz = calc_mz_function(
        test_data["mag_z"].image,
        test_data["t1"].image,
        test_data["inv_pulse_times"],
        test_data["sat_pulse_time"],
        -1,
    )
    # compare
    numpy.testing.assert_array_almost_equal(
        bsup_filter.outputs[BackgroundSuppressionFilter.KEY_MAG_Z].image, mz
    )

