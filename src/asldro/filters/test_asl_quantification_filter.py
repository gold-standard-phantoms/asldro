"""Tests for asl_quantification_filter.py"""

from copy import deepcopy
import pytest

import numpy as np
import numpy.testing
import nibabel as nib

from asldro.filters.basefilter import BaseFilter, FilterInputValidationError
from asldro.containers.image import NiftiImageContainer

from asldro.filters.asl_quantification_filter import AslQuantificationFilter

TEST_VOLUME_DIMENSIONS = (32, 32, 32)
TEST_NIFTI_ONES = nib.Nifti2Image(
    np.ones(TEST_VOLUME_DIMENSIONS),
    affine=np.array(
        (
            (1, 0, 0, -16),
            (0, 1, 0, -16),
            (0, 0, 1, -16),
            (0, 0, 0, 1),
        )
    ),
)
TEST_NIFTI_CON_ONES = NiftiImageContainer(
    nifti_img=TEST_NIFTI_ONES,
)

# input validation dictionary, for each key the list provides:
# [0] bool for Optional, [1] passes, [2:] fail
INPUT_VALIDATION_DICT = {
    "control": [False, TEST_NIFTI_CON_ONES, TEST_NIFTI_ONES, 1.0, "str"],
    "label": [False, TEST_NIFTI_CON_ONES, TEST_NIFTI_ONES, 1.0, "str"],
    "m0": [False, TEST_NIFTI_CON_ONES, TEST_NIFTI_ONES, 1.0, "str"],
    "label_type": [False, "casl", "CSL", 1.0, TEST_NIFTI_CON_ONES],
    "model": [False, "whitepaper", "whitpaper", 1.0, TEST_NIFTI_CON_ONES],
    "label_duration": [False, 1.8, -1.8, "str"],
    "post_label_delay": [False, 1.8, -1.8, "str"],
    "label_efficiency": [False, 0.85, 1.85, "str"],
    "lambda_blood_brain": [False, 0.9, 1.9, "str"],
    "t1_arterial_blood": [False, 1.65, -1.65, "str"],
}


def validate_filter_inputs(flt: BaseFilter, validation_data: dict):
    """Tests a filter with a validation data dictionary.  Checks that FilterInputValidationErrors
    are raised when data is missing or incorrect.
    :param flt: [description]
    :type flt: BaseFilter
    :param validation_data: [description]
    :type validation_data: dict
    """
    test_filter = flt()
    test_data = deepcopy(validation_data)
    # check with inputs that should pass
    for data_key in test_data:
        test_filter.add_input(data_key, test_data[data_key][1])
    test_filter.run()

    for inputs_key in validation_data:
        test_data = deepcopy(validation_data)
        test_filter = flt()
        is_optional: bool = test_data[inputs_key][0]

        # remove key
        test_data.pop(inputs_key)
        for data_key in test_data:
            test_filter.add_input(data_key, test_data[data_key][1])

        # optional inputs should run without issue
        if is_optional:
            test_filter.run()
        else:
            with pytest.raises(FilterInputValidationError):
                test_filter.run()

        # Try data that should fail
        for test_value in validation_data[inputs_key][2:]:
            test_filter = flt()
            for data_key in test_data:
                test_filter.add_input(data_key, test_data[data_key][1])
            test_filter.add_input(inputs_key, test_value)

            with pytest.raises(FilterInputValidationError):
                test_filter.run()


def test_asl_quantification_filter_validate_inputs():
    """Check a FilterInputValidationError is raised when the inputs to the
    AslQuantificationFilter are incorrect or missing"""
    validate_filter_inputs(AslQuantificationFilter, INPUT_VALIDATION_DICT)


def test_asl_quantification_filter_asl_quant_wp_casl():
    """Test that the static function asl_quant_wp_casl produces correct results"""

    control = np.ones(TEST_VOLUME_DIMENSIONS)
    label = (1 - 0.001) * np.ones(TEST_VOLUME_DIMENSIONS)
    m0 = np.ones(TEST_VOLUME_DIMENSIONS)
    lambda_blood_brain = 0.9
    label_duration = 1.8
    post_label_delay = 1.8
    label_efficiency = 0.85
    t1_arterial_blood = 1.65
    calc_cbf = AslQuantificationFilter.asl_quant_wp_casl(
        control,
        label,
        m0,
        lambda_blood_brain,
        label_duration,
        post_label_delay,
        label_efficiency,
        t1_arterial_blood,
    )
    numpy.testing.assert_array_equal(
        calc_cbf,
        np.divide(
            6000
            * lambda_blood_brain
            * (control - label)
            * np.exp(post_label_delay / t1_arterial_blood),
            2
            * label_efficiency
            * t1_arterial_blood
            * m0
            * (1 - np.exp(-label_duration / t1_arterial_blood)),
            out=np.zeros_like(m0),
            where=m0 != 0,
        ),
    )


def test_asl_quantification_filter_with_mock_data():
    """Tests the AslQuantificationFilter with some mock data"""
    label_image_container = TEST_NIFTI_CON_ONES.clone()
    # 1% signal difference
    label_image_container.image = label_image_container.image * 0.99
    input_params = {
        "control": TEST_NIFTI_CON_ONES,
        "label": label_image_container,
        "m0": TEST_NIFTI_CON_ONES,
        "label_type": "casl",
        "model": "whitepaper",
        "lambda_blood_brain": 0.9,
        "label_duration": 1.8,
        "post_label_delay": 1.8,
        "label_efficiency": 0.85,
        "t1_arterial_blood": 1.65,
    }

    asl_quantification_filter = AslQuantificationFilter()
    asl_quantification_filter.add_inputs(input_params)
    asl_quantification_filter.run()

    numpy.testing.assert_array_equal(
        asl_quantification_filter.outputs["perfusion_rate"].image,
        AslQuantificationFilter.asl_quant_wp_casl(
            TEST_NIFTI_CON_ONES.image,
            label_image_container.image,
            TEST_NIFTI_CON_ONES.image,
            input_params["lambda_blood_brain"],
            input_params["label_duration"],
            input_params["post_label_delay"],
            input_params["label_efficiency"],
            input_params["t1_arterial_blood"],
        ),
    )


def test_asl_quantification_filter_with_mock_timeseries():
    """Tests the AslQuantificationFilter with some mock timeseries data"""
    label_image_container = NiftiImageContainer(
        nib.Nifti2Image(0.99 * np.ones((32, 32, 32, 4)), affine=np.eye(4))
    )
    control_image_container = NiftiImageContainer(
        nib.Nifti2Image(np.ones((32, 32, 32, 4)), affine=np.eye(4))
    )

    input_params = {
        "control": control_image_container,
        "label": label_image_container,
        "m0": TEST_NIFTI_CON_ONES,
        "label_type": "casl",
        "model": "whitepaper",
        "lambda_blood_brain": 0.9,
        "label_duration": 1.8,
        "post_label_delay": 1.8,
        "label_efficiency": 0.85,
        "t1_arterial_blood": 1.65,
    }

    asl_quantification_filter = AslQuantificationFilter()
    asl_quantification_filter.add_inputs(input_params)
    asl_quantification_filter.run()

    numpy.testing.assert_array_equal(
        asl_quantification_filter.outputs["perfusion_rate"].image,
        AslQuantificationFilter.asl_quant_wp_casl(
            control_image_container.image[:, :, :, 0],
            label_image_container.image[:, :, :, 0],
            TEST_NIFTI_CON_ONES.image,
            input_params["lambda_blood_brain"],
            input_params["label_duration"],
            input_params["post_label_delay"],
            input_params["label_efficiency"],
            input_params["t1_arterial_blood"],
        ),
    )