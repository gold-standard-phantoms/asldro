"""Tests for asl_quantification_filter.py"""

from copy import deepcopy
import pytest

import numpy as np
import numpy.testing
import nibabel as nib

from asldro.filters.basefilter import BaseFilter, FilterInputValidationError
from asldro.containers.image import NiftiImageContainer
from asldro.utils.filter_validation import validate_filter_inputs

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


@pytest.fixture(name="test_data")
def test_data_fixture() -> NiftiImageContainer:
    image = NiftiImageContainer(
        nib.Nifti2Image(np.ones(TEST_VOLUME_DIMENSIONS), affine=np.eye(4))
    )
    image.metadata = {
        "RepetitionTime": 5.0,
        "RepetitionTimePreparation": 5.0,
        "EchoTime": 0.01,
        "FlipAngle": 90,
        "M0Type": "Included",
        "ComplexImageComponent": "REAL",
        "ImageType": [
            "ORIGINAL",
            "PRIMARY",
            "PERFUSION",
            "NONE",
        ],
        "BackgroundSuppression": False,
        "VascularCrushing": False,
        "LabelingDuration": 1.8,
        "PostLabelingDelay": 1.8,
        "LabelingEfficiency": 0.85,
    }
    return image


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


def test_asl_quantification_filter_with_mock_data_casl(test_data):
    """Tests the AslQuantificationFilter with some mock data for CASL"""
    label_image_container = test_data.clone()
    # 1% signal difference
    label_image_container.image = label_image_container.image * 0.99
    input_params = {
        "control": test_data,
        "label": label_image_container,
        "m0": test_data,
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
            test_data.image,
            label_image_container.image,
            test_data.image,
            input_params["lambda_blood_brain"],
            input_params["label_duration"],
            input_params["post_label_delay"],
            input_params["label_efficiency"],
            input_params["t1_arterial_blood"],
        ),
    )

    # check the image metadata
    assert asl_quantification_filter.outputs["perfusion_rate"].metadata == {
        "ComplexImageComponent": "REAL",
        "ImageType": [
            "DERIVED",
            "PRIMARY",
            "PERFUSION",
            "RCBF",
        ],
        "BackgroundSuppression": False,
        "VascularCrushing": False,
        "LabelingDuration": 1.8,
        "PostLabelingDelay": 1.8,
        "LabelingEfficiency": 0.85,
        "asl_context": "cbf",
        "Units": "ml/100g/min",
    }


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


def test_asl_quantification_filter_asl_quant_wp_pasl():
    """Test that the static function asl_quant_wp_pasl produces correct results"""

    control = np.ones(TEST_VOLUME_DIMENSIONS)
    label = (1 - 0.001) * np.ones(TEST_VOLUME_DIMENSIONS)
    m0 = np.ones(TEST_VOLUME_DIMENSIONS)
    lambda_blood_brain = 0.9
    bolus_duration = 0.8
    inversion_time = 1.8
    label_efficiency = 0.85
    t1_arterial_blood = 1.65
    calc_cbf = AslQuantificationFilter.asl_quant_wp_pasl(
        control,
        label,
        m0,
        lambda_blood_brain,
        bolus_duration,
        inversion_time,
        label_efficiency,
        t1_arterial_blood,
    )
    numpy.testing.assert_array_equal(
        calc_cbf,
        np.divide(
            6000
            * lambda_blood_brain
            * (control - label)
            * np.exp(inversion_time / t1_arterial_blood),
            2 * label_efficiency * bolus_duration * m0,
            out=np.zeros_like(m0),
            where=m0 != 0,
        ),
    )


def test_asl_quantification_filter_with_mock_data_pasl(test_data):
    """Tests the AslQuantificationFilter with some mock data for PASL"""
    label_image_container = test_data.clone()
    # 1% signal difference
    label_image_container.image = label_image_container.image * 0.99
    input_params = {
        "control": test_data,
        "label": label_image_container,
        "m0": test_data,
        "label_type": "pasl",
        "model": "whitepaper",
        "lambda_blood_brain": 0.9,
        "label_duration": 0.8,
        "post_label_delay": 1.8,
        "label_efficiency": 0.85,
        "t1_arterial_blood": 1.65,
    }

    asl_quantification_filter = AslQuantificationFilter()
    asl_quantification_filter.add_inputs(input_params)
    asl_quantification_filter.run()

    numpy.testing.assert_array_equal(
        asl_quantification_filter.outputs["perfusion_rate"].image,
        AslQuantificationFilter.asl_quant_wp_pasl(
            test_data.image,
            label_image_container.image,
            test_data.image,
            input_params["lambda_blood_brain"],
            input_params["label_duration"],
            input_params["post_label_delay"],
            input_params["label_efficiency"],
            input_params["t1_arterial_blood"],
        ),
    )

    # check the image metadata
    assert asl_quantification_filter.outputs["perfusion_rate"].metadata == {
        "ComplexImageComponent": "REAL",
        "ImageType": [
            "DERIVED",
            "PRIMARY",
            "PERFUSION",
            "RCBF",
        ],
        "BackgroundSuppression": False,
        "VascularCrushing": False,
        "LabelingDuration": 1.8,
        "PostLabelingDelay": 1.8,
        "LabelingEfficiency": 0.85,
        "asl_context": "cbf",
        "Units": "ml/100g/min",
    }
