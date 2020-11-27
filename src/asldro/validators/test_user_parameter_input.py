""" Tests some user inputs to the model to make sure the validation is performed correctly """
# pylint: disable=redefined-outer-name
from copy import deepcopy
import pytest
from asldro.data.filepaths import GROUND_TRUTH_DATA
from asldro.validators.parameters import ValidationError
from asldro.validators.user_parameter_input import (
    IMAGE_TYPE_VALIDATOR,
    ASL,
    GROUND_TRUTH,
    STRUCTURAL,
    validate_input_params,
    get_example_input_params,
)


def test_user_input_valid():
    """ Tests a valid set of inputs """
    d = {
        "label_type": "PASL",
        "label_duration": 2.0,
        "signal_time": 2.5,
        "label_efficiency": 0.8,
        "lambda_blood_brain": 0.9,
        "t1_arterial_blood": 85,
        "m0": 0.7,
        "asl_context": "m0scan control label control label",
        "echo_time": [0, 1, 2, 3, 4],
        "repetition_time": [3, 4.5, 5, 6.4, 1.2],
        "rot_z": [-180, 180, 0, 0, 0],
        "rot_y": [0.0, 180.0, 0.0, 180.0, 1.2],
        "rot_x": [-180.0, 0, 0.2, 3.0, 1.3],
        "transl_x": [-1000, 0.0, 5.6, 6.7, 7.8],
        "transl_y": [0.0, 1000.0, 0.3, 100.6, 2.3],
        "transl_z": [5.6, 1.3, 1.2, 1.3, 1.2],
        "desired_snr": 5.0,
        "acq_matrix": [8, 9, 10],
        "acq_contrast": "se",
        "random_seed": 123_871_263,
        "excitation_flip_angle": 35.6,
        "inversion_flip_angle": 164.0,
        "inversion_time": 1.0,
    }
    assert d == IMAGE_TYPE_VALIDATOR[ASL].validate(
        d
    )  # the same dictionary should be returned


def test_asl_user_input_defaults_created():
    """ Test default values for the asl image type """
    correct_defaults = {
        "label_type": "pcasl",
        "asl_context": "m0scan control label",
        "echo_time": [0.01, 0.01, 0.01],
        "repetition_time": [10.0, 5.0, 5.0],
        "rot_z": [0.0, 0.0, 0.0],
        "rot_y": [0.0, 0.0, 0.0],
        "rot_x": [0.0, 0.0, 0.0],
        "transl_x": [0.0, 0.0, 0.0],
        "transl_y": [0.0, 0.0, 0.0],
        "transl_z": [0.0, 0.0, 0.0],
        "label_duration": 1.8,
        "signal_time": 3.6,
        "label_efficiency": 0.85,
        "desired_snr": 100,
        "acq_matrix": [64, 64, 12],
        "acq_contrast": "se",
        "random_seed": 0,
        "excitation_flip_angle": 90.0,
        "inversion_flip_angle": 180.0,
        "inversion_time": 1.0,
    }

    # Validation should include inputs
    assert IMAGE_TYPE_VALIDATOR[ASL].validate({}) == correct_defaults
    # Get the defaults directly
    assert IMAGE_TYPE_VALIDATOR[ASL].get_defaults() == correct_defaults


def test_structural_user_input_defaults_created():
    """ Test default values for the structural image type """
    correct_defaults = {
        "echo_time": 0.005,
        "repetition_time": 0.3,
        "rot_z": 0.0,
        "rot_y": 0.0,
        "rot_x": 0.0,
        "transl_x": 0.0,
        "transl_y": 0.0,
        "transl_z": 0.0,
        "acq_matrix": [197, 233, 189],
        "acq_contrast": "se",
        "excitation_flip_angle": 90.0,
        "inversion_flip_angle": 180.0,
        "inversion_time": 1.0,
        "desired_snr": 50.0,
        "random_seed": 0,
        "output_image_type": "magnitude",
        "modality": "anat",
    }

    # Validation should include inputs
    assert IMAGE_TYPE_VALIDATOR[STRUCTURAL].validate({}) == correct_defaults
    # Get the defaults directly
    assert IMAGE_TYPE_VALIDATOR[STRUCTURAL].get_defaults() == correct_defaults


def test_ground_truth_user_input_defaults_created():
    """ Test default values for the ground_truth image type """
    correct_defaults = {
        "rot_z": 0.0,
        "rot_y": 0.0,
        "rot_x": 0.0,
        "transl_x": 0.0,
        "transl_y": 0.0,
        "transl_z": 0.0,
        "acq_matrix": [64, 64, 12],
    }

    # Validation should include inputs
    assert IMAGE_TYPE_VALIDATOR[GROUND_TRUTH].validate({}) == correct_defaults
    # Get the defaults directly
    assert IMAGE_TYPE_VALIDATOR[GROUND_TRUTH].get_defaults() == correct_defaults


def test_mismatch_asl_context_array_sizes():
    """Check that if the length of any of:
    - echo_time
    - repetition_time
    - rot_z
    - rot_y
    - rot_x
    - transl_x
    - transl_y
    - transl_z
    does not match the number of items in asl_context, a ValidationError
    will be raised with an appropriate error message
    """
    good_input = {
        "label_type": "PASL",
        "asl_context": "m0scan control label",
        "echo_time": [0.01, 0.01, 0.01],
        "repetition_time": [10.0, 5.0, 5.0],
        "rot_z": [0.0, 0.0, 0.0],
        "rot_y": [0.0, 0.0, 0.0],
        "rot_x": [0.0, 0.0, 0.0],
        "transl_x": [0.0, 0.0, 0.0],
        "transl_y": [0.0, 0.0, 0.0],
        "transl_z": [0.0, 0.0, 0.0],
    }
    IMAGE_TYPE_VALIDATOR[ASL].validate(good_input)  # no exception

    for param in [
        "echo_time",
        "repetition_time",
        "rot_x",
        "rot_y",
        "rot_z",
        "transl_z",
        "transl_y",
        "transl_x",
    ]:
        d = deepcopy(good_input)
        d[param] = [0.1, 0.2, 0.3, 0.4]  # wrong number of parameters

        with pytest.raises(
            ValidationError,
            match=f"{param} must be present and have the same number of entries as asl_context",
        ):
            IMAGE_TYPE_VALIDATOR[ASL].validate(d)


@pytest.fixture
def input_params():
    """ A valid input parameter config """
    return {
        "global_configuration": {
            "ground_truth": "hrgt_icbm_2009a_nls_3t",
            "image_override": {"m0": 5.0},
            "parameter_override": {"lambda_blood_brain": 0.85},
            "ground_truth_modulate": {
                "t1": {"scale": 0.5},
                "t2": {"offset": 2},
                "m0": {"scale": 2, "offset": 1.5},
            },
        },
        "image_series": [
            {
                "series_type": "asl",
                "series_description": "user description for asl",
                "series_parameters": {
                    "asl_context": "m0scan control label",
                    "label_type": "pcasl",
                    "acq_matrix": [64, 64, 20],
                },
            },
            {
                "series_type": "structural",
                "series_description": "user description for structural scan",
                "series_parameters": {
                    "acq_contrast": "ge",
                    "echo_time": 0.05,
                    "repetition_time": 0.3,
                    "acq_matrix": [256, 256, 128],
                },
            },
            {
                "series_type": "ground_truth",
                "series_description": "user description for ground truth",
                "series_parameters": {"acq_matrix": [64, 64, 20]},
            },
        ],
    }


@pytest.fixture(name="expected_parsed_input")
def fixture_expected_parsed_input():
    return {
        "global_configuration": {
            "ground_truth": {
                "nii": GROUND_TRUTH_DATA["hrgt_icbm_2009a_nls_3t"]["nii"],
                "json": GROUND_TRUTH_DATA["hrgt_icbm_2009a_nls_3t"]["json"],
            },
            "image_override": {"m0": 5.0},
            "parameter_override": {"lambda_blood_brain": 0.85},
            "ground_truth_modulate": {
                "t1": {"scale": 0.5},
                "t2": {"offset": 2},
                "m0": {"scale": 2, "offset": 1.5},
            },
        },
        "image_series": [
            {
                "series_type": "asl",
                "series_description": "user description for asl",
                "series_parameters": {
                    "asl_context": "m0scan control label",
                    "label_type": "pcasl",
                    "acq_matrix": [64, 64, 20],
                    "echo_time": [0.01, 0.01, 0.01],
                    "repetition_time": [10.0, 5.0, 5.0],
                    "rot_z": [0.0, 0.0, 0.0],
                    "rot_y": [0.0, 0.0, 0.0],
                    "rot_x": [0.0, 0.0, 0.0],
                    "transl_x": [0.0, 0.0, 0.0],
                    "transl_y": [0.0, 0.0, 0.0],
                    "transl_z": [0.0, 0.0, 0.0],
                    "label_duration": 1.8,
                    "signal_time": 3.6,
                    "label_efficiency": 0.85,
                    "desired_snr": 100.0,
                    "acq_contrast": "se",
                    "random_seed": 0,
                    "excitation_flip_angle": 90.0,
                    "inversion_flip_angle": 180.0,
                    "inversion_time": 1.0,
                },
            },
            {
                "series_type": "structural",
                "series_description": "user description for structural scan",
                "series_parameters": {
                    "echo_time": 0.05,
                    "repetition_time": 0.3,
                    "rot_z": 0.0,
                    "rot_y": 0.0,
                    "rot_x": 0.0,
                    "transl_x": 0.0,
                    "transl_y": 0.0,
                    "transl_z": 0.0,
                    "acq_matrix": [256, 256, 128],
                    "acq_contrast": "ge",
                    "excitation_flip_angle": 90.0,
                    "inversion_flip_angle": 180.0,
                    "inversion_time": 1.0,
                    "desired_snr": 50.0,
                    "random_seed": 0,
                    "output_image_type": "magnitude",
                    "modality": "anat",
                },
            },
            {
                "series_type": "ground_truth",
                "series_description": "user description for ground truth",
                "series_parameters": {
                    "acq_matrix": [64, 64, 20],
                    "rot_z": 0.0,
                    "rot_y": 0.0,
                    "rot_x": 0.0,
                    "transl_x": 0.0,
                    "transl_y": 0.0,
                    "transl_z": 0.0,
                },
            },
        ],
    }


def test_valid_input_params(input_params: dict, expected_parsed_input: dict):
    """Test that a valid input parameter file is parsed without
    raising an exception and that the appropriate defaults are inserted"""
    # Should not raise an exception
    parsed_input = validate_input_params(input_params)

    assert parsed_input == expected_parsed_input

    # Also, try changing the ground_truth to the nifti file
    # in the HRGT data (JSON file assumed same name)
    input_params["global_configuration"]["ground_truth"] = GROUND_TRUTH_DATA[
        "hrgt_icbm_2009a_nls_3t"
    ]["nii"]
    # Should not raise an exception
    parsed_input = validate_input_params(input_params)

    assert parsed_input == expected_parsed_input

    # Also, try changing the ground_truth to the nifti file/json file
    # in the HRGT data
    input_params["global_configuration"]["ground_truth"] = {
        "nii": GROUND_TRUTH_DATA["hrgt_icbm_2009a_nls_3t"]["nii"],
        "json": GROUND_TRUTH_DATA["hrgt_icbm_2009a_nls_3t"]["json"],
    }
    # Should not raise an exception
    parsed_input = validate_input_params(input_params)

    assert parsed_input == expected_parsed_input


def test_invalid_data_input_params(input_params: dict):
    """Tests that bad ground_truth data set in the input parameters
    raises appropriate Expections (should always be
    asldro.validators.parameters.ValidationError)"""

    input_params["global_configuration"]["ground_truth"] = "i_dont_exist"
    with pytest.raises(ValidationError):
        validate_input_params(input_params)
    input_params["global_configuration"]["image_override"] = "a_string"
    with pytest.raises(ValidationError):
        validate_input_params(input_params)

    input_params["global_configuration"]["image_override"] = {"m0": "a_string"}
    with pytest.raises(ValidationError):
        validate_input_params(input_params)


def test_bad_series_type_input_params(input_params: dict):
    """Tests that bad series_type data set in the input parameters
    raises appropriate Expections (should always be
    asldro.validators.parameters.ValidationError)"""

    input_params["image_series"][0]["series_type"] = "magic"
    with pytest.raises(ValidationError):
        validate_input_params(input_params)


def test_missing_series_parameters_inserts_defaults(input_params: dict):
    """Tests that if series_parameters are completely missing for
    an image series, the defaults are inserted"""

    input_params["image_series"][0].pop("series_parameters")

    # The default series parameters should be added
    assert validate_input_params(input_params)["image_series"][0] == {
        "series_type": "asl",
        "series_description": "user description for asl",
        "series_parameters": {
            "asl_context": "m0scan control label",
            "label_type": "pcasl",
            "acq_matrix": [64, 64, 12],
            "echo_time": [0.01, 0.01, 0.01],
            "repetition_time": [10.0, 5.0, 5.0],
            "rot_z": [0.0, 0.0, 0.0],
            "rot_y": [0.0, 0.0, 0.0],
            "rot_x": [0.0, 0.0, 0.0],
            "transl_x": [0.0, 0.0, 0.0],
            "transl_y": [0.0, 0.0, 0.0],
            "transl_z": [0.0, 0.0, 0.0],
            "label_duration": 1.8,
            "signal_time": 3.6,
            "label_efficiency": 0.85,
            "desired_snr": 100.0,
            "acq_contrast": "se",
            "random_seed": 0,
            "excitation_flip_angle": 90.0,
            "inversion_flip_angle": 180.0,
            "inversion_time": 1.0,
        },
    }


def test_example_input_params_valid():
    """Just test that the generated example input parameters pass
    the validation (validated internally)"""
    validate_input_params(get_example_input_params())
