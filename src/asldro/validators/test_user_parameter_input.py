""" Tests some user inputs to the model to make sure the validation is performed correctly """
from copy import deepcopy
import pytest
from asldro.validators.parameters import ValidationError
from asldro.validators.user_parameter_input import USER_INPUT_VALIDATOR


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
    assert d == USER_INPUT_VALIDATOR.validate(
        d
    )  # the same dictionary should be returned


def test_user_input_defaults_created():
    """ Test default values are created for missing inputs """
    d = {}
    assert USER_INPUT_VALIDATOR.validate(d) == {
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
        "desired_snr": 10,
        "acq_matrix": [64, 64, 12],
        "acq_contrast": "se",
        "random_seed": 0,
        "excitation_flip_angle": 90.0,
        "inversion_flip_angle": 180.0,
        "inversion_time": 1.0,
    }


def test_mismatch_asl_context_array_sizes():
    """ Check that if the length of any of:
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
    USER_INPUT_VALIDATOR.validate(good_input)  # no exception

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
            USER_INPUT_VALIDATOR.validate(d)
