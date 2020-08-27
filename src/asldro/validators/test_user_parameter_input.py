""" Tests some user inputs to the model to make sure the validation is performed correctly """
import pytest
from asldro.validators.parameters import ValidationError
from asldro.validators.user_parameter_input import USER_INPUT_VALIDATOR


def test_user_input_valid():
    """ Tests a valid set of inputs """
    d = {
        "label_type": "pCASL",
        "label_duration": 2.0,
        "signal_time": 2.5,
        "label_efficiency": 0.8,
        "lambda_blood_brain": 0.9,
        "t1_arterial_blood": 85,
        "m0": 0.7,
        "asl_context_array": "m0scan control label control label control label control label",
        "te_array": [0, 1, 2],
        "tr_array": [3, 4.5, 5],
        "rot_yaw_array": [-180, 180],
        "rot_pitch_array": [0.0, 180.0],
        "rot_roll_array": [-180.0, 0],
        "transl_x_array": [-1000, 0.0],
        "transl_y_array": [0.0, 1000.0],
        "transl_z_array": [5.6],
        "desired_snr": 5.0,
    }
    assert d == USER_INPUT_VALIDATOR.validate(
        d
    )  # the same dictionary should be returned


def test_user_input_defaults_created():
    """ Test default values are created for missing inputs """
    d = {
        "label_type": "pCASL",
        "asl_context_array": "m0scan control label control label control label control label",
    }
    assert USER_INPUT_VALIDATOR.validate(d) == {
        "label_type": "pCASL",
        "asl_context_array": "m0scan control label control label control label control label",
        "label_duration": 1.8,
        "signal_time": 3.6,
        "label_efficiency": 1.0,
        "desired_snr": 10,
    }


def test_user_input_missing_values():
    """ Check that missing non-optional values throws an exception """
    with pytest.raises(ValidationError):
        USER_INPUT_VALIDATOR.validate(
            {
                "asl_context_array": "m0scan control label control label control label control label"
            }
        )

    with pytest.raises(ValidationError):
        USER_INPUT_VALIDATOR.validate({"label_type": "pCASL"})
