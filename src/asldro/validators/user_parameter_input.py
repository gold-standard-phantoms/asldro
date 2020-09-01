"""
A user input validator. Used to initialise the model.
All of the validation rules are contained within this file.
The validator may be used with:
`d = USER_INPUT_VALIDATOR(some_input_dictionary)`
`d` will now contain the input dictionary with any defaults values added.
A ValidationError will be raised if any validation rules fail.
"""

from asldro.validators.parameters import (
    ParameterValidator,
    Parameter,
    range_inclusive_validator,
    greater_than_validator,
    greater_than_equal_to_validator,
    from_list_validator,
    reserved_string_list_validator,
    non_empty_list_validator,
    list_of_type_validator,
    for_each_validator,
)

USER_INPUT_VALIDATOR = ParameterValidator(
    parameters={
        "label_type": Parameter(
            validators=from_list_validator(
                ["CASL", "PCASL", "PASL"], case_insensitive=True
            )
        ),
        "label_duration": Parameter(
            validators=range_inclusive_validator(0, 100), default_value=1.8
        ),
        "signal_time": Parameter(
            validators=range_inclusive_validator(0, 100), default_value=3.6
        ),
        "label_efficiency": Parameter(
            validators=range_inclusive_validator(0, 1), default_value=1.0
        ),
        "lambda_blood_brain": Parameter(
            validators=range_inclusive_validator(0, 1), optional=True
        ),
        "t1_arterial_blood": Parameter(
            validators=range_inclusive_validator(0, 100), optional=True
        ),
        "m0": Parameter(validators=greater_than_equal_to_validator(0), optional=True),
        "asl_context_array": Parameter(
            validators=reserved_string_list_validator(
                ["m0scan", "control", "label"], case_insensitive=True
            )
        ),
        "te_array": Parameter(
            validators=[
                list_of_type_validator((int, float)),
                non_empty_list_validator(),
            ],
            optional=True,
        ),
        "tr_array": Parameter(
            validators=[
                list_of_type_validator((int, float)),
                non_empty_list_validator(),
            ],
            optional=True,
        ),
        "rot_yaw_array": Parameter(
            validators=for_each_validator(range_inclusive_validator(-180, 180)),
            optional=True,
        ),
        "rot_pitch_array": Parameter(
            validators=for_each_validator(range_inclusive_validator(-180, 180)),
            optional=True,
        ),
        "rot_roll_array": Parameter(
            validators=for_each_validator(range_inclusive_validator(-180, 180)),
            optional=True,
        ),
        "transl_x_array": Parameter(
            validators=for_each_validator(range_inclusive_validator(-1000, 1000)),
            optional=True,
        ),
        "transl_y_array": Parameter(
            validators=for_each_validator(range_inclusive_validator(-1000, 1000)),
            optional=True,
        ),
        "transl_z_array": Parameter(
            validators=for_each_validator(range_inclusive_validator(-1000, 1000)),
            optional=True,
        ),
        "desired_snr": Parameter(
            validators=greater_than_validator(0), default_value=10
        ),
    }
)
