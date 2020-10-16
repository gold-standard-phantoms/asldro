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
    Validator,
    Parameter,
    range_inclusive_validator,
    greater_than_equal_to_validator,
    from_list_validator,
    reserved_string_list_validator,
    non_empty_list_validator,
    list_of_type_validator,
    of_length_validator,
    for_each_validator,
)

ASL_CONTEXT = "asl_context"
ECHO_TIME = "echo_time"
REPETITION_TIME = "repetition_time"
ROT_Z = "rot_z"
ROT_Y = "rot_y"
ROT_X = "rot_x"
TRANSL_X = "transl_x"
TRANSL_Y = "transl_y"
TRANSL_Z = "transl_z"

# Creates a validator which checks a parameter is the same
# length as the number of entries in asl_context
asl_context_length_validator_generator = lambda other: Validator(
    func=lambda d: ASL_CONTEXT in d
    and other in d
    and len(d[ASL_CONTEXT].split()) == len(d[other]),
    criteria_message=f"{other} must be present and have the same "
    f"number of entries as {ASL_CONTEXT}",
)

USER_INPUT_VALIDATOR = ParameterValidator(
    parameters={
        "label_type": Parameter(
            validators=from_list_validator(
                ["CASL", "PCASL", "PASL"], case_insensitive=True
            ),
            default_value="pcasl",
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
        ASL_CONTEXT: Parameter(
            validators=reserved_string_list_validator(
                ["m0scan", "control", "label"], case_insensitive=True
            ),
            default_value="m0scan control label",
        ),
        ECHO_TIME: Parameter(
            validators=[
                list_of_type_validator((int, float)),
                non_empty_list_validator(),
            ],
            default_value=[0.01, 0.01, 0.01],
        ),
        REPETITION_TIME: Parameter(
            validators=[
                list_of_type_validator((int, float)),
                non_empty_list_validator(),
            ],
            default_value=[10.0, 5.0, 5.0],
        ),
        ROT_Z: Parameter(
            validators=for_each_validator(range_inclusive_validator(-180, 180)),
            default_value=[0.0, 0.0, 0.0],
        ),
        ROT_Y: Parameter(
            validators=for_each_validator(range_inclusive_validator(-180, 180)),
            default_value=[0.0, 0.0, 0.0],
        ),
        ROT_X: Parameter(
            validators=for_each_validator(range_inclusive_validator(-180, 180)),
            default_value=[0.0, 0.0, 0.0],
        ),
        TRANSL_Z: Parameter(
            validators=for_each_validator(range_inclusive_validator(-1000, 1000)),
            default_value=[0.0, 0.0, 0.0],
        ),
        TRANSL_Y: Parameter(
            validators=for_each_validator(range_inclusive_validator(-1000, 1000)),
            default_value=[0.0, 0.0, 0.0],
        ),
        TRANSL_X: Parameter(
            validators=for_each_validator(range_inclusive_validator(-1000, 1000)),
            default_value=[0.0, 0.0, 0.0],
        ),
        "acq_matrix": Parameter(
            validators=[list_of_type_validator(int), of_length_validator(3)],
            default_value=[64, 64, 12],
        ),
        "acq_contrast": Parameter(
            validators=from_list_validator(["ge", "se"], case_insensitive=True),
            default_value="se",
        ),
        "desired_snr": Parameter(
            validators=greater_than_equal_to_validator(0), default_value=10
        ),
        "random_seed": Parameter(
            validators=greater_than_equal_to_validator(0), default_value=0
        ),
    },
    post_validators=[
        Validator(
            func=lambda d: ASL_CONTEXT in d,
            criteria_message=f"{ASL_CONTEXT} must be supplied",
        )
    ]
    + [
        asl_context_length_validator_generator(param)
        for param in [
            ECHO_TIME,
            REPETITION_TIME,
            ROT_Z,
            ROT_Y,
            ROT_X,
            TRANSL_Z,
            TRANSL_Y,
            TRANSL_X,
        ]
    ],
)
