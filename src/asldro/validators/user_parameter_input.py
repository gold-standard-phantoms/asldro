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

# String constants
ASL_CONTEXT = "asl_context"
LABEL_TYPE = "label_type"
LABEL_DURATION = "label_duration"
SIGNAL_TIME = "signal_time"
LABEL_EFFICIENCY = "label_efficiency"
LAMBDA_BLOOD_BRAIN = "lambda_blood_brain"
T1_ARTERIAL_BLOOD = "t1_arterial_blood"
M0 = "m0"
ECHO_TIME = "echo_time"
REPETITION_TIME = "repetition_time"
ROT_Z = "rot_z"
ROT_Y = "rot_y"
ROT_X = "rot_x"
TRANSL_X = "transl_x"
TRANSL_Y = "transl_y"
TRANSL_Z = "transl_z"
ACQ_MATRIX = "acq_matrix"
ACQ_CONTRAST = "acq_contrast"
DESIRED_SNR = "desired_snr"
RANDOM_SEED = "random_seed"
EXCITATION_FLIP_ANGLE = "excitation_flip_angle"
INVERSION_FLIP_ANGLE = "inversion_flip_angle"
INVERSION_TIME = "inversion_time"

# Creates a validator which checks a parameter is the same
# length as the number of entries in asl_context
asl_context_length_validator_generator = lambda other: Validator(
    func=lambda d: ASL_CONTEXT in d
    and other in d
    and len(d[ASL_CONTEXT].split()) == len(d[other]),
    criteria_message=f"{other} must be present and have the same "
    f"number of entries as {ASL_CONTEXT}",
)

# Default parameters
DEFAULT_PARAMS = {
    ASL_CONTEXT: "m0scan control label",
    LABEL_TYPE: "pcasl",
    LABEL_DURATION: 1.8,
    SIGNAL_TIME: 3.6,
    LABEL_EFFICIENCY: 0.85,
    ECHO_TIME: [0.01, 0.01, 0.01],
    REPETITION_TIME: [10.0, 5.0, 5.0],
    ROT_Z: [0.0, 0.0, 0.0],
    ROT_Y: [0.0, 0.0, 0.0],
    ROT_X: [0.0, 0.0, 0.0],
    TRANSL_X: [0.0, 0.0, 0.0],
    TRANSL_Y: [0.0, 0.0, 0.0],
    TRANSL_Z: [0.0, 0.0, 0.0],
    ACQ_MATRIX: [64, 64, 12],
    ACQ_CONTRAST: "se",
    EXCITATION_FLIP_ANGLE: 90.0,
    INVERSION_FLIP_ANGLE: 180.0,
    INVERSION_TIME: 1.0,
    DESIRED_SNR: 10.0,
    RANDOM_SEED: 0,
}

# Input validator
USER_INPUT_VALIDATOR = ParameterValidator(
    parameters={
        LABEL_TYPE: Parameter(
            validators=from_list_validator(
                ["CASL", "PCASL", "PASL"], case_insensitive=True
            ),
            default_value=DEFAULT_PARAMS.get(LABEL_TYPE),
        ),
        LABEL_DURATION: Parameter(
            validators=range_inclusive_validator(0, 100),
            default_value=DEFAULT_PARAMS.get(LABEL_DURATION),
        ),
        SIGNAL_TIME: Parameter(
            validators=range_inclusive_validator(0, 100),
            default_value=DEFAULT_PARAMS.get(SIGNAL_TIME),
        ),
        LABEL_EFFICIENCY: Parameter(
            validators=range_inclusive_validator(0, 1),
            default_value=DEFAULT_PARAMS.get(LABEL_EFFICIENCY),
        ),
        LAMBDA_BLOOD_BRAIN: Parameter(
            validators=range_inclusive_validator(0, 1),
            optional=True,
            default_value=DEFAULT_PARAMS.get(LAMBDA_BLOOD_BRAIN),
        ),
        T1_ARTERIAL_BLOOD: Parameter(
            validators=range_inclusive_validator(0, 100),
            optional=True,
            default_value=DEFAULT_PARAMS.get(T1_ARTERIAL_BLOOD),
        ),
        M0: Parameter(validators=greater_than_equal_to_validator(0), optional=True),
        ASL_CONTEXT: Parameter(
            validators=reserved_string_list_validator(
                ["m0scan", "control", "label"], case_insensitive=True
            ),
            default_value=DEFAULT_PARAMS.get(ASL_CONTEXT),
        ),
        ECHO_TIME: Parameter(
            validators=[
                list_of_type_validator((int, float)),
                non_empty_list_validator(),
            ],
            default_value=DEFAULT_PARAMS.get(ECHO_TIME),
        ),
        REPETITION_TIME: Parameter(
            validators=[
                list_of_type_validator((int, float)),
                non_empty_list_validator(),
            ],
            default_value=DEFAULT_PARAMS.get(REPETITION_TIME),
        ),
        ROT_Z: Parameter(
            validators=for_each_validator(range_inclusive_validator(-180, 180)),
            default_value=DEFAULT_PARAMS.get(ROT_Z),
        ),
        ROT_Y: Parameter(
            validators=for_each_validator(range_inclusive_validator(-180, 180)),
            default_value=DEFAULT_PARAMS.get(ROT_Y),
        ),
        ROT_X: Parameter(
            validators=for_each_validator(range_inclusive_validator(-180, 180)),
            default_value=DEFAULT_PARAMS.get(ROT_X),
        ),
        TRANSL_Z: Parameter(
            validators=for_each_validator(range_inclusive_validator(-1000, 1000)),
            default_value=DEFAULT_PARAMS.get(TRANSL_Z),
        ),
        TRANSL_Y: Parameter(
            validators=for_each_validator(range_inclusive_validator(-1000, 1000)),
            default_value=DEFAULT_PARAMS.get(TRANSL_Y),
        ),
        TRANSL_X: Parameter(
            validators=for_each_validator(range_inclusive_validator(-1000, 1000)),
            default_value=DEFAULT_PARAMS.get(TRANSL_X),
        ),
        ACQ_MATRIX: Parameter(
            validators=[list_of_type_validator(int), of_length_validator(3)],
            default_value=DEFAULT_PARAMS.get(ACQ_MATRIX),
        ),
        ACQ_CONTRAST: Parameter(
            validators=from_list_validator(["ge", "se", "ir"], case_insensitive=True),
            default_value=DEFAULT_PARAMS.get(ACQ_CONTRAST),
        ),
        DESIRED_SNR: Parameter(
            validators=greater_than_equal_to_validator(0),
            default_value=DEFAULT_PARAMS.get(DESIRED_SNR),
        ),
        RANDOM_SEED: Parameter(
            validators=greater_than_equal_to_validator(0),
            default_value=DEFAULT_PARAMS.get(RANDOM_SEED),
        ),
        EXCITATION_FLIP_ANGLE: Parameter(
            validators=range_inclusive_validator(-180.0, 180.0),
            default_value=DEFAULT_PARAMS.get(EXCITATION_FLIP_ANGLE),
        ),
        INVERSION_FLIP_ANGLE: Parameter(
            validators=range_inclusive_validator(-180.0, 180.0),
            default_value=DEFAULT_PARAMS.get(INVERSION_FLIP_ANGLE),
        ),
        INVERSION_TIME: Parameter(
            validators=greater_than_equal_to_validator(0.0),
            default_value=DEFAULT_PARAMS.get(INVERSION_TIME),
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
