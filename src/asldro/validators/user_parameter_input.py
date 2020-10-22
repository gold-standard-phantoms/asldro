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
    greater_than_validator,
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
OUTPUT_IMAGE_TYPE = "output_image_type"

# Creates a validator which checks a parameter is the same
# length as the number of entries in asl_context
asl_context_length_validator_generator = lambda other: Validator(
    func=lambda d: ASL_CONTEXT in d
    and other in d
    and len(d[ASL_CONTEXT].split()) == len(d[other]),
    criteria_message=f"{other} must be present and have the same "
    f"number of entries as {ASL_CONTEXT}",
)


# Supported image types
ASL = "asl"
GROUND_TRUTH = "ground_truth"
STRUCTURAL = "structural"

# Input validator
IMAGE_TYPE_VALIDATOR = {
    GROUND_TRUTH: ParameterValidator(
        parameters={
            ROT_X: Parameter(
                validators=range_inclusive_validator(-180.0, 180.0), default_value=0.0
            ),
            ROT_Y: Parameter(
                validators=range_inclusive_validator(-180.0, 180.0), default_value=0.0
            ),
            ROT_Z: Parameter(
                validators=range_inclusive_validator(-180.0, 180.0), default_value=0.0
            ),
            TRANSL_X: Parameter(
                validators=range_inclusive_validator(-1000.0, 1000.0), default_value=0.0
            ),
            TRANSL_Y: Parameter(
                validators=range_inclusive_validator(-1000.0, 1000.0), default_value=0.0
            ),
            TRANSL_Z: Parameter(
                validators=range_inclusive_validator(-1000.0, 1000.0), default_value=0.0
            ),
            ACQ_MATRIX: Parameter(
                validators=[
                    list_of_type_validator(int),
                    of_length_validator(3),
                    for_each_validator(greater_than_validator(0)),
                ],
                default_value=[64, 64, 12],
            ),
        }
    ),
    STRUCTURAL: ParameterValidator(
        parameters={
            ECHO_TIME: Parameter(
                validators=greater_than_validator(0), default_value=0.005
            ),
            REPETITION_TIME: Parameter(
                validators=greater_than_validator(0), default_value=0.3
            ),
            ROT_X: Parameter(
                validators=range_inclusive_validator(-180.0, 180.0), default_value=0.0
            ),
            ROT_Y: Parameter(
                validators=range_inclusive_validator(-180.0, 180.0), default_value=0.0
            ),
            ROT_Z: Parameter(
                validators=range_inclusive_validator(-180.0, 180.0), default_value=0.0
            ),
            TRANSL_X: Parameter(
                validators=range_inclusive_validator(-1000.0, 1000.0), default_value=0.0
            ),
            TRANSL_Y: Parameter(
                validators=range_inclusive_validator(-1000.0, 1000.0), default_value=0.0
            ),
            TRANSL_Z: Parameter(
                validators=range_inclusive_validator(-1000.0, 1000.0), default_value=0.0
            ),
            ACQ_MATRIX: Parameter(
                validators=[
                    list_of_type_validator(int),
                    of_length_validator(3),
                    for_each_validator(greater_than_validator(0)),
                ],
                default_value=[197, 233, 189],
            ),
            ACQ_CONTRAST: Parameter(
                validators=from_list_validator(
                    ["ge", "se", "ir"], case_insensitive=True
                ),
                default_value="se",
            ),
            EXCITATION_FLIP_ANGLE: Parameter(
                validators=range_inclusive_validator(-180.0, 180.0), default_value=90.0
            ),
            INVERSION_FLIP_ANGLE: Parameter(
                validators=range_inclusive_validator(-180.0, 180.0), default_value=180.0
            ),
            INVERSION_TIME: Parameter(
                validators=greater_than_equal_to_validator(0.0), default_value=1.0
            ),
            DESIRED_SNR: Parameter(
                validators=greater_than_equal_to_validator(0), default_value=50.0
            ),
            RANDOM_SEED: Parameter(
                validators=greater_than_equal_to_validator(0), default_value=0
            ),
            OUTPUT_IMAGE_TYPE: Parameter(
                validators=from_list_validator(["complex", "magnitude"]),
                default_value="magnitude",
            ),
        }
    ),
    ASL: ParameterValidator(
        parameters={
            ROT_X: Parameter(
                validators=for_each_validator(range_inclusive_validator(-180, 180)),
                default_value=[0.0, 0.0, 0.0],
            ),
            ROT_Y: Parameter(
                validators=for_each_validator(range_inclusive_validator(-180, 180)),
                default_value=[0.0, 0.0, 0.0],
            ),
            ROT_Z: Parameter(
                validators=for_each_validator(range_inclusive_validator(-180, 180)),
                default_value=[0.0, 0.0, 0.0],
            ),
            TRANSL_X: Parameter(
                validators=for_each_validator(range_inclusive_validator(-1000, 1000)),
                default_value=[0.0, 0.0, 0.0],
            ),
            TRANSL_Y: Parameter(
                validators=for_each_validator(range_inclusive_validator(-1000, 1000)),
                default_value=[0.0, 0.0, 0.0],
            ),
            TRANSL_Z: Parameter(
                validators=for_each_validator(range_inclusive_validator(-1000, 1000)),
                default_value=[0.0, 0.0, 0.0],
            ),
            ACQ_MATRIX: Parameter(
                validators=[
                    list_of_type_validator(int),
                    of_length_validator(3),
                    for_each_validator(greater_than_validator(0)),
                ],
                default_value=[64, 64, 12],
            ),
            LABEL_TYPE: Parameter(
                validators=from_list_validator(
                    ["CASL", "PCASL", "PASL"], case_insensitive=True
                ),
                default_value="pcasl",
            ),
            LABEL_DURATION: Parameter(
                validators=range_inclusive_validator(0, 100), default_value=1.8
            ),
            SIGNAL_TIME: Parameter(
                validators=range_inclusive_validator(0, 100), default_value=3.6
            ),
            LABEL_EFFICIENCY: Parameter(
                validators=range_inclusive_validator(0, 1), default_value=0.85
            ),
            LAMBDA_BLOOD_BRAIN: Parameter(
                validators=range_inclusive_validator(0, 1), optional=True
            ),
            T1_ARTERIAL_BLOOD: Parameter(
                validators=range_inclusive_validator(0, 100), optional=True
            ),
            M0: Parameter(validators=greater_than_equal_to_validator(0), optional=True),
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
            ACQ_CONTRAST: Parameter(
                validators=from_list_validator(
                    ["ge", "se", "ir"], case_insensitive=True
                ),
                default_value="se",
            ),
            DESIRED_SNR: Parameter(
                validators=greater_than_equal_to_validator(0), default_value=10.0
            ),
            RANDOM_SEED: Parameter(
                validators=greater_than_equal_to_validator(0), default_value=0
            ),
            EXCITATION_FLIP_ANGLE: Parameter(
                validators=range_inclusive_validator(-180.0, 180.0), default_value=90.0
            ),
            INVERSION_FLIP_ANGLE: Parameter(
                validators=range_inclusive_validator(-180.0, 180.0), default_value=180.0
            ),
            INVERSION_TIME: Parameter(
                validators=greater_than_equal_to_validator(0.0), default_value=1.0
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
    ),
}

