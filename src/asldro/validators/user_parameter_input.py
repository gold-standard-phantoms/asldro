"""
A user input validator. Used to initialise the model.
All of the validation rules are contained within this file.
The validator may be used with:
`d = USER_INPUT_VALIDATOR(some_input_dictionary)`
`d` will now contain the input dictionary with any defaults values added.
A ValidationError will be raised if any validation rules fail.
"""
import os
from copy import deepcopy
import jsonschema
from asldro.data.filepaths import GROUND_TRUTH_DATA

from asldro.validators.parameters import (
    ParameterValidator,
    Validator,
    ValidationError,
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
from asldro.validators.schemas.index import SCHEMAS
from asldro.utils.general import splitext

INPUT_PARAMETER_SCHEMA = SCHEMAS["input_params"]

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
MODALITY = "modality"

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
SUPPORTED_IMAGE_TYPES = [ASL, GROUND_TRUTH, STRUCTURAL]

# Supported asl contexts
M0SCAN = "m0scan"
CONTROL = "control"
LABEL = "label"
SUPPORTED_ASL_CONTEXTS = [M0SCAN, CONTROL, LABEL]

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
            MODALITY: Parameter(
                validators=from_list_validator(["T1w", "T2w", "FLAIR", "anat"]),
                default_value="anat",
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
                validators=greater_than_equal_to_validator(0), default_value=100.0
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


def validate_input_params(input_params: dict) -> dict:
    """
    Validate the input parameters
    :param input_params: The input parameters asa Python dict
    :returns: The parsed input parameter dictionary, with any defaults added
    :raises asldro.validators.parameters.ValidationError: if the input
        validation does not pass
    """
    # Check that the input parameters validate against the input parameter schema
    # This checks the general structure of the input, but does not validate the
    # series parameters
    try:
        jsonschema.validate(instance=input_params, schema=INPUT_PARAMETER_SCHEMA)
    except jsonschema.exceptions.ValidationError as ex:
        # Make the type of exception raised consistent
        raise ValidationError from ex

    validated_input_params = deepcopy(input_params)
    # For every image series
    for image_series in validated_input_params["image_series"]:
        # Perform the parameter validation based on the 'series_type'
        # (and insert defaults)
        if "series_parameters" not in image_series:
            image_series["series_parameters"] = {}
        image_series["series_parameters"] = IMAGE_TYPE_VALIDATOR[
            image_series["series_type"]
        ].validate(image_series["series_parameters"])

    # Determine whether the ground truth is a valid filename (and exists)
    # or is a pre-existing dataset in the asldro data
    ground_truth_params = validated_input_params["global_configuration"]["ground_truth"]

    if isinstance(ground_truth_params, dict):
        # The input is already a dict with the filename included, so don't do anything
        pass
    elif ground_truth_params in GROUND_TRUTH_DATA.keys():
        # The input is a string - use it to look up the relevant files from the
        # included datasets
        # Replace the 'ground_truth' with the paths to the nii.gz and json files
        validated_input_params["global_configuration"]["ground_truth"] = deepcopy(
            GROUND_TRUTH_DATA[ground_truth_params]
        )
    else:
        # Assume the ground_truth_str is a path to the nifti file, and there is an
        # associated json file
        if not ground_truth_params.endswith((".nii", ".nii.gz")):
            raise ValidationError(
                f"The ground truth {ground_truth_params} must be one of: "
                f'{". ".join(GROUND_TRUTH_DATA.keys())} or be a .nii or .nii.gz file'
            )
        validated_input_params["global_configuration"]["ground_truth"] = {
            "nii": ground_truth_params,
            "json": splitext(ground_truth_params)[0] + ".json",
        }

    ground_truth_dict = validated_input_params["global_configuration"]["ground_truth"]
    for filetype in ["json", "nii"]:
        if not (
            os.path.exists(ground_truth_dict[filetype])
            and os.path.isfile(ground_truth_dict[filetype])
        ):
            raise ValidationError(
                f"Ground truth file {ground_truth_dict[filetype]} does not exist"
            )

    return validated_input_params


def get_example_input_params() -> dict:
    """Generate and validate an example input parameter dictionary.
    Will contain one of each supported image type containing the
    default parameters for each.
    :return: the validated input parameter dictionary
    :raises asldro.validators.parameters.ValidationError: if the input
        validation does not pass
    """
    return validate_input_params(
        {
            "global_configuration": {
                "ground_truth": "hrgt_icbm_2009a_nls_3t",
                "image_override": {},
                "parameter_override": {},
            },
            "image_series": [
                {
                    "series_type": IMAGE_TYPE,
                    "series_description": f"user description for {IMAGE_TYPE}",
                }
                for IMAGE_TYPE in SUPPORTED_IMAGE_TYPES
            ],
        }
    )
