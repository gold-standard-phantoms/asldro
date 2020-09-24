""" A validator for the JSON file used in the ground truth input """
from jsonschema import validate

schema = {
    "type": "object",
    "required": ["quantities", "segmentation", "parameters"],
    "properties": {
        "quantities": {"type": "array", "items": {"type": "string"}},
        "segmentation": {"type": "object", "additionalProperties": {"type": "integer"}},
        "parameters": {
            "type": "object",
            "required": ["lambda_blood_brain", "t1_arterial_blood"],
            "properties": {
                "lambda_blood_brain": {"type": "number"},
                "t1_arterial_blood": {"type": "number"},
            },
        },
    },
    "additionalProperties": False,
}


def validate_input(input_dict: dict):
    """ Validates the provided dictionary against the ground truth schema.
    Raises a jsonschema.exceptions.ValidationError on error """
    validate(instance=input_dict, schema=schema)
