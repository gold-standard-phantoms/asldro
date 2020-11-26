""" A validator for the JSON file used in the ground truth input """
from jsonschema import validate

from asldro.validators.schemas.index import SCHEMAS

schema = SCHEMAS["ground_truth"]


def validate_input(input_dict: dict):
    """Validates the provided dictionary against the ground truth schema.
    Raises a jsonschema.exceptions.ValidationError on error"""
    validate(instance=input_dict, schema=schema)
