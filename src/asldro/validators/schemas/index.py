"""Index of the JSON schema files"""
import os
import json
from jsonschema import Draft7Validator

THIS_DIR = os.path.dirname(os.path.realpath(__file__))


def load_schemas():
    """Return all of the schemas in this directory in a dictionary where
    the keys are the filename (without the .json extension) and the values
    are the JSON schemas (in dictionary format)
    :raises jsonschema.exceptions.SchemaError if any of the JSON files in this
    directory are not valid (Draft 7) JSON schemas"""
    schemas = {}

    for filename in os.listdir(THIS_DIR):
        if (
            os.path.isfile(os.path.join(THIS_DIR, filename))
            and os.path.splitext(filename)[1].lower() == ".json"
        ):
            key = os.path.splitext(filename)[0]
            with open(os.path.join(THIS_DIR, filename)) as file_obj:
                value = json.load(file_obj)
                Draft7Validator.check_schema(value)
            schemas[key] = value

    return schemas


SCHEMAS = load_schemas()
