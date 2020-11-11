""" JSON file loader filter """

import os
import json

from jsonschema.exceptions import ValidationError

from asldro.filters.basefilter import BaseFilter, FilterInputValidationError
from asldro.validators.ground_truth_json import validate_input


class JsonLoaderFilter(BaseFilter):
    """A filter for loading a JSON file.

    Must have a single string input named 'filename'.

    Creates a multiple outputs, based on the root
    key,value pairs in the JSON filter. For example:
    { "foo": 1, "bar": "test"} will create two outputs named
    "foo" and "bar" with integer and string values respectively.
    The outputs may also be nested i.e. object or arrays.
    However - it will only load JSON file matching the validation schema
    in asldro.validators.ground_truth_json
    """

    def __init__(self):
        super().__init__("JsonLoader")

    def _run(self):
        """Load the input `filename`. Create the relevant
        outputs."""
        with open(self.inputs["filename"]) as file:
            self.outputs = json.load(file)

    def _validate_inputs(self):
        """There must be an input named `filename`.
        It must end in .json. It must
        point to a existing file. The file contents must
        be valid as per the JSON ground truth schema"""

        if self.inputs.get("filename", None) is None:
            raise FilterInputValidationError(
                "JsonLoader filter requires a `filename` input"
            )
        if not isinstance(self.inputs["filename"], str):
            raise FilterInputValidationError(
                "JsonLoader filter `filename` input must be a string"
            )
        if not self.inputs["filename"].endswith(".json"):
            raise FilterInputValidationError(
                "JsonLoader filter `filename` must be a .json file"
            )

        if not (
            os.path.exists(self.inputs["filename"])
            and os.path.isfile(self.inputs["filename"])
        ):
            raise FilterInputValidationError(
                f"{self.inputs['filename']} does not exist"
            )

        with open(self.inputs["filename"]) as file:
            try:
                validate_input(json.load(file))
            except ValidationError as error:
                raise FilterInputValidationError from error
