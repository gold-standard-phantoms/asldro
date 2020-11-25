""" JSON file loader filter """

import os
import json

import jsonschema

from asldro.filters.basefilter import BaseFilter, FilterInputValidationError


class JsonLoaderFilter(BaseFilter):
    """A filter for loading a JSON file.

    **Inputs**

    Input parameters are all keyword arguments for the :class:`JsonLoaderFilter.add_inputs()`
    member function.  They are also accessible via class constants, for example
    :class:`JsonLoaderFilter.KEY_FILENAME`.

    :param 'filename': The path to the JSON file to load
    :type 'filename': str
    :param 'schema': (optional) The schema to validate against (in python dict format). Some schemas
    can be found in asldro.validators.schemas, or one can just in input here.
    :param 'schema': dict

    **Outputs**

    Creates a multiple outputs, based on the root
    key,value pairs in the JSON filter. For example:
    { "foo": 1, "bar": "test"} will create two outputs named
    "foo" and "bar" with integer and string values respectively.
    The outputs may also be nested i.e. object or arrays.
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
        point to a existing file.
        If 'schema' is defined, it must exist in the schema database and
        the input JSON must be verified by the schema."""

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

        if "schema" in self.inputs:
            with open(self.inputs["filename"]) as file:
                try:
                    jsonschema.validate(json.load(file), self.inputs["schema"])
                except jsonschema.ValidationError as error:
                    raise FilterInputValidationError from error
