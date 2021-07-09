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
    :type 'schema': dict
    :param 'root_object_name': Optionally place all of the key-value pairs inside this
      object
    :type 'root_object_name': str

    **Outputs**

    Creates a multiple outputs, based on the root
    key,value pairs in the JSON filter. For example:
    { "foo": 1, "bar": "test"} will create two outputs named
    "foo" and "bar" with integer and string values respectively.
    The outputs may also be nested i.e. object or arrays.

    If the input parameter ``'root_object_name'`` is supplied then these outputs
    will be nested within an object taking the name of the value of
    ``'root_object_name'``
    """

    KEY_FILENAME = "filename"
    KEY_SCHEMA = "schema"
    KEY_ROOT_OBJECT_NAME = "root_object_name"

    def __init__(self):
        super().__init__("JsonLoader")

    def _run(self):
        """Load the input `filename`. Create the relevant
        outputs."""
        with open(self.inputs[self.KEY_FILENAME]) as file:
            # if root_object_filename put the contents in there
            if self.inputs.get(self.KEY_ROOT_OBJECT_NAME, None) is not None:
                self.outputs[self.inputs[self.KEY_ROOT_OBJECT_NAME]] = json.load(file)
            else:
                self.outputs = json.load(file)

    def _validate_inputs(self):
        """There must be an input named `filename`.
        It must end in .json. It must
        point to a existing file.
        If 'schema' is defined, it must exist in the schema database and
        the input JSON must be verified by the schema."""

        if self.inputs.get(self.KEY_FILENAME, None) is None:
            raise FilterInputValidationError(
                "JsonLoader filter requires a `filename` input"
            )
        if not isinstance(self.inputs[self.KEY_FILENAME], str):
            raise FilterInputValidationError(
                "JsonLoader filter `filename` input must be a string"
            )
        if not self.inputs[self.KEY_FILENAME].endswith(".json"):
            raise FilterInputValidationError(
                "JsonLoader filter `filename` must be a .json file"
            )

        if not (
            os.path.exists(self.inputs[self.KEY_FILENAME])
            and os.path.isfile(self.inputs[self.KEY_FILENAME])
        ):
            raise FilterInputValidationError(
                f"{self.inputs[self.KEY_FILENAME]} does not exist"
            )

        if "schema" in self.inputs:
            with open(self.inputs[self.KEY_FILENAME]) as file:
                try:
                    jsonschema.validate(json.load(file), self.inputs["schema"])
                except jsonschema.ValidationError as error:
                    raise FilterInputValidationError(error) from error

        if self.inputs.get(self.KEY_ROOT_OBJECT_NAME, None) is not None:
            if not isinstance(self.inputs[self.KEY_ROOT_OBJECT_NAME], str):
                raise FilterInputValidationError(
                    "JsonLoader filter 'root_object_name' must be a string"
                )
