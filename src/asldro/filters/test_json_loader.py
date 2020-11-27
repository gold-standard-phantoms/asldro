""" JsonLoaderFilter tests """
# pylint: disable=duplicate-code
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from asldro.filters.basefilter import FilterInputValidationError
from asldro.filters.json_loader import JsonLoaderFilter


def test_json_loader_input_validation_no_input():
    """Test all of the JsonLoader input validation -
    No input filename (but some input so the filter will run)"""

    json_loader_filter = JsonLoaderFilter()
    json_loader_filter.add_input("dummy", None)
    with pytest.raises(FilterInputValidationError):
        json_loader_filter.run()


def test_json_loader_input_validation_non_string_input():
    """Test all of the JsonLoader input validation -
    Non-string filename"""

    json_loader_filter = JsonLoaderFilter()

    json_loader_filter.add_input("filename", 1)
    with pytest.raises(FilterInputValidationError):
        json_loader_filter.run()


def test_json_loader_input_validation_bad_json_filename():
    """Test all of the JsonLoader input validation -
    Bad JSON filename"""

    json_loader_filter = JsonLoaderFilter()

    with TemporaryDirectory() as temp_dir:
        temp_file = os.path.join(temp_dir, "file.txt")
        Path(temp_file).touch()
        json_loader_filter.add_input("filename", temp_file)
        with pytest.raises(FilterInputValidationError):
            json_loader_filter.run()


def test_json_loader_input_validation_missing_json_file():
    """Test all of the JsonLoader input validation -
    Missing JSON file"""

    json_loader_filter = JsonLoaderFilter()

    with TemporaryDirectory() as temp_dir:
        # Missing JSON file
        temp_file = os.path.join(temp_dir, "file.json")
        json_loader_filter.add_input("filename", temp_file)
        with pytest.raises(FilterInputValidationError):
            json_loader_filter.run()


def test_json_loader_input_validation_schema_mismatch():
    """Test the JsonLoader input validation -
    JSON doesn't match schema"""

    json_loader_filter = JsonLoaderFilter()

    with TemporaryDirectory() as temp_dir:
        # JSON file (quantities mispelt)
        input_dict = {
            "numbers": [1, "not_a_number", 3],
            "string": "foobar",
        }
        temp_file = os.path.join(temp_dir, "file.json")
        with open(temp_file, "w") as file:
            json.dump(input_dict, file)

        json_loader_filter.add_input("filename", temp_file)
        json_loader_filter.add_input(
            "schema",
            {
                "type": "object",
                "required": ["numbers", "string"],
                "properties": {
                    "numbers": {"type": "array", "items": {"type": "number"}},
                    "string": {"type": "string"},
                },
            },
        )
        with pytest.raises(FilterInputValidationError):
            json_loader_filter.run()  # JSON not valid against schema


def test_json_loader_input_validation_correct_functionality():
    """Test the JsonLoader input validation -
    Correct functionality"""

    json_loader_filter = JsonLoaderFilter()

    with TemporaryDirectory() as temp_dir:
        # JSON file (quantities mispelt)
        input_dict = {
            "numbers": [1, 2, 3],
            "string": "foobar",
        }
        temp_file = os.path.join(temp_dir, "file.json")
        with open(temp_file, "w") as file:
            json.dump(input_dict, file)

        json_loader_filter.add_input("filename", temp_file)
        json_loader_filter.add_input(
            "schema",
            {
                "type": "object",
                "required": ["numbers", "string"],
                "properties": {
                    "numbers": {"type": "array", "items": {"type": "number"}},
                    "string": {"type": "string"},
                },
            },
        )
        json_loader_filter.run()
        assert json_loader_filter.outputs["numbers"] == [
            1,
            2,
            3,
        ]
        assert json_loader_filter.outputs["string"] == "foobar"
