""" general.py tests """
import pytest
from asldro.utils.general import map_dict


@pytest.fixture(name="input_dict")
def fixture_input_dict():
    """ A test dictionary """
    return {
        "one": "two",
        "three": "four",
        "five": "six",
        "seven": "eight",
        "nine": "ten",
    }


def test_map_dict(input_dict: dict):
    """ Perform a simple dictionary mapping """
    assert map_dict(
        input_dict=input_dict, io_map={"one": "one_hundred", "five": "five_hundred"}
    ) == {"one_hundred": "two", "five_hundred": "six"}


def test_map_dict_raises_keyerror(input_dict: dict):
    """ Perform a simple dictionary mapping with a missing input dictionary key.
    Check a KeyError is raised """
    with pytest.raises(KeyError):
        _ = map_dict(
            input_dict=input_dict,
            io_map={"doesnotexist": "one_hundred", "five": "five_hundred"},
        ) == {"one_hundred": "two", "five_hundred": "six"}


def test_map_dict_with_optional(input_dict: dict):
    """ Perform a simple dictionary mapping with a missing input dictionary key,
    and optional flag set True. Check a KeyError is not raised and the correct
    output is created, excluding the io_map which does not exist. """
    assert map_dict(
        input_dict=input_dict,
        io_map={
            "doesnotexist": "one_hundred",
            "five": "five_hundred",
            "nine": "nine_hundred",
        },
        io_map_optional=True,
    ) == {"five_hundred": "six", "nine_hundred": "ten"}
