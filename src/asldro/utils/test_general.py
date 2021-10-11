""" general.py tests """
import pytest
from asldro.utils.general import map_dict, generate_random_numbers
import numpy as np
from numpy.random import default_rng
import numpy.testing


@pytest.fixture(name="input_dict")
def fixture_input_dict():
    """A test dictionary"""
    return {
        "one": "two",
        "three": "four",
        "five": "six",
        "seven": "eight",
        "nine": "ten",
    }


def test_map_dict(input_dict: dict):
    """Perform a simple dictionary mapping"""
    assert map_dict(
        input_dict=input_dict, io_map={"one": "one_hundred", "five": "five_hundred"}
    ) == {"one_hundred": "two", "five_hundred": "six"}


def test_map_dict_raises_keyerror(input_dict: dict):
    """Perform a simple dictionary mapping with a missing input dictionary key.
    Check a KeyError is raised"""
    with pytest.raises(KeyError):
        _ = (
            map_dict(
                input_dict=input_dict,
                io_map={"doesnotexist": "one_hundred", "five": "five_hundred"},
            )
            == {"one_hundred": "two", "five_hundred": "six"}
        )


def test_map_dict_with_optional(input_dict: dict):
    """Perform a simple dictionary mapping with a missing input dictionary key,
    and optional flag set True. Check a KeyError is not raised and the correct
    output is created, excluding the io_map which does not exist."""
    assert (
        map_dict(
            input_dict=input_dict,
            io_map={
                "doesnotexist": "one_hundred",
                "five": "five_hundred",
                "nine": "nine_hundred",
            },
            io_map_optional=True,
        )
        == {"five_hundred": "six", "nine_hundred": "ten"}
    )


def test_generate_random_numbers():
    """Checks that generate_random_numbers returns correct values"""
    seed = 12345
    shape_1d = (10,)
    shape_2d = (3, 6)
    shape_3d = (7, 4, 9)

    # test normal distributions
    spec = {"distribution": "gaussian", "mean": 1000.0, "sd": 10.0}
    for shape in [None, shape_1d, shape_2d, shape_3d]:
        rg = default_rng(seed=seed)
        x = rg.normal(1000, 10.0, size=shape)
        y = generate_random_numbers(spec, shape=shape, rng=seed)
        numpy.testing.assert_equal(x, y)

    # test uniform distributions
    spec = {"distribution": "uniform", "max": 150.0, "min": 50.0}
    for shape in [None, shape_1d, shape_2d, shape_3d]:
        rg = default_rng(seed=seed)
        x = rg.uniform(50.0, 150.0, size=shape)
        y = generate_random_numbers(spec, shape=shape, rng=seed)
        numpy.testing.assert_equal(x, y)

    # check that if no specification is supplied then an array of zeros is
    # generated
    # test normal distributions
    spec = {}
    for shape in [None, shape_1d, shape_2d, shape_3d]:
        x = np.zeros(shape)
        y = generate_random_numbers(spec, shape=shape, rng=seed)
        numpy.testing.assert_equal(x, y)

    # check that errors occur if the specification keywords are missing
    spec = {"distribution": "gaussian"}
    with pytest.raises(KeyError):
        generate_random_numbers(spec)

    spec = {"distribution": "uniform"}
    with pytest.raises(KeyError):
        generate_random_numbers(spec)
