""" General utilities """
import os
from typing import Any, Mapping, Dict
import numpy as np
from numpy.random import default_rng


def map_dict(
    input_dict: Mapping[str, Any],
    io_map: Mapping[str, str],
    io_map_optional: bool = False,
) -> Dict[str, str]:
    """
    Maps a dictionary onto a new dictionary by changing some/all of
    the keys.

    :param input_dict: The input dictionary
    :param io_map: The dictionary used to perform the mapping. All keys
      and values must be strings. For example:

        .. code-block:: python

            {
                "one": "two",
                "three": "four"
            }

      will map inputs keys of "one" to "two" AND "three" to "four".
    :param io_map_optional: If this is False, a KeyError will be raised
      if the keys in the io_map are not found in the input_dict.
    :raises KeyError: if keys required in the mapping are not found in the input_dict
    :return: the remapped dictionary
    """
    # Will raise KeyError if key from io_map is missing in input_dict but
    # only if io_map_optional is False
    return {
        map_to: input_dict[map_from]
        for map_from, map_to in io_map.items()
        if (not io_map_optional) or map_from in input_dict
    }


def splitext(path):
    """The normal os.path.splitext treats path/example.tar.gz
    as having a filepath of path/example.tar with a .gz
    extension - this fixes it"""
    for ext in [".tar.gz", ".tar.bz2", ".nii.gz"]:
        if path.lower().endswith(ext.lower()):
            return path[: -len(ext)], path[-len(ext) :]
    return os.path.splitext(path)


def generate_random_numbers(specification: dict, shape=None, rng=None) -> np.ndarray:
    """Generates a set of numbers according to the prescribed distributions,
    returning them as a list

    :param specification: The distribution to use to generate the parameters:

      * 'gaussian' - normal distribution with mean and standard deviation
      * 'uniform' - rectangular distribution with min and max values
      * 'mean' - mean value of the distribution (gaussian only)
      * 'sd' - standard deviation of the distribution (gaussian only)
      * 'min' - minimum value of the distribution (uniform only)
      * 'max' - maximum value of the distribution (uniform only)

    :type specification: dict
    :param shape: length of the list to return
    :type shape: int or tuple of ints
    :param rng: The random number generator to use, defaults to None
    :type rng: [type], optional
    :return: List of the generated numbers
    :rtype: list
    """
    rng = default_rng(rng)
    if specification.get("distribution") is not None:
        distribution = specification["distribution"]
        if distribution == "gaussian":
            out: np.ndarray = rng.normal(
                specification["mean"], specification["sd"], shape
            )
        elif distribution == "uniform":
            out: np.ndarray = rng.uniform(
                specification["min"], specification["max"], shape
            )
    else:
        out = np.zeros(shape)

    return out
