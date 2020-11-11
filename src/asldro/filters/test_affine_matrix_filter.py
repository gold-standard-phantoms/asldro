""" Affine Matrix Filter tests """
# pylint: disable=duplicate-code

from copy import deepcopy
import pytest
import numpy as np
import numpy.testing
from asldro.filters.basefilter import BaseFilter, FilterInputValidationError
from asldro.filters.affine_matrix_filter import AffineMatrixFilter

# input validation dictionary, [0] in each tuple passes, after that should fail validation
INPUT_VALIDATION_DICTIONARY = {
    "rotation": (
        (180.0, -180.0, 0.0),
        1.0,
        (181, 300, 234.2),
        (0.0, 0.0, 0.0, 0.0),
        "str",
    ),
    "rotation_origin": ((1.0, 2.0, 3.0), 1.0, (int(1), int(2), int(3)), "str"),
    "translation": (
        (1.0, 2.0, 3.0),
        1.0,
        (int(1), int(2), int(3)),
        (0.0, 0.0, 0.0, 0.0),
        "str",
    ),
    "scale": (
        (1.0, 2.0, 3.0),
        1.0,
        (int(1), int(2), int(3)),
        (0.0, 0.0, 0.0, 0.0),
        "str",
    ),
    "affine": (np.eye(4), np.eye(5), 0, (0.0, 0.0, 0.0, 0.0), "str"),
    "affine_last": (np.eye(4), np.eye(5), 0, (0.0, 0.0, 0.0, 0.0), "str"),
}

# mock data: rotation, rotation_offset, translation, scale, expected
MOCK_DATA = (
    (
        (0.0, 90.0, 0.0),  # rotate 90 degrees about y
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        (1.0, 0.0, 0.0, 1.0),
        (0.0, 0.0, -1.0, 1.0),
    ),
    (
        (0.0, 180.0, 0.0),  # rotate 180 degrees about y
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        (1.0, 0.0, 0.0, 1.0),
        (-1.0, 0.0, 0.0, 1.0),
    ),
    (
        (0.0, 0.0, 90.0),  # rotate 90 degrees about z
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        (1.0, 0.0, 0.0, 1.0),
        (0.0, 1.0, 0.0, 1.0),
    ),
    (
        (0.0, 0.0, 90.0),  # rotate 90 degrees about z
        (-1.0, 0.0, 0.0),  # rotation offset at (-1,0,0)
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        (1.0, 0.0, 0.0, 1.0),
        (-1.0, 2.0, 0.0, 1.0),
    ),
    (
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),  # translate by (1,0,0)
        (1.0, 1.0, 1.0),
        (1.0, 0.0, 0.0, 1.0),
        (2.0, 1.0, 1.0, 1.0),
    ),
    (
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (10.0, 5.0, 2.5),  # scale by (10,5,2.5)
        (1.0, 1.0, 1.0, 1.0),
        (10.0, 5.0, 2.5, 1.0),
    ),
    (
        (0.0, 0.0, 90.0),  # rotate about z by 90 degrees
        (5.0, 0.0, 0.0),  # rotation origin at (5, 0, 0)
        (0.0, 5.0, 0.0),  # translate by (0, 5, 0)
        (2.0, 2.0, 2.0),  # scale isotropically factor 2
        (1.0, 1.0, 1.0, 1.0),
        (8.0, 2.0, 2.0, 1.0),
    ),
)


def add_multiple_inputs_to_filter(input_filter: BaseFilter, input_data: dict):
    """ Adds the data held within the input_data dictionary to the filter's inputs """
    for key in input_data:
        input_filter.add_input(key, input_data[key])

    return input_filter


@pytest.mark.parametrize("validation_data", [INPUT_VALIDATION_DICTIONARY])
def test_affine_matrix_filter_validate_inputs(validation_data: dict):
    """Check a FilterInputValidationError is raised when the
    inputs to the AffineMatrixFilter are incorrect or missing
    """
    # Check with all data that should pass
    affine_filter = AffineMatrixFilter()
    test_data = deepcopy(validation_data)

    for data_key in test_data:
        affine_filter.add_input(data_key, test_data[data_key][0])

    # should run with no issues
    affine_filter.run()

    for inputs_key in validation_data:
        affine_filter = AffineMatrixFilter()
        test_data = deepcopy(validation_data)

        # remove key
        test_data.pop(inputs_key)
        for data_key in test_data:
            affine_filter.add_input(data_key, test_data[data_key][0])
        # Key not defined - should still pass as all keys are optional
        affine_filter.run()

        # Try data that should fail
        for test_value in validation_data[inputs_key][1:]:
            affine_filter = AffineMatrixFilter()
            for data_key in test_data:
                affine_filter.add_input(data_key, test_data[data_key][0])
            affine_filter.add_input(inputs_key, test_value)
            with pytest.raises(FilterInputValidationError):
                affine_filter.run()


def test_affine_matrix_filter_default_data():
    """Test the AffineMatrixFilter with default data and check the output"""
    # filter with no inputs, should produce a 4x4 identity matrix
    affine_filter = AffineMatrixFilter()
    # add an unused input to set needs_run to True within the filter
    affine_filter.add_input("test", None)
    affine_filter.run()
    numpy.testing.assert_array_equal(affine_filter.outputs["affine"], np.eye(4))

    affine_filter = AffineMatrixFilter()
    params = {
        "rotation": (0.0, 0.0, 0.0),
        "rotation_origin": (0.0, 0.0, 0.0),
        "translation": (0.0, 0.0, 0.0),
        "scale": (1.0, 1.0, 1.0),
        "affine": np.eye(4),
        "affine_last": np.eye(4),
    }
    for key in params:
        affine_filter.add_input(key, params[key])
    affine_filter.run()
    numpy.testing.assert_array_equal(affine_filter.outputs["affine"], np.eye(4))

    # test that with default parameters, the input affine is preserved
    affine_filter = AffineMatrixFilter()
    params = {
        "affine": np.array(
            (
                (1, 2, 3, 4),
                (5, 6, 7, 8),
                (9, 10, 11, 12),
                (0, 0, 0, 1),
            )
        ),
    }
    for key in params:
        affine_filter.add_input(key, params[key])
    affine_filter.run()
    numpy.testing.assert_array_equal(affine_filter.outputs["affine"], params["affine"])

    # test that with default parameters, 'affine_last' is preserved
    affine_filter = AffineMatrixFilter()
    params = {
        "affine_last": np.array(
            (
                (1, 2, 3, 4),
                (5, 6, 7, 8),
                (9, 10, 11, 12),
                (0, 0, 0, 1),
            )
        ),
    }
    for key in params:
        affine_filter.add_input(key, params[key])
    affine_filter.run()
    numpy.testing.assert_array_equal(
        affine_filter.outputs["affine"], params["affine_last"]
    )

    # Check that the 'affine_inverse' is the inverse of 'affine'
    numpy.testing.assert_array_equal(
        affine_filter.outputs["affine_inverse"],
        np.linalg.inv(affine_filter.outputs["affine"]),
    )


@pytest.mark.parametrize(
    "rotation, rotation_origin, translation, scale, vector, expected", MOCK_DATA
)
def test_affine_matrix_filter_mock_data(
    rotation: float,
    rotation_origin: float,
    translation: float,
    scale: float,
    vector: float,
    expected: float,
):
    """Tests the AffineMatrixFilter by computing the affine matrix, transforming
    a supplied vector and comparing with a supplied expected vector.

    :param rotation: Tuple of floats describing rotation angles about x, y, z
    :type rotation: float
    :param rotation_offset: Tuple of floats describing the coordinates of the rotation origin
    :type rotation_offset: float
    :param translation: Tuple of floats describing a translation vector
    :type translation: float
    :param scale: Tuple of floats describing scaling along x, y, z
    :type scale: float
    :param expected: Tuple of floats describing the expected value
    :type expected: float
    """

    affine_filter = AffineMatrixFilter()
    affine_filter.add_input("rotation", rotation)
    affine_filter.add_input("rotation_origin", rotation_origin)
    affine_filter.add_input("translation", translation)
    affine_filter.add_input("scale", scale)
    affine_filter.run()

    # transformed vector and expected should be equal to 15 decimal places
    numpy.testing.assert_array_almost_equal(
        affine_filter.outputs["affine"] @ vector, expected, 15
    )
    numpy.testing.assert_array_equal(
        affine_filter.outputs["affine_inverse"],
        np.linalg.inv(affine_filter.outputs["affine"]),
    )
